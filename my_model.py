from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import condconv_layers
import utils

import functools
import numpy as np
import math
import six
from six.moves import xrange

from absl import logging
import collections

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'local_pooling', 'condconv_num_experts', 'clip_projection_output',
    'blocks_args'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    logging.info('round_filter input=%s output=%s', orig_f, new_filters)
    return int(new_filters)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

    Attributes:
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.

        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._batch_norm = global_params.batch_norm
        self._condconv_num_experts = global_params.condconv_num_experts
        self._data_format = global_params.data_format
        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (
                global_params.use_se and self._block_args.se_ratio is not None and
                0 < self._block_args.se_ratio <= 1)

        self._clip_projection_output = global_params.clip_projection_output

        self.endpoints = None

        self.conv_cls = tf.layers.Conv2D
        self.depthwise_conv_cls = utils.DepthwiseConv2D
        if self._block_args.condconv:
            self.conv_cls = functools.partial(
                condconv_layers.CondConv2D, num_experts=self._condconv_num_experts)
            self.depthwise_conv_cls = functools.partial(
                condconv_layers.DepthwiseCondConv2D,
                num_experts=self._condconv_num_experts)

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        if self._block_args.super_pixel == 1:
            self._superpixel = tf.layers.Conv2D(
                self._block_args.input_filters,
                kernel_size=[2, 2],
                strides=[2, 2],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False)
            self._bnsp = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        if self._block_args.condconv:
            # Add the example-dependent routing function
            self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
                data_format=self._data_format)
            self._routing_fn = tf.layers.Dense(
                self._condconv_num_experts, activation=tf.nn.sigmoid)

        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        # Fused expansion phase. Called if using fused convolutions.
        self._fused_conv = self.conv_cls(
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        self._expand_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn0 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = self.depthwise_conv_cls(
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)

        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        if self._has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = tf.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=True)
            self._se_expand = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=True)

        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn2 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.

        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        logging.info('Built Squeeze and Excitation with tensor shape: %s',
                     (se_tensor.shape))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        logging.info('Block input: %s shape: %s', inputs.name, inputs.shape)
        logging.info('Block input depth: %s output depth: %s',
                     self._block_args.input_filters,
                     self._block_args.output_filters)

        x = inputs

        fused_conv_fn = self._fused_conv
        expand_conv_fn = self._expand_conv
        depthwise_conv_fn = self._depthwise_conv
        project_conv_fn = self._project_conv

        if self._block_args.condconv:
            pooled_inputs = self._avg_pooling(inputs)
            routing_weights = self._routing_fn(pooled_inputs)
            # Capture routing weights as additional input to CondConv layers
            fused_conv_fn = functools.partial(
                self._fused_conv, routing_weights=routing_weights)
            expand_conv_fn = functools.partial(
                self._expand_conv, routing_weights=routing_weights)
            depthwise_conv_fn = functools.partial(
                self._depthwise_conv, routing_weights=routing_weights)
            project_conv_fn = functools.partial(
                self._project_conv, routing_weights=routing_weights)

        # creates conv 2x2 kernel
        if self._block_args.super_pixel == 1:
            with tf.variable_scope('super_pixel'):
                x = self._relu_fn(
                    self._bnsp(self._superpixel(x), training=training))
            logging.info(
                'Block start with SuperPixel: %s shape: %s', x.name, x.shape)

        if self._block_args.fused_conv:
            # If use fused mbconv, skip expansion and use regular conv.
            x = self._relu_fn(self._bn1(fused_conv_fn(x), training=training))
            logging.info('Conv2D: %s shape: %s', x.name, x.shape)
        else:
            # Otherwise, first apply expansion and then apply depthwise conv.
            if self._block_args.expand_ratio != 1:
                x = self._relu_fn(self._bn0(expand_conv_fn(x), training=training))
                logging.info('Expand: %s shape: %s', x.name, x.shape)

            x = self._relu_fn(self._bn1(depthwise_conv_fn(x), training=training))
            logging.info('DWConv: %s shape: %s', x.name, x.shape)

        if self._has_se:
            with tf.variable_scope('se'):
                x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(project_conv_fn(x), training=training)
        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if self._clip_projection_output:
            x = tf.clip_by_value(x, -6, 6)
        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # Apply only if skip connection presents.
                if survival_prob:
                    x = utils.drop_connect(x, training, survival_prob)
                x = tf.add(x, inputs)
        logging.info('Project: %s shape: %s', x.name, x.shape)
        return x


class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
            self._bn0 = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        logging.info('Block input: %s shape: %s', inputs.name, inputs.shape)
        if self._block_args.expand_ratio != 1:
            x = self._relu_fn(self._bn0(self._expand_conv(inputs), training=training))
        else:
            x = inputs
        logging.info('Expand: %s shape: %s', x.name, x.shape)

        self.endpoints = {'expansion_output': x}

        x = self._bn1(self._project_conv(x), training=training)
        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if self._clip_projection_output:
            x = tf.clip_by_value(x, -6, 6)

        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # Apply only if skip connection presents.
                if survival_prob:
                    x = utils.drop_connect(x, training, survival_prob)
                x = tf.add(x, inputs)
        logging.info('Project: %s shape: %s', x.name, x.shape)
        return x


class EfficientNet(tf.keras.Model):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet, self).__init__()

        self._blocks_args = blocks_args
        self._global_params = global_params
        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._batch_norm = global_params.batch_norm

        self.endpoints = None

        self._build()

    def _get_conv_block(self, conv_type):
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[0] 

    def _build(self):
        self._blocks = []
        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon

        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            channel_axis = -1
            self._spatial_dims = [1, 2]

        # Stem part.
        self._conv_stem = tf.layers.Conv2D(
            filters=round_filters(32, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._global_params.data_format,
            use_bias=False)

        self._bn0 = self._batch_norm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            assert block_args.super_pixel in [0, 1, 2]
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(block_args.input_filters,
                                          self._global_params)
            output_filters = round_filters(block_args.output_filters,
                                           self._global_params)
            kernel_size = block_args.kernel_size
            block_args = block_args._replace(
                input_filters=input_filters,
                output_filters=output_filters,
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))

            # The first block needs to take care of stride and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            if not block_args.super_pixel:  # no super_pixel at all
                self._blocks.append(conv_block(block_args, self._global_params))
            else:
                # if superpixel, adjust filters, kernels, and strides.
                depth_factor = int(4 / block_args.strides[0] / block_args.strides[1])
                block_args = block_args._replace(
                    input_filters=block_args.input_filters * depth_factor,
                    output_filters=block_args.output_filters * depth_factor,
                    kernel_size=((block_args.kernel_size + 1) // 2 if depth_factor > 1
                                 else block_args.kernel_size))
                # if the first block has stride-2 and super_pixel trandformation
                if (block_args.strides[0] == 2 and block_args.strides[1] == 2):
                    block_args = block_args._replace(strides=[1, 1])
                    self._blocks.append(conv_block(block_args, self._global_params))
                    block_args = block_args._replace(  # sp stops at stride-2
                        super_pixel=0,
                        input_filters=input_filters,
                        output_filters=output_filters,
                        kernel_size=kernel_size)
                elif block_args.super_pixel == 1:
                    self._blocks.append(conv_block(block_args, self._global_params))
                    block_args = block_args._replace(super_pixel=2)
                else:
                    self._blocks.append(conv_block(block_args, self._global_params))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                self._blocks.append(conv_block(block_args, self._global_params))

        # Head part.
        self._conv_head = tf.layers.Conv2D(
            filters=round_filters(128, self._global_params),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = self._batch_norm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self._global_params.data_format)
        if self._global_params.num_classes:
            self._fc = tf.layers.Dense(
                self._global_params.num_classes,
                kernel_initializer=dense_kernel_initializer)
        else:
            self._fc = None

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True, features_only=False, pooled_features_only=False):
        reduction_idx = 0
        self.endpoints = {}

        # Calls step part
        with tf.variable_scope('stem'):
            x = self._conv_stem(inputs)
            x = self._bn0(x, training=training)
            x = self._relu_fn(x)
        self.endpoints['stem'] = x

        # Calls blocks
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if (block.block_args().super_pixel == 1 and idx == 0):
                reduction_idx += 1
                self.endpoints['reduction_%s' % reduction_idx] = x

            elif ((idx == len(self._blocks) - 1) or
                  self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            with tf.variable_scope('blocks_%s' % idx):
                survival_prob = self._global_params.survival_prob
                if survival_prob:
                    drop_rate = 1.0 - survival_prob
                    survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)

                x = block.call(x, training=training, survival_prob=survival_prob)
                self.endpoints['block_%s' % idx] = x
                if is_reduction:
                    self.endpoints['reduction_%s' % reduction_idx] = x
                if block.endpoints:
                    for k, v in six.iteritems(block.endpoints):
                        self.endpoints['block_%s/%s' % (idx, k)] = v
                        if is_reduction:
                            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['features'] = x

        if not features_only:
            with tf.variable_scope('head'):
                x = self._conv_head(x)
                x = self._bn1(x, training=training)
                x = self._relu_fn(x)
                self.endpoints['head_1x1'] = x

                output = x

                output = self._avg_pooling(output)
                self.endpoints['pooled_features'] = output
                output = self._fc(output)
                self.endpoints['head'] = output

            return output


import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    def __init__(self, confige):
        self._config = confige
        self.total_train = 0
        self.total_valid = 0

        self._validation_dir = {}
        self._train_dir = {}

        self._read_data()

    def _read_data(self):
        # downlaod the dataset from url
        url = self._config.URL
        path_to_zip = keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=True)  # todo
        path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

        # creating train and validation folder
        self._train_dir = os.path.join(path, self._config.TRAIN_FOLDER)
        self._validation_dir = os.path.join(path, self._config.VALID_FOLDER)

        train_cls_dir = []
        valid_cls_dir = []

        for label in (self._config.LABELS):
            train_path = os.path.join(self._train_dir, str(label))
            valid_path = os.path.join(self._validation_dir, str(label))
            self.total_train += len(os.listdir(train_path))
            self.total_valid += len(os.listdir(valid_path))
            train_cls_dir.append(train_path)
            valid_cls_dir.append(valid_path)

    def __getitem__(self, item='train'):
        data_generator = ImageDataGenerator(rescale=1. / 255)
        if item == 'train':
            directory = self._train_dir
        elif item == 'valid':
            directory = self._validation_dir
        else:
            raise Exception("{item} argument is not defined [it should be seted to 'train' or 'valid']".format(item))

        train_data_gen = data_generator.flow_from_directory(batch_size=self._config.BATCH_SIZE,
                                                            directory=directory,
                                                            shuffle=True,
                                                            target_size=(self._config.INPUT_SIZE,
                                                                         self._config.INPUT_SIZE),
                                                            class_mode='binary')

        return train_data_gen


def efficientnet_edgetpu_params(model_name):
    """Get efficientnet-edgetpu params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-edgetpu-S': (1.0, 1.0, 150, 0.2),
        'efficientnet-edgetpu-M': (1.0, 1.1, 240, 0.2),
        'efficientnet-edgetpu-L': (1.2, 1.4, 300, 0.3),
    }
    return params_dict[model_name]


import efficientnet_model
import efficientnet_builder


def efficientnet_edgetpu(width_coefficient=None,
                         depth_coefficient=None,
                         dropout_rate=0.2,
                         survival_prob=0.8):
    """Creates an efficientnet-edgetpu model."""
    # blocks_args = [
    #     'r1_k3_s11_e4_i24_o24_c1_noskip',
    #     'r2_k3_s22_e8_i24_o32_c1',
    #     'r4_k3_s22_e8_i32_o48_c1',
    #     'r5_k5_s22_e8_i48_o96',
    #     'r4_k5_s11_e8_i96_o144',
    #     'r2_k5_s22_e8_i144_o192',
    # ]
    blocks_args = [
        'r1_k3_s11_e4_i24_o24_c1_noskip',
        'r2_k3_s22_e8_i24_o32_c1',
        'r4_k3_s22_e8_i32_o48_c1',
    ]
    global_params = efficientnet_model.GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        survival_prob=survival_prob,
        data_format='channels_last',
        num_classes=2,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.relu,
        # The default is TPU-specific batch norm.
        # The alternative is tf.layers.BatchNormalization.
        batch_norm=utils.TpuBatchNormalization,  # TPU-specific requirement.
        use_se=False)
    decoder = efficientnet_builder.BlockDecoder()
    return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet-edgetpu'):
        width_coefficient, depth_coefficient, _, dropout_rate = (
            efficientnet_edgetpu_params(model_name))
        blocks_args, global_params = efficientnet_edgetpu(width_coefficient,
                                                          depth_coefficient,
                                                          dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    logging.info('global_params= %s', global_params)
    logging.info('blocks_args= %s', blocks_args)
    return blocks_args, global_params


from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, Dropout

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

# FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'
FAKE_DATA_DIR = '/home/mehran/.keras/datasets/cats_and_dogs_filtered/train'

# # flags region
# # region
# flags.DEFINE_bool(
#     'use_tpu', default=False,
#     help=('Use TPU to execute the model for training and evaluation. If'
#           ' --use_tpu=false, will use whatever devices are available to'
#           ' TensorFlow by default (e.g. CPU and GPU)'))
#
# # Cloud TPU Cluster Resolvers
# flags.DEFINE_string(
#     'tpu', default=None,
#     help='The Cloud TPU to use for training. This should be either the name '
#          'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
#
# flags.DEFINE_string(
#     'gcp_project', default=None,
#     help='Project name for the Cloud TPU-enabled project. If not specified, we '
#          'will attempt to automatically detect the GCE project from metadata.')
#
# flags.DEFINE_string(
#     'tpu_zone', default=None,
#     help='GCE zone where the Cloud TPU is located in. If not specified, we '
#          'will attempt to automatically detect the GCE project from metadata.')
#
# # Model specific flags
# flags.DEFINE_string(
#     'data_dir', default=FAKE_DATA_DIR,
#     help=('The directory where the ImageNet input data is stored. Please see'
#           ' the README.md for the expected data format.'))
#
# flags.DEFINE_string(
#     'model_dir', default='Model_DIR2',
#     help=('The directory where the model and training/evaluation summaries are'
#           ' stored.'))
# # set model name here
# flags.DEFINE_string(
#     'model_name',
#     default='efficientnet-edgetpu-S',
#     help=('The model name among existing configurations.'))
#
# flags.DEFINE_string(
#     'mode', default='train_and_eval',
#     help='One of {"train_and_eval", "train", "eval"}.')
#
# flags.DEFINE_string(
#     'augment_name', default=None,
#     help='`string` that is the name of the augmentation method'
#          'to apply to the image. `autoaugment` if AutoAugment is to be used or'
#          '`randaugment` if RandAugment is to be used. If the value is `None` no'
#          'augmentation method will be applied applied. See autoaugment.py for  '
#          'more details.')
#
# flags.DEFINE_integer(
#     'randaug_num_layers', default=2,
#     help='If RandAug is used, what should the number of layers be.'
#          'See autoaugment.py for detailed description.')
#
# flags.DEFINE_integer(
#     'randaug_magnitude', default=10,
#     help='If RandAug is used, what should the magnitude be. '
#          'See autoaugment.py for detailed description.')
#
# flags.DEFINE_integer(
#     'train_steps', default=218949,
#     help=('The number of steps to use for training. Default is 218949 steps'
#           ' which is approximately 350 epochs at batch size 2048. This flag'
#           ' should be adjusted according to the --train_batch_size flag.'))
#
# flags.DEFINE_integer(
#     'input_image_size', default=None,
#     help=('Input image size: it depends on specific model name.'))
#
# flags.DEFINE_integer(
#     'train_batch_size', default=16, help='Batch size for training.')
#
# flags.DEFINE_integer(
#     'eval_batch_size', default=16, help='Batch size for evaluation.')
#
# flags.DEFINE_integer(
#     'num_train_images', default=1281167, help='Size of training data set.')
#
# flags.DEFINE_integer(
#     'num_eval_images', default=50000, help='Size of evaluation data set.')
#
# flags.DEFINE_integer(
#     'steps_per_eval', default=6255,
#     help=('Controls how often evaluation is performed. Since evaluation is'
#           ' fairly expensive, it is advised to evaluate as infrequently as'
#           ' possible (i.e. up to --train_steps, which evaluates the model only'
#           ' after finishing the entire training regime).'))
#
# flags.DEFINE_integer(
#     'eval_timeout',
#     default=None,
#     help='Maximum seconds between checkpoints before evaluation terminates.')
#
# flags.DEFINE_bool(
#     'skip_host_call', default=False,
#     help=('Skip the host_call which is executed every training step. This is'
#           ' generally used for generating training summaries (train loss,'
#           ' learning rate, etc...). When --skip_host_call=false, there could'
#           ' be a performance drop if host_call function is slow and cannot'
#           ' keep up with the TPU-side computation.'))
#
# flags.DEFINE_integer(
#     'iterations_per_loop', default=1251,
#     help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
#           ' If the number of iterations in the loop would exceed the number of'
#           ' train steps, the loop will exit before reaching'
#           ' --iterations_per_loop. The larger this value is, the higher the'
#           ' utilization on the TPU.'))
#
# flags.DEFINE_integer(
#     'num_parallel_calls', default=64,
#     help=('Number of parallel threads in CPU for the input pipeline'))
#
# flags.DEFINE_string(
#     'bigtable_project', None,
#     'The Cloud Bigtable project.  If None, --gcp_project will be used.')
# flags.DEFINE_string(
#     'bigtable_instance', None,
#     'The Cloud Bigtable instance to load data from.')
# flags.DEFINE_string(
#     'bigtable_table', 'imagenet',
#     'The Cloud Bigtable table to load data from.')
# flags.DEFINE_string(
#     'bigtable_train_prefix', 'train_',
#     'The prefix identifying training rows.')
# flags.DEFINE_string(
#     'bigtable_eval_prefix', 'validation_',
#     'The prefix identifying evaluation rows.')
# flags.DEFINE_string(
#     'bigtable_column_family', 'tfexample',
#     'The column family storing TFExamples.')
# flags.DEFINE_string(
#     'bigtable_column_qualifier', 'example',
#     'The column name storing TFExamples.')
#
# flags.DEFINE_string(
#     'data_format', default='channels_last',
#     help=('A flag to override the data format used in the model. The value'
#           ' is either channels_first or channels_last. To run the network on'
#           ' CPU or TPU, channels_last should be used. For GPU, channels_first'
#           ' will improve performance.'))
# flags.DEFINE_integer(
#     'num_label_classes', default=1000, help='Number of classes, at least 2')
#
# flags.DEFINE_float(
#     'batch_norm_momentum',
#     default=None,
#     help=('Batch normalization layer momentum of moving average to override.'))
# flags.DEFINE_float(
#     'batch_norm_epsilon',
#     default=None,
#     help=('Batch normalization layer epsilon to override..'))
#
# flags.DEFINE_bool(
#     'transpose_input', default=True,
#     help='Use TPU double transpose optimization')
#
# flags.DEFINE_bool(
#     'use_bfloat16',
#     default=False,
#     help=('Whether to use bfloat16 as activation for training.'))
#
# flags.DEFINE_string(
#     'export_dir',
#     default=None,
#     help=('The directory where the exported SavedModel will be stored.'))
# flags.DEFINE_bool(
#     'export_to_tpu', default=False,
#     help=('Whether to export additional metagraph with "serve, tpu" tags'
#           ' in addition to "serve" only metagraph.'))
#
# flags.DEFINE_float(
#     'base_learning_rate',
#     default=0.016,
#     help=('Base learning rate when train batch size is 256.'))
#
# flags.DEFINE_float(
#     'momentum', default=0.9,
#     help=('Momentum parameter used in the MomentumOptimizer.'))
#
# flags.DEFINE_float(
#     'moving_average_decay', default=0.9999,
#     help=('Moving average decay rate.'))
#
# flags.DEFINE_float(
#     'weight_decay', default=1e-5,
#     help=('Weight decay coefficiant for l2 regularization.'))
#
# flags.DEFINE_float(
#     'label_smoothing', default=0.1,
#     help=('Label smoothing parameter used in the softmax_cross_entropy'))
#
# flags.DEFINE_float(
#     'dropout_rate', default=None,
#     help=('Dropout rate for the final output layer.'))
#
# flags.DEFINE_float(
#     'survival_prob', default=None,
#     help=('Drop connect rate for the network.'))
#
# flags.DEFINE_float(
#     'mixup_alpha',
#     default=0.0,
#     help=('Alpha parameter for mixup regularization, 0.0 to disable.'))
#
# flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
#                                                  'which the global step information is logged.')
#
# flags.DEFINE_bool(
#     'use_cache', default=False, help=('Enable cache for training input.'))
#
# flags.DEFINE_float(
#     'depth_coefficient', default=None,
#     help=('Depth coefficient for scaling number of layers.'))
#
# flags.DEFINE_float(
#     'width_coefficient', default=None,
#     help=('Width coefficient for scaling channel size.'))
#
# flags.DEFINE_bool(
#     'use_async_checkpointing', default=False, help=('Enable async checkpoint'))
#
# # endregion

def model_fn(features, labels, mode, params):
    """The model_fn to be used with TPUEstimator.

    Args:
      features: `Tensor` of batched images.
      labels: `Tensor` of one hot labels for the data samples
      mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
      params: `dict` of parameters passed to the model from the TPUEstimator,
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `TPUEstimatorSpec` for the model
    """
    if isinstance(features, dict):
        features = features['feature']

    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    has_moving_average_decay = (FLAGS.moving_average_decay > 0)
    # This is essential, if using a keras-derived model.
    tf.keras.backend.set_learning_phase(is_training)
    logging.info('Using open-source implementation.')
    override_params = {}
    if FLAGS.batch_norm_momentum is not None:
        override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
    if FLAGS.batch_norm_epsilon is not None:
        override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
    if FLAGS.dropout_rate is not None:
        override_params['dropout_rate'] = FLAGS.dropout_rate
    if FLAGS.survival_prob is not None:
        override_params['survival_prob'] = FLAGS.survival_prob
    if FLAGS.data_format:
        override_params['data_format'] = FLAGS.data_format
    if FLAGS.num_label_classes:
        override_params['num_classes'] = FLAGS.num_label_classes
    if FLAGS.depth_coefficient:
        override_params['depth_coefficient'] = FLAGS.depth_coefficient
    if FLAGS.width_coefficient:
        override_params['width_coefficient'] = FLAGS.width_coefficient

    def normalize_features(features, mean_rgb, stddev_rgb):
        """Normalize the image given the means and stddevs."""
        features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
        features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
        return features

    def build_model():
        """Build model using the model_name given through the command line."""
        model_builder = None
        if FLAGS.model_name.startswith('efficientnet-edgetpu'):
            model_builder = efficientnet_edgetpu_builder

        normalized_features = normalize_features(features, model_builder.MEAN_RGB,
                                                 model_builder.STDDEV_RGB)
        print(model_builder)
        print(type(model_builder))
        print("ianjm")
        logits, model = model_builder.build_model(
            normalized_features,
            model_name=FLAGS.model_name,
            training=is_training,
            override_params=override_params,
            model_dir=FLAGS.model_dir)
        print(normalized_features)
        print(FLAGS.model_name)
        print(is_training)
        print(override_params)
        print(FLAGS.model_dir)
        # exit()
        return logits

    if params['use_bfloat16']:
        with tf.tpu.bfloat16_scope():
            logits = tf.cast(build_model(), tf.float32)
    else:
        logits = build_model()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, in the model_fn, use params['batch_size'] instead the batch
    # size flags (--train_batch_size or --eval_batch_size).
    batch_size = params['batch_size']  # pylint: disable=unused-variable

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        label_smoothing=FLAGS.label_smoothing)

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    global_step = tf.train.get_global_step()
    if has_moving_average_decay:
        ema = tf.train.ExponentialMovingAverage(
            decay=FLAGS.moving_average_decay, num_updates=global_step)
        ema_vars = utils.get_ema_vars()

    host_call = None
    restore_vars_dict = None
    if is_training:
        # Compute the current epoch and associated learning rate from global_step.
        current_epoch = (
                tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

        scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
        logging.info('base_learning_rate = %f', FLAGS.base_learning_rate)
        learning_rate = utils.build_learning_rate(scaled_lr, global_step,
                                                  params['steps_per_epoch'])
        optimizer = utils.build_optimizer(learning_rate)
        if FLAGS.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tf.tpu.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

        if has_moving_average_decay:
            with tf.control_dependencies([train_op]):
                train_op = ema.apply(ema_vars)

        if not FLAGS.skip_host_call:
            def host_call_fn(gs, lr, ce):
                """Training host call. Creates scalar summaries for training metrics.

                This function is executed on the CPU and should not directly reference
                any Tensors in the rest of the `model_fn`. To pass Tensors from the
                model to the `metric_fn`, provide as part of the `host_call`. See
                https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
                for more information.

                Arguments should match the list of `Tensor` objects passed as the second
                element in the tuple passed to `host_call`.

                Args:
                  gs: `Tensor with shape `[batch]` for the global_step
                  lr: `Tensor` with shape `[batch]` for the learning_rate.
                  ce: `Tensor` with shape `[batch]` for the current_epoch.

                Returns:
                  List of summary ops to run on the CPU host.
                """
                gs = gs[0]
                # Host call fns are executed FLAGS.iterations_per_loop times after one
                # TPU loop is finished, setting max_queue value to the same as number of
                # iterations will make the summary writer only flush the data to storage
                # once per loop.
                with tf2.summary.create_file_writer(
                        FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
                    with tf2.summary.record_if(True):
                        tf2.summary.scalar('learning_rate', lr[0], step=gs)
                        tf2.summary.scalar('current_epoch', ce[0], step=gs)

                        return tf.summary.all_v2_summary_ops()

            # To log the loss, current learning rate, and epoch for Tensorboard, the
            # summary op needs to be run on the host CPU via host_call. host_call
            # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
            # dimension. These Tensors are implicitly concatenated to
            # [params['batch_size']].
            gs_t = tf.reshape(global_step, [1])
            lr_t = tf.reshape(learning_rate, [1])
            ce_t = tf.reshape(current_epoch, [1])

            host_call = (host_call_fn, [gs_t, lr_t, ce_t])

    else:
        train_op = None
        if has_moving_average_decay:
            # Load moving average variables for eval.
            restore_vars_dict = ema.variables_to_restore(ema_vars)

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, logits):
            """Evaluation metric function. Evaluates accuracy.

            This function is executed on the CPU and should not directly reference
            any Tensors in the rest of the `model_fn`. To pass Tensors from the model
            to the `metric_fn`, provide as part of the `eval_metrics`. See
            https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
            for more information.

            Arguments should match the list of `Tensor` objects passed as the second
            element in the tuple passed to `eval_metrics`.

            Args:
              labels: `Tensor` with shape `[batch, num_classes]`.
              logits: `Tensor` with shape `[batch, num_classes]`.

            Returns:
              A dict of the metrics to return from evaluation.
            """
            labels = tf.argmax(labels, axis=1)
            predictions = tf.argmax(logits, axis=1)
            top_1_accuracy = tf.metrics.accuracy(labels, predictions)
            in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
            top_5_accuracy = tf.metrics.mean(in_top_5)

            return {
                'top_1_accuracy': top_1_accuracy,
                'top_5_accuracy': top_5_accuracy,
            }

        eval_metrics = (metric_fn, [labels, logits])

    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    logging.info('number of trainable parameters: %d', num_params)

    def _scaffold_fn():
        saver = tf.train.Saver(restore_vars_dict)
        return tf.train.Scaffold(saver=saver)

    if has_moving_average_decay and not is_training:
        # Only apply scaffold for eval jobs.
        scaffold_fn = _scaffold_fn
    else:
        scaffold_fn = None

    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=eval_metrics,
        scaffold_fn=scaffold_fn)


def main():
    tf.disable_eager_execution()

    model_name = 'efficientnet-edgetpu-S'
    override_params = {'data_format': 'channels_last', 'num_classes': 1}
    blocks_args, global_params = get_model_params(model_name, override_params)
    model = EfficientNet(blocks_args, global_params)
    #
    print(blocks_args)
    print(global_params)
    exit()

    input_size = 150
    input_image = Input(shape=(input_size, input_size, 3))
    features = model(input_image)
    #logits = tf.identity(features, 'logits')

    # =====
    # model = Model(features, input_image)
    # model = model(input_image)
    # model.summary()
    # ======
    adam = optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=False,
                           epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    from configs.config_handler import Config
    path = 'configs/config.json'
    cfg = Config(path=path)

    data_generator = DataLoader(cfg)

    train_data_gen = data_generator['train']
    val_data_gen = data_generator['valid']

    saveddir = os.path.join(cfg.SAVED_FOLDER, cfg.MODEL_NAME)
    print(saveddir)
    if not os.path.exists(saveddir):
        os.makedirs(saveddir)

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=saveddir,
                                                  save_weights_only=False,
                                                  verbose=1, period=1)

    model.fit_generator(
        train_data_gen,
        steps_per_epoch=data_generator.total_train // cfg.BATCH_SIZE,
        epochs=cfg.EPOCHS,
        validation_data=val_data_gen,
        validation_steps=data_generator.total_valid // cfg.BATCH_SIZE, verbose=1,
        callbacks=[cp_callback])


if __name__ == '__main__':
    main()
