import os
import sys
import time
import tensorflow.compat.v1 as tf
import numpy as np

_NUM_CLASSES = 10
_MODEL_DIR = "model_name"
_NUM_CHANNELS = 1
_IMG_SIZE = 28
_LEARNING_RATE = 0.05
_NUM_EPOCHS = 1
_BATCH_SIZE = 100


class Model(object):

    def __call__(self, inputs):
        net = tf.layers.conv2d(inputs, 32, [5, 5],
                               activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], 2,
                                      name='pool1')
        net = tf.layers.conv2d(net, 64, [5, 5],
                               activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], 2,
                                      name='pool2')
        net = tf.layers.flatten(net)

        logits = tf.layers.dense(net, _NUM_CLASSES,
                                 activation=None, name='fc1')
        return logits

def model_fn(features, labels, mode):
    model = Model()
    global_step = tf.train.get_global_step()

    images = tf.reshape(features, [-1, _IMG_SIZE, _IMG_SIZE,
                                   _NUM_CHANNELS])

    logits = model(images)
    predicted_logit = tf.argmax(input=logits, axis=1,
                                output_type=tf.int32)
    probabilities = tf.nn.softmax(logits)
    # PREDICT
    predictions = {
        "probabilities": probabilities
    }
    aa = tf.estimator.export.ServingInputReceiver(logits, predictions)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits, scope='loss')

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_logit, name='acc')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, scope='loss')
        tf.summary.scalar('loss', cross_entropy)
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_logit, name='acc')
    tf.summary.scalar('accuracy', accuracy[1])  # EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            eval_metric_ops={'accuracy/accuracy': accuracy},
            evaluation_hooks=None)

    # Create a SGR optimizer
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=_LEARNING_RATE)
    train_op = optimizer.minimize(
        cross_entropy, global_step=global_step)

    # Create a hook to print acc, loss & global step every 100 iter.
    train_hook_list = []
    train_tensors_log = {'accuracy': accuracy[1],
                         'loss': cross_entropy,
                         'global_step': global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=100))

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            train_op=train_op,
            training_hooks=train_hook_list)

def one_hot(label):
    num = np.shape(label)[0]
    labels = np.zeros(shape=(num, _NUM_CLASSES))
    for i in range(num):
        labels[i, label[i]] = 1
    return labels

# def export(est, export_dir, input_image_size=None):
#     """Export graph to SavedModel and TensorFlow Lite.
#
#     Args:
#       est: estimator instance.
#       export_dir: string, exporting directory.
#       input_image_size: int, input image size.
#
#     Raises:
#       ValueError: the export directory path is not specified.
#     """
#     if not export_dir:
#         raise ValueError('The export directory path is not specified.')
#
#     batch_size = None  # Use fixed batch size for condconv.
#
#     print('Starting to export model.')
#     image_serving_input_fn = imagenet_input.build_image_serving_input_fn(
#         input_image_size, batch_size=batch_size)
#     est.export_saved_model(
#         export_dir_base=export_dir,
#         serving_input_receiver_fn=image_serving_input_fn)



def MNIST_classifier_estimator():
    # Load training and eval data
    (train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / 255.0
    eval_data = eval_data / 255.0

    # train_labels = one_hot(train_labels)
    # eval_labels = one_hot(eval_labels)

    train_labels = np.ndarray.astype(train_labels, dtype='int32')
    eval_labels = np.ndarray.astype(eval_labels, dtype='int32')
    # Create a input function to train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=_BATCH_SIZE,
        num_epochs=1,
        shuffle=True)  # Create a input function to eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        y=eval_labels,
        batch_size=_BATCH_SIZE,
        num_epochs=1,
        shuffle=False)  # Create a estimator with model_fn
    image_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                              model_dir=_MODEL_DIR)  # Finally, train and evaluate the model after each epoch

    # from tensorflow.python.tools import freeze_graph

    for _ in range(_NUM_EPOCHS):
        image_classifier.train(input_fn=train_input_fn)

    def cnn_serving_input_receiver_fn():
        inputs = {'Input': tf.placeholder(tf.float32, [None, 28, 28])}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    print('saving mdeol...')
    image_classifier.export_saved_model(os.path.join('mnist_model', 'serving'),
                                        cnn_serving_input_receiver_fn)
    print('model is saved!:)')

    # MODEL_NAME = 'model_name'
    # input_graph_path = 'model_name/graph.pbtxt'
    # checkpoint_path = 'model_name/checkpoint'
    # input_saver_def_path = ''
    # input_binary = False
    # output_node_names = "O"
    # restore_op_name = "save/restore_all"
    # filename_tensor_name = "save/Const:0"
    # output_frozen_graph_name = 'frozen_' + MODEL_NAME + '.pb'
    # output_optimized_graph_name = 'optimized_' + MODEL_NAME + '.pb'
    # clear_devices = True
    #
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
    #                           input_binary, checkpoint_path, output_node_names,
    #                           restore_op_name, filename_tensor_name,
    #                           output_frozen_graph_name, clear_devices, "")


def main():
    MNIST_classifier_estimator()


if __name__ == '__main__':
    main()
