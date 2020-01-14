import efficientnet_builder
from tensorflow.keras.layers import Input

input_size = 224
input_image = Input(shape=(input_size, input_size, 3))

features, endpoints = efficientnet_builder.build_model_base(input_image, 'efficientnet-b0', training=False)
