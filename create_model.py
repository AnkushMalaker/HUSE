import tensorflow as tf
from tensorflow import keras


# Define Custom layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[input_shape[-1][-1],
                                             self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


def build_model(input_image_encoding_size, input_text_encoding_size, num_classes):
    '''
    Input arguments:    input_image_encoding_size   - dimention of the encoded images 
                        input_text_encoding_size    - dimention of encoded text
                        num_classes                 - number of classes
    Outputs:  Model with classification layer
    '''
    # image tower
    encoded_image_input = keras.Input(
        shape=(input_image_encoding_size))  # Define input layer
    # Define first hidden layer and point the input layer to hidden layer
    hlayer1 = keras.layers.Dense(512, activation='relu')(encoded_image_input)
    # Define drop out layer and point first hidden layer to dropout layer
    dlayer1 = keras.layers.Dropout(0.15)(hlayer1)
    hlayer2 = keras.layers.Dense(512, activation='relu')(dlayer1)
    dlayer2 = keras.layers.Dropout(0.15)(hlayer2)
    hlayer3 = keras.layers.Dense(512, activation='relu')(dlayer2)
    dlayer3 = keras.layers.Dropout(0.15)(hlayer3)
    hlayer4 = keras.layers.Dense(512, activation='relu')(dlayer3)
    dlayer4 = keras.layers.Dropout(0.15)(hlayer4)
    hlayer5 = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(
        0.01), name='image_universal_embedding_output_layer')(dlayer4)

    # text tower
    encoded_text_input = keras.Input(shape=(input_text_encoding_size))
    thlayer1 = keras.layers.Dense(512, activation='relu')(encoded_text_input)
    tdlayer1 = keras.layers.Dropout(0.15)(thlayer1)
    thlayer2 = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(
        0.01), name='text_universal_embedding_output_layer')(tdlayer1)

    # Creates the hidden layer and points the 5th hidden layer of imagetower and 2nd hidden layer of texttower to it.
    shared_hidden_layer = MyDenseLayer(num_classes)([hlayer5, thlayer2])
    softmax_output_layer = tf.keras.layers.Softmax()(shared_hidden_layer)

    complete_model = keras.Model(inputs=[encoded_image_input, encoded_text_input], outputs=[
                                 softmax_output_layer, hlayer5, thlayer2])  # completes the model and assigns inputs and outputs
    return complete_model
