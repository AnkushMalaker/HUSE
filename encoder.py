'''
Following encoders available:
resnet50
ResNet101
'''
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet50, ResNet101


def get_resnet50(img_shape=(80,80,3)):
    '''
    Arguments: input image shape
    default = (80,80)
    '''
    image_input = keras.Input(shape=img_shape) #Define the input layer in our embedding_extraction_model
    temp_model = resnet50.ResNet50(weights='imagenet',include_top=False)(image_input) #Set resnet50 to temp_model
    image_embedding_output_layer = keras.layers.GlobalAveragePooling2D()(temp_model) 
    img_embedding_extractor_model = Model(inputs=image_input, outputs=image_embedding_output_layer) #Create a new model and add GlobalAveragePooling2D layer to it.
    return img_embedding_extractor_model

def get_resnet101(img_shape=(80,80,3)):
    '''
    Arguments: input image shape
    default = (80,80)
    '''
    image_input = keras.Input(shape=img_shape) #Define the input layer in our embedding_extraction_model
    temp_model = ResNet101.ResNet101(weights='imagenet',include_top=False)(image_input) #Set resnet50 to temp_model
    image_embedding_output_layer = keras.layers.GlobalAveragePooling2D()(temp_model) 
    img_embedding_extractor_model = Model(inputs=image_input, outputs=image_embedding_output_layer) #Create a new model and add GlobalAveragePooling2D layer to it.
    return img_embedding_extractor_model


