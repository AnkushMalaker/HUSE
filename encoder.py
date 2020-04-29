'''
Following image encoders available:
resnet50
ResNet101

Text encoding is done via BERT.
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet50, ResNet101
import tokenization
import tensorflow_hub as hub


def get_resnet50(img_shape=(80, 80, 3)):
    '''
    Arguments: input image shape
    default = (80,80)
    '''
    image_input = keras.Input(
        shape=img_shape)  # Define the input layer in our embedding_extraction_model
    temp_model = resnet50.ResNet50(weights='imagenet', include_top=False)(
        image_input)  # Set resnet50 to temp_model
    image_embedding_output_layer = keras.layers.GlobalAveragePooling2D()(temp_model)
    # Create a new model and add GlobalAveragePooling2D layer to it.
    img_embedding_extractor_model = Model(
        inputs=image_input, outputs=image_embedding_output_layer)
    return img_embedding_extractor_model


def get_resnet101(img_shape=(80, 80, 3)):
    '''
    Arguments: input image shape
    default = (80,80)
    '''
    image_input = keras.Input(
        shape=img_shape)  # Define the input layer in our embedding_extraction_model
    temp_model = ResNet101.ResNet101(weights='imagenet', include_top=False)(
        image_input)  # Set resnet50 to temp_model
    image_embedding_output_layer = keras.layers.GlobalAveragePooling2D()(temp_model)
    # Create a new model and add GlobalAveragePooling2D layer to it.
    img_embedding_extractor_model = Model(
        inputs=image_input, outputs=image_embedding_output_layer)
    return img_embedding_extractor_model


def get_universal_sentence_encoder():
    print("Downloading universal sentence encoder")
    # Using pretrained Universal Sentence Encoder
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    return model

# BERT Model


def get_bert(max_seq_len):
    max_seq_length = max_seq_len
    load_tokenizer_workaround = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                                               trainable=True)
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                                trainable=True)([input_word_ids, input_mask, segment_ids])
    model = keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=bert_layer)

    vocab_file = load_tokenizer_workaround.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = load_tokenizer_workaround.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    return tokenizer, model
