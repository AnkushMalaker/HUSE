import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import random
from tensorflow.keras.preprocessing import image
import encoder


def avg_of_array(set_of_tokens):  # function to calculate average of all tokens
    number_of_tokens = len(set_of_tokens[0])
    temp_array = np.zeros(768,)
    for i in range(number_of_tokens):
        temp_array = temp_array + set_of_tokens[i][1][0]
    avg_array = np.array(temp_array) / number_of_tokens
    return avg_array


def plot_model(model):
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96)


def cos_dist(label1, label2):
    return np.inner(label1, label2)/(np.linalg.norm(label1)*np.linalg.norm(label2))


def get_adj_graph(classes):
    model = encoder.get_universal_sentence_encoder()
    print("Loaded universal encoder")

    def embed(input):
        return model(input)

    # Embed the classes and store them in a 'class_embeddings
    class_embeddings = embed(classes)

    n = len(class_embeddings)
    adj_graph_classes = np.zeros([n, n])  # Initialize the matrix

    for i in range(n):
        for j in range(n):
            adj_graph_classes[i][j] = cos_dist(
                class_embeddings[i], class_embeddings[j])

    return adj_graph_classes


def clean_text(text_list):
    for i in range(len(text_list)):  # cleaning data
        sentence = text_list[i]
        sentence.strip()
        sentence = sentence.split(' ')
        while("" in sentence):
            sentence.remove("")
        text_list[i] = ' '.join(sentence)
    return text_list

# Forked the following from https://github.com/vineetm/tfhub-bert


def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return input_ids, input_mask, segment_ids


def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(
            sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids


def pack_batch(image_encodings, text_encodings, training_classes, dataset_size, batch_size):
    '''
    Returns x1 image encodings, x1 text encodings, x2 image encodings, x2 text encodings, y1 batch, y2 batch
    '''
    x1_images_batch = []
    x1_text_batch = []
    x2_images_batch = []
    x2_text_batch = []
    y1_batch = []
    y2_batch = []
    for _ in range(batch_size):
        index1 = random.randint(0, dataset_size-1)
        index2 = random.randint(0, dataset_size-1)

        x1_images_batch.append(image_encodings[index1])
        x1_text_batch.append(text_encodings[index1])

        x2_images_batch.append(image_encodings[index2])
        x2_text_batch.append(text_encodings[index2])

        y1_batch.append(training_classes[index1])
        y2_batch.append(training_classes[index2])

    return np.array(x1_images_batch), np.array(x1_text_batch), np.array(x2_images_batch), np.array(x2_text_batch), np.array(y1_batch), np.array(y2_batch)


def encode_and_pack_batch(batch_size, image_encoder, text_encoder, image_names, text_list, training_classes, img_shape, tokenizer):
    '''
    Encodes images and text and then packs a batch
    Returns x1 image encodings, x1 text encodings, x2 image encodings, x2 text encodings, y1 batch, y2 batch
    '''
    num_samples = len(image_names)
    images1 = []
    images2 = []

    input_ids1 = []
    masks1 = []
    segments1 = []
    input_ids2 = []
    masks2 = []
    segments2 = []

    y1_batch = []
    y2_batch = []

    indexes1 = random.sample(range(0, num_samples), batch_size)
    indexes2 = random.sample(range(0, num_samples), batch_size)

    # load images into images1 and images2, convert features to be fed to BERT and load into text_features1 and text_features2
    for i in range(batch_size):

        # Batch images
        image_name1 = image_names[indexes1[i]]
        image_path1 = 'images/' + image_name1
        img1 = image.load_img(image_path1, target_size=img_shape)
        img1 = image.img_to_array(img1)
        image_name2 = image_names[indexes2[i]]
        image_path2 = 'images/' + image_name2
        img2 = image.load_img(image_path2, target_size=img_shape)
        img2 = image.img_to_array(img2)
        images1.append(img1)
        images2.append(img2)

        # batch labels
        y1_batch.append(training_classes[indexes1[i]])
        y2_batch.append(training_classes[indexes2[i]])

        # batch text
        inputid1, inputmask1, inputsegment1 = convert_sentence_to_features(
            text_list[indexes1[i]], tokenizer, 512)
        inputid2, inputmask2, inputsegment2 = convert_sentence_to_features(
            text_list[indexes2[i]], tokenizer, 512)
        input_ids1.append(inputid1)
        masks1.append(inputmask1)
        segments1.append(inputsegment1)
        input_ids2.append(inputid2)
        masks2.append(inputmask2)
        segments2.append(inputsegment2)

    image_encodings1 = image_encoder(np.array(images1))
    image_encodings2 = image_encoder(np.array(images2))
    text_encodings1, _ = text_encoder(
        [np.array(input_ids1), np.array(masks1), np.array(segments1)])
    text_encodings2, _ = text_encoder(
        [np.array(input_ids2), np.array(masks2), np.array(segments2)])
    # There are two outputs from text_encoder. First, is pooled output and second is sequence output
    # [batch_size, , 768]
    # We simply use the entire sequence representation.

    return image_encodings1, text_encodings1, image_encodings2, text_encodings2, np.array(y1_batch), np.array(y2_batch)
