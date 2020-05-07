import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import random
from tensorflow.keras.preprocessing import image
import encoder
import os.path


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


def encode_and_pack_batch(batch_size, image_encoder, text_encoder, image_names, text_list, training_classes, img_shape, tokenizer, img_folder_path):
    '''
    Encodes images and text and then packs a batch
    Returns x1 image encodings, x1 text encodings, x2 image encodings, x2 text encodings, y1 batch, y2 batch
    '''
    if (os.path.isdir('batched_data')):
        pass
    else:
        os.mkdir('batched_data')
    if (os.path.isfile('batched_data/img_encodings.npy')):
        image_encodings = np.load(
            'batched_data/img_encodings.npy',  allow_pickle=True)
        text_encodings = np.load(
            'batched_data/text_encodings.npy',  allow_pickle=True)
        y_batch = np.load('batched_data/classes.npy',  allow_pickle=True)
    else:
        num_samples = len(image_names)
        index = 0
        for i in range((num_samples//batch_size)+1):
            images = []
            input_ids = []
            segments = []
            masks = []
            y_batch = []
            print("Encoding batch: %d out of %d" %
                  (i, num_samples//batch_size+1))
            for j in range(batch_size):
                index = batch_size*i + j
                if index>=num_samples:
                    break
                image_name = image_names[index]
                image_path = img_folder_path + image_name
                img = image.load_img(image_path, target_size=img_shape)
                img = image.img_to_array(img)
                images.append(img)

                # batch labels
                y_batch.append(training_classes[index])
                inputid, inputmask, inputsegment = convert_sentence_to_features(
                    text_list[index], tokenizer, 512)

                input_ids.append(inputid)
                masks.append(inputmask)
                segments.append(inputsegment)

            one_batch_image_encodings = image_encoder(np.array(images))
            one_batch_text_encodings, _ = text_encoder(
                [np.array(input_ids), np.array(masks), np.array(segments)])

            if (i == 0):
                image_encodings = one_batch_image_encodings
                text_encodings = one_batch_text_encodings
            else:
                image_encodings = np.concatenate((
                    image_encodings, one_batch_image_encodings))
                text_encodings = np.concatenate((
                    text_encodings, one_batch_text_encodings))
        y_batch = np.array(y_batch)
        np.save('batched_data/img_encodings',
                image_encodings, allow_pickle=True)
        np.save('batched_data/text_encodings',
                text_encodings, allow_pickle=True)
        np.save('batched_data/classes', np.array(y_batch), allow_pickle=True)
    image_encodings = tf.data.Dataset.from_tensor_slices(image_encodings)
    text_encodings = tf.data.Dataset.from_tensor_slices(text_encodings)
    y_batch = tf.data.Dataset.from_tensor_slices(y_batch)

    training_batch1 = tf.data.Dataset.zip(
        (image_encodings, text_encodings, y_batch)).batch(batch_size).shuffle(num_samples)
    training_batch2 = tf.data.Dataset.zip(
        (image_encodings, text_encodings, y_batch)).batch(batch_size).shuffle(num_samples)

    return training_batch1, training_batch2
