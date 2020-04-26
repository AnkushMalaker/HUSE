import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import random
from tensorflow.keras.preprocessing import image


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
    print("Downloading universal sentence encoder")
    # Using pretrained Universal Sentence Encoder
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print("module %s loaded" % module_url)

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
        text_list[i] = sentence
    return text_list


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


def encode_and_pack_batch(batch_size, image_encoder, text_encoder, image_names, text_list, training_classes, img_shape):
    '''
    Encodes images and text and then packs a batch
    Returns x1 image encodings, x1 text encodings, x2 image encodings, x2 text encodings, y1 batch, y2 batch
    '''
    num_samples = len(image_names)
    images1 = []
    images2 = []
    x1_text_batch = []
    x2_text_batch = []
    y1_batch = []
    y2_batch = []
    indexes1 = random.sample(range(0, num_samples), batch_size)
    indexes2 = random.sample(range(0, num_samples), batch_size)

    # load images into images1 and images2
    for i in range(batch_size):
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
        y1_batch.append(training_classes[indexes1[i]])
        y2_batch.append(training_classes[indexes2[i]])
        result1 = text_encoder(text_list[indexes1[i]])
        resutl2 = text_encoder(text_list[indexes2[i]])
        avg_array1 = avg_of_array(result1)
        avg_array2 = avg_of_array(resutl2)
        x1_text_batch.append(avg_array1)
        x2_text_batch.append(avg_array2)

    image_encodings1 = image_encoder(np.array(images1))
    image_encodings2 = image_encoder(np.array(images2))

    return image_encodings1, np.array(x1_text_batch), image_encodings2, np.array(x2_text_batch), np.array(y1_batch), np.array(y2_batch)
