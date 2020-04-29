from create_model import build_model
import encoder
import utils
import random
import tensorflow as tf
import numpy as np
from numpy import argmax
import sys
import getopt
import pandas as pd
# from bert_embedding import BertEmbedding   #Remove if build pass


def classification_loss(y_pred, y_true):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    cl1 = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred[0], from_logits=False)
    cl2 = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred[1], from_logits=False)
    c_loss = cl1 + cl2
    return c_loss


def graph_loss(image_tower_output1, image_tower_output2, text_tower_output1, text_tower_output2, y1, y2, adj_graph, graph_threshold):

    batch_loss = []
    i = argmax(y1, axis=1)
    j = argmax(y2, axis=1)

    for k in range(len(i)):
        if (adj_graph[i[k]][j[k]] < graph_threshold):
            # Image_tower_embedding
            I = utils.cos_dist(
                image_tower_output1[k], image_tower_output2[k]) - adj_graph[i[k]][j[k]]
            # Text tower embedding
            T = utils.cos_dist(
                text_tower_output1[k], text_tower_output2[k]) - adj_graph[i[k]][j[k]]
            g_loss = I + T
            batch_loss.append(g_loss)
        else:
            g_loss = 0
            batch_loss.append(g_loss)

    return tf.convert_to_tensor(batch_loss, dtype='float32')


def gap_loss(output_image_encoding, output_text_encoding):
    gap_loss = utils.cos_dist(output_image_encoding, output_text_encoding)
    return tf.convert_to_tensor(gap_loss)


def loss(model, x1, x2, y1, y2, loss_weights, graph_threshold, adj_graph):
    softmax_outputs, image_tower_output1, text_tower_output1 = model(x1)
    _, image_tower_output2, text_tower_output2 = model(x2)
    # get classification loss for this training_example
    c_loss = classification_loss(softmax_outputs, y1)
    grph_loss = graph_loss(image_tower_output1, image_tower_output2,
                           text_tower_output1, text_tower_output2, y1, y2, adj_graph, graph_threshold)
    # get gap loss for this training example
    g_loss = gap_loss(image_tower_output1, text_tower_output1)
    joint_loss = loss_weights[0]*c_loss + \
        loss_weights[1]*grph_loss + loss_weights[2]*g_loss
    return joint_loss


# Define gradient function for reducing classification and gap loss
def grad(model, x1, x2, y1, y2, loss_weights, graph_threshold, adj_graph):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x1, x2, y1, y2, loss_weights,
                          graph_threshold, adj_graph)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def main(argv):
    CSV_FILE_PATH = 'data.csv'
    num_epochs = 250000
    BATCH_SIZE = 1
    img_shape = (224, 224, 3)  # Reduce based on RAM
    GRAPH_THRESHOLD = 0.5
    LEARNING_RATE = 1.6192e-05

    # Give importance to classification, semantic and gap loss respectively.
    LOSS_WEIGHTS = [0.6, 0.2, 0.2]
    IMAGE_ENCODER = 'resnet50'
    TEXT_ENCODER = 'bert'

    try:
        opts, args = getopt.getopt(
            argv, "i:t:b:", ["image_encoder=", "text_encoder=", "batch_size="])
    except getopt.GetoptError:
        print('test -i <image_folder> -c <csv_filename>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--image_encoder"):
            IMAGE_ENCODER = arg
        elif opt in ("-c", "--text_encoder"):
            TEXT_ENCODER = arg
        elif opt in ("-b", "--batch_size"):
            BATCH_SIZE = int(arg)
            print("Set batch_size to %d" % BATCH_SIZE)

    df = pd.read_csv(CSV_FILE_PATH)
    num_samples = df.shape[0]

    class_names = df.classes.unique()

    # CONVERT TO CATEGORICAL
    temp = list(df.classes)
    training_class_intmap = temp.copy()

    # map each color to an integer
    mapping = {}

    for x in range(len(class_names)):
        mapping[class_names[x]] = x

    # integer representation
    for x in range(df.shape[0]):
        training_class_intmap[x] = mapping[training_class_intmap[x]]

    training_classes = tf.keras.utils.to_categorical(training_class_intmap)
    image_names = df.image
    text_list = df.text

    text_list = utils.clean_text(text_list)
    num_classes = len(class_names)

    adj_graph_classes = utils.get_adj_graph(class_names)

    if (IMAGE_ENCODER == 'resnet50'):
        image_embedding_extractor_model = encoder.get_resnet50(img_shape)
        image_encoder_size = 2048
    elif (IMAGE_ENCODER == 'resnet101'):
        image_embedding_extractor_model = encoder.get_resnet101(img_shape)
    if (TEXT_ENCODER == 'bert'):
        tokenizer, text_embedding_extractor_model = encoder.get_bert(512)
        text_encoder_size = 768

    complete_model = build_model(
        image_encoder_size, text_encoder_size, num_classes)

    train_loss_results = []
    # train_accuracy_results = []
    # Define the optimize and specify the learning rate
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        # epoch_accuracy = tf.keras.metrics.CategoricalAccuracy() #Uncomment if you want to track
        # Training loop - using batches of 1024
        # encode_and_pack_batch(batch_size, image_encoder, text_encoder, image_names, text_list, training_classes, img_shape):
        xi1, xt1, xi2, xt2, y1, y2 = utils.encode_and_pack_batch(
            BATCH_SIZE, image_embedding_extractor_model,  text_embedding_extractor_model, image_names, text_list, training_classes, img_shape,
            tokenizer)
        x1 = [xi1, xt1]
        x2 = [xi2, xt2]
        # Optimize the model
        loss_value, grads = grad(
            complete_model, x1, x2, y1, y2, LOSS_WEIGHTS, GRAPH_THRESHOLD, adj_graph_classes)
        optimizer.apply_gradients(
            zip(grads, complete_model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        if epoch % 5 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(
                epoch, epoch_loss_avg.result()))


if __name__ == "__main__":
    main(sys.argv[1:])
