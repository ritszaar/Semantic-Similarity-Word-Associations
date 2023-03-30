import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import PIL
import random
import string
import requests
import psycopg2
import datasets
import numpy as np
import tensorflow as tf

def connect():
    print("Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="word_associations",
            user="miranda",
            password="1234")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def get_associated_words(cue):
    cursor = conn.cursor()
    cursor.execute("SELECT target, strength FROM usf_word_associations WHERE cue='{}'".format(cue))
    rows = cursor.fetchall()
    return rows

def db_format(label):
    return label.replace('_', ' ').upper()

def resize_image_for_resnet(img):
    rgb_img = img.convert('RGB')
    img_array = np.array(rgb_img)
    resized_img_array = cv2.resize(img_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return PIL.Image.fromarray(resized_img_array)

def get_predictions(img):
    # # Load the pre-trained ResNet50 model
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

    # Convert the image to a numpy array
    x = tf.keras.preprocessing.image.img_to_array(img)

    # Reshape the array to match the input shape of the model
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    # Use the model to classify the image
    print("Getting predictions from the model...")
    preds = model.predict(x)

    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=4)[0]
    pred_associated_words = []
    for decoded_pred in decoded_preds:
        pred_associated_words.append((db_format(decoded_pred[1]), decoded_pred[2]))
    return pred_associated_words

class CIFARData:
    def __init__(self):
        # Load the CIFAR 100 dataset
        print("Loading the CIFAR 100 dataset...")
        dataset = datasets.load_dataset('cifar100')

        # Get the fine labels and store the mappings from fine_label to id and vice versa
        fine_labels = dataset["train"].features["fine_label"].names
        fine_label2id, id2fine_label = dict(), dict()

        for i, fine_label in enumerate(fine_labels):
            fine_label2id[fine_label] = i
            id2fine_label[i] = fine_label

        # Get the coarse labels and store the mappings from coarse_label to id and vice versa
        coarse_labels = dataset["train"].features["coarse_label"].names
        coarse_label2id, id2coarse_label = dict(), dict()

        for i, coarse_label in enumerate(coarse_labels):
            coarse_label2id[coarse_label] = i
            id2coarse_label[i] = coarse_label

        self.dataset = dataset["train"]
        self.fine_labels  = fine_labels
        self.coarse_labels = coarse_labels
        self.fine_label2id = fine_label2id
        self.id2fine_label = id2fine_label
        self.coarse_label2id = coarse_label2id
        self.id2coarse_label = id2coarse_label
    
    def get_fine_label_by_index(self, index):
        return self.id2fine_label[self.dataset[index]["fine_label"]]

class Node:
    def __init__(self, id, type, content):
        self.id = id
        self.type = type
        if (type == "image"):
            self.index = content
        else:
            self.word = content

class Graph:
    def __init__(self, cifar_data):
        self.all_words = set()
        self.associated_words = []
        for i in range(len(cifar_data.dataset)):
            self.associated_words.append(get_associated_words(cifar_data.get_fine_label_by_index(i)))
            for associated_word in self.associated_words[i]:
                self.all_words.add(associated_word[0])  
            if (i + 1) % 500 == 0:
                print("Added similar words for ({}/{})".format(i + 1, len(cifar_data.dataset)))
    
        self.n_word_nodes = len(self.all_words)
        self.n_image_nodes = len(cifar_data.dataset)
        self.n = self.n_word_nodes + self.n_image_nodes
        print(self.n_word_nodes)
        print(self.n_image_nodes)
        print(self.n)
    
conn = connect()

if conn is not None:
    print("Successfully connected to PostgreSQL database.\n\n")

# print(get_predictions('octopus.jpg'))
# print(get_associated_words("CAB"))

cifar_data = CIFARData()
index = random.choice(range(len(cifar_data.dataset)))
training_image = cifar_data.dataset[index]["img"]
fine_label = cifar_data.id2fine_label[cifar_data.dataset[index]["fine_label"]]
coarse_label = cifar_data.id2coarse_label[cifar_data.dataset[index]["coarse_label"]]
training_image.show()
print(fine_label, coarse_label)
 
print(get_predictions(resize_image_for_resnet(cifar_data.dataset[index]["img"])))
