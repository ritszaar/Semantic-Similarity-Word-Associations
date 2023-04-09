import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import PIL
import queue
import random
import bisect
import pickle
import datasets
import psycopg2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

topIWK = 4
topWWK = 4

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

def db_format(label):
    return label.replace('_', ' ').upper()

def resize_image_for_resnet(img):
    rgb_img = img.convert('RGB')
    img_array = np.array(rgb_img)
    resized_img_array = cv2.resize(img_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return PIL.Image.fromarray(resized_img_array)

def get_predictions(img):
    # Load the pre-trained ResNet50 model
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

    # Convert the image to a numpy array
    x = tf.keras.preprocessing.image.img_to_array(img)

    # Reshape the array to match the input shape of the model
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    # Use the model to classify the image
    preds = model.predict(x)

    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=10)[0]
    pred_associated_words = []
    for decoded_pred in decoded_preds:
        pred_associated_words.append((db_format(decoded_pred[1]), decoded_pred[2]))
    return pred_associated_words

class CIFARData:
    def __init__(self, which="train"):
        # Load the CIFAR 100 dataset
        print("\nLoading the CIFAR 100 {} dataset...".format(which))
        dataset = datasets.load_dataset('cifar100')
        print("\nSuccessfully loaded the CIFAR 100 {} dataset.\n".format(which))

        # Get the fine labels and store the mappings from fine_label to id and vice versa
        fine_labels = dataset[which].features["fine_label"].names
        fine_label2id, id2fine_label = dict(), dict()

        for i, fine_label in enumerate(fine_labels):
            fine_label2id[fine_label] = i
            id2fine_label[i] = fine_label

        # Get the coarse labels and store the mappings from coarse_label to id and vice versa
        coarse_labels = dataset[which].features["coarse_label"].names
        coarse_label2id, id2coarse_label = dict(), dict()

        for i, coarse_label in enumerate(coarse_labels):
            coarse_label2id[coarse_label] = i
            id2coarse_label[i] = coarse_label

        self.dataset = dataset[which]
        self.fine_labels  = fine_labels
        self.coarse_labels = coarse_labels
        self.fine_label2id = fine_label2id
        self.id2fine_label = id2fine_label
        self.coarse_label2id = coarse_label2id
        self.id2coarse_label = id2coarse_label
    
    def get_fine_label_by_index(self, index):
        return self.id2fine_label[self.dataset[index]["fine_label"]]
    
def get_new_words(conn, all_words, base_new_words):
    all_new_words = set()
    for base_new_word in base_new_words:
        all_new_words.add(base_new_word)
        cue = base_new_word.split(" ")[-1].replace("'", "''")
        cursor = conn.cursor()
        cursor.execute("SELECT target, strength FROM usf_word_associations WHERE cue='{}'".format(cue))
        rows = cursor.fetchall()
        derived_words_count = min(len(rows) if rows is not None else 0, topWWK)
        for i in range(derived_words_count):
            if bisect.bisect_left(all_words, rows[i][0]) == len(all_words):
                all_new_words.add(rows[i][0])
        cursor.close()
    all_new_words = sorted(all_new_words)
    return all_new_words
class WordAssociationsNetwork:
    def __init__(self):
        with open("words.pickle", "rb") as f:
            data = pickle.load(f)
            self.all_words = data["all_words"]
            self.word2id   = data["word2id"]
            self.id2word   = data["id2word"]
        
        with open("image_links.pickle", "rb") as f:
            data = pickle.load(f)
            self.image_links = data["image_links"]  

        with open("word_links.pickle", "rb") as f:
            data = pickle.load(f)
            self.word_links = data["word_links"]

        self.n_images = len(self.image_links)
        self.n_words = len(self.all_words)
        self.n = self.n_images + self.n_words
        self.g = [[] for i in range(self.n + 1)]

        # Image Links
        for i in range(self.n_images):
            for j in range(len(self.image_links[i])):
                u = i + 1
                v = self.image_links[i][j][0] + self.n_images + 1
                if self.image_links[i][j][1] != 0:
                    w = 1.00 / self.image_links[i][j][1]
                    self.g[u].append((v, w))
                    self.g[v].append((u, w))

        # Word Links
        for i in range(len(self.all_words)):
            for j in range(i, len(self.all_words)):
                u = i + self.n_images + 1
                v = j + self.n_images + 1
                winv = max(self.word_links[i][j], self.word_links[j][i])
                if winv != -1:
                    w = 1.00/winv
                    self.g[u].append((v, w))
                    self.g[v].append((u, w))

    def isEtherealNode(self, u):
        return u == -1
    
    def isTestNode(self, u):
        return u == 0

    def isImageNode(self, u):
        return u > 0 and u <= self.n_images; 

    def isWordNode(self, u):
        return u > self.n_images and u <= self.n 
    
    def dijkstra(self, sc, topK):
        vis  = [False for i in range(self.n + 1)]
        par = [-1 for i in range(self.n + 1)]
        dist = [1e9 for i in range(self.n + 1)]
        imgdist = [1e9 for i in range(self.n_images + 1)]

        dist[sc] = 0
        imgdist[sc] = 0
        pq = queue.PriorityQueue()
        pq.put((dist[sc], sc))

        while not pq.empty():
            u = pq.get()[1]
            vis[u] = True
            if self.isImageNode(u) and u != sc:
                imgdist[u] = min(imgdist[u], dist[u])
                continue
            for (v, w) in self.g[u]:
                if not vis[v]:
                    if dist[v] > dist[u] + w:
                        par[v] = u
                        dist[v] = dist[u] + w
                        pq.put((dist[v], v))

        predictions = []
        for u in range(1, self.n_images + 1):
            if imgdist[u] < 1e9:
                predictions.append((imgdist[u], u))
        predictions.sort()
        predictions = [x[1] for x in predictions]
        # predictions = sorted(range(len(imgdist)), key = lambda index: imgdist[index])
        print(len(predictions))
        # predictions.reverse()
        predictions = predictions[0:topK]

        paths = []
        for prediction in predictions:
            u = par[prediction]
            path = []
            while u != 0:
                path.append(self.id2word[u - self.n_images - 1])
                u = par[u]
            path.reverse()
            paths.append(path)

        return [x - 1 for x in predictions], paths
    
    def predict(self, img, topK):
        image_link = get_predictions(resize_image_for_resnet(img))

        base_new_words = set()
        for i in range(topIWK):
            if bisect.bisect_left(self.all_words, image_link[i][0]) == self.n_words:
                base_new_words.add(image_link[i][0])
        
        if len(base_new_words) == 0:
            print("All fine!\n")
        else:
            print("Oh boi!\n")
            # all_new_words = get_new_words(base_new_words)
            # for i in range(all_new_words):
            #     id = len(self.all_words)
            #     self.all_words.append(all_new_words[i])
            #     self.id2word[id] = all_new_words[i]
            #     self.word2id[all_new_words[i]] = id
            # self.n_words = len(self.all_words)
            # self.n = self.n_images + self.n_words

        # for i in range(len(image_link)):
        #     u = 0
        #     if image_link[i][0] in self.word2id:
        #         v = self.word2id[image_link[i][0]] + self.n_images + 1
        #         w = 1.00 / float(image_link[i][1])
        #         self.g[u].append((v, w))
        #         self.g[v].append((u, w))

        return model.dijkstra(0, topK)
        

train_data = CIFARData("train")
test_data  = CIFARData("test")

topK = 20
model = WordAssociationsNetwork()
index = 276
predictions, paths = model.predict(test_data.dataset[index]["img"], topK)

print("\nRandomly chosen test image index: {}\n".format(index))
print("Predictions: {}".format(predictions))

print("\nPaths: \n")
for i in range(len(predictions)):
    print("Query-{}: Query --> ".format(predictions[i]), end='')
    for j in range(len(paths[i])):
        print("{} --> ".format(paths[i][j]), end='')
    print(predictions[i])

# create figure
fig = plt.figure(figsize=(10, 10))
  
# setting values to rows and column variables
rows = int(topK/4) + 1
columns = 4
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(test_data.dataset[index]["img"])
plt.axis('off')
plt.title(test_data.id2fine_label[test_data.dataset[index]["fine_label"]])
  
for i in range(topK):
    fig.add_subplot(rows, columns, i + columns + 1)
  
    plt.imshow(train_data.dataset[predictions[i]]["img"])
    plt.axis('off')
    plt.title(train_data.id2fine_label[train_data.dataset[predictions[i]]["fine_label"]])
  

plt.show()
