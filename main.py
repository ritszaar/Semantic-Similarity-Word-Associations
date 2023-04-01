import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import queue
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def db_format(label):
    return label.replace('_', ' ').upper()

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
        self.g = [[] for i in range(self.n)]

        # Image Links
        for i in range(self.n_images):
            for j in range(len(self.image_links[i])):
                u = i
                v = int(self.image_links[i][j][0] + self.n_images)
                if self.image_links[i][j][1] != 0:
                    w = 1.00 / self.image_links[i][j][1]
                    self.g[u].append((v, w))
                    self.g[v].append((u, w))

        # Word Links
        for i in range(len(self.all_words)):
            for j in range(i, len(self.all_words)):
                u = i + self.n_images
                v = j + self.n_images
                winv = max(self.word_links[i][j], self.word_links[j][i])
                if winv != -1:
                    w = 1.00/winv
                    self.g[u].append((v, w))
                    self.g[v].append((u, w))

    def isEtherealNode(self, u):
        return u == -1

    def isImageNode(self, u):
        return u >= 0 and u < self.n_images; 

    def isWordNode(self, u):
        return u >= self.n_images and u < self.n 
    
    def dijkstra(self, sc, topK):
        vis  = [False for i in range(self.n + 1)]
        par = [-1 for i in range(self.n + 1)]
        dist = [1e9 for i in range(self.n + 1)]
        imgdist = [1e9 for i in range(self.n_images)]

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

        ans = sorted(
                range(len(imgdist)),
                key = lambda index: imgdist[index]
        )[1:(topK + 1)]
        return ans

model = WordAssociationsNetwork()      
print(model.dijkstra(218, 4))