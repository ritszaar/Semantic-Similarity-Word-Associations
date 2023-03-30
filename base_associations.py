import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import cv2
import PIL
import datasets
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

# @tf.function
# def tf_get_predictions(img):
#     return get_predictions(img)

class CIFARData:
    def __init__(self):
        # Load the CIFAR 100 dataset
        print("\nLoading the CIFAR 100 dataset...")
        dataset = datasets.load_dataset('cifar100')
        print("\nSuccessfully loaded the CIFAR 100 dataset.\n")

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
    
cifar_data = CIFARData()
start_index = int(sys.argv[1])
savepath = "./Base Associations/{}-{}.txt".format(start_index, start_index + 1000 - 1)
print("\nComputing base associations for {}-{}...\n".format(start_index, start_index + 1000 - 1))
with open(savepath, "w") as f:
    for i in range(1000):
        index = start_index + i
        predictions = get_predictions(resize_image_for_resnet(cifar_data.dataset[index]["img"]))
        fine_label = db_format(cifar_data.id2fine_label[cifar_data.dataset[index]["fine_label"]])
        coarse_label = db_format(cifar_data.id2coarse_label[cifar_data.dataset[index]["coarse_label"]])
        print('{} "{}" "{}"'.format(index, fine_label, coarse_label), file=f)
        for prediction in predictions:
            print('"{}" {:.6f}'.format(prediction[0], prediction[1]), file=f) 

        if ((i + 1) % 10 == 0):
            print("\nComputed base associations ({}/{}).\n".format(i + 1, 1000))   

print("\n\nSuccessfully computed base associations for {}-{}.\n".format(start_index, start_index + 1000 - 1))
