import re
import pickle

topIWK = 4
topWWK = 4

def load_data():
    print('Loading data from "word.pickle"...')
    f = open("words.pickle", "rb")
    data = pickle.load(f)
    f.close()
    print('Successfully loaded data from "word.pickle".')
    return data["all_words"], data["word2id"], data["id2word"]

def get_image_links(word2id):
    print("\nGetting the image links...")
    image_links = []
    for i in range(0, 50000, 1000):
        with open("./Base Associations/{}-{}.txt".format(i, i + 1000 - 1), "r") as f:
            for i in range(0, 11000, 11):
                line = f.readline()
                id = int(line.split(" ")[0])
                fine_label_id = word2id[re.findall('"([^"]*)"', line)[0]]
                image_links.append([])
                image_links[id].append((fine_label_id, 0.8))
                for i in range(10):
                    if i < topIWK:
                        line = f.readline()
                        strength = float(line.split(" ")[-1])
                        pred_label_id = word2id[re.findall('"([^"]*)"', line)[0]]
                        image_links[id].append((pred_label_id, strength))
                    else:
                        f.readline()
    print("Successfully obtained the image links.")
    return image_links

print()
all_words, word2id, id2word = load_data()
data = dict()
data["image_links"] = get_image_links(word2id)

print('\nDumping data in "image_links.pickle"...')
with open("image_links.pickle", "wb") as f:
    pickle.dump(data, file=f)
print('Successfully dumped data in "image_links.pickle".')
