import pickle
import datasets

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
    
train_data = CIFARData("train")
test_data = CIFARData("test")

test_results = None
with open("test_results.pickle", "rb") as f:
    data = pickle.load(f)
    test_results = data["test_results"]

for i in range(len(test_results)):
    predictions, paths = test_results[i][0], test_results[i][1]
    target_fine_label = test_data.id2fine_label[test_data.dataset[i]["fine_label"]]
    for j in range(20):
        pred_fine_label = train_data.id2fine_label[train_data.dataset[predictions[j]]["fine_label"]]
        if len(paths[j]) > 3: 
            print("{} -- {} -- {} -- {}".format(i, j, target_fine_label, pred_fine_label))
            for k in range(len(paths[j])):
                print("{} -- ".format(paths[j][k]), end='')
            print()


