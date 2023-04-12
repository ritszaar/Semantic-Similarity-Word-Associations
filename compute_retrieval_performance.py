import sys
import pickle
import datasets
import prettytable
import numpy as np

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
    
embeddings = None
with open("class_embeddings.pickle", "rb") as f:
    data = pickle.load(f)
    embeddings = np.array(data["embedding"])
    
topK = 20
    
def evaluate_predictions(train_data, test_data, test_results):
    avg_precision_at_5       = 0
    avg_precision_at_10      = 0
    avg_precision_at_15      = 0
    avg_precision_at_20      = 0
    h_avg_precision_at_5     = 0
    h_avg_precision_at_10    = 0
    h_avg_precision_at_15    = 0
    h_avg_precision_at_20    = 0
    avg_preprocessing_time   = 0
    avg_total_retrieval_time = 0
    n = 0
    for i in range(len(test_results)):
        predictions = test_results[i][0]
        precisions = []
        h_precisions = []
        target_fine_label = test_data.id2fine_label[test_data.dataset[i]["fine_label"]]
        target_coarse_label = test_data.id2coarse_label[test_data.dataset[i]["coarse_label"]]
        target_fine_label_id = test_data.dataset[i]["fine_label"]

        for j in range(topK):
            pred_fine_label = train_data.id2fine_label[train_data.dataset[predictions[j]]["fine_label"]]
            pred_coarse_label = train_data.id2coarse_label[train_data.dataset[predictions[j]]["coarse_label"]]
            pred_fine_label_id = train_data.dataset[predictions[j]]["fine_label"]
            precisions.append(0)
            if target_fine_label == pred_fine_label:
                precisions[j] = 1
            elif target_coarse_label == pred_coarse_label:
                precisions[j] = 0.5
            h_precisions.append(np.dot(embeddings[target_fine_label_id], embeddings[pred_fine_label_id]))

        if sum(precisions[0:20]) != 0:
            n = n + 1
            avg_precision_at_5       += sum(precisions[0:5])/5
            avg_precision_at_10      += sum(precisions[0:10])/10
            avg_precision_at_15      += sum(precisions[0:15])/15
            avg_precision_at_20      += sum(precisions[0:20])/20
            h_avg_precision_at_5     += sum(h_precisions[0:5])/5
            h_avg_precision_at_10    += sum(h_precisions[0:10])/10
            h_avg_precision_at_15    += sum(h_precisions[0:15])/15
            h_avg_precision_at_20    += sum(h_precisions[0:20])/20
            avg_preprocessing_time   += test_results[i][2]
            avg_total_retrieval_time += test_results[i][3]

    avg_precision_at_5       /= n
    avg_precision_at_10      /= n
    avg_precision_at_15      /= n
    avg_precision_at_20      /= n
    h_avg_precision_at_5     /= n
    h_avg_precision_at_10    /= n
    h_avg_precision_at_15    /= n
    h_avg_precision_at_20    /= n
    avg_preprocessing_time   /= n
    avg_total_retrieval_time /= n
    stat = prettytable.PrettyTable()
    stat.title = "\033[1m\033[92mOverall Retrieval Performance\033[0m"
    stat.field_names = ["\033[93mMetric\033[0m", "\033[93mValue\033[0m"]
    stat.add_row(["Retrieval Time (without overhead)", "{:.2f}s".format(avg_total_retrieval_time - avg_preprocessing_time)])
    stat.add_row(["Retrieval Time (with overhead)", "{:.2f}s".format(avg_total_retrieval_time)])
    stat.add_row(["mAP@5",  "{:.2f}%".format(avg_precision_at_5  * 100)])
    stat.add_row(["mAP@10", "{:.2f}%".format(avg_precision_at_10 * 100)])
    stat.add_row(["mAP@15", "{:.2f}%".format(avg_precision_at_15 * 100)])
    stat.add_row(["mAP@20", "{:.2f}%".format(avg_precision_at_20 * 100)])
    stat.add_row(["mAHP@5",  "{:.2f}%".format(h_avg_precision_at_5  * 100)])
    stat.add_row(["mAHP@10", "{:.2f}%".format(h_avg_precision_at_10 * 100)])
    stat.add_row(["mAHP@15", "{:.2f}%".format(h_avg_precision_at_15 * 100)])
    stat.add_row(["mAHP@20", "{:.2f}%".format(h_avg_precision_at_20 * 100)])
    stat.align["\033[93mMetric\033[0m"] = "l"
    stat.align["\033[93mValue\033[0m"] = "l"
    print()
    print(stat, end='\n\n')

pickle_path = "test_results.pickle"

if len(sys.argv) > 1:
    if sys.argv[1] == "--all":
        pass
    elif sys.argv[1] == "--partial":
        pickle_path = "partial_test_results.pickle"
    else:
        print("usage: python {} --all/--partial".format(sys.argv[0]))
        exit()

data = dict()
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

test_results = data["test_results"]


train_data = CIFARData("train")
test_data  = CIFARData("test")

evaluate_predictions(train_data, test_data, test_results)
