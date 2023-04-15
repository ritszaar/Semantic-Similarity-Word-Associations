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
    avg_precision_at_1       = 0
    avg_precision_at_5       = 0
    avg_precision_at_10      = 0
    avg_precision_at_15      = 0
    avg_precision_at_20      = 0
    h_avg_precision_at_1     = 0
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
            avg_precision_at_1       += sum(precisions[0:1])
            avg_precision_at_5       += sum(precisions[0:5])/5
            avg_precision_at_10      += sum(precisions[0:10])/10
            avg_precision_at_15      += sum(precisions[0:15])/15
            avg_precision_at_20      += sum(precisions[0:20])/20
            h_avg_precision_at_1     += sum(h_precisions[0:1])
            h_avg_precision_at_5     += sum(h_precisions[0:5])/5
            h_avg_precision_at_10    += sum(h_precisions[0:10])/10
            h_avg_precision_at_15    += sum(h_precisions[0:15])/15
            h_avg_precision_at_20    += sum(h_precisions[0:20])/20
            avg_preprocessing_time   += test_results[i][2]
            avg_total_retrieval_time += test_results[i][3]

    avg_precision_at_1       /= n
    avg_precision_at_5       /= n
    avg_precision_at_10      /= n
    avg_precision_at_15      /= n
    avg_precision_at_20      /= n
    h_avg_precision_at_1     /= n
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
    stat.add_row(["mAP@1",  "{:.2f}%".format(avg_precision_at_1  * 100)])
    stat.add_row(["mAP@5",  "{:.2f}%".format(avg_precision_at_5  * 100)])
    stat.add_row(["mAP@10", "{:.2f}%".format(avg_precision_at_10 * 100)])
    stat.add_row(["mAP@15", "{:.2f}%".format(avg_precision_at_15 * 100)])
    stat.add_row(["mAP@20", "{:.2f}%".format(avg_precision_at_20 * 100)])
    stat.add_row(["mAHP@1",  "{:.2f}%".format(h_avg_precision_at_1  * 100)])
    stat.add_row(["mAHP@5",  "{:.2f}%".format(h_avg_precision_at_5  * 100)])
    stat.add_row(["mAHP@10", "{:.2f}%".format(h_avg_precision_at_10 * 100)])
    stat.add_row(["mAHP@15", "{:.2f}%".format(h_avg_precision_at_15 * 100)])
    stat.add_row(["mAHP@20", "{:.2f}%".format(h_avg_precision_at_20 * 100)])
    stat.align["\033[93mMetric\033[0m"] = "l"
    stat.align["\033[93mValue\033[0m"] = "l"
    print()
    print(stat, end='\n\n')

def compute_average_precisions(relevances, k_list):
    precisions = [0] * topK
    precisions[0] = relevances[0]
    for i in range(1, topK):
        precisions[i] = precisions[i - 1] + relevances[i]
    for i in range(topK):
        precisions[i] = 1.00 * precisions[i]/ (1.00 * i + 1)

    average_precisions = [0] * len(k_list)
    dens = [0] * len(k_list)

    for i in range(topK):
        if precisions[i] > 0:
            for j in range(len(k_list)):
                if i <= k_list[j] - 1:
                    average_precisions[j] += precisions[i]
                    dens[j] += 1

    for i in range(len(k_list)):
        if dens[i] > 0:
            average_precisions[i] /= dens[i]

    # for i in range(1, topK):
    #     average_precisions[i] = average_precisions[i - 1] + precisions[i]
    # for i in range(topK):
    #     average_precisions[i] = average_precisions[i]/(i + 1)

    return average_precisions

def compute_metrics(pickle_path, train_data, test_data, embeddings, k_list=[1, 5, 10, 15, 20]):
    test_results = None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        test_results = data["test_results"]

    n1 = 0
    n2 = 0
    mean_rtno = 0
    mean_rto = 0
    mean_average_precisions = [0] * len(k_list)
    mean_average_hierarchical_precisions = [0] * len(k_list)
    for i in range(len(test_results)):
        predictions = test_results[i][0]
        if len(sys.argv) > 1 and sys.argv[1] == "--partial":
            mean_rtno += test_results[i][3] - test_results[i][2]
            mean_rto += test_results[i][3]
        else:
            mean_rtno += test_results[i][3] - test_results[i][2]
            mean_rto += test_results[i][3]
        target_fine_label = test_data.id2fine_label[test_data.dataset[i]["fine_label"]]
        # target_coarse_label = test_data.id2coarse_label[test_data.dataset[i]["coarse_label"]]
        target_fine_label_id = test_data.dataset[i]["fine_label"]

        relevances = []
        hierarchical_relevances = []
        for j in range(topK):
            pred_fine_label = train_data.id2fine_label[train_data.dataset[predictions[j]]["fine_label"]]
            pred_coarse_label = train_data.id2coarse_label[train_data.dataset[predictions[j]]["coarse_label"]]
            pred_fine_label_id = train_data.dataset[predictions[j]]["fine_label"]
            relevances.append(0)
            if target_fine_label == pred_fine_label:
                relevances[j] = 1
            # elif target_coarse_label == pred_coarse_label:
            #     relevances[j] = 0.5
            hierarchical_relevances.append(np.dot(embeddings[target_fine_label_id], embeddings[pred_fine_label_id]))

        average_precisions = compute_average_precisions(relevances, k_list)
        average_hierarchical_precisions = compute_average_precisions(hierarchical_relevances, k_list)
        if sum(relevances[0:20]) > 0:
            n1 = n1 + 1
            mean_average_precisions = [mean_average_precisions[i] + average_precisions[i] for i in range(len(k_list))]
        if sum(hierarchical_relevances[0:20]) > 0:
            n2 = n2 + 1
            mean_average_hierarchical_precisions = [mean_average_hierarchical_precisions[i] + average_hierarchical_precisions[i] for i in range(len(k_list))]

    
    mean_rtno = mean_rtno / len(test_results)
    mean_rto = mean_rto / len(test_results)
    mean_average_precisions = [mean_average_precisions[i]/n1 for i in range(len(k_list))]
    mean_average_hierarchical_precisions = [mean_average_hierarchical_precisions[i]/n2 for i in range(len(k_list))]
    stat = prettytable.PrettyTable()
    stat.title = "\033[1m\033[92mOverall Retrieval Performance\033[0m"
    stat.field_names = ["\033[93mMetric\033[0m", "\033[93mValue\033[0m"]
    stat.add_row(["Retrieval Time (without overhead)", "{:.2f}s".format(mean_rtno)])
    stat.add_row(["Retrieval Time (with overhead)", "{:.2f}s".format(mean_rto)])
    stat.add_row(["mAP@1",   "{:.2f}%".format(mean_average_precisions[0] * 100)])
    stat.add_row(["mAP@5",   "{:.2f}%".format(mean_average_precisions[1] * 100)])
    stat.add_row(["mAP@10",  "{:.2f}%".format(mean_average_precisions[2] * 100)])
    stat.add_row(["mAP@15",  "{:.2f}%".format(mean_average_precisions[3] * 100)])
    stat.add_row(["mAP@20",  "{:.2f}%".format(mean_average_precisions[4] * 100)])
    stat.add_row(["mAHP@1",  "{:.2f}%".format(mean_average_hierarchical_precisions[0]  * 100)])
    stat.add_row(["mAHP@5",  "{:.2f}%".format(mean_average_hierarchical_precisions[1]  * 100)])
    stat.add_row(["mAHP@10", "{:.2f}%".format(mean_average_hierarchical_precisions[2]  * 100)])
    stat.add_row(["mAHP@15", "{:.2f}%".format(mean_average_hierarchical_precisions[3]  * 100)])
    stat.add_row(["mAHP@20", "{:.2f}%".format(mean_average_hierarchical_precisions[4]  * 100)])
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

# evaluate_predictions(train_data, test_data, test_results)
compute_metrics("test_results.pickle", train_data, test_data, embeddings, [1, 5, 10, 15, 20])
