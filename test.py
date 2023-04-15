import pickle
import datasets
import numpy as np

topK = 250

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



def compute_metrics(pickle_path, train_data, test_data, embeddings, k_list=[1, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250]):
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
        mean_rtno += test_results[i][2] - test_results[i][1]
        mean_rto += test_results[i][2]
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
    return mean_rtno, mean_rto, mean_average_precisions, mean_average_hierarchical_precisions

train_data = CIFARData("train")
test_data = CIFARData("test")

embeddings = None
with open("class_embeddings.pickle", "rb") as f:
    data = pickle.load(f)
    embeddings = data["embedding"]

k_list = [1, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250]

IWKResults = [None]
for topIWK in range(1, 11):
    pickle_path = "partial_test_results_{}.pickle".format(topIWK)
    mean_rtno, mean_rto, mean_average_precisions, mean_average_hierarchical_precisions = compute_metrics(pickle_path, train_data, test_data, embeddings, k_list) 
    IWKResult = dict()
    IWKResult["mean_rtno"] = mean_rtno
    IWKResult["mean_rto"] = mean_rto
    IWKResult["mean_average_precisions"] = mean_average_precisions
    IWKResult["mean_average_hierarchical_precisions"] = mean_average_hierarchical_precisions
    IWKResults.append(IWKResult)
    print("Computed experiment results ({}/{}).\n".format(topIWK, 10))

data = dict()
data["IWKResults"] = IWKResults
with open("boosted_results.pickle", "wb") as f:
    pickle.dump(data, file=f)
