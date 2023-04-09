import pickle
import datasets
# print("Retrieval Time: {:.2f} s".format(query_time))
    # evaluate_predictions(train_data, test_data, i, predictions)
    # show_predictions(train_data, test_data, i, predictions)

# class CIFARData:
#     def __init__(self, which="train"):
#         # Load the CIFAR 100 dataset
#         print("\nLoading the CIFAR 100 {} dataset...".format(which))
#         dataset = datasets.load_dataset('cifar100')
#         print("\nSuccessfully loaded the CIFAR 100 {} dataset.\n".format(which))

#         # Get the fine labels and store the mappings from fine_label to id and vice versa
#         fine_labels = dataset[which].features["fine_label"].names
#         fine_label2id, id2fine_label = dict(), dict()

#         for i, fine_label in enumerate(fine_labels):
#             fine_label2id[fine_label] = i
#             id2fine_label[i] = fine_label

#         # Get the coarse labels and store the mappings from coarse_label to id and vice versa
#         coarse_labels = dataset[which].features["coarse_label"].names
#         coarse_label2id, id2coarse_label = dict(), dict()

#         for i, coarse_label in enumerate(coarse_labels):
#             coarse_label2id[coarse_label] = i
#             id2coarse_label[i] = coarse_label

#         self.dataset = dataset[which]
#         self.fine_labels  = fine_labels
#         self.coarse_labels = coarse_labels
#         self.fine_label2id = fine_label2id
#         self.id2fine_label = id2fine_label
#         self.coarse_label2id = coarse_label2id
#         self.id2coarse_label = id2coarse_label
    
#     def get_fine_label_by_index(self, index):
#         return self.id2fine_label[self.dataset[index]["fine_label"]]
    
# topK = 20
    
# def evaluate_predictions(train_data, test_data, test_results):
#     avg_precision_at_5       = 0
#     avg_precision_at_10      = 0
#     avg_precision_at_15      = 0
#     avg_precision_at_20      = 0
#     avg_preprocessing_time   = 0
#     avg_total_retrieval_time = 0
#     n = 0
#     for i in range(len(test_results)):
#         predictions = test_results[i][0]
#         precisions = []
#         target_fine_label = test_data.id2fine_label[test_data.dataset[i]["fine_label"]]
#         target_coarse_label = test_data.id2coarse_label[test_data.dataset[i]["coarse_label"]]

#         for j in range(topK):
#             pred_fine_label = train_data.id2fine_label[train_data.dataset[predictions[j]]["fine_label"]]
#             pred_coarse_label = train_data.id2coarse_label[train_data.dataset[predictions[j]]["coarse_label"]]
#             precisions.append(0)
#             if target_fine_label == pred_fine_label:
#                 precisions[j] = 1
#             elif target_coarse_label == pred_coarse_label:
#                 precisions[j] = 0.5

#         if True:
#             n = n + 1
#             avg_precision_at_5       += sum(precisions[0:5])/5
#             avg_precision_at_10      += sum(precisions[0:10])/10
#             avg_precision_at_15      += sum(precisions[0:15])/15
#             avg_precision_at_20      += sum(precisions[0:20])/20
#             avg_preprocessing_time   += test_results[i][2]
#             avg_total_retrieval_time += test_results[i][3]

#     avg_precision_at_5       /= n
#     avg_precision_at_10      /= n
#     avg_precision_at_15      /= n
#     avg_precision_at_20      /= n
#     avg_preprocessing_time   /= n
#     avg_total_retrieval_time /= n
#     print("Non-confused Results:         {:.2f}".format(n))
#     print("Average Precision@5:          {:.2f}".format(avg_precision_at_5))
#     print("Average Precision@10:         {:.2f}".format(avg_precision_at_10))
#     print("Average Precision@15:         {:.2f}".format(avg_precision_at_15))
#     print("Average Precision@20:         {:.2f}".format(avg_precision_at_20))
#     print("Average Processing Time:      {:.2f}".format(avg_preprocessing_time))
#     print("Average Total Retrieval Time: {:.2f}".format(avg_total_retrieval_time))

# data = dict()
# with open("test_results.pickle", "rb") as f:
#     data = pickle.load(f)

# test_results = data["test_results"]


# train_data = CIFARData("train")
# test_data  = CIFARData("test")

# evaluate_predictions(train_data, test_data, test_results)

