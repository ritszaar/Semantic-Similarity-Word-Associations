import pickle
import prettytable
import matplotlib.pyplot as plt

IWKResults = []
with open("boosted_results.pickle", "rb") as f:
    data = pickle.load(f)
    IWKResults = data["IWKResults"]

print(IWKResults[4])

# figure = plt.figure()
# ax1 = figure.add_subplot(1, 2, 1)
# ax2 = figure.add_subplot(1, 2, 2, sharey=ax1)

topKs = [1, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250]

# table = prettytable.PrettyTable()
# table.title = "topIWK vs mAHP@K"
# table.field_names = ["Dummy", "mAHP@1", "mAHP@5", "mAHP@10", "mAHP@15", "mAHP@20", "mAHP@25", "mAHP@50", "mAHP@100", "mAHP@150", "mAHP@200", "mAHP@250"]
# for i in range(1, 11):
#     row = ["topIWK = {}".format(i)] + ["{:.6f}".format(x) for x in IWKResults[i]["mean_average_precisions"]]
#     table.add_row(row)

# print(table, end='\n\n')

table = prettytable.PrettyTable()
table.title = "Time vs TopIWK"
table.field_names = ["topIWK", "Mean Time with No Overhead", "Min Time With Overhead"]
for i in range(1, 11):
    table.add_row([i, "{:.4f}".format(IWKResults[i]["mean_rtno"]), "{:.4f}".format(IWKResults[i]["mean_rto"])])

print(table, end='\n\n')




# topKs = [1, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = []

# plt.figure(figsize=(8,6))

# y1 = []
# y2 = []
# for topIWK in range(1, 11):
#     y1.append(IWKResults[topIWK]["mean_average_precisions"][4])
#     y2.append(IWKResults[topIWK]["mean_average_hierarchical_precisions"][4])

# plt.plot(x, y1, label="mAP@20")
# plt.plot(x, y2, label="mAHP@20")
# plt.title("Comparison of mAP@20 and mAHP@20")
# plt.xticks(x)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# plt.xlabel("topIWK")
# plt.ylabel("Precision")
# plt.show()


# for i in range(len(topKs)):
#     y = []
#     for topIWK in range(1, 11):
#         y.append(IWKResults[topIWK]["mean_average_precisions"][i])
#     plt.plot(x, y, label='mAP@{}'.format(topKs[i]))
#     plt.xticks(x)
#     plt.title("Variation of mAP with topIWK")
#     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#     plt.tight_layout()
#     plt.xlabel("topIWK")
#     plt.ylabel("Precision")

# for i in range(len(topKs)):
#     y = []
#     for topIWK in range(1, 11):
#         y.append(IWKResults[topIWK]["mean_average_hierarchical_precisions"][i])
#     ax2.plot(x, y, label='mHAP@{}'.format(topKs[i]))
#     # ax1.title("Variation of mHAP with topIWK")
#     ax2.legend(loc='best')
    # ax1.xlabel("topIWK")
    # ax1.ylabel("Precision")

# y1 = []
# y2 = []
# for topIWK in range(1, 11):
#     y1.append(IWKResults[topIWK]["mean_rtno"])
#     y2.append(IWKResults[topIWK]["mean_rto"])
# plt.plot(x, y1, label='Retrieval Time (No Overhead)')
# plt.plot(x, y2, label='Retrieval Time (Overhead)')
# plt.title("Variation of Retrieval Time with topIWK")

# plt.xticks(x)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# plt.xlabel("topIWK")
# plt.ylabel("Time (in s)")

# plt.show()
