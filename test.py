import pickle
# print("Retrieval Time: {:.2f} s".format(query_time))
    # evaluate_predictions(train_data, test_data, i, predictions)
    # show_predictions(train_data, test_data, i, predictions)

data = dict()
with open("test_results.pickle", "rb") as f:
    data = pickle.load(f)

print(len(data["test_results"]))