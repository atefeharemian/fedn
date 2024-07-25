# import os
# import sys

# import math
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, Subset
# import numpy as np

# from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
# from data import load_data
# from model import load_parameters

# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.abspath(dir_path))

# HELPER_MODULE = "numpyhelper"
# helper = get_helper(HELPER_MODULE)

# NUM_CLASSES = 10


# def validate(in_model_path, out_json_path, data_path=None):
#     """Validate model.

#     :param in_model_path: The path to the input model.
#     :type in_model_path: str
#     :param out_json_path: The path to save the output JSON to.
#     :type out_json_path: str
#     :param data_path: The path to the data file.
#     :type data_path: str
#     """
#     # Load data
#     x_test, y_test = load_data(data_path, is_train=False)

#     # Load model
#     model = load_parameters(in_model_path)
#     model.eval()

#     precision_list = []
#     recall_list = []
#     f1_list = []

#     # Evaluate
#     criterion = nn.CrossEntropyLoss()
#     with torch.no_grad():
#         test_out = model(x_test)
#         _, predicted = torch.max(test_out.data, 1)
#         correct = (predicted == y_test).sum().item()
#         test_loss = criterion(test_out, y_test)
#         test_accuracy = torch.sum(torch.argmax(test_out, dim=1) == y_test) / len(
#             test_out
#         )
#         precision, recall, f1 = precision_recall_f1(y_test, predicted)
#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)
#     total = y_test.size(0)
#     accuracy = 100 * correct / total
#     avg_precision = np.mean(precision_list)
#     avg_recall = np.mean(recall_list)
#     avg_f1 = np.mean(f1_list)

#     print(f"Accuracy: {accuracy}%")
#     print(f"Precision: {avg_precision}")
#     print(f"Recall: {avg_recall}")
#     print(f"F1 Score: {avg_f1}")
#     print(f"Evaluation completed.\n")

#     # JSON schema
#     report = {
#         "test_loss": test_loss.item(),
#         "test_accuracy": test_accuracy.item(),
#         "precision": avg_precision,
#         "recall": avg_recall,
#         "f1": avg_f1,
#         "accuracy": accuracy,
#     }

#     # Save JSON
#     save_metrics(report, out_json_path)


# # Custom metrics
# import torch

# def precision_recall_f1(y_true, y_pred, average="macro"):
#     epsilon = 1e-7
    
#     y_true = y_true.cpu()
#     y_pred = y_pred.cpu()
    
#     unique_labels = torch.unique(y_true)
#     precisions = []
#     recalls = []
#     f1s = []
    
#     for label in unique_labels:
#         true_positives = ((y_pred == label) & (y_true == label)).sum().item()
#         predicted_positives = (y_pred == label).sum().item()
#         possible_positives = (y_true == label).sum().item()
        
#         precision = true_positives / (predicted_positives + epsilon)
#         recall = true_positives / (possible_positives + epsilon)
#         f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
#         precisions.append(precision)
#         recalls.append(recall)
#         f1s.append(f1)
    
#     if average == "macro":
#         precision = sum(precisions) / len(unique_labels)
#         recall = sum(recalls) / len(unique_labels)
#         f1 = sum(f1s) / len(unique_labels)
#     elif average == "micro":
#         precision = sum(precisions) / len(unique_labels)
#         recall = sum(recalls) / len(unique_labels)
#         f1 = sum(f1s) / len(unique_labels)
#     else:
#         raise ValueError("Unsupported average type")
    
#     return precision, recall, f1
# # def precision_recall_f1(y_true, y_pred, average="macro"):
# #     epsilon = 1e-7
# #     y_true = y_true.cpu()
# #     y_pred = y_pred.cpu()

# #     true_positives = ((y_pred == y_true) & (y_true == 1)).sum()
# #     predicted_positives = (y_pred == 1).sum()
# #     possible_positives = (y_true == 1).sum()

# #     precision = true_positives / (predicted_positives + epsilon)
# #     recall = true_positives / (possible_positives + epsilon)
# #     f1 = 2 * (precision * recall) / (precision + recall + epsilon)

# #     return precision.item(), recall.item(), f1.item()


# if __name__ == "__main__":
#     validate(sys.argv[1], sys.argv[2])
import os
import sys

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
from data import load_data
from model import load_parameters

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_test, y_test = load_data(data_path, is_train=False)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    precision_list = []
    recall_list = []
    f1_list = []

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        test_out = model(x_test)
        _, predicted = torch.max(test_out.data, 1)
        correct = (predicted == y_test).sum().item()
        test_loss = criterion(test_out, y_test)
        test_accuracy = torch.sum(torch.argmax(test_out, dim=1) == y_test) / len(
            test_out
        )
        precision, recall, f1 = precision_recall_f1(y_test, predicted)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    total = y_test.size(0)
    accuracy = 100 * correct / total
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {avg_precision}")
    print(f"Recall: {avg_recall}")
    print(f"F1 Score: {avg_f1}")
    print(f"Evaluation completed.\n")

    # JSON schema
    report = {
        "test_loss": test_loss.item(),
        "test_accuracy": test_accuracy.item(),
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "accuracy": accuracy,
    }

    # Save JSON
    save_metrics(report, out_json_path)


# Custom metrics
from sklearn.metrics import precision_score, recall_score, f1_score

def precision_recall_f1(y_true, y_pred, average="macro"):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    
    return precision, recall, f1


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
