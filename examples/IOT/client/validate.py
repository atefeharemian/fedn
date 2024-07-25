import os
import re
import sys

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
from data import load_data
from model import load_parameters, save_parameters, save_embeddings, load_embeddings

from fedn.common.log_config import logger


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

def validate_vfl(in_model_path, out_json_path, data_path=None, local_model_path="/app/client/model/local_model.pt", batch_size=256):
    """ Validate model for VFL. based on train.py"""

    # Set the seed for generating random numbers
    torch.manual_seed(0)

    # Load data
    x_test = load_data(data_path, is_train=False)

    # Load parameters and initialize model
    gradients = load_parameters(in_model_path)
    # perform backprop on local model using gradients before starting validation

    # Getting Client ID
    fedn_data_path = os.getenv('FEDN_DATA_PATH')
    match = re.search(r'/clients/(\d+)/', fedn_data_path)
    client_id = match.group(1)

    # the corresponding gradients for this client based on client_id
    gradients = gradients[client_id]

    # Load local model
    # model = torch.load(local_model_path)
    model = load_parameters(local_model_path)

    #Update model parameters with the gradients received from the combiner
    for param in model.parameters():
        param.grad = gradients[param.shape]  # Assuming the gradients are stored in the same shape as the parameters
    # access the gradients for the current client using the client_id and perform backprop using Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    optimizer.step()

    # client training -> embeddings -> combiner takes embeddings -> combiner performs perdiction 
    # -> compbiner saves gradients -> start validation/model_update -> client validation -> embeddings -> final validation by combiner
    

    ############ Validation logic
    dataset_size = len(x_test)
    model.eval()

    # Evaluate
    with torch.no_grad():
        test_embeddings = model(x_test)

    # Save embeddings for this client (mandatory)
    save_embeddings(test_embeddings, out_json_path)
    logger.info("Model training completed.")

    # Save JSON
    # save_metrics(report, out_json_path)

# Custom metrics
def precision_recall_f1(y_true, y_pred, average="macro"):
    epsilon = 1e-7
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    true_positives = ((y_pred == y_true) & (y_true == 1)).sum()
    predicted_positives = (y_pred == 1).sum()
    possible_positives = (y_true == 1).sum()

    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (possible_positives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision.item(), recall.item(), f1.item()


if __name__ == "__main__":
    validate_vfl(sys.argv[1], sys.argv[2])