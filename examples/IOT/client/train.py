import os
import sys

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
from model import load_parameters, save_parameters
from data import load_data

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10


def train(
    in_model_path,
    out_model_path,
    data_path=None,
    batch_size=256,
    epochs=10,
    lr=0.001,
    n_folds=16,
):
    """Complete a model update.

    Load model parameters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated parameters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    :param n_folds: The number of folds for cross-validation.
    :type n_folds: int
    """
    # Set the seed for generating random numbers
    torch.manual_seed(0)

    # Load data
    x_train, y_train = load_data(data_path)

    # Load parameters and initialize model
    model = load_parameters(in_model_path)

    # Prepare the indices for K-Fold cross-validation
    dataset_size = len(x_train)
    indices = list(range(dataset_size))
    fold_size = dataset_size // n_folds

    # Create a TensorDataset from x_train and y_train
    dataset = TensorDataset(x_train, y_train)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for fold in range(n_folds):
        # Split indices into train and validation
        val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = [i for i in indices if i not in val_indices]

        # Create data loaders
        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=batch_size, shuffle=False
        )

        for e in range(epochs):  # epoch loop
            model.train()
            for b, (batch_x, batch_y) in enumerate(train_loader):  # batch loop
                # Train on batch
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                # print hello work
                optimizer.step()
                # Log
                if b % 100 == 0:
                    print(
                        f"Fold {fold}/{n_folds-1} | Epoch {e}/{epochs-1} | Batch: {b}/{len(train_loader)-1} | Loss: {loss.item()}"
                    )

            model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for b, (batch_x, batch_y) in enumerate(val_loader):  # validation loop
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    loss = criterion(outputs, batch_y)
                    # Log
                    if b % 100 == 0:
                        print(
                            f"Fold {fold}/{n_folds-1} | Epoch {e+1}/{epochs} | Validation Batch: {b}/{len(val_loader)-1} | Loss: {loss.item()}"
                        )
            print(f"Validation Accuracy: {100 * correct / total}%")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "n_folds": n_folds,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])