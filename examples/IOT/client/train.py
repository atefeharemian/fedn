import os
import sys
import tempfile

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
from model import load_parameters, save_parameters, save_embeddings, load_embeddings
from data import load_data

from fedn.common.log_config import logger

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10
CLIENT_ID = None

# Set the seed for generating random numbers to ensure reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def train(
    in_model_path,
    out_model_path,
    data_path=None,
    batch_size=256,
    epochs=50,
    lr=0.001,
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
    x_train = load_data(data_path)

    # Load parameters and initialize model
    model = load_parameters(in_model_path)

    # Prepare the indices for K-Fold cross-validation
    dataset_size = len(x_train)

    # Create a TensorDataset from x_train
    dataset = TensorDataset(x_train)

    # Set the seed for generating random numbers to ensure reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()

    all_batch_embeddings = []

    for e in range(epochs):  # epoch loop
        model.train()
        batch_embeddings = []
        for b, (batch_x,) in enumerate(train_loader):  # batch loop
            # Calculate embeddings and send to combiner
            embeddings = model(batch_x)
            batch_embeddings.append(embeddings.detach().numpy())
            # gradients = process_embeddings_by_combiner(embeddings)
            # the corresponding gradients for this client based on CLIENT_ID
            # gradients = gradients[CLIENT_ID]
            # Update model parameters based on received gradients
            # for param, grad in zip(model.parameters(), gradients):
            #     param.grad = grad

            # optimizer.step()

            # Log
            if b % 100 == 0:
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{len(train_loader)-1}")
        all_batch_embeddings.append(batch_embeddings)

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": dataset_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

    # Save embeddings for this client (mandatory)
    save_embeddings(all_batch_embeddings, out_model_path)
    logger.info("Model training completed.")

def process_embeddings_by_combiner(embeddings):
    """Process embeddings by combiner.

    :param embeddings: The embeddings to process.
    :type embeddings: torch.Tensor
    """
    pass


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])