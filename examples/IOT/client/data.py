import os
import docker
from pathlib import Path
from urllib.parse import urlparse
import requests

from math import floor
import torch
import pandas as pd

TRAIN_DATA_FRAC = 0.8

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


# Helper function to download a file
def download_file(url, filename=None, filedir=None):
    if filename is None:
        a = urlparse(url)
        filename = os.path.basename(a.path)
    if filedir is not None:
        filename = os.path.join(filedir, filename)
    Path(filedir).mkdir(parents=True, exist_ok=True)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
    return filename


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # Check if the data is already downloaded
    if os.path.exists(os.path.join(out_dir, "nodes5_normal.csv")) and os.path.exists(
        os.path.join(out_dir, "nodes5_base.csv")
    ):
        print(f"Files already downloaded to {out_dir}")
        return

    # # Set URLs for the dataset( should set the url address of  data set here to download)
    # nodes5_agg_b_url = "www.example.com/nodes5_base.csv"
    # nodes5_base_url = "www.example.com/nodes5_base.csv"

    # # Download the files to the data directory
    # normal_data = download_file(nodes5_agg_b_url, "nodes5_normal.csv", out_dir)
    # base_data= download_file(nodes5_base_url, "nodes5_base.csv", out_dir)

    print(f"Downloaded files to {out_dir}")


def _get_data_path():
    """For test automation using docker-compose."""
    # Figure out FEDn client number from container name
    client = docker.from_env()
    container = client.containers.get(os.environ["HOSTNAME"])
    number = container.name[-1]

    # Return data path
    return f"/var/data/clients/{number}/IOT_normal_base.pt"


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path+'/data/clients/1/IOT_normal_base.pt')

    data = torch.load(data_path)
    
    if is_train:
        X = data["x_train"]
        # y = data["y_train"]
    else:
        X = data["x_test"]
        # y = data["y_test"]

    # return X, y
    return X


def process_to_tensors(train_data, test_data):
    x_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]
    x_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # drop the index column but it's unnamed
    x_train = x_train.drop("Unnamed: 0", axis=1)
    x_test = x_test.drop("Unnamed: 0", axis=1)

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    return (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    # calculate leftover data points
    leftover = n % parts

    lengths = [local_n + 1 if i < leftover else local_n for i in range(parts)]

    # result = torch.utils.data.random_split(dataset, lengths. torch.Generator().manual_seed(42))
    result = []

    start = 0
    for length in lengths:
        result.append(dataset[start : start + length])
        start += length

    return result


# vertically split the data into n_splits
def splitset_v(dataset, parts, label=False):
    print(f"dataset shape: {dataset.shape}")
    n = dataset.shape[1]
    local_n = floor(n / parts)
    # calculate leftover data points
    leftover = n % parts

    lengths = [local_n + 1 if i < leftover else local_n for i in range(parts)]

    # result = torch.utils.data.random_split(dataset, lengths. torch.Generator().manual_seed(42))
    result = []

    start = 0
    for length in lengths:
        result.append(dataset[:, start : start + length])
        start += length

    return result

def split(out_dir="data", n_splits=2):

    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # Load and convert to dict
    normal_data = pd.read_csv(f"{out_dir}/nodes5_normal.csv")
    base_data = pd.read_csv(f"{out_dir}/nodes5_base.csv")

    # Concat both data
    all_data = pd.concat([normal_data, base_data], ignore_index=True)

    #### train and test data split

    # extract data points with label 1 and 0 from normal and base data respectively
    all_data_1 = all_data[all_data["label"] == 1]
    all_data_0 = all_data[all_data["label"] == 0]

    # generate train and test data with 80 to 20 ratio from each lable
    train_data_1 = all_data_1.sample(frac=TRAIN_DATA_FRAC)
    test_data_1 = all_data_1.drop(train_data_1.index)
    train_data_0 = all_data_0.sample(frac=TRAIN_DATA_FRAC)
    test_data_0 = all_data_0.drop(train_data_0.index)

    # concatenate train and test data
    train_data_df = pd.concat([train_data_1, train_data_0], ignore_index=True)
    test_data_df = pd.concat([test_data_1, test_data_0], ignore_index=True)

    # initialize train_data and test_data tensors
    train_data = torch.utils.data.TensorDataset()
    test_data = torch.utils.data.TensorDataset()

    train_data.data, train_data.targets, test_data.data, test_data.targets = (
        process_to_tensors(train_data_df, test_data_df)
    )

    # shuffle the train data

    # split each of train and test data into n_splits
    # data = {
    #     "x_train": splitset(train_data.data, n_splits),
    #     "y_train": splitset(train_data.targets, n_splits),
    #     "x_test": splitset(test_data.data, n_splits),
    #     "y_test": splitset(test_data.targets, n_splits),
    # }
    data = {
        "x_train": splitset_v(train_data.data, n_splits),
        "y_train": train_data.targets,
        "x_test": splitset_v(test_data.data, n_splits),
        "y_test": test_data.targets,
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        labels_dir = f"{out_dir}/clients/{str(i+1)}/labels"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
            os.mkdir(labels_dir)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "x_test": data["x_test"][i],
            },
            f"{subdir}/IOT_normal_base.pt",
        )
        torch.save(
            {
                "y_train": data["y_train"],
                "y_test": data["y_test"],
            },
            f"{labels_dir}/IOT_normal_base_labels.pt",
        )


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        get_data()
        split()
