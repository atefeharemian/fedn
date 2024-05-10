import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

# class ClientModel(nn.Module):
#     def __init__(self, noFtr):
#         super(ClientModel, self).__init__()
#         self.layer1 = nn.Linear(noFtr, np.ceil(noFtr / 2).astype(int))
#         self.layer2 = nn.Linear(
#             np.ceil(noFtr / 2).astype(int), 7  # output embeddings of size 7
#         )

#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         embeddings = torch.relu(self.layer2(x))
#         return embeddings
# class CombinerModel(nn.Module):
#     def __init__(self, noFtr):
#         super(CombinerModel, self).__init__()
#         self.layer3 = nn.Linear(noFtr, np.ceil(noFtr / 2).astype(int))
#         self.layer4 = nn.Linear(
#             np.ceil(noFtr / 2).astype(int), 2  # output size of 2
#         )
        
#     def forward(self, x):
#         x = torch.relu(self.layer3(x))
#         x = torch.softmax(self.layer4(x), dim=1)
#         return x

def compile_model():
    """Compile the pytorch model.

    :return: The compiled model.
    :rtype: torch.nn.Module
    """

    class ClientModel(nn.Module):
        def __init__(self, noFtr):
            super(ClientModel, self).__init__()
            self.layer1 = nn.Linear(noFtr, np.ceil(noFtr / 2).astype(int))
            self.layer2 = nn.Linear(
                np.ceil(noFtr / 2).astype(int), 7  # output embeddings of size 7
            )

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            embeddings = torch.relu(self.layer2(x))
            return embeddings

    return ClientModel(7)  # Assuming 7 features for each client


def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
