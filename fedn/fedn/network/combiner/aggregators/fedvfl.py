import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from fedn.common.exceptions import InvalidParameterError
from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase

# Set the seed for generating random numbers to ensure reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class CombinerModel(nn.Module):
    def __init__(self, noFtr):
        super(CombinerModel, self).__init__()
        self.layer3 = nn.Linear(noFtr, np.ceil(noFtr / 2).astype(int))
        self.layer4 = nn.Linear(
            np.ceil(noFtr / 2).astype(int), 2  # output size of 2
        )

class Aggregator(AggregatorBase):
    """ Vertical Federated aggregator (FedVFL).

    This aggregator takes the embeddings from the clients, 
    combines (concat) them and predicts the model output based on the labels.pt file.
    computes gradients for the CombinerModel, retaining the gradients to save to the
    statestore so the clients can later fetch them and update their ClientModel parameters.



    :param id: A reference to id of :class: `fedn.network.combiner.Combiner`
    :type id: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.network.combiner.roundhandler.RoundHandler`
    :type control: class: `fedn.network.combiner.roundhandler.RoundHandler`

    """

    def __init__(self, storage, server, modelservice, round_handler):

        super().__init__(storage, server, modelservice, round_handler)

        self.name = "fedvfl"
        self.storage = storage
        self.server = server
        self.modelservice = modelservice
        self.round_handler = round_handler

        # Set the seed for generating random numbers to ensure reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(0)
        self.model = CombinerModel(7*5)  # Assuming 7 features for each client and 5 clients
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()


    def combine_models(self, helper=None, delete_models=True, parameters=None):
        """Combine models.

        This aggregator takes the embeddings from the clients, 
        combines (concat) them and predicts the model output based on the labels.pt file.
        computes gradients for the CombinerModel, retaining the gradients to save to the
        statestore so the clients can later fetch them and update their ClientModel parameters.
        """
        logger.info("Combining models")
        
        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0

        # Define parameter schema
        parameter_schema = {
            'optimizer': str,
            'lr': float,
            'criterion': str,
        }

        # Define default parameters
        default_parameters = {
            'optimizer': 'Adam',
            'lr': 0.001,
            'criterion': 'CrossEntropyLoss',
        }

        # Validate parameters
        if parameters:
            try:
                parameters.validate(parameter_schema)
            except InvalidParameterError as e:
                logger.error("Aggregator {} recieved invalid parameters. Reason {}".format(self.name, e))
                return None, data
        else:
            logger.info("Aggregator {} using default parameteres.", format(self.name))
            parameters = self.default_parameters


        # Override missing paramters with defaults
        for key, value in default_parameters.items():
            if key not in parameters:
                parameters[key] = value

        model = self.model
        nr_aggregated_models = 0
        total_examples = 0

        logger.info(
            "AGGREGATOR({}): Aggregating client embeddings... ".format(self.name))

        while not self.model_updates.empty():
            try:
                # Get next model embedding from queue
                logger.info("AGGREGATOR({}): Getting next model embeddings update from queue.".format(self.name))
                model_update = self.next_model_update()

                # Load model parameters and metadata
                logger.info("AGGREGATOR({}): Loading model metadata {}.".format(self.name, model_update.model_update_id))
                model_next, metadata = self.load_model_update(model_update, helper)

                logger.info(
                    "AGGREGATOR({}): Processing model update {}, metadata: {}  ".format(self.name, model_update.model_update_id, metadata))

                # Assign total number of examples
                total_examples = metadata['num_examples']

                if nr_aggregated_models == 0:
                    client_embeddings = {}
                    client_embeddings[str(nr_aggregated_models + 1)] = model_next
                else:
                    client_embeddings[str(nr_aggregated_models + 1)] = model_next
                    # model = helper.increment_average(
                    #     model, model_next, metadata['num_examples'], total_examples)

                nr_aggregated_models += 1
                # Delete model from storage
                if delete_models:
                    self.modelservice.temp_model_storage.delete(model_update.model_update_id)
                    logger.info(
                        "AGGREGATOR({}): Deleted model update {} from storage.".format(self.name, model_update.model_update_id))
                self.model_updates.task_done()
            except Exception as e:
                logger.error(
                    "AGGREGATOR({}): Error encoutered while processing model update {}, skipping this update.".format(self.name, e))
                self.model_updates.task_done()

        if parameters['optimizer'] == 'Adam':
            # model = self.serveropt_adam(helper, pseudo_gradient, model_old, parameters)
            client_gradients = self.process_embeddings(helper, model, client_embeddings, parameters)
            
        else:
            logger.error("Unsupported combiner optimizer passed to FedVFL.")
            return None, data

        data['nr_aggregated_models'] = nr_aggregated_models

        logger.info("AGGREGATOR({}): Aggregation completed, aggregated {} models.".format(self.name, nr_aggregated_models))
        return client_gradients, data

    def next_model_update():
        """ Get the next model update from the queue.

        :param helper: A helper object.
        :type helper: object
        :return: The model update.
        :rtype: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        pass
    
    def next_model_update():
        """ Get the next model update from the queue.

        :param helper: A helper object.
        :type helper: object
        :return: The model update.
        :rtype: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        pass

    def load_model_update(self, model_update, helper):
        """ Load the memory representation of the model update.

        Load the model update paramters and the
        associate metadata into memory.
        """
        pass
    
    def process_embeddings(helper, model, client_embeddings, parameters):
        """Process embeddings by combiner.

        :param embeddings: The embeddings to process.
        :type embeddings: torch.Tensor
        """
        pass