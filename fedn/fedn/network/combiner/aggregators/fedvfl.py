import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import json


from fedn.common.exceptions import InvalidParameterError
from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase

# Set the seed for generating random numbers to ensure reproducibility
def seed_worker(worker_id):
    """[FEDVFL] Seed the workers for reproducibility.
    
    :param worker_id: Worker ID.
    :type worker_id: int
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    math.random.seed(worker_seed)

class CombinerModel(nn.Module):
    # [FEDVFL] Combiner model
    def __init__(self, noFtr):
        super(CombinerModel, self).__init__()
        self.layer3 = nn.Linear(noFtr, np.ceil(noFtr / 2).astype(int))
        self.layer4 = nn.Linear(
            np.ceil(noFtr / 2).astype(int), 2  # output size of 2
        )
    def forward(self, x):
        x = torch.relu(self.layer3(x))
        x = torch.softmax(self.layer4(x), dim=1)
        return x

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

        # [FEDVFL]
        # Set the seed for generating random numbers to ensure reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(0)
        self.model = CombinerModel(7*5)  # Assuming 7 features for each client and 5 clients
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.default_parameters = {
            'optimizer': 'Adam',
            'lr': 0.001,
            'criterion': 'CrossEntropyLoss',
        }
        self.client_embeddings = {}


    def combine_models(self, helper=None, delete_models=False, parameters=None):
        """[FEDVFL] Combine models.

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
        # if parameters:
        #     try:
        #         parameters.validate(parameter_schema)
        #     except InvalidParameterError as e:
        #         logger.error("Aggregator {} recieved invalid parameters. Reason {}".format(self.name, e))
        #         return None, data
        # else:
        #     logger.info("Aggregator {} using default parameteres.", format(self.name))
        #     parameters = default_parameters

        parameters = default_parameters

        # Override missing paramters with defaults
        for key, value in default_parameters.items():
            if key not in parameters:
                parameters[key] = value

        model = self.model
        nr_aggregated_models = 0
        total_examples = 0

        logger.info(
            "AGGREGATOR({}): Aggregating client embeddings... ".format(self.name))
        
        client_embeddings = self.client_embeddings

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
            client_gradients = self.process_embeddings(helper, model, client_embeddings, parameters, task='train')
            
        else:
            logger.error("Unsupported combiner optimizer passed to FedVFL.")
            return None, data

        data['nr_aggregated_models'] = nr_aggregated_models

        logger.info("AGGREGATOR({}): Aggregation completed, aggregated {} client model embeddings.".format(self.name, nr_aggregated_models))
        return client_gradients, data

    def process_embeddings(self, helper, model, client_embeddings, parameters=None, task='train'):
        """[FEDVFL] Process embeddings by combiner.
        
        :param helper: A helper object.
        :type helper: class: `fedn.network.combiner.helpers.Helper`
        :param model: A model object.
        :type model: class: `torch.nn.Module`
        :param client_embeddings: A dictionary containing client embeddings.
        :type client_embeddings: dict
        :param parameters: A dictionary containing parameters.
        :type parameters: dict
        :return: A dictionary containing client gradients.
        :rtype: dict
        """
        TASK = task # 'train' or 'test'
        # Load the labels
        labels = torch.load('/app/data/IOT_normal_base_labels.pt')
        if TASK == 'train':
            labels = labels['y_train']
        if TASK == 'test':
            labels = labels['y_test']
            
            # Validation
            validation_results = {}
            x_input, _ = self.make_x_input_tensor(client_embeddings)
            precision_list = []
            recall_list = []
            f1_list = []

            criterion = nn.CrossEntropyLoss()
            model.eval()
            # Evaluate
            with torch.no_grad():
                test_out = model(x_input)
                _, predicted = torch.max(test_out.data, 1)
                correct = (predicted == labels).sum().item()
                test_loss = criterion(test_out, labels)
                test_accuracy = torch.sum(torch.argmax(test_out, dim=1) == labels) / len(
                    test_out
                )
                precision, recall, f1 = self._precision_recall_f1(labels, predicted)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
            total = labels.size(0)
            accuracy = 100 * correct / total
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)

            # Populate validation results
            validation_results['accuracy'] = accuracy
            validation_results['precision'] = avg_precision
            validation_results['recall'] = avg_recall
            validation_results['f1'] = avg_f1
            validation_results['test_loss'] = test_loss.item()

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {avg_precision}")
            logger.info(f"Recall: {avg_recall}")
            logger.info(f"F1: {avg_f1}")
            logger.info(f"Test loss: {test_loss.item()}")

            validation_results_path = '/app/data/validation_results.json'
            with open(validation_results_path, 'w') as file:
                json.dump(validation_results, file)
            logger.info(f"Validation results saved to {validation_results_path}")

            return validation_results


        # # Flatten and concatenate client embeddings
        x_input, _ = self.make_x_input_tensor(client_embeddings) # dataset_size x 35
        x_input.requires_grad_(True) # Enable gradient computation for x_input

        # Ensure labels match the number of samples
        if len(labels) != x_input.shape[0]:
            raise ValueError(f"Number of labels ({len(labels)}) doesn't match number of samples ({x_input.shape[0]})")

        # Create data loader
        dataset = TensorDataset(x_input, labels)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=self.g)

        # Train
        optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(1):  # You can adjust the number of epochs
            logger.info(f"Training Epoch {epoch + 1}/{1}")
            for batch_idx, data in enumerate(train_loader):
                logger.info(f"Epoch {epoch + 1}/{1} | batch {batch_idx + 1}/{len(train_loader)}")
                batch_x, batch_y = data
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # save client_embeddings for later use
        torch.save(x_input, '/app/data/x_input_with_grads.pt')

        # Prepare gradients for clients
        client_gradients = {}
        start_idx = 0
        for client_id, embeddings in client_embeddings.items():
            feature_size = embeddings[0][0].shape[1]  # Get feature size from first batch of first epoch
            end_idx = start_idx + feature_size
            client_gradients[client_id] = x_input.grad[:, start_idx:end_idx].clone().detach().numpy()
            start_idx = end_idx


        np.savez_compressed('/app/data/client_gradients.npz', **client_gradients)

        return client_gradients

    def merge_client_data(self, client_data):
        """
        [FEDVFL] Merge embeddings for each batch within an epoch into a single ndarray.

        :param client_data: A list of lists, where each inner list contains embeddings for a batch within an epoch.
        :return: A list of ndarrays, where each ndarray contains all embeddings for a given epoch.
        """

        # Merge all batch ndarrays into one ndarray for each epoch
        merged_client_data = [np.vstack(epoch_data) for epoch_data in client_data]
        # Now, when calling merged_client_data[0], it will return an ndarray with size of dataset_size x 7
        # merged_client_data[0].shape -> (23526, 7)

        return merged_client_data

    def make_x_input_tensor(self, client_embeddings_dict):
        """
        [FEDVFL] Make a tensor from the client embeddings.

        :param client_embeddings_dict: A dictionary where the key is the client ID and the value is a list of lists, where each inner list contains embeddings for a batch within an epoch.
        :return: A tensor containing all client embeddings.
        """

        all_clients_data = []
        
        for client_id in sorted(client_embeddings_dict.keys()):
            client_data = client_embeddings_dict[client_id]
            merged_client_data = self.merge_client_data(client_data)
            all_clients_data.append(merged_client_data[0])

        # Stack all data vertically
        final_data_np = np.hstack(all_clients_data)

        # Convert to torch tensor
        final_data_tensor = torch.from_numpy(final_data_np)
        
        return final_data_tensor, final_data_np
    
    def validate_models(self, model, helper=None, delete_models=False, parameters=None):
        """[FEDVFL] Validate the model.
        
        :param model: A model object.
        :type model: class: `torch.nn.Module`
        """
        logger.info("Validating models")
        
        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0

        model = self.model
        nr_aggregated_models = 0
        total_examples = 0

        logger.info(
            "AGGREGATOR({}): Aggregating client embeddings... ".format(self.name))
        
        client_embeddings = self.client_embeddings

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

        validation_results = self.process_embeddings(helper, model, client_embeddings, task='test')

        data['nr_aggregated_models'] = nr_aggregated_models

        logger.info("AGGREGATOR({}): Aggregation of validations completed, aggregated {} client model embeddings.".format(self.name, nr_aggregated_models))
        return validation_results, data
    
    # Custom metrics
    def _precision_recall_f1(y_true, y_pred, average="macro"):
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