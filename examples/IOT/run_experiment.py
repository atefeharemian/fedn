from fedn import APIClient
import time
import uuid
import json
import matplotlib.pyplot as plt
import numpy as np
import collections

DISCOVER_HOST = "127.0.0.1"
DISCOVER_PORT = 8092
client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

# client.set_package("package.tgz", "numpyhelper")
# client.set_initial_model("seed.npz")
seed_model = client.get_initial_model()

session_id = "experiment_fedavg001"

session_config_fedavg = {
    "helper": "numpyhelper",
    "session_id": session_id,
    "aggregator": "fedavg",
    "model_id": seed_model["model_id"],
    "rounds": 2,
    "round_timeout": 40000,
}

# result_fedavg = client.start_session(**session_config_fedavg)

# while not client.session_is_finished(session_id):
#     time.sleep(2)

models = client.list_models(session_id)

validations = []
acc = collections.OrderedDict()
for model in models["result"]:
    model_id = model["model"]
    validations = client.list_validations(modelId=model_id)

    for _, validation in validations.items():
        metrics = json.loads(validation["data"])
        try:
            acc[model_id].append(metrics["test_accuracy"])
        except KeyError:
            acc[model_id] = [metrics["test_accuracy"]]

mean_acc_fedavg = []
for model, data in acc.items():
    mean_acc_fedavg.append(np.mean(data))

    # write the acc to file (append)
    with open(f"{model}.txt", "w") as f:
        for item in data:
            f.write(f"{item}\n")

# write the validations object to file
with open("validations_fedavg_2_round.json", "w") as f:
    json.dump(validations, f, indent=4)


mean_acc_fedavg.reverse()
print(f"Mean accuracy FedAvg: {mean_acc_fedavg}")

x = range(1, len(mean_acc_fedavg) + 1)
plt.plot(x, mean_acc_fedavg)
plt.legend(["Training Accuracy (FedAvg)"])
# save the plot as a file
plt.savefig("fedavg_acc.png")

######################

# session_config_fedopt = {
#     "helper": "numpyhelper",
#     "session_id": "experiment_fedopt",
#     "aggregator": "fedopt",
#     "model_id": seed_model["model_id"],
#     "rounds": 10,
# }

# result_fedopt = client.start_session(**session_config_fedopt)

# # # while not client.session_is_finished("experiment_fedopt"):
# time.sleep(120)

# models = client.list_models(session_id="experiment_fedopt")

# validations = []
# acc = collections.OrderedDict()
# for model in models["result"]:
#     model_id = model["model"]
#     validations = client.list_validations(modelId=model_id)
#     for _, validation in validations.items():
#         metrics = json.loads(validation["data"])
#         try:
#             acc[model_id].append(metrics["test_accuracy"])
#         except KeyError:
#             acc[model_id] = [metrics["test_accuracy"]]

# mean_acc_fedopt = []
# for model, data in acc.items():
#     mean_acc_fedopt.append(np.mean(data))
# mean_acc_fedopt.reverse()
# print(f"Mean accuracy FedAdam: {mean_acc_fedopt}")

# x = range(1, len(mean_acc_fedavg) + 1)
# plt.plot(x, mean_acc_fedavg, x, mean_acc_fedopt)
# plt.legend(["FedAvg", "FedAdam"])
# # save the plot as a file
# plt.savefig("fedavg_fedopt_acc.png")
