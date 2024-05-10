from fedn import APIClient
client = APIClient(host="localhost", port=8092)
client.set_active_package("package.tgz", helper="numpyhelper")
client.set_active_model("seed.npz")