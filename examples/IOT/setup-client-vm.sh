#!/bin/bash

###### DOCKER ######

# Create docker cache directory
mkdir ~/volume/docker

# Add docker config
sudo touch /etc/docker/daemon.json

# Add docker config data-root

echo '{"data-root": "/home/ubuntu/volume/docker"}' | sudo tee /etc/docker/daemon.json

# Restart docker
sudo systemctl restart docker

####### PIP #######
#install pip
sudo apt-get install python3-pip
#install venv
sudo apt-get install python3-venv
# Install ipykernel
pip install ipykernel

python3 -m venv .IOTH

# Create cache folder for pip
mkdir ~/volume/.cache/
mkdir ~/volume/.cache/pip

# Append directory to venv activate file
echo "export PIP_CACHE_DIR=/home/ubuntu/volume/.cache/pip" >> ~/volume/fedn/examples/IOT/.IOTH/bin/activate


# source venv
source ~/volume/fedn/examples/IOT/.IOTH/bin/activate

# Install packages
pip install -r ~/volume/fedn/examples/IOT/requirements.txt


# Add your virtualenv as a kernel
python -m ipykernel install --user --name=.IOTH --display-name "IOTH"


# Install fedn package
pip install fedn==0.9.0

#install Jupiter notebook
pip install Jupiter notebook

# make client.yaml file
touch client.yaml

# add client.yaml script
echo "network_id: fedn-network" >> ~/volume/fedn/examples/IOT/client.yaml
echo "discover_host: 192.168.2.114" >> ~/volume/fedn/examples/IOT/client.yaml
echo "discover_port: 8092" >> ~/volume/fedn/examples/IOT/client.yaml
# echo "network_id: fedn-network" | sudo tee -a client.yaml
# echo "discover_host: 192.168.2.56" | sudo tee -a client.yaml
# echo "discover_port: 8092" | sudo tee -a client.yaml



