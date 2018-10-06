#!/usr/bin/env bash

echo "Installing needed libraries"
sudo apt-get update
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl   

sudo pip3 install -r requirements.txt


echo "Downloading CIFAR10 dataset"
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz

echo "Downloading CIFAR100 dataset"
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
