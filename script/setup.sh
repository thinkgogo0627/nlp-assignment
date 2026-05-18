#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Create Conda environment with Python 3.10.16
conda create -n Agent python=3.10.16 -y

# Activate the environment
conda activate Agent

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found"
fi
