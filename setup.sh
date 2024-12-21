#!/bin/bash

# Update pip
pip install --upgrade pip

# Install neuralforecast
pip install neuralforecast

# Install tsfm from GitHub
pip install git+https://github.com/IBM/tsfm.git

# Install Hugging Face Transformers library
pip install transformers

echo "Setup complete. All required packages are installed."
