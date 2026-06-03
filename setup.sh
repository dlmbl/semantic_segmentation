#!/bin/bash

# Create environment name based on the exercise name
conda create -n 03-segmentation python=3.11 -y
conda activate 03-segmentation

pip install uv
uv pip install -r requirements.txt

python -m ipykernel install --user --name "03-segmentation"

# Download and extract data, etc.
echo -e "\n downloading data...\n"
aws s3 cp s3://dl-at-mbl-data/2024/02_unet . --recursive --no-sign-request

aws s3 cp "s3://dl-at-mbl-2023-data/woodshole_new.zip" "." --no-sign-request
unzip woodshole_new.zip
mkdir tissuenet_data
mv woodshole_new/* tissuenet_data
rm woodshole_new.zip
rm -r woodshole_new

# Return to base environment
conda deactivate
conda activate base
