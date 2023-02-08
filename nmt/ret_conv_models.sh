#!/bin/sh

# Delete some working directories
rm -rf ./models
rm -rf ./outs

# Download the requireded models from HuggingFace
git clone https://huggingface.co/jkorsvik/opus-mt-eng-nor ./models/en-no



# Convert the model to ONNX
python3 convert.py -i ./models/en-no -o ./outs/en-no