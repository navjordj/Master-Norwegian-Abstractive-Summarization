FROM pytorch/pytorch:latest
RUN apt-get -y update && apt-get install -y git

RUN apt-get install git-lfs

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel
RUN pip install datasets sacremoses sentencepiece transformers accelerate bitsandbytes git+https://github.com/huggingface/transformers.git

COPY translate.py translate.py 