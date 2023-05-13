FROM pytorch/pytorch:latest
RUN apt-get -y update && apt-get install -y git && apt-get install -y wget

RUN apt-get install git-lfs 


COPY requirements.txt /tmp
WORKDIR /tmp

RUN pip install -r requirements.txt

WORKDIR /

ADD training root/training

WORKDIR root/training

RUN wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;

WORKDIR /

RUN git config --global credential.helper store