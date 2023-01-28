FROM pytorch/pytorch:latest
RUN apt-get -y update && apt-get install -y git

RUN apt-get install git-lfs


COPY requirements.txt /tmp
WORKDIR /tmp

RUN pip install -r requirements.txt

WORKDIR /

ADD training root/training