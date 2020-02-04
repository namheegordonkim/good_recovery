FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python3.7 python3-pip git
RUN pip3 install scikit-learn tqdm torch matplotlib numpy
RUN git clone https://github.com/UBCMOCCA/gym.git /gym

WORKDIR /gym
RUN python3 setup.py develop

RUN mkdir /app
WORKDIR /app

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu