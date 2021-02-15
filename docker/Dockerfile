FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libgl1-mesa-glx \
         tmux \
         python3-setuptools \
         python3-pip \
         libglib2.0-0 \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# Must upgrade pip for TF1.15
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==1.15
RUN pip3 install graph_nets dm-sonnet==1.36 tensorflow_probability==0.8.0 gym==0.11.0 progress stable_baselines==2.9.0

