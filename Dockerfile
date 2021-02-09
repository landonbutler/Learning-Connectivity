FROM tensorflow/tensorflow:1.15.0-gpu-py3


RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libgl1-mesa-glx \
         tmux \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


RUN pip install graph_nets "tensorflow>=1.15,<2" "dm-sonnet<2" "tensorflow_probability<0.9" gym==0.11.0 progress stable_baselines==2.9.0
