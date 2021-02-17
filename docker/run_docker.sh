#!/bin/bash
use_gpu=4
image_name=kate_tensorflow
container_name="${USER}_tf${use_gpu}"
project_volume=/raid0/docker-raid/${USER}/gnn:/gnn/ 

xhost +local:root
#export CUDA_VISIBLE_DEVICES=${use_gpu}
docker run -it \
    --name=${container_name}\
    --net=host \
    --privileged \
    -p 0.0.0.0:6006:6006 \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v ${project_volume} \
    ${image_name} \
    /bin/bash
