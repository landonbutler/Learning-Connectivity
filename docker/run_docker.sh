#!/bin/bash
image_name=kate_tensorflow
project_volume=/raid0/docker-raid/kate/gnn:/gnn/ 

xhost +local:root
docker run -it \
    --name="kate_tf" \
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

