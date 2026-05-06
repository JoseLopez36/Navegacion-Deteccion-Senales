#!/usr/bin/env bash
xhost +local:docker

docker run --rm -it \
  --net host \
  --privileged \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/autoware_data/maps:/home/aw/autoware_data/maps \
  -v $HOME/autoware_data/ml_models:/home/aw/autoware_data/ml_models \
  -v $HOME/autoware:/home/aw/autoware \
  -v $PWD:/home/aw/workspace \
  -w /home/aw/workspace \
  --runtime=nvidia \
  ghcr.io/autowarefoundation/autoware:universe-cuda-jazzy \
  bash -c "source /opt/autoware/setup.bash && exec bash"