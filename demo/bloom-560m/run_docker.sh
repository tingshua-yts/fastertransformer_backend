#!/bin/bash
set -x
name=pytorch_docker2
image=ai-studio-registry-vpc.cn-beijing.cr.aliyuncs.com/kube-ai/triton_with_ft:22.03-main-2edb257e-transformers
flag=$(sudo docker ps  | grep "$name" | wc -l)
workspace=`pwd`
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --net=host \
	-v /mnt:/mnt --shm-size=16g \
    -w $workspace   -it $image /bin/bash
#    -w /workspace   -it pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel /bin/bash
fi
sudo docker exec -it "$name" bash