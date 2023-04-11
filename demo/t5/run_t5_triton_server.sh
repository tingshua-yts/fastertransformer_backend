set -x
set -e
nvidia-docker run  --rm --net=host \
                   --env="CUDA_VISIBLE_DEVICES=0,1" \
                   -v /mnt:/mnt ai-studio-registry-vpc.cn-beijing.cr.aliyuncs.com/kube-ai/triton_with_ft:22.03 \
                   /opt/tritonserver/bin/tritonserver \
                   --model-repository=/mnt/project/fastertransformer_backend/demo/t5/model_repo