set -x
set -e
nvidia-docker run  --rm --net=host \
                   --shm-size=16g \
                   --env="CUDA_VISIBLE_DEVICES=0,1" \
                   -v /mnt:/mnt ai-studio-registry-vpc.cn-beijing.cr.aliyuncs.com/kube-ai/triton_with_ft:22.03-main-2edb257e-transformers \
                   /opt/tritonserver/bin/tritonserver \
                   --model-repository=/mnt/project/fastertransformer_backend/all_models/bloom