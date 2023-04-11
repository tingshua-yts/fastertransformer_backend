python3 /mnt/project/FasterTransformer/examples/pytorch/gpt/utils/huggingface_bloom_convert.py\
        -i /mnt/model/bloom-560m \
        -o  /mnt/model/bloom-560m-ft\
        -p 64 \
        -tp 2 -v