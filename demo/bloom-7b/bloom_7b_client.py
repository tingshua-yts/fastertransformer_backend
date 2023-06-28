
import os, sys
#from tkinter import _Padding
import numpy as np
import json
import torch
#import tritongrpcclient
import argparse
import time
from transformers import AutoTokenizer
import tritonclient.grpc as grpcclient

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/model/bloom-7b1', padding_side='right')
tokenizer.pad_token_id = tokenizer.eos_token_id

def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype='uint8')

def tokeninze(query):


    # encode
    encoded_inputs = tokenizer(query, padding=True, return_tensors='pt')
    input_token_ids = encoded_inputs['input_ids'].int()
    input_lengths = encoded_inputs['attention_mask'].sum(
            dim=-1, dtype=torch.int32).view(-1, 1)
    return input_token_ids.numpy().astype('uint32'), input_lengths.numpy().astype('uint32')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="fastertransformer",
                        help="Model name")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    # 创建client
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)


    output_name = "OUTPUT"

    # 1）构造input数据
    # image_data = load_image(args.image)
    # image_data = np.expand_dims(image_data, axis=0)

    # 2) 设置input
    inputs = []
    ## 2.1) input_ids
    query="deepspeed is"
    input_ids, input_lengths = tokeninze(query)
    inputs.append(grpcclient.InferInput("input_ids", input_ids.shape, "UINT32"))
    inputs[0].set_data_from_numpy(input_ids)

    ## 2.2) input_length
    inputs.append(grpcclient.InferInput("input_lengths", input_lengths.shape, "UINT32"))
    inputs[1].set_data_from_numpy(input_lengths)


    ## 2.3） output length
    output_len=32
    output_len_np = np.array([[output_len]], dtype=np.uintc)
    inputs.append(grpcclient.InferInput("request_output_len", output_len_np.shape, "UINT32"))
    inputs[2].set_data_from_numpy(output_len_np)


    # 3) 设置output
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output_ids"))

    # 4) 发起请求
    start_time = time.time()
    results = triton_client.infer(model_name=args.model_name, inputs=inputs,  outputs=outputs)
    latency = time.time() - start_time

    # 5) 结果处理：转化为numpy 类型，计算max，转化label
    output0_data = results.as_numpy("output_ids")
    print(output0_data.shape)
    result = tokenizer.batch_decode(output0_data[0])
    print(result)
