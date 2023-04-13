
import os, sys
import numpy as np
import json
import tritongrpcclient
import argparse
import time


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype='uint8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="ensemble",
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
                        default=True,
                        help='Enable verbose output')
    args = parser.parse_args()

    # 创建client
    try:
        triton_client = tritongrpcclient.InferenceServerClient(
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
    ## 2.1) query
    query="deepspeed is"
    query_np = np.array([[query.encode("utf-8")]])
    inputs.append(tritongrpcclient.InferInput("INPUT_0", query_np.shape, "BYTES"))
    inputs[0].set_data_from_numpy(query_np)
    ## 2.2) bad words
    bad_words_dict="sexy"
    bad_words_np = np.array([[bad_words_dict.encode("utf-8")]])
    inputs.append(tritongrpcclient.InferInput("INPUT_2", bad_words_np.shape, "BYTES"))
    inputs[1].set_data_from_numpy(bad_words_np)
    ## 2.3） stop words
    stop_words_dict="sexy"
    stop_words_np = np.array([[stop_words_dict.encode("utf-8")]])
    inputs.append(tritongrpcclient.InferInput("INPUT_3", stop_words_np.shape, "BYTES"))
    inputs[2].set_data_from_numpy(stop_words_np)
    ## 2.4） output length
    output_len=1024
    output_len_np = np.array([[output_len]], dtype=np.uintc)
    inputs.append(tritongrpcclient.InferInput("INPUT_1", output_len_np.shape, "UINT32"))
    inputs[3].set_data_from_numpy(output_len_np)
    print(f"shapes: {query_np.shape} {bad_words_np.shape} {stop_words_np.shape} {output_len_np.shape} ")

    # 3) 设置output
    outputs = []
    outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT_0"))
    outputs.append(tritongrpcclient.InferRequestedOutput("sequence_length"))
    outputs.append(tritongrpcclient.InferRequestedOutput("cum_log_probs"))
    outputs.append(tritongrpcclient.InferRequestedOutput("output_log_probs"))

    # 4) 发起请求
    start_time = time.time()
    results = triton_client.infer(model_name=args.model_name, inputs=inputs,  outputs=outputs)
    latency = time.time() - start_time

    # 5) 结果处理：转化为numpy 类型，计算max，转化label
    output0_data = results.as_numpy("OUTPUT")
    print(output0_data.shape)
