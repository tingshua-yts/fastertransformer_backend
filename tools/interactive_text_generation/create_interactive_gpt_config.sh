MODEL_PATH=$1
TP=$2
PP=$3
DATA_TYPE=$4
MODEL_TYPE=$5

if [ $MODEL_PATH ]; then
  :
else
	echo "MODEL_PATH IS NOT EXISTS"
    exit
fi

if [ $TP ]; then
  :
else
	echo "TP IS NOT EXISTS"
    exit
fi

if [ $PP ]; then
  :
else
	echo "PP IS NOT EXISTS"
    exit
fi

if [ $DATA_TYPE ]; then
  :
else
	echo "DATA_TYPE IS NOT EXISTS"
    exit
fi

if [ "$MODEL_TYPE" = "GPT" ]; then
  :
elif [ "$MODEL_TYPE" = "GPT-J" ]; then
  :
elif [ "$MODEL_TYPE" = "GPT-NeoX" ]; then
  :
else
	echo "MODEL_TYPE ${MODEL_TYPE} is not suppoted"
  exit
fi

echo "
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

name: \"fastertransformer\"
backend: \"fastertransformer\"
default_model_filename: \"${MODEL_TYPE}\"
max_batch_size: 1

model_transaction_policy {
  decoupled: False
}

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

sequence_batching {
  max_sequence_idle_microseconds: 60000000
  direct { }
  control_input [
    {
      name: \"START\"
      control [
        {
          kind: CONTROL_SEQUENCE_START
          int32_false_true: [ 0, 1 ]
        }
      ]
    },
    {
      name: \"END\"
      control [
        {
          kind: CONTROL_SEQUENCE_END
          int32_false_true: [ 0, 1 ]
        }
      ]
    }
  ]
}

input [
  {
    name: \"input_ids\"
    data_type: TYPE_UINT32
    dims: [ -1, -1]
  },
  {
    name: \"input_lengths\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
  },
  {
    name: \"request_output_len\"
    data_type: TYPE_UINT32
    dims: [ -1, -1 ]
  },
  {
    name: \"session_len\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"memory_len\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"runtime_top_k\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"runtime_top_p\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"beam_search_diversity_rate\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"temperature\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"len_penalty\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"repetition_penalty\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"random_seed\"
    data_type: TYPE_UINT64
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"is_return_log_probs\"
    data_type: TYPE_BOOL
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"beam_width\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"start_id\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"end_id\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"stop_words_list\"
    data_type: TYPE_INT32
    dims: [ -1, 2, -1 ]
    optional: true
  },
  {
    name: \"bad_words_list\"
    data_type: TYPE_INT32
    dims: [ -1, 2, -1 ]
    optional: true
  },
  {
    name: \"prompt_learning_task_name_ids\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"request_prompt_embedding\"
    data_type: TYPE_FP16
    dims: [ -1, -1, -1 ]
    optional: true
  },
  {
    name: \"request_prompt_lengths\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"request_prompt_type\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"top_p_decay\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"top_p_min\"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  },
  {
    name: \"top_p_reset_ids\"
    data_type: TYPE_UINT32
    dims: [ -1, 1 ]
    reshape: { shape: [ -1 ] }
    optional: true
  }
]
output [
  {
    name: \"output_ids\"
    data_type: TYPE_UINT32
    dims: [ -1, -1, -1 ]
  },
  {
    name: \"sequence_length\"
    data_type: TYPE_UINT32
    dims: [ -1, -1 ]
  },
  {
    name: \"cum_log_probs\"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  },
  {
    name: \"output_log_probs\"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
  }
]
parameters {
  key: \"tensor_para_size\"
  value: {
    string_value: \"${TP}\"
  }
}
parameters {
  key: \"pipeline_para_size\"
  value: {
    string_value: \"${PP}\"
  }
}
parameters {
  key: \"data_type\"
  value: {
    string_value: \"${DATA_TYPE}\"
  }
}
parameters {
  key: \"model_type\"
  value: {
    string_value: \"${MODEL_TYPE}\"
  }
}
parameters {
  key: \"model_checkpoint_path\"
  value: {
    string_value: \"${MODEL_PATH}\"
  }
}
parameters {
  key: \"int8_mode\"
  value: {
    string_value: \"0\"
  }
}
parameters {
  key: \"enable_custom_all_reduce\"
  value: {
    string_value: \"0\"
  }
}" > config.pbtxt