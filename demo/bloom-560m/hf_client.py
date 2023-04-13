from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", cache_dir="/mnt/model/bloom-560m")

# tokenize
encoded_inputs = tokenizer("deepspeed is", padding=True, return_tensors='pt')

# forward
model_outputs = model.generate(input_ids=encoded_inputs["input_ids"], attention_mask=encoded_inputs["attention_mask"])

# detokenize
text = tokenizer.decode( model_outputs[0],skip_special_tokens=True,  )
print(f"naive result: {text}")


##### pipeline result
from transformers import pipeline
generator = pipeline("text-generation", model="bigscience/bloom-560m")
result = generator("deepspeed is")
print(f"pipeline result: {result}")