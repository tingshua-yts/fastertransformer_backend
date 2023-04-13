from transformers import pipeline
from transformers import AutoTokenizer
import torch
# unmasker = pipeline('fill-mask', model='bert-base-uncased')
# res = unmasker("Hello I'm a [MASK] model.")
# print(res)
    # create tokenizer
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom', padding_side='right')
tokenizer.pad_token_id = tokenizer.eos_token_id
def tokeninze(query):


    # encode
    encoded_inputs = tokenizer(query, padding=True, return_tensors='pt')
    input_token_ids = encoded_inputs['input_ids'].int()
    input_lengths = encoded_inputs['attention_mask'].sum(
            dim=-1, dtype=torch.int32).view(-1, 1)
    return input_token_ids.numpy().astype('uint32'), input_lengths.numpy().astype('uint32')

ids1, ids2 = tokeninze("deepspeed is")
print(type(ids1))
decoded = tokenizer.batch_decode(ids1)
print(decoded)