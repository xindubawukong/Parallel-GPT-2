
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

import tqdm

# torch.set_num_threads(1)

model.eval()
input_ids = torch.arange(0, 1024)
for i in tqdm.tqdm(range(1000)):
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
        logits = output.logits