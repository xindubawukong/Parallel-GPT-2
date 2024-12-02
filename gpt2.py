import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import subprocess
import argparse


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, past_key_values=None):
        output = self.model(input_ids,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            use_cache=True
                            )
        return (output.logits, output.past_key_values, output.hidden_states)


def get_gpt2_model(model_name):
    subprocess.call('mkdir -p model', shell=True)

    gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
    gpt2_wrapped = GPT2Wrapper(gpt2)
    traced_model = torch.jit.trace(
        gpt2_wrapped, example_inputs=torch.randint(0, 50257, (1, 20)))

    model_path = f'model/{model_name}_model.pt'
    config_path = f'model/{model_name}_config.json'
    traced_model.save(model_path)
    gpt2.config.to_json_file(config_path)
    print(f'\n\n{model_name} model saved to {model_path}')
    print(f'{model_name} config saved to {config_path}')


def encode_text(model_name, text):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.flatten().tolist()
    print(input_ids)


def decode_text(model_name, input_ids):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input_ids = list(map(int, input_ids.split(' ')))
    print(tokenizer.decode(input_ids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--encode', type=str, default='')
    parser.add_argument('--decode', type=str, default='')

    args = parser.parse_args()

    if args.save_model:
        get_gpt2_model(args.model_name)
    elif args.encode:
        encode_text(args.model_name, args.encode)
    elif args.decode:
        decode_text(args.model_name, args.decode)
    else:
        print('Please provide an argument')


# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# # Load model and tokenizer
# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # Set the model to evaluation mode
# model.eval()

# # Initial prompt to start generation
# prompt = "Once upon a time"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# # Variables to store generated tokens and past key values
# generated_tokens = input_ids
# past_key_values = None

# # Number of tokens to generate
# max_new_tokens = 20

# for i in range(max_new_tokens):
#     # Only pass the last generated token to the model, along with past_key_values
#     outputs = model(input_ids=generated_tokens[:, -1:], past_key_values=past_key_values, use_cache=True)

#     # Get the logits and past_key_values (cache)
#     logits = outputs.logits
#     past_key_values = outputs.past_key_values  # Update cache with new past_key_values

#     # Greedy decoding: pick the token with the highest probability
#     next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

#     # Append the generated token to the sequence
#     generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

#     # Decode and print the generated token
#     generated_text = tokenizer.decode(next_token[0])
#     print(generated_text, end="")

# # Decode the entire generated sequence
# full_generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
# print("\n\nFull generated text:\n", full_generated_text)
