import os
import pickle

import tiktoken
import torch

from model import GPTConfig, GPT

device = 'cuda'
dataset = 'proto_test'

if __name__ == '__main__':
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda l: enc.decode(l)

    ckpt_path = os.path.join(data_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptConf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptConf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
