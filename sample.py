import os
import pickle

import torch

from model import GPTConfig, GPT

out_dir = 'out'

device = 'cuda'
dataset = 'proto_test'

if __name__ == '__main__':
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptConf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptConf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
