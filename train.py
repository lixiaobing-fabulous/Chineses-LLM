from model import GPTConfig, GPT
import torch
import os
import numpy as np

learning_rate = 1e-3
max_iters = 5000
eval_interval = 100
eval_iters = 200
block_size = 32
batch_size = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gpt_conf = GPTConfig()
model = GPT(gpt_conf)
model.to(device)

with open('data/proto_test/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
dataset = 'proto_test'
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + block_size + 1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Trainer:
    def __init__(self, model):
        # self.config = config
        self.model = model

    def run(self):
        model = self.model
        optimizer = model.configure_optimizers(learning_rate)
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(f"stemp {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            xb, yb = get_batch('train')

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


Trainer(model).run()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

# 1.7808 non embedding sharing
# 1.8875 embedding sharing
# 1.8178 non embedding sharing with init by hand
# 1.8113 embedding sharing with init by hand
# 1.8072 embedding sharing with init by hand and apply special scaled init to the residual projection
# 3.3 without residual
# 2.0 without layer normalization
