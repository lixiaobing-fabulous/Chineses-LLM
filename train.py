from model import GPTConfig, GPT
import torch
import os
import numpy as np
import pickle


block_size = 32
n_layer = 4
n_head = 4
n_embed = 64

vocab_size = 65
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 5000
eval_interval = 100
eval_iters = 200
batch_size = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    dataset = 'proto_test'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size, bias=bias,
                      vocab_size=vocab_size, dropout=dropout)
    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)
    model.to(device)


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
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                xb, yb = get_batch('train')

                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            checkpoint = {
                'model_args': model_args,
                'model': model.state_dict(),
            }
            print(f"saving checkpoint to {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(data_dir, 'ckpt.pt'))


    Trainer(model).run()
# 1.7808 non embedding sharing
# 1.8875 embedding sharing
# 1.8178 non embedding sharing with init by hand
# 1.8113 embedding sharing with init by hand
# 1.8072 embedding sharing with init by hand and apply special scaled init to the residual projection
# 3.3 without residual
# 2.0 without layer normalization
