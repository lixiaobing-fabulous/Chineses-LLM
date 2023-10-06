import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embed: int = 64
    dropout: float = 0.0
    bias: bool = False


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to openai GPT)
    Reference: Gaussian Error Linear Units (GELU)  paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_length, embedding dimensionality
        attn_layer = self.c_attn(x)  # calculate query, key, values for all heads
        q, k, v = attn_layer.split(self.n_embed, dim=2)  # split query, key, values
        q = q.view(B, T, self.n_head, self.n_embed // self.n_head)  # (B, T, n_head, head_size)
        k = k.view(B, T, self.n_head, self.n_embed // self.n_head)  # (B, T, n_head, head_size)
        v = v.view(B, T, self.n_head, self.n_embed // self.n_head)  # (B, T, n_head, head_size)

        q = q.transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.transpose(1, 2)  # (B, n_head, T, head_size)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v  # （B, nh, T, T) x (B, nh, T, hs)->(B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        # do some validation
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.n_embed is not None
        assert config.n_layer is not None
        assert config.dropout is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),
            wpe=nn.Embedding(config.block_size, config.n_embed),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embed, config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        # self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('project.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, learning_rate):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        word_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(word_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
            'gpt-xl': dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
        }[model_type]
        print('forcing vocab_size=50257, block_size=1024, bias=True')
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_huggingface = GPT2LMHeadModel.from_pretrained(model_type)
        sd_huggingface = model_huggingface.state_dict()
        sd_keys_huggingface = sd_huggingface.keys()
        sd_keys_huggingface = [k for k in sd_keys_huggingface if not k.endswith('.attn.masked_bias')]
        sd_keys_huggingface = [k for k in sd_keys_huggingface if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_huggingface) == len(
            sd_keys), f"mismatched keys: {len(sd_keys_huggingface)} != {len(sd_keys)}"
        for k in sd_keys_huggingface:
            if any(k.endswith(w) for w in transposed):
                assert sd_huggingface[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_huggingface[k].t())
            else:
                assert sd_huggingface[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_huggingface[k])
        return model

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_truncated = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_truncated)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
