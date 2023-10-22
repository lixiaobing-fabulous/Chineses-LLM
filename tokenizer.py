import pickle

import tiktoken

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


class Tokenizer:
    @staticmethod
    def new_tokenizer(meta_file):
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        type = meta['tokenizer']
        if type == 'simple':
            return SimpleTokenizer(meta)
        elif type == 'gpt2':
            return GPT2Tokenizer()
        elif type == 'chatglm':
            return ChatGlmTokenizer()
        else:
            return GPT2Tokenizer()



class SimpleTokenizer:
    def __init__(self, meta):
        self.meta = meta
        self.stoi, self.itos = meta['stoi'], meta['itos']

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])


class GPT2Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding('gpt2')

    def encode(self, s):
        return self.enc.encode_ordinary(s)

    def decode(self, ids):
        return self.enc.decode(ids)


class ChatGlmTokenizer:
    def __init__(self):
        self.tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

    def encode(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
