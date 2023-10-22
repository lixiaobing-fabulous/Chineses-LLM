import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer

if __name__ == '__main__':
    alpaca_file_path = os.path.join(os.path.dirname(__file__), 'alpaca_gpt4_data_zh.json')

    with open(alpaca_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    #
    q_lst = []
    a_lst = []
    for per in data:
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)

    belle_file_path = os.path.join(os.path.dirname(__file__), 'Belle_open_source_1M.json')
    f = open(belle_file_path, 'r', encoding='utf-8')

    # s
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    doc_ids = []
    tokenizer = ChatGLMTokenizer(vocab_file='../../chatglm_tokenizer/tokenizer.model')

    for prompt, answer in tqdm(zip(q_lst, a_lst)):
        input_ids = tokenizer.encode(prompt) + [tokenizer.special_tokens['<bos>']] + tokenizer.encode(answer) + [
            tokenizer.special_tokens['<eos>']]
        doc_ids += input_ids
    arr = np.array(doc_ids, dtype=np.uint16)

    n = len(arr)
    train_ids = arr[:int(n * 0.9)]
    val_ids = arr[int(n * 0.9):]
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    meta = {
        'vocab_size': 64793,
        'tokenizer': 'chatglm',
    }

    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
# train has 60,623,536 tokens
# val has 6,735,949 tokens
