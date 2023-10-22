import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer

if __name__ == '__main__':
    data_list = ['medical_book_zh.json', 'train_encyclopedia.json']
    doc_ids = []
    tokenizer = ChatGLMTokenizer(vocab_file='../../chatglm_tokenizer/tokenizer.model')
    for file in data_list:
        input_file_path = os.path.join(os.path.dirname(__file__), file)
        with open(input_file_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = json.loads(line)
                text = line['text']
                text_id = tokenizer.encode(text, add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id) > 5:
                    doc_ids += text_id
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
# train has 126,114,189 tokens
# val has 14,012,688 tokens
