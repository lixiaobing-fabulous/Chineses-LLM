import os.path

from datasets import load_dataset
import tiktoken
import numpy as np
from tqdm import tqdm

enc = tiktoken.get_encoding('gpt2')

if __name__ == '__main__':
    dataset = load_dataset('openwebtext', num_proc=8)
    split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=1234, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')


    def process(sentence):
        ids = enc.encode_ordinary(sentence['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out


    tokenized = split_dataset.map(
        process,
        remove_columns='text',
        desc='tokenizing the splits',
        num_proc=8
    )
    for split, data in tokenized.items():
        arr_len = np.sum(data['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # can do since enc.max_token_value = 50256 is < 2**16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = data.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
