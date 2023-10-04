from datasets import load_dataset
import tiktoken

if __name__ == '__main__':
    dataset = load_dataset('openwebtext', num_proc=8)

