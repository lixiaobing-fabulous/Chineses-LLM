import os
import torch
from tokenizer import Tokenizer
from model import GPTConfig, GPT

device = 'cuda'
dataset = 'sft_medical'
init_from_pretrained = True

if __name__ == '__main__':
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    tokenizer = Tokenizer.new_tokenizer(meta_path)

    if init_from_pretrained:
        model = GPT.from_pretrained('gpt2')
        tokenizer = Tokenizer.new_tokenizer('gpt2')

    else:
        ckpt_path = os.path.join(data_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptConf = GPTConfig(**checkpoint['model_args'])
        print(gptConf)
        model = GPT(gptConf)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print(model)
    model.eval()
    model.to(device)
    # context = torch.zeros((1,1), dtype=torch.long, device=device)
    context = torch.tensor(tokenizer.encode("Translator communicated acceleration's "), dtype=torch.long).unsqueeze(0).to(device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
