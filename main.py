import tiktoken # byte-pair tokenizer
import torch
from torch.utils.data import Dataset, DataLoader




class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super(GPTDataset, self).__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index] 

def create_loader(txt, batch_size=4, max_length=128, stride=128, shuffle=True, drop_last=True, num_worker=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt=raw_text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_worker
    )
    return dataloader

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_loader(raw_text, batch_size=8, max_length=4, stride = 4)

vocab_size = 50257
output_dim = 256

data_iter = iter(dataloader)
first_data = next(data_iter)
data, target = first_data

# token embedding
# for every token there is an vector weigth with out_dim in the layer. And for every token ID we do a lookup table
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embedding = token_embedding_layer(data)

# for every pos in the context_lenght have an vector in the weight
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding = pos_embedding_layer(torch.arange(context_length))

input_embedding = token_embedding + pos_embedding
print(input_embedding)