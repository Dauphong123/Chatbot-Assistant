from torch.utils.data import Dataset
import torch
import numpy as np


class GPTDataset(Dataset):
    def __init__(self, file_path, max_length, stride):
        super(GPTDataset, self).__init__()

        self.token_ids = np.memmap(file_path, dtype=np.uint16, mode="r")

        self.max_length = max_length
        self.stride = stride

        self.num_samples = max(
            0,
            (len(self.token_ids) - max_length - 1) // stride + 1,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start_index = index * self.stride

        input_chunk = self.token_ids[start_index : start_index + self.max_length]

        input_ids = torch.tensor(input_chunk, dtype=torch.long)

        target_chunk = self.token_ids[
            start_index + 1 : start_index + self.max_length + 1
        ]

        target_ids = torch.tensor(target_chunk, dtype=torch.long)

        return input_ids, target_ids
