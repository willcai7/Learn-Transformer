import torch 
import numpy as np 
# from typing import tuple 

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
):

    indices = np.random.choice(len(dataset) - context_length, batch_size)
    input = torch.stack([torch.tensor(dataset[idx:idx + context_length]) for idx in indices])
    target = torch.stack([torch.tensor(dataset[idx + 1: idx + 1 + context_length]) for idx in indices])
    return input.to(device), target.to(device)

class Dataset:
    def __init__(self, dataset_name, context_length, batch_size, device):
        dataset_path = f'data/{dataset_name}'
        slef.train_data = np.memmap(f'{dataset_path}/train.bin', dtype=np.uint16,mode='r').astype(np.int64) 
        self.valid_data = np.memmap(f'{dataset_path}/valid.bin', dtype=np.uint16,mode='r').astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.valid_data
        return get_batch(data, self.batch_size, self.context_length, self.device)