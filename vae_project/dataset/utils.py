import torch
from torch.utils.data import Subset

def subset(dataset, n:int, random=False):
    subset_indices = torch.randperm(len(dataset))[:n].tolist() if random else range(n)
    return Subset(dataset, subset_indices)
