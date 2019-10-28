import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class LMDataset(Dataset):

    def __init__(self, path):
        super(LMDataset, self).__init__()
        data = np.load(path)
        self.src = torch.from_numpy(data['src']).long()
        self.trg = torch.from_numpy(data['trg']).long()
        self.len = self.src.size(0)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.src[item], self.trg[item]