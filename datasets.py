import torch
from torch.utils.data import Dataset
import random

class CuboidDataset(Dataset):
    def __init__(self, X, Y_mask, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(Y_mask).float()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # random rotation 0, 90, 180, 270 degrees around z-axis
            k = random.randint(0, 3)
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])

            # random flip along x-axis
            if random.random() > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])

            # random flip along y-axis
            if random.random() > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])

        return x, y