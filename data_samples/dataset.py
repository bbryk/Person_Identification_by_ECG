import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from itertools import chain
import json
import torch
import torch.nn as nn
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class IDDataset(Dataset):
    def __init__(self, data_folder, to_test, num_samples):
        self.data_paths = []
        self.labels = []
        ind = 0
        b_i = np.load("diplom_test/bad_indices.npy")

        b_i_set = set(b_i)
        empty_dirs = []
        train_range = list(range(1, 600))
        test_range = list(range(601, 1122))

        sample_range = test_range if to_test else train_range
        seed = 1736
        if seed is not None:
            random.seed(seed)
            random.shuffle(sample_range)

        for label in sample_range:
            if label in empty_dirs or label in b_i_set:
                continue
            ind += 1
            class_folder = os.path.join(data_folder, str(label))
            for filename in os.listdir(class_folder):
                if filename.endswith('.npy'):
                    self.data_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(ind)
            if ind > num_samples - 1:
                break

        print(f"IND: {ind}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        data = np.load(file_path)
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, found in file: {file_path}")

        data = data.reshape(1, -1)
        data_tensor = torch.from_numpy(data).float()

        assert not torch.isnan(data_tensor).any(), f"NaN values found in data at index: {idx}, file: {file_path}"

        label = self.labels[idx] - 1
        return data_tensor, label
