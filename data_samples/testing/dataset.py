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
        empty_dirs = [7, 51, 92, 96, 154, 179, 186, 400, 167, 2, 3, 17, 21, 25, 31, 32, 47, 48, 52, 57, 61, 64, 68, 71,
                      74, 77, 83, 104, 108, 114, 121, 124, 129, 130, 132, 134, 144, 146, 149, 154, 157, 159, 167, 169,
                      170, 172, 187, 191, 197, 210, 211, 218, 219, 225, 245, 249, 251, 252, 253, 254, 261, 270, 273,
                      277, 278, 279, 280, 284, 293, 304, 305, 311, 314, 315, 316, 317, 318, 320, 321, 322, 323, 331,
                      335, 351, 359, 362, 370, 373, 377, 380, 384, 388, 389, 391, 392, 394, 399, 402, 416, 423, 424,
                      428, 434, 454, 455, 458, 464, 469, 471, 484, 489, 492, 499, 501, 508, 514, 517, 521, 526, 528,
                      529, 530, 545, 550, 557, 559, 564, 569, 574, 576, 581, 589, 604, 608, 611, 612, 619, 620, 624,
                      627, 629, 638, 653, 659, 662, 665, 667, 670, 678, 681, 683, 684, 688, 699, 702, 709, 723, 724,
                      729, 730, 740, 746, 747, 753, 760, 761, 766, 767, 773, 775, 784, 787, 793, 806, 816, 840, 850,
                      879, 902, 909, 910, 935, 940, 956, 1001, 1011, 1027, 1048, 1075, 1079, 1089]
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
