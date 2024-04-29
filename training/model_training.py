import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from itertools import chain
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from itertools import chain
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class ResNet1D(nn.Module):
    def __init__(self):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.resblocks1 = nn.Sequential(
            ResidualBlock(64, 64, kernel_size=7),
            ResidualBlock(64, 64, kernel_size=7),
            ResidualBlock(64, 128, kernel_size=7),
            ResidualBlock(128, 128, kernel_size=7),
            ResidualBlock(128, 256, kernel_size=7),
            # ResidualBlock(256, 256, kernel_size=7)
        )



        self.fc = nn.Linear(256 , 128)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.resblocks1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)



        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AngularSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):

        super(AngularSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # scale parameter
        self.m = m  # margin parameter
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        normalized_weight = F.normalize(self.weight, dim=1)
        normalized_input = F.normalize(input, dim=1)

        cosine = torch.mm(normalized_input, normalized_weight.t())
        clipped_cosine = torch.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        angle = torch.acos(clipped_cosine)

        marginal_angle = angle + self.m
        marginal_cosine = torch.cos(marginal_angle)

        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * marginal_cosine) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = F.cross_entropy(output, label)
        return loss

import random
class IDDataset(Dataset):
    def __init__(self, data_folder,num_train_subjects):
        self.data_paths = []
        self.labels = []
        ind = 0

        train_range = list(range(1, 600))
        sample_range = train_range
        seed = 1736
        if seed is not None:
            random.seed(seed)
            random.shuffle(sample_range)
        for label in sample_range:

            subfolder_names = [int(name) for name in os.listdir(data_folder)
                               if os.path.isdir(os.path.join(data_folder, name))]
            # print(subfolder_names)
            if label not in subfolder_names:
                continue

            print(f"{ind}: {label}")

            ind += 1



            class_folder = os.path.join(data_folder, str(label))
            for filename in os.listdir(class_folder):
                if filename.endswith('.npy'):
                    self.data_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(ind)
            if ind>num_train_subjects-1:
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


import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--num_train_subjects', type=int, default=100, help='Number of training subjects')
    parser.add_argument('--m', type=float, default=0.1, help='Margin parameter m for loss calculation')
    parser.add_argument('--data_folder', type=str, default='../ddiplom_test/git_ecg_samples',
                        help='Data folder containing ECG data')
    args = parser.parse_args()

    print(os.getcwd())
    # for num_train_subjects in [400, 250, 100]:
    num_train_subjects = 100
    m = 0.1
    data_folder = '../diplom_test/git_ecg_samples'

    num_train_subjects = args.num_train_subjects
    m = args.m
    data_folder = args.data_folder


    print(f"num_train: {num_train_subjects}")
    print(f"m: {m}")
    print(f"data_folder: {data_folder}")
    dataset = IDDataset(data_folder, num_train_subjects=num_train_subjects)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    torch.autograd.set_detect_anomaly(True)


    model = ResNet1D()


    criterion = AngularSoftmaxLoss(in_features=128, out_features=num_train_subjects, s=30 , m=0.5)
    optimizer = torch.optim.Adam(chain(model.parameters(), criterion.parameters()), lr=0.001)






    # Hyperparameters
    num_epochs = 5
    k_folds = 2
    batch_size = 32
    # m_values = [0.1, 0.3, 0.5]

    dataset = dataset# your dataset here
    kfold = KFold(n_splits=k_folds, shuffle=True)
    os.makedirs(f"logs_loss", exist_ok=True)

    log_file_path = f"logs_loss/log_{num_train_subjects}_losses"
    with open(log_file_path, 'a') as file:


        file.write("########################################################\n")

        print("Logging completed.")
        # for m in m_values:
        print(f"Testing m = {m}")
        file.write("########################################################\n")

        file.write(f"Testing m = {m}\n")
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            file.write(f"Fold\n")
            train_subsampler = Subset(dataset, train_ids)
            test_subsampler = Subset(dataset, test_ids)
            train_dataloader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)

            model = ResNet1D()
            criterion = AngularSoftmaxLoss(in_features=128, out_features=num_train_subjects, s=30, m=m)
            optimizer = torch.optim.Adam(chain(model.parameters(), criterion.parameters()), lr=0.001)

            epoch_train_losses = []

            model.train()

            for epoch in range(num_epochs):
                c = 0

                running_loss = 0.0
                for inputs, labels in train_dataloader:
                    c += 1
                    print(f"{c}/{len(train_dataloader)}")
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                    optimizer.step()
                    running_loss += loss.item()
                epoch_loss = running_loss / len(train_dataloader)
                epoch_train_losses.append(epoch_loss)
                print(f'Fold {fold +1}, m = {m}, Epoch {epoch +1}, Loss: {running_loss / len(train_dataloader):.4f}')

            model.eval()
            val_loss = 0.0
            epoch_val_losses = []

            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            epoch_val_losses.append(val_loss)
            file.write("Train loss: ")
            file.write(', '.join(map(str, epoch_train_losses)))
            file.write("\n")

            file.write("Validation loss: ")
            file.write(f"{val_loss / len(test_dataloader)}\n")
            file.write("\n")

        file.write("########################################################\n")

        num = num_train_subjects
        # print(f"Avg Validation Loss for m = {m}: {sum(val_losses_per_m[m]) / len(val_losses_per_m[m]):.4f}")
        os.makedirs(f"models/models_{num}", exist_ok=True)
        # os.makedirs(sample_ecg_test_dir, exist_ok=True)
        torch.save(model.state_dict(), f'models/models_{num}/ver2_m_{str(m)[0]+str(m)[2]}_{num}.pth')
        torch.save(criterion.state_dict(), f'models/models_{num}/ver2_m_{str(m)[0]+str(m)[2]}_{num}_criterion.pth')
        print(f'Model saved to models/models_{num}/ver2_m_{str(m)[0]+str(m)[2]}_{num}.pth')
        print(f'Criterion saved to models/models_{num}/ver2_m_{str(m)[0]+str(m)[2]}_{num}_criterion.pth')

        # Determine the best m value
        # average_val_losses = {m: sum(losses) / len(losses) for m, losses in val_losses_per_m.items()}
        # best_m = min(average_val_losses, key=average_val_losses.get)
        # print(f"Best margin parameter m is {best_m} with an average validation loss of {average_val_losses[best_m]:.4f}")



