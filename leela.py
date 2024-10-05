#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

name = "Gukesh"

class ChessValueDataset(Dataset):
    global name
    def __init__(self):
        dat = np.load(f"processed/{name}_1M.npz")
        self.X = dat['arr_0']
        self.Y = dat['arr_1']
        print(self.Y)
        print("loaded", self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class SEBlock(nn.Module):
    def __init__(self, filters, se_channels):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters, se_channels)
        self.fc2 = nn.Linear(se_channels, 2 * filters)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, filters, _, _ = x.size()
        z = self.global_pool(x).view(batch_size, filters)
        
        z = self.relu(self.fc1(z))
        z = self.fc2(z)
        
        # Split into W and B
        w, b = z[:, :filters], z[:, filters:]
        
        # Sigmoid activation on W
        w = self.sigmoid(w).view(batch_size, filters, 1, 1)
        b = b.view(batch_size, filters, 1, 1) 
        
        # SE layer output
        return (w * x) + b

# Residual Block with optional SE layer
class ResidualBlock(nn.Module):
    def __init__(self, filters, se_channels=None):
        super(ResidualBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Second convolution layer
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Conditional SE layer
        self.use_se = se_channels is not None
        if self.use_se:
            self.se = SEBlock(filters, se_channels)
        
    def forward(self, x):
        residual = x  # Save input for skip connection
        
        # Two convolutions with ReLU
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        
        # SE layer (if applicable)
        if self.use_se:
            out = self.se(out)
        
        # Adding residual (skip connection) and applying ReLU
        out += residual
        out = self.relu(out)
        return out

# Residual Tower composed of multiple residual blocks
class ResidualTower(nn.Module):
    def __init__(self, blocks, filters, se_channels=None):
        super(ResidualTower, self).__init__()
        
        # Stacking multiple residual blocks
        self.tower = nn.ModuleList([
            ResidualBlock(filters, se_channels) for _ in range(blocks)
        ])
        
    def forward(self, x):
        for block in self.tower:
            x = block(x)
        return x
class Leela_Network(nn.Module):
    def __init__(self, input_channels=5, filters=112, blocks=4, se_channels=None):
        super(Leela_Network, self).__init__()
        
        # Initial convolution to increase the channel count from input to filters
        self.initial_conv = nn.Conv2d(input_channels, filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # Residual tower
        self.residual_tower = ResidualTower(blocks, filters, se_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # From (Batch, Filters, 8, 8) to (Batch, Filters, 1, 1)
        self.fc = nn.Linear(filters, 1)
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = self.relu(x)
        
        # Residual tower
        x = self.residual_tower(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)  # Flatten (Batch_Size, Filters, 1, 1) -> (Batch_Size, Filters)
        x = self.fc(x).view(-1)
        
        return x



if __name__ == "__main__":
    input_channels = 5  # Starting channels (input dimensions)
    filters = 112  # Number of filters in convolutional layers
    blocks = 4  # Number of residual blocks
    se_channels = 16  # Number of SE channels
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()


    chess_dataset = ChessValueDataset()
    train_loader = DataLoader(chess_dataset, batch_size=256, shuffle=True)
    model = Leela_Network(input_channels=input_channels, filters=filters, blocks=blocks, se_channels=se_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    floss = nn.MSELoss()

    model.train()

    for epoch in range(100):
        all_loss = 0
        num_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)  # (batch_size, 1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            optimizer.zero_grad()
            value_output = model(data)
            loss = floss(value_output, target.view(-1))
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        avg_loss = all_loss / num_loss
        print(f"Epoch {epoch:03d}: Loss = {avg_loss:.6f}")

        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        if not os.path.exists("nets"):
            os.makedirs("nets")

        torch.save(model.state_dict(), f"nets/l_{name.lower()}.pth")

    writer.close()
