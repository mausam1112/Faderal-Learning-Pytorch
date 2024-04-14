import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p) -> None: # k, s, p -> Kernel size, s-stride, p-padding
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        # return F.relu(self.conv(x))
        return F.relu(self.pooling(self.conv(x)))


class ClassificationNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 32, 3, 2, 1)
        self.conv2 = ConvBlock(32, 64, 3, 2, 1)
        self.conv3 = ConvBlock(64, 128, 3, 2, 1)
        self.fc1 = nn.Linear(8960, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)