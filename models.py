
import torch.nn as nn
import torch.nn.functional as F
class MLPProjectionHeader(nn.Module):
    def __init__(self, input_dim, output_dim=256) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()
        self.l = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.l(x)
        
class CIFAR10Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x