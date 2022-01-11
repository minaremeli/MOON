import torch.nn as nn
import torch.nn.functional as F


class MLPProjectionHead(nn.Module):
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


class CifarEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Cifar10Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CifarEncoder()
        self.projection_head = MLPProjectionHead(84, 256)
        self.output_layer = OutputLayer(256, 10)

    def forward(self, x):
        proj = self.projection_head(self.encoder(x))
        out = self.output_layer(proj)
        return (proj, out)


class Cifar100Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
