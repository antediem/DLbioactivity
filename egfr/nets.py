
import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

        # Convolutionals
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)

        # Fully connected
        self.fc1 = nn.Linear(16 * 9 * 36, 120)
        self.fc2 = nn.Linear(120, 84)

        # Batch norms
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)

        # Dropouts
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.batch_norm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 9 * 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DenseNet(nn.Module):
    def __init__(self, input_dim):
        super(DenseNet, self).__init__()

        # Fully connected
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)

        # Batch norms
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(128)

        # Dropouts
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        return x


class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        self.fc1 = nn.Linear(84 + 64, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x



