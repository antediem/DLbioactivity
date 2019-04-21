
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
        self.fc2 = nn.Linear(128, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.fc = nn.Linear(21, 1)

    def forward(self, x_mat, x_com):
        x = torch.bmm(x_mat, x_com.unsqueeze(-1)).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x


class UnitedNet(nn.Module):
    def __init__(self, dense_dim, use_mat=True):
        super(UnitedNet, self).__init__()
        self.use_mat = use_mat

        # PARAMS FOR CNN NET
        # Convolutionals
        self.conv_conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv_pool = nn.MaxPool2d(2, 2)
        self.conv_conv2 = nn.Conv2d(6, 16, kernel_size=3)

        # Fully connected
        self.conv_fc1 = nn.Linear(16 * 18 * 72, 120)
        self.conv_fc2 = nn.Linear(120, 84)

        # Batch norms
        self.conv_batch_norm1 = nn.BatchNorm2d(6)
        self.conv_batch_norm2 = nn.BatchNorm2d(16)

        # Dropouts
        self.conv_dropout = nn.Dropout2d()

        # PARAMS FOR DENSE NET
        # Fully connected
        self.dense_fc1 = nn.Linear(dense_dim, 512)
        self.dense_fc2 = nn.Linear(512, 128)
        self.dense_fc3 = nn.Linear(128, 64)

        # Batch norms
        self.dense_batch_norm1 = nn.BatchNorm1d(512)
        self.dense_batch_norm2 = nn.BatchNorm1d(128)

        # Dropouts
        self.dense_dropout = nn.Dropout()

        # PARAMS FOR COMBINED NET
        self.comb_fc1 = nn.Linear(84 + 64, 128)
        self.comb_fc2 = nn.Linear(128, 150)
        self.comb_fc3 = nn.Linear(150, 1)

        # PARAMS FOR ATTENTION NET
        if self.use_mat:
            self.att_fc = nn.Linear(21, 1)

    def forward(self, x_non_mord, x_mord, x_mat):

        # FORWARD CNN
        x_non_mord = F.relu(self.conv_conv1(x_non_mord))
        x_non_mord = self.conv_dropout(x_non_mord)
        x_non_mord = self.conv_batch_norm1(x_non_mord)
        x_non_mord = self.conv_pool(x_non_mord)

        x_non_mord = F.relu(self.conv_conv2(x_non_mord))
        x_non_mord = self.conv_dropout(x_non_mord)
        x_non_mord = self.conv_batch_norm2(x_non_mord)
        # x_non_mord = self.conv_pool(x_non_mord)

        x_non_mord = x_non_mord.view(-1, 16 * 18 * 72)
        x_non_mord = F.relu(self.conv_fc1(x_non_mord))
        x_non_mord = F.relu(self.conv_fc2(x_non_mord))

        # FORWARD DENSE
        x_mord = F.relu(self.dense_fc1(x_mord))
        x_mord = self.dense_batch_norm1(x_mord)
        x_mord = self.dense_dropout(x_mord)

        x_mord = F.relu(self.dense_fc2(x_mord))
        x_mord = self.dense_batch_norm2(x_mord)
        x_mord = self.dense_dropout(x_mord)

        x_mord = F.relu(self.dense_fc3(x_mord))

        # FORWARD COMBINE
        x_comb = torch.cat([x_non_mord, x_mord], dim=1)
        x_comb = F.relu(self.comb_fc1(x_comb))
        x_comb = F.relu(self.comb_fc2(x_comb))

        # FORWARD ATTENTION
        if self.use_mat:
            x_mat = torch.bmm(x_mat, x_comb.unsqueeze(-1)).squeeze(-1)
            x_mat = torch.sigmoid(self.att_fc(x_mat))
            return x_mat
        else:
            x_comb = torch.sigmoid(self.comb_fc3(x_comb))
            return x_comb


