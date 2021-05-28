#nets.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv, MFConv
from torch_geometric.nn import global_mean_pool
import os


class GraphNet(nn.Module):
    def __init__(self, dropout):
        super(GraphNet, self).__init__()

        # PARAMS FOR CNN NET
        #Graph convolution
        #self.graph_conv1 = GCNConv(8, 32)
        #self.graph_conv2 = GCNConv(32, 32)
        #self.graph_conv3 = GCNConv(32, 32)
        self.graph_conv1 = MFConv(8, 32)
        self.graph_conv2 = MFConv(32, 32)
        self.graph_conv3 = MFConv(32, 32)

        self.dense_fc1 = nn.Linear(862, 512)
        self.dense_fc2 = nn.Linear(512, 128)
        self.dense_fc3 = nn.Linear(128, 64)

        # Batch norms
        self.dense_batch_norm1 = nn.BatchNorm1d(512)
        self.dense_batch_norm2 = nn.BatchNorm1d(128)
        self.dense_batch_norm3 = nn.BatchNorm1d(64)

        # Dropouts
        self.dense_dropout = nn.Dropout(dropout)
    

        self.linear = nn.Linear(32 + 64, 1)

    def forward(self, x, edge_index, batch, x_mord):
        # FORWARD CNN
        x = self.graph_conv1(x, edge_index)
        x = x.relu()
        x = self.graph_conv2(x, edge_index)
        x = x.relu()
        x = self.graph_conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        
        x_mord = F.relu(self.dense_fc1(x_mord))
        x_mord = self.dense_batch_norm1(x_mord)
        x_mord = self.dense_dropout(x_mord)

        x_mord = F.relu(self.dense_fc2(x_mord))
        x_mord = self.dense_batch_norm2(x_mord)
        x_mord = self.dense_dropout(x_mord)

        x_mord = F.relu(self.dense_fc3(x_mord))
        x_mord = self.dense_batch_norm3(x_mord)
        x_mord = self.dense_dropout(x_mord)

        x = torch.cat([x, x_mord], dim=1)

        return torch.sigmoid(self.linear(x))