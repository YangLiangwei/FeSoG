import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size = (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size = (2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        self.W_1 = nn.Parameter(torch.randn(in_features, out_features))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        W_h = torch.matmul(h, self.W)
        W_adj = torch.mm(adj, self.W)
        a_input = torch.cat((W_h.repeat(W_adj.shape[0], 1), W_adj), dim = 1)
        attention = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)

        attention = F.softmax(attention, dim = -1)
        W_adj_transform = torch.mm(adj, self.W_1)
        h = torch.matmul(attention, W_adj_transform)
        return h
