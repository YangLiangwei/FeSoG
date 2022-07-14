import torch
import torch.nn as nn
from GAT import GraphAttentionLayer

class model(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.GAT_neighbor = GraphAttentionLayer(embed_size, embed_size)
        self.GAT_item = GraphAttentionLayer(embed_size, embed_size)
        self.relation_neighbor = nn.Parameter(torch.randn(embed_size))
        self.relation_item = nn.Parameter(torch.randn(embed_size))
        self.relation_self = nn.Parameter(torch.randn(embed_size))
        self.c = nn.Parameter(torch.randn(2 * embed_size))

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())


    def forward(self, feature_self, feature_neighbor, feature_item):
        if type(feature_item) == torch.Tensor:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            f_i = self.GAT_item(feature_self, feature_item)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_i = torch.matmul(self.c, torch.cat((f_i, self.relation_item)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim = -1)
            e_tensor = torch.stack([e_n, e_i, e_s])
            e_tensor = m(e_tensor)
            r_n, r_i, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n + r_i * f_i
        else:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim = -1)
            e_tensor = torch.stack([e_n, e_s])
            e_tensor = m(e_tensor)
            r_n, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n

        return user_embedding

