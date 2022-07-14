import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv

class model(nn.Module):
    def __init__(self, embed_size, head_num):
        super().__init__()
        self.GAT_layer = GATConv(embed_size, embed_size, head_num)

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    # def forward(self, graph, features, item_index):
    #     features = self.GAT_layer(graph, features)
    #     n = features.shape[0]
    #     features = features.reshape(n, -1)

    #     user_embedding = features[0, :]
    #     item_embedding = features[item_index:, :]
    #     return self.predict(user_embedding, item_embedding)

    def forward(self, graph, features_in, item_index):
        features = self.GAT_layer(graph, features_in)
        n = features.shape[0]
        features = features.reshape(n, -1)
        user_embedding = features[0, :]
        return user_embedding

    # def predict_item(self, graph, features, item_index):
    #     features = self.GAT_layer(graph, features)
    #     n = features.shape[0]
    #     features = features.reshape(n, -1)

    #     user_embedding = features[0, :]
    #     item_embedding = features[-1, :]
    #     return self.predict(user_embedding, item_embedding)


