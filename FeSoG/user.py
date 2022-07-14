import torch
import copy
from random import sample
import torch.nn as nn
import numpy as np
import dgl
import pdb
from model import model

class user():
    def __init__(self, id_self, items, ratings, neighbors, embed_size, clip, laplace_lambda, negative_sample):
        self.negative_sample = negative_sample
        self.clip = clip
        self.laplace_lambda = laplace_lambda
        self.id_self = id_self
        self.items = items
        self.embed_size = embed_size
        self.ratings = ratings
        self.neighbors = neighbors
        self.model = model(embed_size)
        self.graph = self.build_local_graph(id_self, items, neighbors)
        self.graph = dgl.add_self_loop(self.graph)
        self.user_feature = torch.randn(self.embed_size)

    def build_local_graph(self, id_self, items, neighbors):
        G = dgl.DGLGraph()
        dic_user = {self.id_self: 0}
        dic_item = {}
        count = 1
        for n in neighbors:
            dic_user[n] =  count
            count += 1
        for item in items:
            dic_item[item] = count
            count += 1
        G.add_edges([i for i in range(1, len(dic_user))], 0)
        G.add_edges(list(dic_item.values()), 0)
        G.add_edges(0, 0)
        return G

    def user_embedding(self, embedding):
        return embedding[torch.tensor(self.neighbors)], embedding[torch.tensor(self.id_self)]

    def item_embedding(self, embedding):
        return embedding[torch.tensor(self.items)]

    def GNN(self, embedding_user, embedding_item, sampled_items):
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        sampled_items_embedding = embedding_item[torch.tensor(sampled_items)]
        items_embedding_with_sampled = torch.cat((items_embedding, sampled_items_embedding), dim = 0)
        user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
        predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
        self.user_feature = user_feature.detach()
        return predicted

    def update_local_GNN(self, global_model, rating_max, rating_min, embedding_user, embedding_item):
        self.model = copy.deepcopy(global_model)
        self.rating_max = rating_max
        self.rating_min = rating_min
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        if len(self.items) > 0:
            items_embedding = self.item_embedding(embedding_item)
        else:
            items_embedding = False
        user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
        self.user_feature = user_feature.detach()

    def loss(self, predicted, sampled_rating):
        true_label = torch.cat((torch.tensor(self.ratings).to(sampled_rating.device), sampled_rating))
        return torch.sqrt(torch.mean((predicted - true_label) ** 2))

    def predict(self, item_id, embedding_user, embedding_item):
        self.model.eval()
        item_embedding = embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())

    def negative_sample_item(self, embedding_item):
        item_num = embedding_item.shape[0]
        ls = [i for i in range(item_num) if i not in self.items]
        sampled_items = sample(ls, self.negative_sample)
        sampled_item_embedding = embedding_item[torch.tensor(sampled_items)]
        predicted = torch.matmul(self.user_feature, sampled_item_embedding.t())
        predicted = torch.round(torch.clip(predicted, min = self.rating_min, max = self.rating_max))
        return sampled_items, predicted

    def LDP(self, tensor):
        tensor_mean = torch.abs(torch.mean(tensor))
        tensor = torch.clamp(tensor, min = -self.clip, max = self.clip)
        noise = np.random.laplace(0, tensor_mean * self.laplace_lambda)
        tensor += noise
        return tensor

    def train(self, embedding_user, embedding_item):
        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        embedding_user.grad = torch.zeros_like(embedding_user)
        embedding_item.grad = torch.zeros_like(embedding_item)

        self.model.train()
        sampled_items, sampled_rating = self.negative_sample_item(embedding_item)
        returned_items = self.items + sampled_items
        predicted = self.GNN(embedding_user, embedding_item, sampled_items)
        loss = self.loss(predicted, sampled_rating)
        self.model.zero_grad()
        loss.backward()
        model_grad = []
        for param in list(self.model.parameters()):
            grad = self.LDP(param.grad)
            model_grad.append(grad)

        item_grad = self.LDP(embedding_item.grad[returned_items, :])
        returned_users = self.neighbors + [self.id_self]
        user_grad = self.LDP(embedding_user.grad[returned_users, :])
        res = (model_grad, item_grad, user_grad, returned_items, returned_users, loss.detach())
        return res
