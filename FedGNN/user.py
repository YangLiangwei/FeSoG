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
        self.model = model(embed_size, 1)
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

    def GNN(self, embedding_user, embedding_item):
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        features =  torch.cat((self_embedding.unsqueeze(0), neighbor_embedding, items_embedding), 0)
        user_feature = self.model(self.graph, features, len(self.neighbors) + 1)
        self.user_feature = user_feature.detach()
        predicted = torch.matmul(user_feature, items_embedding.t())
        return predicted

    def update_local_GNN(self, global_model):
        self.model = copy.deepcopy(global_model)

    def loss(self, predicted):
        return torch.mean((predicted - torch.tensor(self.ratings))**2)

    def predict(self, item_id, embedding_user, embedding_item):
        item_embedding = embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())

    # def predict(self, item_id, embedding_user, embedding_item):
    #     item_embedding = embedding_item[item_id]
    #     neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
    #     if len(self.items) > 0:
    #         items_embedding = self.item_embedding(embedding_item)
    #         features =  torch.cat((self_embedding.unsqueeze(0), neighbor_embedding, items_embedding, item_embedding.unsqueeze(0)), 0)
    #     else:
    #         features =  torch.cat((self_embedding.unsqueeze(0), neighbor_embedding, item_embedding.unsqueeze(0)), 0)

    #     graph = copy.deepcopy(self.graph)
    #     graph.add_edges(0, len(self.graph.nodes()))
    #     graph.add_edges(len(self.graph.nodes()), 0)
    #     graph.add_edges(len(self.graph.nodes()), len(self.graph.nodes()))
    #     predicted = self.model.predict_item(graph, features, len(self.neighbors) + 1)
    #     return predicted

    def negative_sample_item(self, grad):
        item_num, embed = grad.shape
        ls = [i for i in range(item_num) if i not in self.items]
        sampled_items = sample(ls, self.negative_sample)
        grad_value = torch.masked_select(grad, grad != 0)
        mean = torch.mean(grad_value)
        var = torch.std(grad_value)
        # for item in sampled_items:
        #     grad[item] += torch.randn(self.embed_size) * var + mean
        grad[torch.tensor(sampled_items)] += torch.randn((len(sampled_items), self.embed_size)) * var + mean
        returned_items = sampled_items + self.items
        return returned_items

    def LDP(self, tensor):
        tensor = torch.clamp(tensor, min=-self.clip, max=self.clip)
        loc = torch.zeros_like(tensor)
        scale = torch.ones_like(tensor) * self.laplace_lambda
        tensor = tensor + torch.distributions.laplace.Laplace(loc, scale).sample()
        return tensor

    def train(self, embedding_user, embedding_item):
        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        predicted = self.GNN(embedding_user, embedding_item)
        loss = self.loss(predicted)
        self.model.zero_grad()
        loss.backward()
        model_grad = []
        for param in list(self.model.parameters()):
            grad = self.LDP(param.grad)
            model_grad.append(grad)

        returned_items = self.negative_sample_item(embedding_item.grad)
        item_grad = self.LDP(embedding_item.grad[returned_items, :])
        returned_users = self.neighbors + [self.id_self]
        user_grad = self.LDP(embedding_user.grad[returned_users, :])
        res = (model_grad, item_grad, user_grad, returned_items, returned_users)
        return res
