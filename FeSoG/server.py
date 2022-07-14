import torch
import os
import numpy as np
import torch.nn as nn
import dgl
from random import sample
from multiprocessing import Pool, Manager
# from torch.multiprocessing import Pool, Manager
from model import model
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

class server():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr, device, rating_max, rating_min, weight_decay):
        self.user_list_with_coldstart = user_list
        self.user_list = self.generate_user_list(self.user_list_with_coldstart)
        self.batch_size = user_batch
        self.user_embedding = torch.randn(len(users), embed_size).share_memory_()
        self.item_embedding = torch.randn(len(items), embed_size).share_memory_()
        self.model = model(embed_size)
        self.lr = lr
        self.rating_max = rating_max
        self.rating_min = rating_min
        self.distribute(self.user_list_with_coldstart)
        self.weight_decay = weight_decay

    def generate_user_list(self, user_list_with_coldstart):
        ls = []
        for user in user_list_with_coldstart:
            if len(user.items) > 0:
                ls.append(user)
        return ls

    def aggregator(self, parameter_list):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_embedding)
        gradient_user = torch.zeros_like(self.user_embedding)
        loss = 0
        item_count = torch.zeros(self.item_embedding.shape[0])
        user_count = torch.zeros(self.user_embedding.shape[0])

        for parameter in parameter_list:
            [model_grad, item_grad, user_grad, returned_items, returned_users, loss_user] = parameter
            num = len(returned_items)
            item_count[returned_items] += 1
            user_count[returned_users] += num
            loss += loss_user ** 2 * num

            number += num
            if not flag:
                flag = True
                gradient_model = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i] * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i] * num
        loss = torch.sqrt(loss / number)
        print('trianing average loss:', loss)
        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)
        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number
        return gradient_model, gradient_item, gradient_user

    def distribute(self, users):
        for user in users:
            user.update_local_GNN(self.model, self.rating_max, self.rating_min, self.user_embedding, self.item_embedding)

    def distribute_one(self, user):
        user.update_local_GNN(self.model)

    def predict(self, valid_data):
        # print('predict')
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])

        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[i]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)

    def train_one(self, user, user_embedding, item_embedding):
        print(user)
        self.parameter_list.append(user.train(user_embedding, item_embedding))

    def train(self):
        parameter_list = []
        users = sample(self.user_list, self.batch_size)
        # print('distribute')
        self.distribute(users)

        for user in users:
            parameter_list.append(user.train(self.user_embedding, self.item_embedding))

        # print('aggregate')
        gradient_model, gradient_item, gradient_user = self.aggregator(parameter_list)

        ls_model_param = list(self.model.parameters())

        item_index = gradient_item.sum(dim = -1) != 0
        user_index = gradient_user.sum(dim = -1) != 0
        # print('renew')
        for i in range(len(ls_model_param)):
            ls_model_param[i].data = ls_model_param[i].data - self.lr * gradient_model[i] - self.weight_decay * ls_model_param[i].data
        self.item_embedding[item_index] = self.item_embedding[item_index] -  self.lr * gradient_item[item_index] - self.weight_decay * self.item_embedding[item_index]
        self.user_embedding[user_index] = self.user_embedding[user_index] -  self.lr * gradient_user[user_index] - self.weight_decay * self.user_embedding[user_index]
