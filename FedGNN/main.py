import pickle
import torch
import numpy as np
from user import user
from server import server
from sklearn import metrics
import math
import argparse
import warnings
import sys
import faulthandler
faulthandler.enable()
warnings.filterwarnings('ignore')
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="args for FedGNN")
parser.add_argument('--embed_size', type=int, default=8)
parser.add_argument('--lr', type=float, default = 1)
parser.add_argument('--data', default='filmtrust')
parser.add_argument('--user_batch', type=int, default=64)
parser.add_argument('--clip', type=float, default = 0.1)
parser.add_argument('--laplace_lambda', type=float, default = 0.1)
parser.add_argument('--negative_sample', type = int, default = 1000)
parser.add_argument('--valid_step', type = int, default = 20)
args = parser.parse_args()

embed_size = args.embed_size
user_batch = args.user_batch
lr = args.lr

def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)

def loss(server, valid_data):
    label = valid_data[:, -1]
    predicted = server.predict(valid_data)
    mae = sum(abs(label - predicted)) / len(label)
    rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return mae, rmse

# read data
data_file = open('../data/' + args.data + '_FedMF.pkl', 'rb')
[train_data, valid_data, test_data, user_id_list, item_id_list, social] = pickle.load(data_file)
data_file.close()
valid_data = processing_valid_data(valid_data)
test_data = processing_valid_data(test_data)

# build user_list
user_list = []
for u in user_id_list:
    ratings = train_data[u]
    items = []
    rating = []
    for i in range(len(ratings)):
        item, rate, _  = ratings[i]
        items.append(item)
        rating.append(rate)
    user_list.append(user(u, items, rating, list(social[u]), embed_size, args.clip, args.laplace_lambda, args.negative_sample))

# build server
server = server(user_list, user_batch, user_id_list, item_id_list, embed_size, lr)
count = 0

# train and evaluate
rmse_best = 9999
while 1:
    for i in range(args.valid_step):
        print(i)
        server.train()
    print('valid')
    mae, rmse = loss(server, valid_data)
    print('valid mae: {}, valid rmse:{}'.format(mae, rmse))
    if rmse < rmse_best:
        rmse_best = rmse
        count = 0
        mae_test, rmse_test = loss(server, test_data)
    else:
        count += 1
    if count > 5:
        print('not improved for 5 epochs, stop trianing')
        break
print('final test mae: {}, test rmse: {}'.format(mae_test, rmse_test))
