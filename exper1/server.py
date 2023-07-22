#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: zs
@Description: 描述
@time: 2022/10/5 10:12
@version: 2.0
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN
from clients import ClientsGroup, client
import torch.nn as nn

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
# 用多少个GPU
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
#
parser.add_argument('-cf', '--cfraction', type=float, default=0.2, help='C fraction, 0 means 1 client, 1 means total clients')
# 本地epoch
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
# 本地训练批次
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
# model名字
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
# 模型验证轮次，也就是多少次通信验证一次聚合模型的结果
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
# 多少次通信频率保存一次模型
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
# 通信次数
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
# 保存模型路径
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
# 是否是IID数据,0代表是
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

class mas(object):

  def __init__(self, model: nn.Module, dataloader, device, prev_guards=[None]):
    self.model = model
    self.dataloader = dataloader
    self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
    self.p_old = {}
    self.device = device
    self.previous_guards_list = prev_guards
    self._precision_matrices = self.calculate_importance()

    for n, p in self.params.items():
      self.p_old[n] = p.clone().detach()

  def calculate_importance(self):
    precision_matrices = {}
    for n, p in self.params.items():
      precision_matrices[n] = p.clone().detach().fill_(0)
      for i in range(len(self.previous_guards_list)):
        if self.previous_guards_list[i]:
          precision_matrices[n] += self.previous_guards_list[i][n]

    self.model.eval()
    if self.dataloader is not None:
      num_data = len(self.dataloader)
      for data in self.dataloader:
        self.model.zero_grad()
        output = self.model(data[0].to(self.device))
        l2_norm = output.norm(2, dim=1).pow(2).mean()
        l2_norm.backward()

        for n, p in self.model.named_parameters():
          precision_matrices[n].data += p.grad.data ** 2 / num_data

      precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices

  def penalty(self, model: nn.Module):
    loss = 0
    for n, p in model.named_parameters():
      _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
      loss += _loss.sum()
    return loss

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net  = Mnist_2NN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
    # 分配客户端组
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)

    testDataLoader = myClients.test_data_loader
    # 抽取的客户端的比例
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
    # 全局参数
    global_parameters = {}
    # 网络的参数
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    result = []

    mas = None

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        print(clients_in_comm)
        sum_parameters = None

        for client in clients_in_comm:
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters,mas=None)

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            net.load_state_dict(global_parameters, strict=True)

            # mas_dataset = DataLoader(myClients.clients_set[client].train_ds, batch_size=10, shuffle=True)
            # mas = mas(net, mas_dataset, 'cpu')

            sum_accu = 0
            num = 0
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                preds = net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            print('accuracy: {}'.format(sum_accu / num))
            result.append(sum_accu / num)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
    print('准确率是',result)
