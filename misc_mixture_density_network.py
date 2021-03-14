import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as ts
import torch.optim as optim

import mdn_base
from mdn_base import MDN_Module as MDN # classic_MDN_Module just use exp for sigma
# from mdn_base import classic_MDN_Module as MDN # classic_MDN_Module just use exp for sigma


class Mixture_Density_Network(nn.Module):
    """ 
    """
    def __init__(self, data_shape, labels_shape, num_gaus, layer_param=None, verbose=True):
        super(Mixture_Density_Network, self).__init__()
        self.data_shape = data_shape
        self.labels_shape = labels_shape
        self.is_mdn = (num_gaus is not None)
        self.vb = verbose

        self.dim_input = data_shape[1]
        self.dim_prob = labels_shape[1]
        self.num_gaus = num_gaus
        self.Loss = []
        self.Val_loss= []

        if layer_param is not None:
            self.layer_param = layer_param
        else:
            self.layer_param = [256, 64, 32, 8] # beta: 64,32,8; betaP: 256+beta; One: [512, 256, 64, 64, 32]

    def build_Network(self):
        self.gen_Model()
        self.gen_Optimizer(self.model.parameters())
        if self.vb:
            print(self.model)

    def gen_Model(self):
        lp = self.layer_param
        self.model = nn.Sequential(
                nn.Linear(self.dim_input, lp[0]), 
                nn.ReLU()
                )
        for i in range(len(lp)-1):
            self.model.add_module('Linear'+str(i+1), nn.Linear(lp[i], lp[i+1]))
            self.model.add_module('BN'+str(i+1), nn.BatchNorm1d(lp[i+1], affine=False))
            self.model.add_module('ReLU'+str(i+1), nn.ReLU())
        if self.is_mdn:
            self.model.add_module('MDN', MDN(lp[-1], self.dim_prob, self.num_gaus)) # dim_fea, dim_prob, num_gaus
        else:
            self.model.add_module('Linear'+str(len(lp)), nn.Linear(lp[-1], self.dim_prob)) # dim_fea, dim_prob
        return self.model

    def gen_Optimizer(self, parameters):
        self.optimizer = optim.Adam(parameters, lr=1e-3)
        return self.optimizer

    def predict(self, data):
        alp, mu, sigma = self.model(data)
        return alp, mu, sigma

    def validate(self, data, labels):
        if self.is_mdn:
            alp, mu, sigma = self.model(data)
            loss = mdn_base.loss_NLL(alp, mu, sigma, labels)
        else:
            mse_loss = nn.MSELoss()
            loss = mse_loss(self.model(data), labels)
        return loss

    def train_batch(self, batch, label):
        self.model.zero_grad() # clear the gradient buffer for updating
        loss = self.validate(batch, label)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, data_handler, batch_size, epoch):
        print('\nTraining...')
        dh = data_handler
        data_val, labels_val, _, _ = dh.split_train_val()
        n = int(dh.return_num_data()*(1-dh.val_p))
        cnt = 0
        for ep in range(epoch):
            for _ in range(int(n/batch_size)+1):
                cnt += batch_size
                batch, label = dh.return_batch(batch_size=batch_size)
                loss = self.train_batch(batch, label) # train here
                val_loss = self.validate(ts(data_val).float(), ts(labels_val).float())
                self.Loss.append(loss.item())
                self.Val_loss.append(val_loss.item())
                assert(~np.isnan(loss.item())),("Loss goes to NaN!")
                if (cnt%2000 == 0) & (self.vb):
                    print("\rLoss/Val_loss: {}/{}, {}k/{}k, Epoch {}/{}   ".format(
                        round(loss.item(),4), round(val_loss.item(),4), cnt/1000, n/1000, ep+1, epoch), end='')
            cnt = 0
            print()
        print('\nTraining Complete!')


    def plot_history_loss(self):
        plt.plot(self.Loss, label='loss')
        plt.plot(self.Val_loss, label='val_loss')
        plt.xlabel('#batch')
        plt.legend()
        plt.show()
