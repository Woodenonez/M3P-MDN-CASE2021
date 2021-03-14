import sys, math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import torch
import torch.tensor as ts

import mdn_base
from data_handler import Data_Handler as DH
from misc_mixture_density_network import Mixture_Density_Network as MDN

from filter import KalmanFilter, model_CV, fill_diag

def plot_fok(ax, x, y, yaw=0):
    l_side = 10
    s_side = 6
    outline = np.array([[-l_side/2, l_side/2, l_side/2, -l_side/2, -l_side/2],
                        [s_side/2,  s_side/2, -s_side/2, -s_side/2, s_side/2]])
    rot = np.array([[ math.cos(yaw), math.sin(yaw)],
                    [-math.sin(yaw), math.cos(yaw)]])
    outline = (outline.T.dot(rot)).T
    outline[0, :] += x
    outline[1, :] += y
    ax.plot(np.array(outline[0, :]).flatten(),
            np.array(outline[1, :]).flatten(), "-r")

print("Program: evaluation\n")

# data_path = './CASE/data_v3/210309_p5m20_2000s_h.2_665_301_249_2.csv'
data_path = 'eval_data.csv'
model_path = './CASE/model_v3/Model_CASE_p5m20_512_256_256_128_64_v3'

df = pd.read_csv(data_path) # for test
data = df.to_numpy()[:,:-2]
data[:,:-2] = data[:,:-2]
labels = df.to_numpy()[:,-2:]
N = data.shape[0]
print('There are {} samples.'.format(N))

# maxT = 20
past = 5
num_gaus = 10
layer_param = [512, 256, 256, 128, 64]

data_shape = (1, past*2+4)
label_shape = (1, 2)

myMDN = MDN(data_shape, label_shape, num_gaus=num_gaus, layer_param=layer_param, verbose=False)
myMDN.build_Network()
model = myMDN.model
model.load_state_dict(torch.load(model_path))
model.eval()
print(myMDN.layer_param)

first_loop = 1
Loss_MDN = []
Loss1_MDN = []
Loss_KF = []
# fig,ax = plt.subplots()
for idx in range(N):
    
    if (idx%2000 == 0):
        print("\r{}/{} ".format(idx,N), end='')

    sample = data[idx,:] # list: [x_h, y_h, x_t, y_t, T, type]
    label = labels[idx,:]

    ### Kalman filter
    X0 = np.array([[sample[0],sample[1],0,0]]).transpose()
    kf_model = model_CV(X0, Ts=1)
    P0 = fill_diag((1,1,1,1))
    Q  = fill_diag((1,1,1,1))
    R  = fill_diag((1,1))
    KF = KalmanFilter(kf_model,P0,Q,R)
    Y = [np.array(sample[2:4]), np.array(sample[4:6]), 
         np.array(sample[6:8]), np.array(sample[8:10]), np.array(sample[10:12])]
    for kf_i in range(len(Y)+int(sample[-2])):
        if kf_i<len(Y):
            KF.one_step(np.array([[0]]), np.array(Y[kf_i]).reshape(2,1))
        else:
            KF.predict(np.array([[0]]),evolve_P=False)
            KF.append_state(KF.X)

    ### MDN
    beta_test = ts(np.tile(np.array([sample]), (1,1)).astype(np.float32))
    alp, mu, sigma = model(beta_test)
    alp1, mu1, sigma1 = mdn_base.take_mainCompo(alp, mu, sigma, main=1)
    alp, mu, sigma = mdn_base.take_mainCompo(alp, mu, sigma, main=3)
    alp, mu, sigma = mdn_base.take_goodCompo(alp, mu, sigma, thre=0.1)

    ### Loss
    _, loss_KF = mdn_base.loss_MaDist(ts([1]), ts([KF.X[:2].reshape(-1)]),
                                      ts([[KF.P[0,0],KF.P[1,1]]]), ts(label))
    Loss_KF.append(loss_KF.detach().float().item())

    _, loss_MDN = mdn_base.loss_MaDist(alp[0],mu[0],sigma[0],ts(label))
    Loss_MDN.append(loss_MDN.detach().float().item())

    _, loss1_MDN = mdn_base.loss_MaDist(alp1[0],mu1[0],sigma1[0],ts(label))
    Loss1_MDN.append(loss1_MDN.detach().float().item())

    if idx==5000:
        print([KF.X[:2].reshape(-1)], mu1[0])
        fig, ax1 = plt.subplots()   
        h1 = ax1.hist(np.array(Loss_KF)[np.array(Loss_KF)<10], bins=20, alpha=0.5, label='KF')
        h2 = ax1.hist(np.array(Loss_MDN)[np.array(Loss_MDN)<10], bins=20, alpha=0.5, label='MDN')
        h3 = ax1.hist(np.array(Loss1_MDN)[np.array(Loss1_MDN)<10], bins=20, alpha=0.5, label='MDN-1')
        plt.plot(h1[1][:-1]+(h1[1][-1]-h1[1][-2])/2, h1[0],'bx--',label='KF')
        plt.plot(h2[1][:-1]+(h2[1][-1]-h2[1][-2])/2, h2[0],'yx-',label='MDN')
        plt.plot(h3[1][:-1]+(h3[1][-1]-h3[1][-2])/2, h3[0],'gx-',label='MDN-1')
        plt.xlabel('Weighted Mahanobis distance', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        # ax1.set_xlim((0,100))
        # ax1.set_ylim((0,200))
        plt.legend(prop={'size': 20})
        plt.show()
        print('\n', sum(Loss_KF)/len(Loss_KF), sum(Loss1_MDN)/len(Loss1_MDN), sum(Loss_MDN)/len(Loss_MDN))
        print(max(Loss_KF), max(Loss1_MDN), max(Loss_MDN))
        print(sum(np.array(Loss_KF)>5),sum(np.array(Loss1_MDN)>5),sum(np.array(Loss_MDN)>5))
        sys.exit(0)


