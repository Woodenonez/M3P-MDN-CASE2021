import sys, math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import torch
import torch.tensor as ts

import mdn_base
import CASE.sim_data3
from data_handler import Data_Handler as DH
from misc_mixture_density_network import Mixture_Density_Network as MDN

from filter import KalmanFilter, model_CV, fill_diag

def plot_fok(ax, x, y, yaw=0):
    l_side = .8
    s_side = .8
    outline = np.array([[-l_side/2, l_side/2, l_side/2, -l_side/2, -l_side/2],
                        [s_side/2,  s_side/2, -s_side/2, -s_side/2, s_side/2]])
    rot = np.array([[ math.cos(yaw), math.sin(yaw)],
                    [-math.sin(yaw), math.cos(yaw)]])
    outline = (outline.T.dot(rot)).T
    outline[0, :] += x
    outline[1, :] += y
    ax.plot(np.array(outline[0, :]).flatten(),
            np.array(outline[1, :]).flatten(), "-r")

print("Program: load\n")

df = pd.read_csv('./CASE/data_v3/210309_p5m20_2000s_h.2_665_301_249_2.csv')
data = df.to_numpy()[:,:-2]
data[:,:-2] = data[:,:-2]
labels = df.to_numpy()[:,-2:]

# maxT = 10
past = 5
num_gaus = 10
layer_param = [512, 256, 256, 128, 64]

data_shape = (1, past*2+4)
label_shape = (1, 2)

myMDN = MDN(data_shape, label_shape, num_gaus=num_gaus, layer_param=layer_param, verbose=False)
myMDN.build_Network()
model = myMDN.model
print(myMDN.layer_param)

load_path = './CASE/model_v3/Model_CASE_p5m20_512_256_256_128_64_v3'
model.load_state_dict(torch.load(load_path))
model.eval()


fig, ax = plt.subplots()

# p3m10_2: 1:39615-39660, 2:45084-45179, 3:50748-50919
# p5m10_2: 1: 3189-3249,  2: 8679-8774,  3:54434-54604
idc = np.linspace(1062,1068,num=1259-1247).astype('int') # 474
first_loop = 1
for idx in idc:
    plt.cla()
    CASE.sim_data3.load_Map(ax)
    
    sample = data[idx,:] # x_h, y_h, x_t, y_t, T, type
    label = labels[idx,:]

    # fig,ax = plt.subplots()
    # X0 = np.array([[sample[0],sample[1],0,0]]).transpose()
    # kf_model = model_CV(X0, Ts=1)
    # P0 = fill_diag((1,1,1,1))
    # Q  = fill_diag((1,1,1,1))
    # R  = fill_diag((1,1))
    # KF = KalmanFilter(kf_model,P0,Q,R)
    # Y = [np.array(sample[2:4]), np.array(sample[4:6]), 
    #      np.array(sample[6:8]), np.array(sample[8:10]), np.array(sample[10:12])]
    # U = np.array([[0]])
    # for i in range(len(Y)+int(sample[-2])):
    #     ax.cla()
    #     CASE.sim_data.load_Map(ax)
    #     print(KF.X)
    #     if i<len(Y):
    #         KF.one_step(U, np.array(Y[i]).reshape(2,1))
    #     else:
    #         KF.predict(U,evolve_P=False)
    #         KF.append_state(KF.X)
    #     mdn_base.draw_GauEllipse(ax, ts([KF.X[:2].reshape(-1)])*100, ts([[KF.P[0,0],KF.P[1,1]]]), fc='g', nsigma=1, extend=False)
    #     plt.plot(KF.Xs[0,:]*100,  KF.Xs[1,:]*100,  'bo-')
    #     plt.plot(np.array(Y)[:,0]*100, np.array(Y)[:,1]*100, 'rx')
    #     plt.pause(1)
    # plt.show()
    # sys.exit(0)

    traj_input = sample[:-2]
    T = sample[-2]
    obj_type = sample[-1].astype(int)

    ### Kalman filter
    X0 = np.array([[traj_input[0],traj_input[1],0,0]]).transpose()
    kf_model = model_CV(X0, Ts=1)
    P0 = fill_diag((1,1,10,10))
    Q  = fill_diag((1,1,1,1))
    R  = fill_diag((1,1))
    KF = KalmanFilter(kf_model,P0,Q,R)
    Y = [np.array(traj_input[2:4]), np.array(traj_input[4:6]), 
         np.array(traj_input[6:8]), np.array(traj_input[8:10]), np.array(traj_input[10:12])]
    for kf_i in range(len(Y)+int(T)):
        if kf_i<len(Y):
            KF.one_step(np.array([[0]]), np.array(Y[kf_i]).reshape(2,1))
        else:
            KF.predict(np.array([[0]]),evolve_P=False)
            KF.append_state(KF.X)

    ### MDN

    past_x = traj_input[::2]
    past_y = traj_input[1::2]

    beta_test = ts(np.tile(np.array([sample]), (1,1)).astype(np.float32))
    alp, mu, sigma = model(beta_test)
    alp, mu, sigma = mdn_base.take_mainCompo(alp, mu, sigma, main=5)
    alp, mu, sigma = mdn_base.take_goodCompo(alp, mu, sigma, thre=0.1)

    if obj_type==0:
        sigma = sigma+0.2
    elif obj_type==1:
        sigma = sigma+0.5
    else:
        sigma = sigma+0.6

    plt.plot(past_x[:-1], past_y[:-1], 'k.', label="Past Position")
    plt.plot(traj_input[-2], traj_input[-1], 'ko', label="Current Position")
    plt.plot(label[0], label[1], 'bo', label="Future truth")

    plt.plot(KF.X[0], KF.X[1], 'cx', label="KF Prediction")

    alp_det = alp.detach().numpy()[0]
    alp_idx = np.argsort(alp_det)

    pred_x = mu.detach().numpy()[0][:,0]
    pred_y = mu.detach().numpy()[0][:,1]
    style = ['gx','go','g.','g.','g.']
    for i in range(len(alp_det)):
        if i==0:
            plt.plot(pred_x[alp_idx[-1-i]], pred_y[alp_idx[-1-i]], style[i], label="First MDN Prediction")
        elif i==1:
            plt.plot(pred_x[alp_idx[-1-i]], pred_y[alp_idx[-1-i]], style[i], label="Second MDN Prediction")
        else:
            plt.plot(pred_x[alp_idx[-1-i]], pred_y[alp_idx[-1-i]], style[i], label="Other MDN Prediction")

    plt.xlabel("x [m]", fontsize=14)
    plt.ylabel("y [m]", fontsize=14)
    plt.legend()
    # plt.title("Comparison between the KF and MDN predictions")

    mdn_base.draw_GauEllipse(ax, mu[0], sigma[0], fc='y', nsigma=2, extend=0, label='MDN 2-sigma')
    mdn_base.draw_GauEllipse(ax, mu[0], sigma[0], fc='r', nsigma=1, extend=0, label='MDN 1-sigma')
    # mdn_base.draw_probDistribution(ax, alp, mu*100, sigma, main=None, nsigma=3, colorbar=first_loop)
    # print(sigma[0], ts([[KF.P[0,0],KF.P[1,1]]]))
    mdn_base.draw_GauEllipse(ax, ts([KF.X[:2].reshape(-1)]), ts([[KF.P[0,0],KF.P[1,1]]]), fc='m', nsigma=1, extend=0, label='KF 1-sigma')

    plt.legend(prop={'size': 14}, loc='upper right')

    # plot_fok(ax, label[0], label[1], 0)

    # if idx == idc[-1]:
    #     plt.text(5,5,'Done!',fontsize=20)
    first_loop = 0

    # plt.savefig('{}'.format(idx))
    plt.pause(1)

plt.show()


