import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import tensor as ts

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import utils
from mdn import mdn_base
from data_handler.data_handler import Data_Handler as DH
from mdn.misc_mixture_density_network import Mixture_Density_Network as MDN

from data_handler.factory_traffic_dataset import load_Map
from util.zfilter import KalmanFilter, model_CV, fill_diag

'''
File info:
    Name    - [gen_pred_json]
    Author  - [Ze]
    Exe     - [Yes]
File description:
    This script generates prediction json files.
'''

print("Program: pred_json\n")

### Customize ###
model_path = './CASE/model/Model_CASE_p5m20_512_256_256_128_64'
json_file = './CASE/traj'
traj_list = utils.read_obj_from_json(json_file)

maxT = 20
past = 5
num_gaus = 10
layer_param = [512, 256, 256, 128, 64]

data_shape = (1, past*2+4)
label_shape = (1, 2)

### Load MDN model ###
myMDN = MDN(data_shape, label_shape, num_gaus=num_gaus, layer_param=layer_param, verbose=False)
myMDN.build_Network()
model = myMDN.model
print(myMDN.layer_param)
model.load_state_dict(torch.load(model_path))
model.eval()

cnt = 0
for traj in traj_list[:]:
    cnt += 1
    obj_type = traj['type'] # get the type of this object

    pred_list_kf = []
    pred_list_mdn = []
    for j in range(past+1, len(traj['traj_x'])-maxT):
        dict_obj_kf = {}  # at one time step
        dict_obj_mdn = {} # at one time step
        data_x = traj['traj_x'][j-past-1:j]
        data_y = traj['traj_y'][j-past-1:j] + [obj_type]
        for i in range(1,maxT+1):
            data = np.array([data_x+[i], data_y]).reshape(-1,order='F')
            label = np.array([traj['traj_x'][j+i-1],traj['traj_y'][j+i-1]])

            data[:-2] = data[:-2]
            label = label

            traj_input = data[:-2]
            tt = data[-2]

            ### Kalman filter
            X0 = np.array([[traj_input[0],traj_input[1],0,0]]).transpose()
            kf_model = model_CV(X0, Ts=1)
            P0 = fill_diag((1,1,10,10))
            Q  = fill_diag((1,1,1,1))
            R  = fill_diag((1,1))
            KF = KalmanFilter(kf_model,P0,Q,R)
            Y = [np.array(traj_input[2:4]), np.array(traj_input[4:6]), 
                np.array(traj_input[6:8]), np.array(traj_input[8:10]), np.array(traj_input[10:12])]
            for kf_i in range(len(Y)+int(tt)):
                if kf_i<len(Y):
                    KF.one_step(np.array([[0]]), np.array(Y[kf_i]).reshape(2,1))
                else:
                    KF.predict(np.array([[0]]),evolve_P=False)
                    KF.append_state(KF.X)
            pred_kf = [[ 1, *[KF.X[0][0], KF.X[1][0]], *[KF.P[0,0],KF.P[1,1]] ]]

            ### MDN
            beta_test = ts(np.tile(np.array([data]), (1,1)).astype(np.float32))
            alp, mu, sigma = model(beta_test)
            alp, mu, sigma = mdn_base.take_mainCompo(alp, mu, sigma, main=5)
            alp, mu, sigma = mdn_base.take_goodCompo(alp, mu, sigma, thre=0.1)
            alp, mu, sigma = alp[0].detach().numpy().astype(float), mu[0].detach().numpy(), sigma[0].detach().numpy()
            alp = alp/sum(alp) # re-normalize the weights
            pred_mdn = []
            for k in range(len(alp)):
                pred_mdn.append([alp[k], *mu[k].tolist(), *sigma[k].tolist()])

            pred_str = 'pred{}'.format(i) # pred1, pred2, ..., predT
            dict_obj_kf['info']  = [j,data[-4],data[-3]]
            dict_obj_kf[pred_str]  = pred_kf
            dict_obj_mdn['info'] = [j,data[-4],data[-3]]
            dict_obj_mdn[pred_str] = pred_mdn
        pred_list_kf.append(dict_obj_kf)
        pred_list_mdn.append(dict_obj_mdn)

    utils.save_obj_as_json(pred_list_kf, '{}_{}_kf'.format(obj_type,cnt))
    utils.save_obj_as_json(pred_list_mdn, '{}_{}_mdn'.format(obj_type,cnt))
