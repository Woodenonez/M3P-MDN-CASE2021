import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch import tensor as ts

import mdn_base
from factory_traffic_dataset import load_Map
from data_handler import Data_Handler as DH
from misc_mixture_density_network import Mixture_Density_Network as MDN

import timeit

print("Program: load\n")

model_path = './CASE/model/Model_CASE_p5m20_512_256_256_128_64'
data_path  = './CASE/data/210309_p5m20_2000s_h.2_700_314_249.csv'

df = pd.read_csv(data_path)
data = df.to_numpy()[:,:-2]
data[:,:-2] = data[:,:-2]
labels = df.to_numpy()[:,-2:]

maxT = 20
past = 5
num_gaus = 10
layer_param = [512, 256, 256, 128, 64]

data_shape = (1, past*2+4)
label_shape = (1, 2)

myMDN = MDN(data_shape, label_shape, num_gaus=num_gaus, layer_param=layer_param, verbose=False)
myMDN.build_Network()
model = myMDN.model
print(myMDN.layer_param)
model.load_state_dict(torch.load(model_path))
model.eval() # with BN layer, must run eval first

# fig, ax = plt.subplots()
# load_Map(ax)

print('There are {} samples.'.format(data.shape[0]))
idx = 12598 # change the index here

sample = data[idx,:] # x_h, y_h, x_t, y_t, T, type
label = labels[idx,:]

traj_input = sample[:-2]
T = sample[-2]
obj_type = sample[-1].astype(int)

past_x = traj_input[::2]
past_y = traj_input[1::2]

beta_test = ts(np.array([sample]).astype(np.float32))

start_time = timeit.default_timer()
alp, mu, sigma = model(beta_test)
inference_time = timeit.default_timer() - start_time
print(f'The inference time is ~{round(inference_time*1e3,2)} ms/sample.')

alp, mu, sigma = mdn_base.take_mainCompo(alp, mu, sigma, main=3)
# alp, mu, sigma = mdn_base.take_goodCompo(alp, mu, sigma, thre=0.1)
alp = alp/sum(alp[0])

# plt.plot(traj_input[-2], traj_input[-1], 'bx', label="current")
# plt.plot(label[0], label[1], 'rx', label="future truth")
# plt.plot(past_x[:-1], past_y[:-1], 'kx', label="past")

alp_det = alp.detach().numpy()[0]
alp_idx = np.argsort(alp_det)

pred_x = mu.detach().numpy()[0][:,0]
pred_y = mu.detach().numpy()[0][:,1]
style = ['gx','go','g.','g.','g.']
# for i in range(len(alp_det)):
#     if i==0:
#         plt.plot(pred_x[alp_idx[-1-i]], pred_y[alp_idx[-1-i]], style[i], label="First MDN Prediction")
#     elif i==1:
#         plt.plot(pred_x[alp_idx[-1-i]], pred_y[alp_idx[-1-i]], style[i], label="Second MDN Prediction")
#     else:
#         plt.plot(pred_x[alp_idx[-1-i]], pred_y[alp_idx[-1-i]], style[i], label="Other MDN Prediction")

# plt.xlabel("x"), plt.ylabel("y")
# plt.legend()
# plt.title("MDN CASE Demo")

# mdn_base.draw_probDistribution(ax, alp, mu, sigma, main=None, nsigma=2, step=0.01, colorbar=True) # step should be proper
# plt.show()
