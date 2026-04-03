import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import tensor as ts
import torch.nn as nn
import torch.optim as optim

from mdn import mdn_base
from mdn.misc_mixture_density_network import Mixture_Density_Network as MDN

from data_handler.data_handler import Data_Handler as DH

'''
File info:
    Name    - [main_train_case]
    Author  - [Ze]
    Date    - [Sep. 2020] -> [Mar. 2021]
    Exe     - [Yes]
File description:
    The training script.
'''

print("Program: train\n")

### Customize
n_compo = 10
epoch = 2
batch_size = 2
layer_param = [512, 256, 256, 128, 64]

save_path = None #'./CASE/model/NEW_model' # if None, don't save
data_path = './CASE/data/210309_p5m20_2000s_h.2_700_314_249.csv'

### Prepare data
myDH = DH(file_dir=data_path)
print("Data prepared. #Samples:{}".format(myDH.data_shape[0]))

### Initialize the model
myMDN = MDN(myDH.data_shape, myDH.label_shape, n_compo, layer_param=layer_param)
myMDN.build_Network()
model = myMDN.model
opt = myMDN.optimizer

### Train the model
myMDN.train(myDH, batch_size, epoch)
if save_path is not None:
    torch.save(model.state_dict(), save_path)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTraining done: {} parameters.".format(nparams))

### Visualize
myMDN.plot_history_loss()
