import os, sys
import math, random

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_handler.factory_traffic_dataset import *

'''
File info:
    Name    - [gen_sim_data]
    Author  - [Ze]
    Date    - [Sep. 2020] -> [Jan. 2021]
    Exe     - [Yes]
File description:
    This script generates synthetic dataset in the form of CSV files.
File content:
    gen_dataset <func> - Generate a dataset from all objects' trajectories.
Comments:
    # The 'info' input is different for different obj_type
    #
    # obj_type=0 (pedestrian), info=[next, (x1,y1), (x2,y2), ..., (xn,yn)]
    #                               (next=2,...,n is the next path point to go)
    # obj_type=1 (forklift),   info=['d1','d2']
    #                               (direction_1 to direction_2)
    # obj_type=2 (MP),         info=None 
    #                               (it runs in a periodical way)
'''

def gen_dataset(obj_dict, past, maxT):
    '''
    Description:
        Generate a dataset from all objects' trajectories.
    Arguments:
        obj_dict <dict> - A dictionary of MovingObject objects.
        past     <int>  - The number of past steps used.
        maxT     <int>  - Maximal predictive time offset.
    Return:
        data   <ndarry> - Data. 
        labels <ndarry> - Labels. 
        info   <dict>   - A dictionary contains objects information.
    Comments:
        Sample: x_h, y_h, x_t, y_t, T, type; ...; ...
        'data_raw' is a list containing lots of trajectories: np.array(type,type; x_vec,y_vec)
    '''
    data_raw = []
    info = {'#objs':len(list(obj_dict)), '#tp0':0, '#tp1':0, '#tp2':0}
    for name, obj in obj_dict.items():
        if obj.tp == 0:
            info['#tp0'] += 1
        elif obj.tp == 1:
            info['#tp1'] += 1
        elif obj.tp == 2:
            info['#tp2'] += 1
        data_raw.append( np.vstack((np.array([obj.tp, obj.tp]), obj.traj)) ) # traj: np.array(x_vec, y_vec)
    past = int(past or 0)
    data = np.empty((0, 2*past+4), int) # placeholder: x_h, y_h, x, y, T, type (training data)
    labels = np.empty((0,2), int)       # placeholder: x, y (future GT position)
    cnt = 0
    cnt_max = len(data_raw)
    print()
    for traj in data_raw:
        cnt += 1
        if cnt%100 == 0:
            print('\r{}/{}  '.format(cnt,cnt_max), end='')
        obj_type = traj[0,0]
        traj = traj[1:,:] # x, y (two columns)
        for j in range(20, maxT+1): # indicating the maxT
            tempT = traj[0:past+1,:].reshape(1,-1)  # first sample
            labelT = traj[past+1+j-1,:].reshape(1,-1) # first label
            for i in range(1,np.shape(traj)[0]-j-past-1):
                temp = traj[i:i+past+1,:].reshape(1,-1) # +1 is the current position # x_h, y_h, x_t, y_t
                tempT = np.concatenate((tempT, temp), axis=0)  # x_h, y_h, x_t, y_t; ...; ...
                label = traj[i+past+1+j-1,:].reshape(1,-1)
                labelT = np.concatenate((labelT, label), axis=0)

            tempT = np.concatenate((tempT,np.ones((tempT.shape[0],1))*j,np.ones((tempT.shape[0],1))*obj_type), 
                                        axis=1) # x_h, y_h, x_t, y_t, T, type; ...; ...
            data = np.concatenate((data, tempT), axis=0)
            labels = np.concatenate((labels, labelT), axis=0)
    print()
    return data.astype(np.float32), labels.astype(np.float32), info

print("Generate synthetic dataset to CSV.")

save_path = 'out.csv' # save as csv file
past = 5
maxT = 20
h = 0.2 # sampling time, in CASE it is 0.2 [s/time step]
u = 1 # map step, in CASE it is 1 [m/map step]
sim_time = 2000 # [second]

ID = 0
obj_dict_a = {}
obj_dict_h = {}
print()

for k in range(int(sim_time/h)+1): # k is the [time step] (maybe not in second)

    print('\r{}/{}s. Active:{}. History:{}.  '
        .format(int(k*h), sim_time, len(list(obj_dict_a)), len(list(obj_dict_h))), end='')

    if (np.random.poisson(5,1)>8): # Poisson distribution to decide if there is a new pedestrian
        ID += 1
        key = 'obj{}'.format(ID)
        idx = random.randint(0,pedestrian_path()-1) # pedestrian_path() no input -> length of path list
        info = pedestrian_path(idx)
        obj_dict_a[key] = Moving_Object(0, ID, info[1], [1,0], info=info)

    if (np.random.poisson(5,1)>9.5): # Poisson distribution to decide if there is a new forklift
        ID += 1
        key = 'obj{}'.format(ID)
        vv = 2 # speed
        choice = random.choice([1,2,3,4,5])
        if choice in [1]:
            obj_dict_a[key] = Moving_Object(1,ID,[0,1.0],[vv,0],info=['w','n'])
        elif choice in [2]:
            obj_dict_a[key] = Moving_Object(1,ID,[4.5,10.0],[0,-vv],info=['n','w'])
        elif choice in [3]:
            obj_dict_a[key] = Moving_Object(1,ID,[10.0,2.0],[-vv,0],info=['e','n'])
        elif choice in [4]:
            obj_dict_a[key] = Moving_Object(1,ID,[0,1.0],[vv,0],info=['w','e'])
        elif choice in [5]:
            obj_dict_a[key] = Moving_Object(1,ID,[4.5,10.0],[0,-vv],info=['n','e'])
        elif choice in [6]:
            obj_dict_a[key] = Moving_Object(1,ID,[10.0,2.0],[-vv,0],info=['e','w'])
        # choice += 1

    if (k%int(8/h)==0):# & (yes): # AGVs run in a regular period (8s)
        ID += 1
        key = 'obj{}'.format(ID)
        obj_dict_a[key] = Moving_Object(2,ID,[0,5.5],[1,0],info=None)
        yes = 0

    for name in list(obj_dict_a): # check for active objects
        if out_of_border(obj_dict_a[name].p[0], obj_dict_a[name].p[1]) | obj_dict_a[name].terminate:
            obj_dict_h[name] = obj_dict_a[name]
            del obj_dict_a[name]

    for name1, obj1 in obj_dict_a.items(): # check interactions
        for name2, obj2 in obj_dict_a.items():
            obj_dict_a[name1] = interaction(obj1,obj2)
            if obj_dict_a[name1].stop:
                break

    obj_tp_list = []
    obj_shape_list = []
    for name, obj in obj_dict_a.items(): # go one step for all active objects
        obj.one_step(sampling_time=h, map_unit=u, ax=None)
        obj_tp_list.append(obj.tp)
        obj_shape_list.append(obj.shape)

print('\nGenerating dataset...')

data, labels, info = gen_dataset(obj_dict_h, past=past, maxT=maxT) # x_h, y_h, x_t, y_t, T, type; ...; ...
data_df = {}
sample_len = (past+1)*2+2
for i in range(sample_len):
    if i==(sample_len-1):
        data_df['type'] = data[:,i]
    elif i==(sample_len-2):
        data_df['T'] = data[:,i]
    else:
        data_df[str(i)] = data[:,i]
data_df['label_x'] = list(labels[:,0])
data_df['label_y'] = list(labels[:,1])

df = pd.DataFrame(data=data_df)
df.to_csv(save_path, index=False)
print(info)

    