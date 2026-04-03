import os, sys
import math, random

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_handler.factory_traffic_dataset import *
from utils import *

'''
File info:
    Name    - [gen_traj_json]
    Author  - [Ze]
    Exe     - [Yes]
File description:
    This script generates json files of trajectories.
Comments:
    # The 'info' input is different for different obj_type
    #
    # obj_type=0 (pedestrian), info=[next, (x1,y1), (x2,y2), ..., (xn,yn)] 
    #                               (next=2,...,n is the next path point to go)
    # obj_type=1 (forklift),   info=['d1','d2'] 
    #                               (direction_1 to direction_2)
    # obj_type=2 (MP),        info=None 
    #                               (it runs in a periodical way)
'''

print("Generate sample trajectories to JSON.")

save_path = './traj' # save as json file
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

    if (k%int(8/h)==0): # AGVs run in a regular period (8s)
        ID += 1
        key = 'obj{}'.format(ID)
        obj_dict_a[key] = Moving_Object(2,ID,[0,5.5],[1,0],info=None)

    for name in list(obj_dict_a): # check for active objects
        if out_of_border(obj_dict_a[name].p[0], obj_dict_a[name].p[1]) | obj_dict_a[name].terminate:
            obj_dict_h[name] = obj_dict_a[name]
            del obj_dict_a[name]

    ######################################################
    if (len(obj_dict_h)==31): # how many trajectories wanted
        traj_list_json = []
        for name in list(obj_dict_h):
            traj_json = prepare_traj_for_json(obj_dict_h[name])
            traj_list_json.append(traj_json)
        save_obj_as_json(traj_list_json, save_path)
        print()
        break
    ######################################################

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


    