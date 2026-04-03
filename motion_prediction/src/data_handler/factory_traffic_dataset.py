import sys

import math, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

'''
File info:
    Name    - [factory_traffic_dataset]
    Author  - [Ze]
    Date    - [Sep. 2020] -> [Mar. 2021]
    Exe     - [Yes]
File description:
    Generate moving objects in the factory traffic dataset (FTD).
File content:
    Moving_Object        <class> - Define a moving object.
    random_walk          <func>  - Random walk model.
    coord_turn           <func>  - Coordinated turn model.
    load_Map             <func>  - Load the map.
    out_of_border        <func>  - Judge if a point is out of the map border.
    pedestrian_path      <func>  - Define pathes for pedestrians.
    interaction          <func>  - ...
    interaction_pred     <func>  - ...
    interaction_pred_out <func>  - ...
    interaction_dire     <func>  - ...
Comments:
    10x10 meters area: working areas, sidewalks, lanes.
    Pedestrian(0), forklift(1), mobile platform-MP(2).
    I'm too tired to make more comments...
'''

class Moving_Object():
    '''
    # The 'info' input is different for different obj_type
    #
    # obj_type=0 (pedestrian), info=[next, (x1,y1), (x2,y2), ..., (xn,yn)] 
    #                               (next=2,...,n is the next path point to go)
    # obj_type=1 (forklift),   info=['d1','d2'] 
    #                               (direction_1 to direction_2)
    # obj_type=2 (MP),         info=None 
    #                               (it runs in a periodical way)
    '''
    def __init__(self, obj_type, ID, current_pos, velocity, info, priority=0):
        super(Moving_Object, self).__init__()
        self.tp = obj_type
        self.id = ID
        self.p = np.array(current_pos) # [map unit]
        self.v = np.array(velocity)    # [m/s]
        self.info = info
        self.pr = priority
        self.shape = None

        self.size_param = {0:.2, 1:(1.0,.6), 2:(1.6,.8)} # type 0 - circle, type 1/2 - rectangle
        self.color_param = {'ec':'r', 'fc':'y'}

        self.terminate = False # if the obj reaches its end of life?
        self.stop = False      # if the obj is in a pause now

        if self.tp == 0:
            self.model = self.model_pedestrian
            self.info[0] = 2 # ensure the next path point is valid
        elif self.tp == 1:
            self.model = self.model_forklift
        elif self.tp == 2:
            self.model = self.model_MP

        self.traj = np.array([current_pos])

    def one_step(self, sampling_time=1, map_unit=1, ax=None):
        # sampling time unit is [second]
        # map size unit is [meter]
        # only plot if ax is specified
        self.p, shape = self.model(sampling_time, map_unit, stay=self.stop) # forklift returns yaw rather than shape
        self.traj = np.vstack((self.traj, self.p)) # add the current position to the trajectory
        if ax is not None:
            self.plot_result(ax, shape)
        if self.tp == 1:
            shape = self.plot_fok(self.p[0],self.p[1],yaw=shape,ax=None) # extended shape
        self.shape = shape

    def model_MP(self, h, u, stay=False): 
        # h=sampling time [s/time step], u=map unit [m/map step]
        # v=velocity [m/s] --> v*h/u [map step/time step]
        l_side = self.size_param[self.tp][0]
        s_side = self.size_param[self.tp][1]
        if stay:
            centre = random_walk(self.p, [0,0])
        else:
            centre = random_walk(self.p, self.v*h/u)
        shape = patches.Rectangle( (centre[0]-l_side/2, centre[1]-s_side/2), l_side, s_side, 
                                   ec=self.color_param['ec'], fc=self.color_param['fc'] )
        return centre, shape

    def model_pedestrian(self, h, u, stay=False):
        # info=[next, (x1,y1), (x2,y2), ..., (xn,yn)]
        # h=sampling time [s/time step], u=map unit [m/map step]
        # v=velocity [m/s] --> v*h/u [map step/time step]
        turn_rate = math.pi/12
        turn_prob = 0.1
        next_goal = np.array(self.info[self.info[0]])
        if (np.linalg.norm(next_goal-self.p)<0.3/u):
            if (self.info[0]==(len(self.info)-1)):
                self.terminate = True
                return self.p, self.shape
            else:
                self.info[0] += 1

        head = (next_goal-self.p)/np.linalg.norm(next_goal-self.p)
        speed = np.linalg.norm(self.v)
        speed = min(1,max(0.4, speed+random.uniform(-0.1,0.2))) # the speed is in [0.2,1] m/s
        self.v = speed*head
        if stay:
            centre = random_walk(self.p, [0,0])
        else:
            centre = random_walk(self.p, self.v*h/u, turn_rate=turn_rate, turn_prob=turn_prob)
        shape = patches.Circle(centre, radius=self.size_param[self.tp], 
                               ec=self.color_param['ec'], fc='b')
        return centre, shape

    def model_forklift(self, h, u, stay=False):
        # the shape info in plot_fok
        # h=sampling time [s/time step], u=map unit [m/map step]
        # v=velocity [m/s] --> v*h/u [map step/time step]
        start_dir = self.info[0]
        goal_dir = self.info[1]
        last_v = self.v
        speed = np.linalg.norm(self.v)
        turn_rate = 10*math.pi/21*h*speed/2/2 # math.pi/21 is designed for 2m/s
        if   start_dir == 'n':
            if goal_dir == 'w':
                if self.p[1] > 3.3:
                    self.v = np.array([0, -speed])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = -math.pi/2
                elif (self.v[1]<-0.001):
                    centre = coord_turn(self.p, self.v*h/u/2, -turn_rate)
                    self.v = (centre-self.p)/h*u*2
                    yaw = math.atan2(self.v[1],self.v[0])
                else:
                    self.v = np.array([-speed, 0])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = math.pi
            else: # go 'e'
                if self.p[1] > 2.3:
                    self.v = np.array([0, -speed])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = -math.pi/2
                elif (self.v[1]<-0.001) | (self.p[1]>1.3):
                    centre = coord_turn(self.p, self.v*h/u/2, turn_rate)
                    self.v = (centre-self.p)/h*u*2
                    yaw = math.atan2(self.v[1],self.v[0])
                else:
                    self.v = np.array([speed, 0])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = math.pi
        elif start_dir == 'w':
            if goal_dir == 'e':
                self.v = np.array([speed, 0])
                centre = random_walk(self.p, self.v*h/u)
                yaw = 0
            else: # go 'n'
                if self.p[0] < 4.3:
                    self.v = np.array([speed, 0])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = 0
                elif (self.v[0]>0.001):# | (self.p[0]<55):
                    centre = coord_turn(self.p, self.v*h/u/2, turn_rate)
                    self.v = (centre-self.p)/h*u*2
                    yaw = math.atan2(self.v[1],self.v[0])
                else:
                    self.v = np.array([0, speed])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = math.pi/2
        elif start_dir == 'e':
            if goal_dir == 'w':
                self.v = np.array([-speed, 0])
                centre = random_walk(self.p, self.v*h/u)
                yaw = math.pi
            else: # go 'n'
                if self.p[0] > 6.8:
                    self.v = np.array([-speed, 0])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = math.pi
                elif (self.v[0]<-0.001):# | (self.p[0]<54):
                    centre = coord_turn(self.p, self.v*h/u/2, -turn_rate)
                    self.v = (centre-self.p)/h*u*2
                    yaw = math.atan2(self.v[1],self.v[0])
                else:
                    self.v = np.array([0, speed])
                    centre = random_walk(self.p, self.v*h/u)
                    yaw = math.pi/2
        if stay:
            centre = self.p
            self.v = last_v

        return centre,yaw

    def plot_result(self, ax, shape):
        if self.tp == 1:
            shape = self.plot_fok(self.p[0],self.p[1],yaw=shape,ax=ax) # extended shape
        else:
            ax.add_patch(shape)
        ax.plot(self.p[0], self.p[1], 'rx')

    def plot_fok(self, x, y, yaw, ax=None):
        l_side = self.size_param[self.tp][0]
        s_side = self.size_param[self.tp][1]
        outline = np.array([[-l_side/2, l_side/2, l_side/2, -l_side/2, -l_side/2],
                            [s_side/2,  s_side/2, -s_side/2, -s_side/2, s_side/2]])
        rot = np.array([[ math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(rot)).T
        outline[0, :] += x
        outline[1, :] += y
        if ax is not None:
            ax.plot( np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-r" )
        shape = patches.Rectangle( (x-(s_side+l_side)/4, y-(s_side+l_side)/4), 
                                      (s_side+l_side)/2,   (s_side+l_side)/2 )
        return shape    


def random_walk(current_pos, velocity, turn_rate=0, turn_prob=0): # default CV model
        x = current_pos[0]
        y = current_pos[1]
        if random.randint(0,1)>=turn_prob: # keep direction
            x = x + velocity[0]
            y = y + velocity[1]
        else: # random turn
            w = math.atan2(velocity[1],velocity[0])
            if random.randint(0,1)>0.5: # right, otherwise left
                turn_rate = -turn_rate
            x = x + np.linalg.norm(velocity) * np.cos(turn_rate+w)
            y = y + np.linalg.norm(velocity) * np.sin(turn_rate+w)
        return np.array([x,y])

def coord_turn(current_pos, velocity, turn_rate): # CT model
    x = current_pos[0]
    y = current_pos[1]
    w = math.atan2(velocity[1],velocity[0])
    vx = np.linalg.norm(velocity) * np.cos(turn_rate+w)
    vy = np.linalg.norm(velocity) * np.sin(turn_rate+w)
    return np.array([x+vx, y+vy])

def noise(shape, amp=1):
    return amp*np.random.standard_normal(shape)


def load_Map(ax):
    rec1 = patches.Rectangle(( 0, 2.5), 4.0, 2.0)
    rec2 = patches.Rectangle(( 0, 6.5), 4.0, 3.5)
    rec3 = patches.Rectangle((6.0, 2.5), 4.0, 2.0)
    rec4 = patches.Rectangle((6.0, 6.5), 4.0, 3.5)
    rec5 = patches.Rectangle((0, 0), 10.0, .5)
    Rec1 = [rec1, rec2, rec3, rec4, rec5] # map - static obstacles 1 （grey）

    rec1 = patches.Rectangle(( 0, 3.0), 3.5, 1.5)
    rec2 = patches.Rectangle(( 0, 6.5), 3.5, 3.5)
    rec3 = patches.Rectangle((6.5, 3.0), 3.5, 1.5)
    rec4 = patches.Rectangle((6.5, 6.5), 3.5, 3.5)
    Rec2 = [rec1, rec2, rec3, rec4] # map - static obstacles 2

    pc1 = PatchCollection(Rec1, color=[0.5,0.5,0.5])
    pc2 = PatchCollection(Rec2, color=[0,0,0])
    
    ax.add_collection(pc1)
    ax.add_collection(pc2)
    ax.axis('equal')
    ax.set(xlim=(-.5, 10.5), ylim=(-.5, 10.5))

    ax.text( 5.0, 10.0,'N')
    ax.text( 5.0,  -.4,'S')
    ax.text( -.5,  5.0,'W')
    ax.text(10.0,  5.0,'E')

    # plt.plot((0,100),(0,0),'k') # borders
    ax.plot((0,10.0),(5.5,5.5),'y--') # ref line 1 - AGV

def out_of_border(x,y):
    if (x>11.0) | (x<-1.0) | (y>11.0) | (y<-1.0):
        return True
    else:
        return False 

def pedestrian_path(idx=-1):
    '''
            p7  p12
            p6  p11
        AGV-->-->-->-->    
            p5  p10
        p2  p4  p9  p14
        p1  p3  p8  p13

        (idx=-1 will return the number of possible pathes)
    '''
    pp = [(.3,.3), (.3,2.7), (3.8,.3), (3.8,2.7),(3.8,4.5),(3.8,6.5),(3.8,9.7),
          (6.2,.3),(6.2,2.7),(6.2,4.5),(6.2,6.5),(6.2,9.7),(9.7,.3), (9.7,2.7)]
    # group 1
    P1_1 = [2, pp[0], pp[12]]
    P1_2 = [2, pp[12], pp[0]]
    P2_1 = [2, pp[0], pp[2], pp[6]]
    P2_2 = [2, pp[6], pp[2], pp[0]]
    P3_1 = [2, pp[12], pp[7], pp[11]]
    P3_2 = [2, pp[11], pp[7], pp[12]]
    P4_1 = [2, pp[0], pp[7], pp[11]]
    P4_2 = [2, pp[11], pp[7], pp[0]]
    P5_1 = [2, pp[12], pp[2], pp[6]]
    P5_2 = [2, pp[6], pp[2], pp[12]]
    # group 2
    P6_1 = [2, pp[1], pp[13]]
    P6_2 = [2, pp[13], pp[1]]
    P7_1 = [2, pp[1], pp[3], pp[6]]
    P7_2 = [2, pp[6], pp[3], pp[1]]
    P8_1 = [2, pp[13], pp[8], pp[11]]
    P8_2 = [2, pp[11], pp[8], pp[13]]
    P9_1 = [2, pp[1], pp[8], pp[11]]
    P9_2 = [2, pp[11], pp[8], pp[1]]
    P10_1 = [2, pp[13], pp[3], pp[6]]
    P10_2 = [2, pp[6], pp[3], pp[13]]
    # use the upper two crosses
    P11_1 = [2, pp[1], pp[3], pp[5], pp[10], pp[11]]
    P11_2 = [2, pp[11], pp[10], pp[5], pp[3], pp[1]]
    P12_1 = [2, pp[13], pp[8], pp[10], pp[5], pp[6]]
    P12_2 = [2, pp[6], pp[5], pp[10], pp[8], pp[13]]
    # path list
    path_list = [P1_1, P1_2, P2_1, P2_2, P3_1, P3_2, P4_1, P4_2, P5_1, P5_2,
                 P6_1, P6_2, P7_1, P7_2, P8_1, P8_2, P9_1, P9_2, P10_1, P10_2,
                 P11_1, P11_2, P12_1, P12_2]
    if idx==-1:
        return len(path_list)
    else:
        return path_list[idx]


def angle_vectors(vec1,vec2):
    return np.arccos(np.dot(obj1.v/np.linalg.norm(obj1.v),obj2.v/np.linalg.norm(obj2.v)))

def interaction(obj1,obj2):
    # 0-Pedestrain, 1-Forklift, 2-AGV
    safe_dist = [1.5, 2.0, 2.5, 2.0] # 0-1, 0-2, 1-1, 1-2
    obj1.stop = False
    if   obj1.tp == 0:
        if obj2.tp == 1:    # ped -> fok
            if interaction_dire(obj1,obj2) | (obj1.p[1]<.5):
                return obj1
            elif (obj1.p[1]>2.0) & (obj1.p[1]<3.5) & (obj1.p[0]>3.5) & (obj1.p[0]<6.5):
                if interaction_pred(obj1,obj2,safe_dist[0],scope=4):
                    if not interaction_pred_out(obj1,obj2,scope=2):
                        obj1.stop = True
            elif (np.linalg.norm(obj1.p-obj2.p) < safe_dist[0]):
                if not interaction_pred_out(obj1,obj2,scope=2):
                    obj1.stop = True
        elif obj2.tp == 2:  # ped -> AGV
            if not interaction_dire(obj1,obj2):
                if interaction_pred(obj1,obj2,safe_dist[0],scope=5):
                    obj1.stop = True
                    if ((obj1.v[1]>0) & (obj1.p[1]>5.4)) or ((obj1.v[1]<0) & (obj1.p[1]<5.6)):
                        obj1.stop = False
                        obj1.v = np.array([1,1])
                    if ((obj1.v[1]>0) & (obj1.p[1]<4.5)) or ((obj1.v[1]<0) & (obj1.p[1]>6.5)):
                        obj1.stop = False
    elif obj1.tp == 1:
        if (obj2.tp == 1): # fok -> fok
            if not (np.abs(angle_vectors(obj1.v, obj2.v))>0.7*math.pi):
                if np.linalg.norm(obj1.p-obj2.p) < safe_dist[2]:
                    if obj1.pr==obj2.pr:
                        obj1.pr += 1
                    elif obj1.pr<obj2.pr:
                        obj1.stop = True
        if (obj2.tp == 2): # fok -> AGV
            if interaction_pred(obj1,obj2,safe_dist[3],scope=4):
                if not interaction_pred_out(obj1,obj2,scope=5):
                    obj1.stop = True
    return obj1

def interaction_pred(obj1,obj2,safe_dist,scope=1):
    v1 = obj1.v
    v2 = obj2.v
    if np.linalg.norm(obj1.v)<0.5:
        v1 = 2*v1
    for i in range(scope+1):
        p1 = obj1.p + scope/10 * v1
        p2 = obj2.p + scope/10 * v2
        if (np.linalg.norm(p1-p2) < safe_dist):
            return True
    return False

def interaction_pred_out(obj1,obj2,scope=1):
    v1 = obj1.v
    p1 = obj1.p + scope/10 * obj1.v
    p2 = obj2.p
    if (np.linalg.norm(p1-p2) < np.linalg.norm(obj1.p-obj2.p)):
        return False
    return True

def interaction_dire(obj1,obj2):
    flag = (np.abs(angle_vectors(obj1.v, obj2.v))>0.9*math.pi) | (np.abs(angle_vectors(obj1.v, obj2.v))<0.1*math.pi)
    return flag


if __name__ == "__main__":
    '''
    # The 'info' input is different for different obj_type
    #
    # obj_type=0 (pedestrian), info=[next, (x1,y1), (x2,y2), ..., (xn,yn)] 
    #                               (next=2,...,n is the next path point to go)
    # obj_type=1 (forklift),   info=['d1','d2'] 
    #                               (direction_1 to direction_2)
    # obj_type=2 (MP),        info=None 
    #                               (it runs in a periodical way)
    '''

    print("Test demo data generator (CASE2021).")

    boost = 1
    h = 0.2 # sampling time, in CASE it is 0.2 [s/time step]
    u = 1 # map step, in CASE it is 1 [m/map step]
    sim_time = 2000 # [second]

    fig, ax1 = plt.subplots(1,1)

    ID = 0
    obj_dict_a = {}
    obj_dict_h = {}
    print()

    for k in range(int(sim_time/h)+1): # k is the [time step] (maybe not in second)

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

        ax1.cla()
        load_Map(ax1)
        ax1.set_xlabel("x [m]", fontsize=14), ax1.set_ylabel("y [m]", fontsize=14)

        legend = [  Line2D([0], [0], linestyle='--', color='y', label='Reference Line'),
                    Line2D([0], [0], color='m', label='Traversed Path'),
                    Line2D([0], [0], marker='x', color='r', label='Object\'s Centre'),
                    patches.Patch(ec='r', fc='b', label='Pedestrian'),
                    patches.Patch(ec='r', fc='w', label='Forklift'),
                    patches.Patch(ec='r', fc='y', label='Mobile Platform'),
                    patches.Patch(fc='grey', label='Sidewalk'),
                    patches.Patch(fc='k', label='Working Area')]
        ax1.legend(handles=legend, prop={'size': 14}, loc='upper right')
        ax1.set_title('The Factory Traffic Dataset (FTD) Simulator, t={}'.format(round(k*h,1)))

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
            obj.one_step(sampling_time=h, map_unit=u, ax=ax1)
            if obj.tp != 2: # if not AGV, then plot traj
                ax1.plot(obj.traj[:,0], obj.traj[:,1], 'm', alpha=0.5)
            obj_tp_list.append(obj.tp)
            obj_shape_list.append(obj.shape)
        
        plt.pause(h/(max(boost*100,1)))

    plt.show()
