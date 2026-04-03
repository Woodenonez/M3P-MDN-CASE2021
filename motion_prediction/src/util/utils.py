import sys
import json

'''
File info:
    Name    - [utils]
    Author  - [Ze]
    Exe     - [No]
File description:
    Some functions used to handle json files.
File content:
    save_obj_as_json      <func>  - Write a list of "objects" to   a json file.
    read_obj_from_json    <func>  - Read  a list of "objects" from a json file.
    prepare_traj_for_json <func>  - Prepare a dictionary from a MovingObject.
Comments:
    A stardard json file storing "predictions" should be:
    {'info':[t1,x,y], 'pred_T1':[[a1,x1,y1,sx1,sy1], ..., [am,xm,ym,sxm,sym]], 'pred_T2':..., ...}
    {'info':[t2,x,y], 'pred_T1':[[a1,x1,y1,sx1,sy1], ..., [am,xm,ym,sxm,sym]], 'pred_T2':..., ...} ...
    One file for one object. Each row is a pred_t.

    A stardard json file storing "trajectories" should be:
    {'type':type, 'traj_x':[x1,x2,x3,...], 'traj_y':[y1,y2,y3,...]}
    {'type':type, 'traj_x':[x1,x2,x3,...], 'traj_y':[y1,y2,y3,...]} ...
    One file for multiple trajectories. Each row is a trajectory.
'''
def save_obj_as_json(obj_list, json_file_path):
    # pred_list: [pred_t1, pred_t2, ...]
    # traj_list: [traj1,   traj2,   ...]
    with open(json_file_path,'w+') as jf:
        for obj in obj_list:
            json.dump(obj, jf)
            jf.write('\n')

def read_obj_from_json(json_file):
    obj_list = []
    with open(json_file,'r+') as jf:
        for obj in jf:
            try:
                obj_list.append(json.loads(obj))
            except: pass
    return obj_list

'''
Moving_Object.traj - [nx2] np.ndarray: [[x1,y1]; [x2,y2]; [x3,y3]; ...]
Moving_Object.tp   - type of the object
'''
def prepare_traj_for_json(moving_obj):
    traj_json = {}
    traj_json['type'] = int(moving_obj.tp)
    traj_json['traj_x'] = list(moving_obj.traj[:,0].astype(float))
    traj_json['traj_y'] = list(moving_obj.traj[:,1].astype(float))
    return traj_json



if __name__ == '__main__':
    pass



