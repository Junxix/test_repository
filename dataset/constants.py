import numpy as np

from utils.constants import *


TO_TENSOR_KEYS = [
    'input_coords_list', 
    'input_feats_list', 
    'action', 
    'action_normalized',
    'human_tracks_abs', 
    'human_tracks_rel', 
    'robot_tracks_abs', 
    'robot_tracks_rel',
    'human_semantics',
    'robot_semantics'
]

REL_TRANS_MAX = 0.1  
REL_GRIPPER_MAX = 0.05  

# camera intrinsics
INTRINSICS = {
    "043322070878": np.array([[909.72656250, 0, 645.75042725, 0],
                              [0, 909.66497803, 349.66162109, 0],
                              [0, 0, 1, 0]]),
    "104122063550": np.array([[914.81945801,   0.        , 630.63891602,   0.        ],
       [  0.        , 913.88464355, 352.51571655,   0.        ],
       [  0.        ,   0.        ,   1.        ,   0.        ]])
}

# inhand camera serial
INHAND_CAM = ["043322070878"]

# transformation matrix from inhand camera (corresponds to INHAND_CAM[0]) to tcp
INHAND_CAM_TCP = np.array([[-8.67318887e-02,-9.96112058e-01,-1.54352617e-02,-4.07872510e-03],
 [ 9.95746677e-01, -8.61958566e-02, -3.25391952e-02,  6.86846646e-02],
 [ 3.10822006e-02, -1.81917964e-02,  9.99351332e-01,  2.52770562e-01],
 [ 2.42187609e-08, -1.13413007e-09,  6.51893954e-08,  1.00000010e+00]])