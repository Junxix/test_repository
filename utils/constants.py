import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.7, -0.8, 0.1]), np.array([0.7, 0.6, 1.5]) 
TRACK_MIN, TRACK_MAX = np.array([-0.7, -0.8, -0.15]), np.array([0.7, 0.6, 1.5]) 

# 相对运动的范围 (通过compute_relative_track_statistics.py计算得出)
# 默认值,运行统计脚本后更新
REL_TRACK_MIN = np.array([-0.15, -0.3, -0.5])
REL_TRACK_MAX = np.array([0.15, 0.15, 0.3])

MAX_GRIPPER_WIDTH = 0.11 # meter

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.6, 0.3])
WORKSPACE_MAX = np.array([0.5, 0.4, 1.3])

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter

'''''

Statistic       X (m)           Y (m)           Z (m)          
------------------------------------------------------------
min                    -0.1766        -0.3118        -0.5424
p0.1                   -0.0529        -0.1574        -0.1902
p0.5                   -0.0341        -0.1253        -0.1547
p1                     -0.0261        -0.1094        -0.1364
p5                     -0.0097        -0.0344        -0.0424
mean                   -0.0005        -0.0045        -0.0052
median                  0.0000        -0.0000         0.0001
p95                     0.0025         0.0010         0.0031
p99                     0.0164         0.0126         0.0153
p99.5                   0.0252         0.0210         0.0197
p99.9                   0.1111         0.0769         0.0701
max                     0.6791         0.1645         0.3146
std                     0.0081         0.0195         0.0242

'''''
