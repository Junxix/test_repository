

"""
Evaluation Agent.
"""

import time
from easydict import EasyDict as edict

import numpy as np
from device.robot.flexiv_api import FlexivApi
from util.transformation import xyz_rot_transform
from device.gripper.dahuan import DahuanModbusGripper
from device.camera.realsense import RealSenseRGBDCamera

class Agent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_ip,
        pc_ip,
        gripper_port,
        camera_serial,
        **kwargs
    ): 
        self.camera_serial = camera_serial

        print("Init robot, gripper, and camera.")
        self.robot = FlexivApi(serial="Rizon4-062027", with_streaming = True)
        self.robot.send_tcp_pose(self.ready_pose)
        time.sleep(1.5)
        
        self.gripper = DahuanModbusGripper(port = gripper_port)
        self.gripper.set_force(30)
        self.gripper.set_width(0)
        time.sleep(0.5)

        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        for _ in range(30): 
            self.camera.get_rgbd_image()
        print("Initialization Finished.")
    
    @property
    def intrinsics(self):
        return np.array([[914.81945801,   0.        , 630.63891602],
       [  0.        , 913.88464355, 352.51571655],
       [  0.        ,   0.        ,   1.        ]])
    
    @property
    def ready_pose(self):
        return np.array([0.4, 0.0, 0.22, 0.0, 0.0, 1.0, 0.0])
    
    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()
        return colors, depths
    
    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention = None, blocking = False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.05)
            
    def get_tcp_pose(self):
        tcp_pose = self.robot.get_tcp_pose()
        return tcp_pose
    
    def set_gripper_width(self, width, blocking = True):
        width = int(np.clip(width / 0.095 * 1000., 0, 1000))
        while True:
            try:
                self.gripper.set_width(width)
                break
            except:
                print("set_gripper_width error")
        time1 = time.time()
        while True:
            time.sleep(0.1)
            if self.get_gripper_width()-width/1000.*0.095 < 0.005:
                break
            if time.time()-time1 > 0.5:
                break
        # if blocking:
        #     time.sleep(0.5)
    
    
    def get_gripper_width(self):
        while True:
            try:
                width = self.gripper.get_info()[0]
                break
            except:
                print("get_gripper_width error")
        return width / 1000. * 0.095

    def stop(self):
        self.robot.stop()
    