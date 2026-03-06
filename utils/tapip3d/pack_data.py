import os
import numpy as np
import cv2
import json
from tqdm import tqdm
from transforms3d.quaternions import quat2mat

def load_color_images(color_root):
    image_names = sorted(os.listdir(color_root))
    images = []
    for image_name in tqdm(image_names):
        image_name = os.path.join(color_root, image_name)
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        assert image is not None
        images.append(image)
    return images

def display_images(images, duration=50):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for image in tqdm(images, desc="Displaying..."):
        cv2.imshow('image', image)
        cv2.waitKey(duration)
    cv2.destroyAllWindows()

def load_gripper_poses(dataset_root, task_name, cam_id):
    # load tcps
    tcp_root = os.path.join(dataset_root, task_name, cam_id, 'tcp')
    tcp_path_list = sorted(os.listdir(tcp_root))
    timestamps = [x.split('.')[0] for x in tcp_path_list]
    tcp_path_list = [os.path.join(tcp_root, x) for x in tcp_path_list]
    tcps = []
    for tcp_path in tcp_path_list:
        tcp = np.load(tcp_path)[:7]
        tcps.append(tcp.astype(np.float32))
    tcps = np.array(tcps)

    # load gripper widths
    gripper_widths = []
    gripper_info_root = os.path.join(dataset_root, task_name, cam_id, 'gripper_info')
    gripper_info_path_list = sorted(os.listdir(gripper_info_root))
    for gripper_info_path in gripper_info_path_list:
        gripper_info_path = os.path.join(gripper_info_root, gripper_info_path)
        gripper_info = np.load(gripper_info_path)
        gripper_width = gripper_info[0] / 1000. * 0.095 # for cfg1
        gripper_widths.append(gripper_width)
    gripper_widths = np.array(gripper_widths)

    return tcps, gripper_widths, timestamps

def load_robot_start_time(task_root):
    """
    加载human.json文件中的robot_start_time
    """
    human_json_path = os.path.join(task_root, 'human.json')
    if not os.path.exists(human_json_path):
        print(f"Warning: human.json not found at {human_json_path}")
        return None
    
    try:
        with open(human_json_path, 'r') as f:
            human_data = json.load(f)
        print(human_data)
        robot_start_time = human_data.get('robor_start_time', None)
        if robot_start_time is None:
            print(f"Warning: robot_start_time not found in {human_json_path}")
        return robot_start_time
    except Exception as e:
        print(f"Error loading human.json: {e}")
        return None

def find_robot_start_frame_index(timestamps, robot_start_time):
    """
    根据robot_start_time找到对应的帧索引
    """
    if robot_start_time is None:
        return None
    
    # 将时间戳转换为整数进行比较
    try:
        robot_start_time_int = int(robot_start_time)
        timestamps_int = [int(ts) for ts in timestamps]
        
        # 找到第一个大于等于robot_start_time的帧索引
        for i, ts in enumerate(timestamps_int):
            if ts >= robot_start_time_int:
                return i
        
        # 如果没有找到，返回最后一个帧的索引
        return len(timestamps) - 1
    except:
        print(f"Error converting timestamps to int: robot_start_time={robot_start_time}")
        return None

def diff(tcps, widths, id1, id2, trans_delta=0.005, rot_delta=np.pi/12, width_delta=0.005):
    '''tcps, widths: numpy.ndarray'''
    if id1 == id2:
        return False
    if id1 > id2:
        id1, id2 = id2, id1

    def _diff_along_axis(sequence, delta):
        if np.any(np.abs(sequence[id1]-sequence[id2]) > delta):
            return True
        return False
        
    def _diff_rotation(quat_sequence, delta):
        mat1 = quat2mat(quat_sequence[id1])
        mat2 = quat2mat(quat_sequence[id2])
        rot_diff = np.matmul(mat1, mat2.T)
        rot_diff = np.diag(rot_diff).sum()
        rot_diff = min(max((rot_diff-1)/2, -1), 1)
        rot_diff = np.arccos(rot_diff)
        return rot_diff > delta

    if _diff_along_axis(tcps[:,:3], trans_delta):
        return True
    if _diff_along_axis(widths, width_delta):
        return True
    if _diff_rotation(tcps[:,3:7], rot_delta):
        return True

    return False

def filter_frames(tcps, widths, timestamps, robot_start_frame_index=None, trans_delta=0.005, rot_delta=np.pi/12, width_delta=0.005):
    """
    过滤帧，但保留robot_start_time之前的所有帧
    """
    kept_frame_ids = []
    
    # 如果有robot_start_frame_index，先添加所有该时间之前的帧
    if robot_start_frame_index is not None:
        print(f"保留robot_start_time之前的所有帧 (0到{robot_start_frame_index-1})")
        kept_frame_ids.extend(range(robot_start_frame_index))
        start_id = robot_start_frame_index
    else:
        start_id = 0
    
    # 对于robot_start_time之后的帧，使用原有的过滤逻辑
    if start_id < len(tcps):
        if start_id not in kept_frame_ids:
            kept_frame_ids.append(start_id)
        
        id1 = start_id
        while True:
            found_next = False
            for id2 in range(id1+1, len(tcps)):
                if diff(tcps, widths, id1, id2, trans_delta, rot_delta, width_delta):
                    id1 = id2
                    if id2 not in kept_frame_ids:
                        kept_frame_ids.append(id2)
                    found_next = True
                    break
            if not found_next:
                break
    
    # 确保帧ID按顺序排列
    kept_frame_ids = sorted(list(set(kept_frame_ids)))
    
    print(f"总共保留 {len(kept_frame_ids)} 帧，原始帧数: {len(tcps)}")
    if robot_start_frame_index is not None:
        before_start = len([i for i in kept_frame_ids if i < robot_start_frame_index])
        after_start = len([i for i in kept_frame_ids if i >= robot_start_frame_index])
        print(f"robot_start_time之前: {before_start} 帧, 之后: {after_start} 帧")
    
    return kept_frame_ids

def filter_and_display_images_by_scene(dataset_root, task_id, user_id, scene_id, cam_id, trans_delta=0.005, rot_delta=np.pi/12, width_delta=0.005, duration=50):
    task_name = '%s_%s_%s_cfg_0001' % (task_id, user_id, scene_id)
    print("\"%s\": \"%s\"," % (task_name, cam_id))

    color_root = os.path.join(dataset_root, task_name, cam_id, 'color')
    color_images = load_color_images(color_root)

    tcps, gripper_widths, timestamps = load_gripper_poses(dataset_root, task_name, cam_id)
    
    # 加载robot_start_time
    task_root = os.path.join(dataset_root, task_name)
    robot_start_time = load_robot_start_time(task_root)
    robot_start_frame_index = find_robot_start_frame_index(timestamps, robot_start_time)
    
    kept_frame_ids = filter_frames(tcps, gripper_widths, timestamps, robot_start_frame_index, trans_delta, rot_delta, width_delta)
    
    color_images_filtered = []
    for i in kept_frame_ids:
        color_images_filtered.append(color_images[i])

    display_images(color_images_filtered, duration)

def filter_and_save_images_by_scene(dataset_root, target_root, task_id, user_id, scene_id, trans_delta=0.005, rot_delta=np.pi/12, width_delta=0.005):
    task_name = '%s_%s_%s_cfg_0001' % (task_id, user_id, scene_id)
    
    # copy meta data
    src_task_root = os.path.join(dataset_root, task_name)
    tgt_task_root = os.path.join(target_root, task_name)
    
    # 检查源任务目录是否存在
    if not os.path.exists(src_task_root):
        print(f"Warning: Source task directory not found: {src_task_root}")
        return
    
    os.makedirs(tgt_task_root, exist_ok=True)
    
    # 复制元数据
    for folder in ['pedal_command', 'robot_command']:
        src_folder = os.path.join(src_task_root, folder)
        if os.path.exists(src_folder):
            os.system('cp -r %s %s' % (src_folder, tgt_task_root))
    
    # 复制metadata.json和human.json
    for file in ['metadata.json', 'human.json']:
        src_file = os.path.join(src_task_root, file)
        if os.path.exists(src_file):
            os.system('cp %s %s' % (src_file, tgt_task_root))

    # 加载robot_start_time
    robot_start_time = load_robot_start_time(src_task_root)
    print(robot_start_time)
    
    # get cam_ids from src_task_root
    cam_ids = os.listdir(src_task_root)
    cam_ids = [x for x in cam_ids if 'cam' in x]
    
    for cam_id in cam_ids:
        print("\"%s\": \"%s\"," % (task_name, cam_id))
        src_color_root = os.path.join(dataset_root, task_name, cam_id, 'color')
        src_depth_root = os.path.join(dataset_root, task_name, cam_id, 'depth')
        
        if not os.path.exists(src_color_root):
            print(f"Warning: Color directory not found: {src_color_root}")
            continue
            
        tcps, gripper_widths, timestamps = load_gripper_poses(dataset_root, task_name, cam_id)
        
        # 找到robot_start_time对应的帧索引
        robot_start_frame_index = find_robot_start_frame_index(timestamps, robot_start_time)
        
        kept_frame_ids = filter_frames(tcps, gripper_widths, timestamps, robot_start_frame_index, trans_delta, rot_delta, width_delta)

        # copy data
        src_task_cam_root = os.path.join(src_task_root, cam_id)
        tgt_task_cam_root = os.path.join(tgt_task_root, cam_id)
        os.makedirs(tgt_task_cam_root, exist_ok=True)
        
        # 复制相机相关数据
        for folder in ['gripper_command', 'gripper_info', 'joint', 'tcp']:
            src_folder = os.path.join(src_task_cam_root, folder)
            if os.path.exists(src_folder):
                os.system('cp -r %s %s' % (src_folder, tgt_task_cam_root))

        tgt_color_root = os.path.join(tgt_task_cam_root, 'color')
        tgt_depth_root = os.path.join(tgt_task_cam_root, 'depth')
        os.makedirs(tgt_color_root, exist_ok=True)
        os.makedirs(tgt_depth_root, exist_ok=True)
        
        # 复制保留的帧
        for kept_frame_id in tqdm(kept_frame_ids, f'Copying {cam_id}...'):
            image_name = '%s.png' % timestamps[kept_frame_id]
            src_color_file = os.path.join(src_color_root, image_name)
            src_depth_file = os.path.join(src_depth_root, image_name)
            
            if os.path.exists(src_color_file):
                os.system('cp %s %s' % (src_color_file, tgt_color_root))
            if os.path.exists(src_depth_file):
                os.system('cp %s %s' % (src_depth_file, tgt_depth_root))

if __name__ == '__main__':
    dataset_root = '/home/ubuntu/data/realdata_20260202_val'
    target_root = '/home/ubuntu/data/realdata_sampled_20260202_val_2'
    split = 'train'
    task_id = 108
    user_id = 555
    start = 4
    end = 4
    calib_root = '/home/ubuntu/data/calib'
    calib_timestamp = '1768480265011'

    for i in range(start, end + 1):
        filter_and_save_images_by_scene(dataset_root, os.path.join(target_root,split), 'task_%04d'%task_id, 'user_%04d'%user_id, 'scene_%04d'%i, trans_delta=0.005, rot_delta=np.pi/24, width_delta=0.005)

    with open('timestamp.txt', 'w') as f:
        f.write(calib_timestamp)

    task_names = []
    for i in range(start, end + 1):
        task_name = 'task_%04d_user_%04d_scene_%04d_cfg_0001' % (task_id, user_id, i)
        task_names.append(task_name+'\n')
        task_root = os.path.join(target_root, split, task_name)
        if os.path.exists(task_root):
            os.system('cp timestamp.txt %s' % task_root)

    if os.path.exists(calib_root):
        os.system('cp -r %s/%s %s' % (calib_root, calib_timestamp, target_root))
        with open('%s/%s/task_ids.txt' % (target_root, calib_timestamp), 'w') as f:
            f.writelines(task_names)