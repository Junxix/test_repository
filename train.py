import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import HistRISE
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss


default_args = edict({
    "data_path": "data/collect_pens",
    "aug": False,
    "aug_jitter": False,
    "num_action": 20,
    "num_history": 5,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "ckpt_dir": "logs/collect_pens",
    "resume_ckpt": None,
    "resume_epoch": -1,
    "lr": 3e-4,
    "batch_size": 240,
    "num_epochs": 1000,
    "save_epochs": 50,
    "num_workers": 24,
    "seed": 233,
    "vis_data": False,
    "num_targets": 4,
    "track_encoder_ckpt": None,    # new
    "value_encoder_ckpt": None,    # new
    "value_seq_len": 16,           # new
})


def train(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    # set up device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    if RANK == 0: print("Loading dataset ...")
    dataset = RealWorldDataset(
        path = args.data_path,
        split = 'train',
        num_obs = 1,
        num_action = args.num_action,
        num_history = args.num_history,
        voxel_size = args.voxel_size,
        aug = args.aug,
        aug_jitter = args.aug_jitter, 
        with_cloud = False,
        vis = args.vis_data,
        num_targets = args.num_targets 
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size // WORLD_SIZE,
        num_workers = args.num_workers,
        collate_fn = collate_fn,
        sampler = sampler,
        drop_last = True
    )

    track_config = {
        'input_dim': 3,       
        'patch_size': 4,      
        'embed_dim': 256,
        'query_dim': 256,
        'num_queries': 1,    
        'num_layers': 4,   
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'use_rope': False,
        'use_time_embedding': True,
        'output_dim': None    
    }

    if RANK == 0: 
        print(f"Loading policy with {args.num_targets} targets...")

    policy = HistRISE(
        num_action = args.num_action,
        num_history = args.num_history,
        input_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout,
        track_config = track_config,
        num_targets = args.num_targets,
        track_encoder_ckpt = args.track_encoder_ckpt,   # new
        value_encoder_ckpt = args.value_encoder_ckpt,   # new
        value_seq_len = args.value_seq_len
    ).to(device)
    
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    

    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 2000,
        num_training_steps = len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []

    policy.train()

    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0

        for data in pbar:
            # cloud data processing
            cloud_coords = data['input_coords_list']
            cloud_feats = data['input_feats_list']
            action_data = data['action_normalized']
            
            # Extract track and semantic data
            human_tracks_abs = data.get('human_tracks_abs', None)
            human_tracks_rel = data.get('human_tracks_rel', None)
            robot_tracks_abs = data.get('robot_tracks_abs', None)
            robot_tracks_rel = data.get('robot_tracks_rel', None)
            human_track_lengths = data.get('human_track_lengths', None)
            robot_track_lengths = data.get('robot_track_lengths', None)
            human_semantics = data.get('human_semantics', None)
            robot_semantics = data.get('robot_semantics', None)
            
            # ===== 新增: 提取 robot_total_length =====
            robot_total_length = data.get('robot_total_length', None)
            # ===== 修改结束 =====

            # Move to device
            cloud_feats = cloud_feats.to(device)
            cloud_coords = cloud_coords.to(device)
            action_data = action_data.to(device)
            
            # if human_tracks_abs is not None:
            #     human_tracks_abs = human_tracks_abs.to(device)
            if human_tracks_rel is not None:
                human_tracks_rel = human_tracks_rel.to(device)
            # if robot_tracks_abs is not None:
            #     robot_tracks_abs = robot_tracks_abs.to(device)
            if robot_tracks_rel is not None:
                robot_tracks_rel = robot_tracks_rel.to(device)
            if human_track_lengths is not None:
                human_track_lengths = human_track_lengths.to(device)
            if robot_track_lengths is not None:
                robot_track_lengths = robot_track_lengths.to(device)
            if human_semantics is not None:
                human_semantics = human_semantics.to(device)
            if robot_semantics is not None:
                robot_semantics = robot_semantics.to(device)
            
            # ===== 新增: 移动 robot_total_length 到 device =====
            if robot_total_length is not None:
                robot_total_length = robot_total_length.to(device)
            # ===== 修改结束 =====
            
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            
            # forward
            loss = policy(
                cloud=cloud_data, 
                actions=action_data, 
                human_tracks_abs=human_tracks_abs,
                human_tracks_rel=human_tracks_rel,
                robot_tracks_abs=robot_tracks_abs,
                robot_tracks_rel=robot_tracks_rel,
                human_track_lengths=human_track_lengths,
                robot_track_lengths=robot_track_lengths,
                human_semantics=human_semantics,
                robot_semantics=robot_semantics,
                robot_total_length=robot_total_length,
                batch_size=action_data.shape[0]
            )
            
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / num_steps
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        if RANK == 0:
            print("Train loss: {:.6f}".format(avg_loss))
            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--aug', action = 'store_true', help = 'whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action = 'store_true', help = 'whether to add color jitter augmentation')
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_history', action = 'store', type = int, help = 'number of history actions', required = False, default = 5)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which epoch', required = False, default = -1)
    parser.add_argument('--lr', action = 'store', type = float, help = 'learning rate', required = False, default = 3e-4)
    parser.add_argument('--batch_size', action = 'store', type = int, help = 'batch size', required = False, default = 240)
    parser.add_argument('--num_epochs', action = 'store', type = int, help = 'training epochs', required = False, default = 1000)
    parser.add_argument('--save_epochs', action = 'store', type = int, help = 'saving epochs', required = False, default = 50)
    parser.add_argument('--num_workers', action = 'store', type = int, help = 'number of workers', required = False, default = 24)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis_data', action = 'store_true', help = 'whether to visualize the input data and ground truth actions.')
    parser.add_argument('--num_targets', action = 'store', type = int, help = 'number of targets to use', required = False, default = 3)
    parser.add_argument('--track_encoder_ckpt', type=str, default=None)
    parser.add_argument('--value_encoder_ckpt', type=str, default=None)
    parser.add_argument('--value_seq_len', type=int, default=16)
    
    train(vars(parser.parse_args()))