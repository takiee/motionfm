import torch
from torch.utils import data
import numpy as np
import os
from os.path import join 
import random
from tqdm import *
import spacy
import pickle
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
import trimesh
from utils.data_util import axis2rot6d,global2local_axis,global2local_axis_by_matrix,obj_global2local_matrix, obj_matrix2rot6d
from tqdm import *
from manotorch.manolayer import ManoLayer
from random import choice
from sklearn.neighbors import KDTree
from pysdf import SDF
manolayer = ManoLayer(
    mano_assets_root='/root/code/CAMS/data/mano_assets/mano',
    side='right'
)
right_manolayer = ManoLayer(
    mano_assets_root='/root/code/CAMS/data/mano_assets/mano',
    side='right'
)
left_manolayer = ManoLayer(
    mano_assets_root='/root/code/CAMS/data/mano_assets/mano',
    side='left'
)
def read_xyz(path):
    data = []
    with open(path,'r') as f:
        line = f.readline()
        ls = line.strip().split(' ')
        data.append([float(ls[0]),float(ls[1]),float(ls[2])])
        while line:
            ls = f.readline().strip().split(' ')
            # print(ls)
            if ls != ['']:
                data.append([float(ls[0]),float(ls[1]),float(ls[2])])
            else:
                line = None
    data = np.array(data)
    return data

def convert_T_to_obj_frame(points, obj_pose):
    # points(frames,21,3)
    # obj_pose (frames,3,4)

    obj_T = obj_pose[:,:3,3].unsqueeze(-2) # B, 1, 3
    points = points - obj_T
    points = torch.einsum('...ij->...ji', [points])
    obj_R = obj_pose[:,:3,:3] # B, 3, 3
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    new_points = torch.einsum('bpn,bnk->bpk',obj_R,points)
    new_points = torch.einsum('...ij->...ji', [new_points])
    return new_points

def convert_R_to_obj_frame(hand_rot, obj_pose):
    # hand_rot: B，3，3
    obj_R = obj_pose[:,:3,:3] # B, 3, 3
    obj_R = torch.einsum('...ij->...ji', [obj_R])
    hand_rot_in_obj = torch.einsum('bji,bjk->bik', obj_R, hand_rot)

    return hand_rot_in_obj

def compute_angular_velocity_nofor(rotation_matrices):
    # rotation_matrices: (T, 3, 3), where T is the number of time steps
    R_next = rotation_matrices[1:]  # (T-1, 3, 3)
    R_current = rotation_matrices[:-1]  # (T-1, 3, 3)
    
    # Compute difference matrix R_next * R_current^T - I
    R_diff = R_next @ R_current.transpose(-1, -2) - torch.eye(3).to(rotation_matrices.device)
    
    # Extract the angular velocity matrix (anti-symmetric part)
    angular_velocity = R_diff
    
    return angular_velocity

def compute_angular_acceleration_nofor(angular_velocity):
    # angular_velocity: (T-1, 3, 3), where T-1 is the number of time steps
    omega_next = angular_velocity[1:]  # (T-2, 3, 3)
    omega_current = angular_velocity[:-1]  # (T-2, 3, 3)
    
    # Compute the difference in angular velocity
    omega_diff = omega_next - omega_current
    
    # Compute angular acceleration
    angular_acceleration = omega_diff
    
    return angular_acceleration


class GazeHOIDataset_o2h_mid(data.Dataset):
    def __init__(self, mode='mid',datapath='/root/code/seqs/gazehoi_list_train_0718.txt',split='train',hint_type='goal_pose'):
        if split == 'test':
            # datapath = '/root/code/seqs/gazehoi_list_train_new.txt'
            datapath = '/root/code/seqs/gazehoi_list_test_0718.txt'
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()[:32]
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.hint_type = hint_type
        self.datalist = []
        self.fps = 6
        self.target_length = 150
        print("Start processing data.")
        for seq in tqdm(self.seqs):
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            mano_right_path = join(seq_path, 'mano/poses_right.npy')
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            
            active_obj = meta['active_obj']
            obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
            obj_mesh = trimesh.load(obj_mesh_path)
            obj_sdf = SDF(obj_mesh.vertices,obj_mesh.faces)
            obj_verts = torch.tensor(np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))).float()
            obj_pose = torch.tensor(np.load(join(seq_path,active_obj+'_pose_trans.npy'))).float()
            # obj_rot = obj_pose[:,:3,:3]
            # obj_trans = obj_pose[:,:3,3]

            hand_params = torch.tensor(np.load(mano_right_path))

            # 统一seq长度 150帧 -- 降低帧率 30fps--6fps
            seq_len = hand_params.shape[0]
            if seq_len >= self.target_length:
                indices = torch.linspace(0, seq_len - 1, steps=self.target_length).long()
                hand_params = hand_params[indices]
                obj_pose = obj_pose[indices]
            else:
                padding_hand = hand_params[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1)
                hand_params = torch.cat((hand_params, padding_hand), dim=0)
                padding_obj = obj_pose[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1,1)
                obj_pose = torch.cat((obj_pose, padding_obj), dim=0)
            
            assert hand_params.shape[0] == self.target_length and obj_pose.shape[0] == self.target_length

            # 降低帧率 30fps--6fps  150帧--30帧
            step = int(30/self.fps)
            for i in range(step):
                hand_params_lowfps = hand_params[i::step]
                obj_pose_lowfps = obj_pose[i::step]

                hand_trans = hand_params_lowfps[:,:3]
                hand_rot = hand_params_lowfps[:,3:6]
                hand_rot_matrix = axis_angle_to_matrix(hand_rot)
                hand_theta = hand_params_lowfps[:,3:51]
                mano_beta = hand_params_lowfps[:,51:]
                mano_output = manolayer(hand_theta, mano_beta)
                hand_joints = mano_output.joints - mano_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1) # B, 21, 3
                # 物体坐标系下的手部关键点
                hand_joints_in_obj = convert_T_to_obj_frame(hand_joints, # B, 21, 3
                                                    obj_pose_lowfps)        # B, 3, 4

                # 手物是否接触 0-1值
                hand_contact = obj_sdf(hand_joints_in_obj.reshape(-1,3)).reshape(-1,21) < 0.01 # TODO: 阈值调整
                hand_contact = torch.tensor(hand_contact)
                # print(hand_contact)

                # 手物之间的offset
                hand_obj_dis = torch.norm(hand_joints_in_obj.unsqueeze(2) - obj_verts.unsqueeze(0).unsqueeze(0).repeat(hand_params_lowfps.shape[0],1,1,1),dim=-1) # B,21,1,3 - B,1,500,3 = B,21,500
                obj_ids = torch.argmin(hand_obj_dis,dim=-1) # B,21
                closest_obj_verts = obj_verts[obj_ids] # B,21,3
                hand_obj_offset = hand_joints_in_obj - closest_obj_verts

                # 手部21个节点的线速度 线加速度
                hand_lin_vel = hand_joints_in_obj[1:] - hand_joints_in_obj[:-1] # B-1,21,3
                hand_lin_acc = hand_lin_vel[1:] - hand_lin_vel[:-1] # B-2,21,3

                # 手部根节点的角速度 角加速度
                # TODO: 需要将手部的旋转也转到物体坐标系下
                hand_rot_in_obj = convert_R_to_obj_frame(hand_rot_matrix, obj_pose_lowfps)
                hand_ang_vel = compute_angular_velocity_nofor(hand_rot_in_obj)
                hand_ang_acc = compute_angular_acceleration_nofor(hand_ang_vel)
                data = {"seq":seq,
                        # "hand_kp":hand_joints, # B,21,3
                        "hand_kp":hand_joints_in_obj, # B,21,3
                        "hand_contact":hand_contact, # B,21
                        "hand_obj_offset":hand_obj_offset, # B,21,3
                        "hand_lin_vel": hand_lin_vel, # B-1, 21, 3
                        "hand_lin_acc": hand_lin_acc, # B-2, 21, 3
                        "hand_ang_vel": hand_ang_vel, # B-1,3,3
                        "hand_ang_acc": hand_ang_acc,  # B-2,3,3
                        "obj_verts": obj_verts, # 500
                        "obj_pose": obj_pose_lowfps, # B,3,4
                        "hand_shape": mano_beta # B,10
                        }
                self.datalist.append(data)


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        seq = data['seq']
        # (B,291)
        hand_kp = data['hand_kp'].reshape(-1,63) #  B,63
        hand_contact = data['hand_contact'] # B,21
        hand_obj_offset = data['hand_obj_offset'].reshape(-1,63) # B,63
        hand_lin_vel = torch.cat((data['hand_lin_vel'],data['hand_lin_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,63) # B,63
        hand_lin_acc = torch.cat((data['hand_lin_acc'],data['hand_lin_acc'][-2:]),dim=0).reshape(-1,63) # B,63
        hand_ang_vel = torch.cat((data['hand_ang_vel'],data['hand_ang_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,9) # B,9
        hand_ang_acc = torch.cat((data['hand_ang_acc'],data['hand_ang_acc'][-2:]),dim=0).reshape(-1,9) # B,9

        hand_all = torch.cat((hand_kp,hand_contact, hand_obj_offset,hand_lin_vel,hand_lin_acc,hand_ang_vel,hand_ang_acc),dim=-1).numpy()
        # print("hand_all.shape: ", hand_all.shape)

        obj_pose = data['obj_pose'].reshape(-1,12).numpy()
        obj_verts = data['obj_verts'].numpy()
        hand_shape = data['hand_shape'].numpy()

        return seq, hand_all, obj_pose, obj_verts, hand_kp.shape[0],hand_shape

class GazeHOIDataset_o2h_mid_2hand_assemobj(data.Dataset):
    def __init__(self, mode='mid_2hand_assmobj',datapath='/root/code/seqs/gazehoi_list_train_0718.txt',split='train',hint_type='goal_pose'):
    # def __init__(self, mode='mid_2hand_assmobj',datapath='/root/code/seqs/gazehoi_list_train_0718.txt',split='train',hint_type='goal_pose'):
    # def __init__(self, mode='mid_2hand_assmobj',datapath='/root/code/seqs/gazehoi_list_train_0718_assem.txt',split='train',hint_type='goal_pose'):
        if split == 'test' or split == 'val':
            # datapath = '/root/code/seqs/gazehoi_list_train_new.txt'
            # datapath = '/root/code/seqs/gazehoi_list_train_0718_assem.txt'
            # datapath = '/root/code/seqs/gazehoi_list_test_0718_assem.txt'
            datapath = '/root/code/seqs/gazehoi_list_test_0718.txt'
            # datapath = '/root/code/seqs/gazehoi_list_test_0820_assem.txt'
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        self.pred_obj_root = '/nas/gazehoi-diffusion/final_result/0817_for_demo_assem/' # TODO:修改预测物体存储路径
        # self.pred_obj_root = '/nas/gazehoi-diffusion/final_result/0725_stage0_1obj_000100000_seed10_final_repeat_20/' # TODO:修改预测物体存储路径
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        print(len(self.seqs))
        self.hint_type = hint_type
        self.datalist = []
        self.fps = 6
        self.target_length = 150
        print("Start processing data.")
        for seq in tqdm(self.seqs):
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            right_mano_path = join(seq_path, 'mano/poses_right.npy')
            left_mano_path = join(seq_path, 'mano/poses_left.npy')
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            right_hand_params = torch.tensor(np.load(right_mano_path))
            left_hand_params = torch.tensor(np.load(left_mano_path))
            goal_index = meta['goal_index']
            active_obj = meta['active_obj']
            obj_mesh_path = join(self.obj_path,active_obj,'simplified_scan_processed.obj')
            obj_mesh = trimesh.load(obj_mesh_path)
            obj_sdf = SDF(obj_mesh.vertices,obj_mesh.faces)
            obj_verts = torch.tensor(np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))).float()
            obj_normals = torch.tensor(np.load(join(self.obj_path,active_obj,'resampled_500_trans_normal.npy'))).float()
            gt_obj_pose = torch.tensor(np.load(join(seq_path,active_obj+'_pose_trans.npy'))).float()
            if split == 'train' or split == 'val':
                obj_pose = torch.tensor(np.load(join(seq_path,active_obj+'_pose_trans.npy'))).float()
                # obj_rot = obj_pose[:,:3,:3]
                # obj_trans = obj_pose[:,:3,3]

                # 统一seq长度 150帧 -- 降低帧率 30fps--6fps
                seq_len = right_hand_params.shape[0]
                if seq_len >= self.target_length:
                    indices = torch.linspace(0, seq_len - 1, steps=self.target_length).long()
                    right_hand_params = right_hand_params[indices]
                    left_hand_params = left_hand_params[indices]
                    obj_pose = obj_pose[indices]
                    new_goal_index = (goal_index / seq_len) * self.target_length
                    goal_index = int(new_goal_index)
                else:
                    right_padding_hand = right_hand_params[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1)
                    right_hand_params = torch.cat((right_hand_params, right_padding_hand), dim=0)
                    left_padding_hand = left_hand_params[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1)
                    left_hand_params = torch.cat((left_hand_params, left_padding_hand), dim=0)
                    padding_obj = obj_pose[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1,1)
                    obj_pose = torch.cat((obj_pose, padding_obj), dim=0)
                assert right_hand_params.shape[0] == self.target_length and left_hand_params.shape[0] == self.target_length 

                # 降低帧率 30fps--6fps  150帧--30帧
                step = int(30/self.fps)
                for i in range(step):
                    goal_index_i = (goal_index // step) + i
                    
                    right_hand_params_lowfps = right_hand_params[i::step]
                    left_hand_params_lowfps = left_hand_params[i::step]
                    if split == 'test':
                        obj_pose_lowfps = obj_pose
                    else:
                        obj_pose_lowfps = obj_pose[i::step]
                    
                    gt_obj_pose_lowfps = gt_obj_pose[i::step]
                    # assert right_hand_params_lowfps.shape[0] == 30 and left_hand_params_lowfps.shape[0] == 30 and obj_pose_lowfps.shape[0] == 30

                    # 提取右手参数
                    right_hand_trans = right_hand_params_lowfps[:, :3]
                    right_hand_rot = right_hand_params_lowfps[:, 3:6]
                    right_hand_rot_matrix = axis_angle_to_matrix(right_hand_rot)
                    right_hand_theta = right_hand_params_lowfps[:, 3:51]
                    right_mano_beta = right_hand_params_lowfps[:, 51:]

                    # 提取左手参数
                    left_hand_trans = left_hand_params_lowfps[:, :3]
                    left_hand_rot = left_hand_params_lowfps[:, 3:6]
                    left_hand_rot_matrix = axis_angle_to_matrix(left_hand_rot)
                    left_hand_theta = left_hand_params_lowfps[:, 3:51]
                    left_mano_beta = left_hand_params_lowfps[:, 51:]

                    # MANO输出右手
                    right_mano_output = right_manolayer(right_hand_theta, right_mano_beta)
                    right_hand_joints = right_mano_output.joints - right_mano_output.joints[:, 0].unsqueeze(1) + right_hand_trans.unsqueeze(1)  # B, 21, 3

                    # MANO输出左手
                    left_mano_output = left_manolayer(left_hand_theta, left_mano_beta)
                    left_hand_joints = left_mano_output.joints - left_mano_output.joints[:, 0].unsqueeze(1) + left_hand_trans.unsqueeze(1)  # B, 21, 3

                    # 物体坐标系下的手部关键点
                    right_hand_joints_in_obj = convert_T_to_obj_frame(right_hand_joints, obj_pose_lowfps)  # B, 21, 3
                    left_hand_joints_in_obj = convert_T_to_obj_frame(left_hand_joints, obj_pose_lowfps)  # B, 21, 3

                    # 手物是否接触 0-1值
                    right_hand_contact = abs(obj_sdf(right_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)) < 0.005  # TODO: 阈值调整
                    right_hand_contact = torch.tensor(right_hand_contact)
                    left_hand_contact = abs(obj_sdf(left_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)) < 0.005  # TODO: 阈值调整
                    left_hand_contact = torch.tensor(left_hand_contact)
                    # print(abs(obj_sdf(right_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)).min(),abs(obj_sdf(left_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)).min())
                    # print(right_hand_contact)
                    # print(left_hand_contact)

                    # 手物之间的offset
                    right_hand_obj_dis = torch.norm(right_hand_joints_in_obj.unsqueeze(2) - obj_verts.unsqueeze(0).unsqueeze(0).repeat(right_hand_params_lowfps.shape[0], 1, 1, 1), dim=-1)  # B, 21, 500
                    right_obj_ids = torch.argmin(right_hand_obj_dis, dim=-1)  # B, 21
                    right_closest_obj_verts = obj_verts[right_obj_ids]  # B, 21, 3
                    right_hand_obj_offset = right_hand_joints_in_obj - right_closest_obj_verts  # B, 21, 3
                    # print(right_hand_obj_offset.shape,obj_verts.shape,right_obj_ids.shape)

                    left_hand_obj_dis = torch.norm(left_hand_joints_in_obj.unsqueeze(2) - obj_verts.unsqueeze(0).unsqueeze(0).repeat(left_hand_params_lowfps.shape[0], 1, 1, 1), dim=-1)  # B, 21, 500
                    left_obj_ids = torch.argmin(left_hand_obj_dis, dim=-1)  # B, 21
                    left_closest_obj_verts = obj_verts[left_obj_ids]  # B, 21, 3
                    left_hand_obj_offset = left_hand_joints_in_obj - left_closest_obj_verts  # B, 21, 3

                    # 左右手之间的offset
                    left_right_offset = left_hand_joints_in_obj - right_hand_joints_in_obj # B, 21, 3

                    # 手部21个节点的线速度线加速度
                    right_hand_lin_vel = right_hand_joints_in_obj[1:] - right_hand_joints_in_obj[:-1]  # B-1, 21, 3
                    right_hand_lin_acc = right_hand_lin_vel[1:] - right_hand_lin_vel[:-1]  # B-2, 21, 3

                    left_hand_lin_vel = left_hand_joints_in_obj[1:] - left_hand_joints_in_obj[:-1]  # B-1, 21, 3
                    left_hand_lin_acc = left_hand_lin_vel[1:] - left_hand_lin_vel[:-1]  # B-2, 21, 3

                    # 左右手的相对线速度、相对线加速度
                    left_rel_right_lin_vel = left_hand_lin_vel - right_hand_lin_vel # B-1, 21, 3 
                    left_rel_right_lin_acc = left_hand_lin_acc - right_hand_lin_acc # B-2, 21, 3


                    # 手部根节点的角速度角加速度
                    # TODO: 需要将手部的旋转也转到物体坐标系下
                    right_hand_rot_in_obj = convert_R_to_obj_frame(right_hand_rot_matrix, obj_pose_lowfps)
                    right_hand_ang_vel = compute_angular_velocity_nofor(right_hand_rot_in_obj)
                    right_hand_ang_acc = compute_angular_acceleration_nofor(right_hand_ang_vel)

                    left_hand_rot_in_obj = convert_R_to_obj_frame(left_hand_rot_matrix, obj_pose_lowfps)
                    left_hand_ang_vel = compute_angular_velocity_nofor(left_hand_rot_in_obj)
                    left_hand_ang_acc = compute_angular_acceleration_nofor(left_hand_ang_vel)

                    # 左右手的相对角速度、相对角加速度
                    left_rel_right_ang_vel = left_hand_ang_vel - right_hand_ang_vel # B-1,3,3
                    left_rel_right_ang_acc = left_hand_ang_acc - right_hand_ang_acc # B-2,3,3

                    data = {
                        "seq": seq,
                        "goal_index": goal_index_i,
                        "right_hand_kp": right_hand_joints_in_obj,  # B, 21, 3
                        "right_hand_contact": right_hand_contact,  # B, 21
                        "right_hand_obj_offset": right_hand_obj_offset,  # B, 21, 3
                        "right_hand_lin_vel": right_hand_lin_vel,  # B-1, 21, 3
                        "right_hand_lin_acc": right_hand_lin_acc,  # B-2, 21, 3
                        "right_hand_ang_vel": right_hand_ang_vel,  # B-1, 3, 3
                        "right_hand_ang_acc": right_hand_ang_acc,  # B-2, 3, 3
                        "left_hand_kp": left_hand_joints_in_obj,  # B, 21, 3
                        "left_hand_contact": left_hand_contact,  # B, 21
                        "left_hand_obj_offset": left_hand_obj_offset,  # B, 21, 3
                        "left_hand_lin_vel": left_hand_lin_vel,  # B-1, 21, 3
                        "left_hand_lin_acc": left_hand_lin_acc,  # B-2, 21, 3
                        "left_hand_ang_vel": left_hand_ang_vel,  # B-1, 3, 3
                        "left_hand_ang_acc": left_hand_ang_acc,  # B-2, 3, 3
                        "left_rel_right_lin_vel": left_rel_right_lin_vel,  # B-1, 21, 3
                        "left_rel_right_lin_acc": left_rel_right_lin_acc,  # B-2, 21, 3
                        "left_rel_right_ang_vel": left_rel_right_ang_vel,  # B-1, 3, 3
                        "left_rel_right_ang_acc": left_rel_right_ang_acc,  # B-2, 3, 3
                        "left_right_offset": left_right_offset, #B,21,3
                        "obj_verts": obj_verts,  # 500,3
                        "obj_normals": obj_normals,  # 500,3
                        "obj_pose": obj_pose_lowfps,  # B, 3, 4
                        "gt_obj_pose": gt_obj_pose_lowfps,  # B, 3, 4
                        "right_hand_shape": right_mano_beta,  # B, 10
                        "left_hand_shape": left_mano_beta  # B, 10
                    }

                    self.datalist.append(data)
            
            elif split == 'test':
                right_hand_params = torch.tensor(np.load(join(self.pred_obj_root,seq,'gt_right_mano.npy')))
                left_hand_params = torch.tensor(np.load(join(self.pred_obj_root,seq,'gt_left_mano.npy')))
                gt_obj_pose = torch.tensor(np.load(join(self.pred_obj_root,seq,'gt_obj_pose.npy'))).float()
                assert right_hand_params.shape[0] == gt_obj_pose.shape[0]
                for index in range(1):
                # for index in range(20):
                    obj_pose = torch.tensor(np.load(join(self.pred_obj_root,seq,'pred_obj_pose.npy'))).float()
                    # obj_pose = torch.tensor(np.load(join(self.pred_obj_root,seq,'obj_repeat_20',str(index),'pred_obj_pose.npy'))).float()
                    
                    # 统一seq长度 150帧 -- 降低帧率 30fps--6fps
                    seq_len = right_hand_params.shape[0]
                    self.target_length = 30
                    if seq_len >= 30:
                        indices = torch.linspace(0, seq_len - 1, steps=self.target_length).long()
                        right_hand_params = right_hand_params[indices]
                        left_hand_params = left_hand_params[indices]
                        new_goal_index = (goal_index / seq_len) * self.target_length
                        goal_index = int(new_goal_index)
                    else:
                        right_padding_hand = right_hand_params[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1)
                        right_hand_params = torch.cat((right_hand_params, right_padding_hand), dim=0)
                        left_padding_hand = left_hand_params[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1)
                        left_hand_params = torch.cat((left_hand_params, left_padding_hand), dim=0)
                        # padding_obj = obj_pose[-1].unsqueeze(0).repeat(self.target_length - seq_len, 1,1)
                
                    

                    # 降低帧率 30fps--6fps  150帧--30帧
                    step = int(30/self.fps)
                    for i in range(step):
                        goal_index_i = (goal_index // step) + i
                        
                        right_hand_params_lowfps = right_hand_params
                        left_hand_params_lowfps = left_hand_params
                        obj_pose_lowfps = obj_pose
                        gt_obj_pose_lowfps = gt_obj_pose
                        assert right_hand_params_lowfps.shape[0] == 30 and left_hand_params_lowfps.shape[0] == 30 and obj_pose_lowfps.shape[0] == 30
                        # print(right_hand_params_lowfps.shape, left_hand_params_lowfps.shape,obj_pose_lowfps.shape)
                        # 提取右手参数
                        right_hand_trans = right_hand_params_lowfps[:, :3]
                        right_hand_rot = right_hand_params_lowfps[:, 3:6]
                        right_hand_rot_matrix = axis_angle_to_matrix(right_hand_rot)
                        right_hand_theta = right_hand_params_lowfps[:, 3:51]
                        right_mano_beta = right_hand_params_lowfps[:, 51:]

                        # 提取左手参数
                        left_hand_trans = left_hand_params_lowfps[:, :3]
                        left_hand_rot = left_hand_params_lowfps[:, 3:6]
                        left_hand_rot_matrix = axis_angle_to_matrix(left_hand_rot)
                        left_hand_theta = left_hand_params_lowfps[:, 3:51]
                        left_mano_beta = left_hand_params_lowfps[:, 51:]

                        # MANO输出右手
                        right_mano_output = right_manolayer(right_hand_theta, right_mano_beta)
                        right_hand_joints = right_mano_output.joints - right_mano_output.joints[:, 0].unsqueeze(1) + right_hand_trans.unsqueeze(1)  # B, 21, 3

                        # MANO输出左手
                        left_mano_output = left_manolayer(left_hand_theta, left_mano_beta)
                        left_hand_joints = left_mano_output.joints - left_mano_output.joints[:, 0].unsqueeze(1) + left_hand_trans.unsqueeze(1)  # B, 21, 3

                        # 物体坐标系下的手部关键点
                        right_hand_joints_in_obj = convert_T_to_obj_frame(right_hand_joints, obj_pose_lowfps)  # B, 21, 3
                        left_hand_joints_in_obj = convert_T_to_obj_frame(left_hand_joints, obj_pose_lowfps)  # B, 21, 3
                        B = right_hand_joints_in_obj.shape
                        # 手物是否接触 0-1值
                        right_hand_contact = abs(obj_sdf(right_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)) < 0.005  # TODO: 阈值调整
                        right_hand_contact = torch.tensor(right_hand_contact)
                        left_hand_contact = abs(obj_sdf(left_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)) < 0.005  # TODO: 阈值调整
                        left_hand_contact = torch.tensor(left_hand_contact)
                        # print(abs(obj_sdf(right_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)).min(),abs(obj_sdf(left_hand_joints_in_obj.reshape(-1, 3)).reshape(-1, 21)).min())
                        # print(right_hand_contact)
                        # print(left_hand_contact)

                        # 手物之间的offset
                        right_hand_obj_dis = torch.norm(right_hand_joints_in_obj.unsqueeze(2) - obj_verts.unsqueeze(0).unsqueeze(0).repeat(right_hand_params_lowfps.shape[0], 1, 1, 1), dim=-1)  # B, 21, 500
                        right_obj_ids = torch.argmin(right_hand_obj_dis, dim=-1)  # B, 21
                        right_closest_obj_verts = obj_verts[right_obj_ids]  # B, 21, 3
                        right_hand_obj_offset = right_hand_joints_in_obj - right_closest_obj_verts  # B, 21, 3
                        # print(right_hand_obj_offset.shape,obj_verts.shape,right_obj_ids.shape)

                        left_hand_obj_dis = torch.norm(left_hand_joints_in_obj.unsqueeze(2) - obj_verts.unsqueeze(0).unsqueeze(0).repeat(left_hand_params_lowfps.shape[0], 1, 1, 1), dim=-1)  # B, 21, 500
                        left_obj_ids = torch.argmin(left_hand_obj_dis, dim=-1)  # B, 21
                        left_closest_obj_verts = obj_verts[left_obj_ids]  # B, 21, 3
                        left_hand_obj_offset = left_hand_joints_in_obj - left_closest_obj_verts  # B, 21, 3

                        # 左右手之间的offset
                        left_right_offset = left_hand_joints_in_obj - right_hand_joints_in_obj # B, 21, 3

                        # 手部21个节点的线速度线加速度
                        right_hand_lin_vel = right_hand_joints_in_obj[1:] - right_hand_joints_in_obj[:-1]  # B-1, 21, 3
                        right_hand_lin_acc = right_hand_lin_vel[1:] - right_hand_lin_vel[:-1]  # B-2, 21, 3

                        left_hand_lin_vel = left_hand_joints_in_obj[1:] - left_hand_joints_in_obj[:-1]  # B-1, 21, 3
                        left_hand_lin_acc = left_hand_lin_vel[1:] - left_hand_lin_vel[:-1]  # B-2, 21, 3

                        # 左右手的相对线速度、相对线加速度
                        left_rel_right_lin_vel = left_hand_lin_vel - right_hand_lin_vel # B-1, 21, 3 
                        left_rel_right_lin_acc = left_hand_lin_acc - right_hand_lin_acc # B-2, 21, 3


                        # 手部根节点的角速度角加速度
                        # TODO: 需要将手部的旋转也转到物体坐标系下
                        right_hand_rot_in_obj = convert_R_to_obj_frame(right_hand_rot_matrix, obj_pose_lowfps)
                        right_hand_ang_vel = compute_angular_velocity_nofor(right_hand_rot_in_obj)
                        right_hand_ang_acc = compute_angular_acceleration_nofor(right_hand_ang_vel)

                        left_hand_rot_in_obj = convert_R_to_obj_frame(left_hand_rot_matrix, obj_pose_lowfps)
                        left_hand_ang_vel = compute_angular_velocity_nofor(left_hand_rot_in_obj)
                        left_hand_ang_acc = compute_angular_acceleration_nofor(left_hand_ang_vel)

                        # 左右手的相对角速度、相对角加速度
                        left_rel_right_ang_vel = left_hand_ang_vel - right_hand_ang_vel # B-1,3,3
                        left_rel_right_ang_acc = left_hand_ang_acc - right_hand_ang_acc # B-2,3,3

                        data = {
                            "seq": seq,
                            "goal_index": goal_index_i,
                            "right_hand_kp": right_hand_joints_in_obj,  # B, 21, 3
                            "right_hand_contact": right_hand_contact,  # B, 21
                            "right_hand_obj_offset": right_hand_obj_offset,  # B, 21, 3
                            "right_hand_lin_vel": right_hand_lin_vel,  # B-1, 21, 3
                            "right_hand_lin_acc": right_hand_lin_acc,  # B-2, 21, 3
                            "right_hand_ang_vel": right_hand_ang_vel,  # B-1, 3, 3
                            "right_hand_ang_acc": right_hand_ang_acc,  # B-2, 3, 3
                            "left_hand_kp": left_hand_joints_in_obj,  # B, 21, 3
                            "left_hand_contact": left_hand_contact,  # B, 21
                            "left_hand_obj_offset": left_hand_obj_offset,  # B, 21, 3
                            "left_hand_lin_vel": left_hand_lin_vel,  # B-1, 21, 3
                            "left_hand_lin_acc": left_hand_lin_acc,  # B-2, 21, 3
                            "left_hand_ang_vel": left_hand_ang_vel,  # B-1, 3, 3
                            "left_hand_ang_acc": left_hand_ang_acc,  # B-2, 3, 3
                            "left_rel_right_lin_vel": left_rel_right_lin_vel,  # B-1, 21, 3
                            "left_rel_right_lin_acc": left_rel_right_lin_acc,  # B-2, 21, 3
                            "left_rel_right_ang_vel": left_rel_right_ang_vel,  # B-1, 3, 3
                            "left_rel_right_ang_acc": left_rel_right_ang_acc,  # B-2, 3, 3
                            "left_right_offset": left_right_offset, #B,21,3
                            "obj_verts": obj_verts,  # 500,3
                            "obj_normals": obj_normals,  # 500,3
                            "obj_pose": obj_pose_lowfps,  # B, 3, 4
                            "gt_obj_pose": gt_obj_pose_lowfps,  # B, 3, 4
                            "right_hand_shape": right_mano_beta,  # B, 10
                            "left_hand_shape": left_mano_beta  # B, 10
                        }

                self.datalist.append(data)
                # self.datalist.append(data)
                # if split == 'test':
                #         self.datalist.append(data)
                # if split == 'test':
                #         break


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        seq = data['seq']
        # (B,291)
        hand_kp = torch.cat((data['left_hand_kp'],data['right_hand_kp']),dim=1).reshape(-1,126) #  B,42,3 -- B,126
        hand_contact = torch.cat((data['left_hand_contact'],data['right_hand_contact']),dim=1) # B,42
        hand_obj_offset = torch.cat((data['left_hand_obj_offset'],data['right_hand_obj_offset']),dim=1).reshape(-1,126) # B,126

        left_hand_lin_vel = torch.cat((data['left_hand_lin_vel'],data['left_hand_lin_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,63) # B,63
        right_hand_lin_vel = torch.cat((data['right_hand_lin_vel'],data['right_hand_lin_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,63) # B,63
        hand_lin_vel = torch.cat((left_hand_lin_vel, right_hand_lin_vel),dim=1) # B,126

        left_hand_lin_acc = torch.cat((data['left_hand_lin_acc'],data['left_hand_lin_acc'][-2:]),dim=0).reshape(-1,63) # B,63
        right_hand_lin_acc = torch.cat((data['right_hand_lin_acc'],data['right_hand_lin_acc'][-2:]),dim=0).reshape(-1,63) # B,63
        hand_lin_acc = torch.cat((left_hand_lin_acc, right_hand_lin_acc),dim=1) # B,126

        left_hand_ang_vel = torch.cat((data['left_hand_ang_vel'],data['left_hand_ang_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,9) # B,9
        right_hand_ang_vel = torch.cat((data['right_hand_ang_vel'],data['right_hand_ang_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,9) # B,9
        hand_ang_vel = torch.cat((left_hand_ang_vel, right_hand_ang_vel),dim=1) # B,18


        left_hand_ang_acc = torch.cat((data['left_hand_ang_acc'],data['left_hand_ang_acc'][-2:]),dim=0).reshape(-1,9) # B,9
        right_hand_ang_acc = torch.cat((data['right_hand_ang_acc'],data['right_hand_ang_acc'][-2:]),dim=0).reshape(-1,9) # B,9
        hand_ang_acc = torch.cat((left_hand_ang_acc, right_hand_ang_acc),dim=1) # B,18

        # "left_rel_right_lin_vel": left_rel_right_lin_vel,  # B-1, 21, 3
        # "left_rel_right_lin_acc": left_rel_right_lin_acc,  # B-2, 21, 3
        # "left_rel_right_ang_vel": left_rel_right_ang_vel,  # B-1, 3, 3
        # "left_rel_right_ang_acc": left_rel_right_ang_acc,  # B-2, 3, 3
        hand_rel_lin_vel = torch.cat((data['left_rel_right_lin_vel'],data['left_rel_right_lin_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,63) # B,63
        hand_rel_lin_acc = torch.cat((data['left_rel_right_lin_acc'],data['left_rel_right_lin_acc'][-2:]),dim=0).reshape(-1,63) # B,63
        hand_rel_ang_vel = torch.cat((data['left_rel_right_ang_vel'],data['left_rel_right_ang_vel'][-1].unsqueeze(0)),dim=0).reshape(-1,9) # B,9
        hand_rel_ang_acc = torch.cat((data['left_rel_right_ang_acc'],data['left_rel_right_ang_acc'][-2:]),dim=0).reshape(-1,9) # B,9

        left_right_offset = data['left_right_offset'].reshape(-1,63)


        hand_all = torch.cat((hand_kp,hand_contact, hand_obj_offset,left_right_offset, hand_lin_vel,hand_lin_acc,hand_ang_vel,hand_ang_acc,hand_rel_lin_vel,hand_rel_lin_acc,hand_rel_ang_vel,hand_rel_ang_acc),dim=-1).numpy()
        # hand_all = torch.cat((hand_kp,hand_contact, hand_obj_offset,hand_lin_vel,hand_lin_acc,hand_ang_vel,hand_ang_acc,hand_rel_lin_vel,hand_rel_lin_acc,hand_rel_ang_vel,hand_rel_ang_acc),dim=-1).numpy()
        # print("hand_all.shape: ", hand_all.shape)

        obj_pose = data['obj_pose'].reshape(-1,12).numpy()
        gt_obj_pose = data['gt_obj_pose'].reshape(-1,12).numpy()
        obj_verts = data['obj_verts'].numpy()
        obj_normals = data['obj_normals'].numpy()
        hand_shape = torch.cat((data['left_hand_shape'],data['right_hand_shape']),dim=1).numpy()

        goal_index = data['goal_index']
        if goal_index>=30:
            goal_index = -1
        goal_pose = hand_all[goal_index]

        return seq, hand_all, obj_pose, obj_verts,obj_normals, goal_pose, hand_kp.shape[0],hand_shape #TODO: add gt_obj_pose

        # return seq, hand_kp, hand_contact, hand_obj_offset, hand_lin_vel, hand_lin_acc, hand_ang_vel, hand_ang_acc
        

class GazeHOIDataset_g2ho(data.Dataset):
    def __init__(self, mode='stage1', datapath='/root/code/seqs/gazehoi_list_train_0718.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_0718.txt'
        print(datapath)
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.hint_type = hint_type
        self.datalist = []
        self.fps = 6
        self.target_length = 150
        print(len(self.seqs))
        for seq in tqdm(self.seqs):
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            
            right_mano_path = join(seq_path, 'mano/poses_right.npy')
            left_mano_path = join(seq_path, 'mano/poses_left.npy')
            right_hand_params = np.load(right_mano_path)[:,:51]
            left_hand_params = np.load(left_mano_path)[:,:51]
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path)
            # print(gaze.shape, right_hand_params.shape)

            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            active_obj = meta['active_obj']

            obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)
            new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)

            self.target_length = 150
            seq_len = right_hand_params.shape[0]
            if seq_len >= self.target_length:
                indices = torch.linspace(0, seq_len - 1, steps=self.target_length).long()
                right_hand_params = right_hand_params[indices]
                left_hand_params = left_hand_params[indices]
                obj_pose = obj_pose[indices].reshape(-1,12)
                gaze = gaze[indices]
            else:
                pad_width = ((0,self.target_length-seq_len), (0, 0))
                right_hand_params = np.pad(right_hand_params,pad_width,mode='edge')
                left_hand_params = np.pad(left_hand_params,pad_width,mode='edge')
                gaze = np.pad(gaze,pad_width,mode='edge')
                obj_pose = np.pad(obj_pose.reshape(-1,12),pad_width,mode='edge')
            
            step = int(30/self.fps)
            for i in range(step):
                # print(i)
                right_hand_params_lowfps = right_hand_params[i::step]
                left_hand_params_lowfps = left_hand_params[i::step]
                obj_pose_lowfps = obj_pose[i::step]
                gaze_lowfps = gaze[i::step]
                # print(left_hand_params_lowfps.shape,right_hand_params_lowfps.shape,obj_pose_lowfps.shape)
                motion = np.concatenate((left_hand_params_lowfps,right_hand_params_lowfps,obj_pose_lowfps),axis=-1)
                hand_pose = np.concatenate((left_hand_params_lowfps,right_hand_params_lowfps),axis=-1)
                data = {'motion': motion, 
                        'gaze':gaze_lowfps,
                        'hand_pose':hand_pose,
                        'obj_verts':new_verts,
                        'obj_pose': obj_pose_lowfps,
                        'seq_length': obj_pose_lowfps.shape[0],
                        'seq':seq}
                self.datalist.append(data)
                # if split == 'test':
                break

    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        return data['motion'], data['gaze'], data['obj_verts'],data['obj_pose'],data['hand_pose'],data['seq_length'],data['seq']


def read_xyz(path):
    data = []
    with open(path,'r') as f:
        line = f.readline()
        ls = line.strip().split(' ')
        data.append([float(ls[0]),float(ls[1]),float(ls[2])])
        while line:
            ls = f.readline().strip().split(' ')
            # print(ls)
            if ls != ['']:
                data.append([float(ls[0]),float(ls[1]),float(ls[2])])
            else:
                line = None
    data = np.array(data)
    return data


class GazeHOIDataset_stage0_1obj(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_0718.txt', split='train',hint_type='goal_pose'):
    # def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_0303.txt', split='train',hint_type='goal_pose'):
        # super().__init__()
        if split == 'test':
            # datapath = '/root/code/seqs/gazehoi_list_test_0303.txt'
            datapath = '/root/code/seqs/gazehoi_list_test_0718.txt'
        print(datapath)
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.obj_global_mean = np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_global_obj_mean.npy')
        self.obj_global_std = np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_global_obj_std.npy')
        self.obj_local_mean = np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_local_obj_mean.npy')
        self.obj_local_std = np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_local_obj_std.npy')
        self.hint_type = hint_type
        self.table_plane = read_xyz("/root/code/gazehoi-diffusion/dataset/table_plane_750.xyz")
        self.datalist = []
        self.fps = 6
        self.target_length = 150
        for seq in self.seqs:
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            # mano_right_path = join(seq_path, 'mano/poses_right.npy')
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            active_obj = meta['active_obj']
            # gaze_obj = meta['gaze_obj']
            goal_index = meta['goal_index']

            obj_name_list = meta['obj_name_list']
            obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)

            new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)

            right_mano_path = join(seq_path, 'mano/poses_right.npy')
            left_mano_path = join(seq_path, 'mano/poses_left.npy')
            right_hand_params = np.load(right_mano_path)
            left_hand_params = np.load(left_mano_path)

            # 统一seq长度 150帧 -- 降低帧率 30fps--6fps
            seq_len = obj_pose.shape[0]
            if seq_len >= self.target_length:
                indices = torch.linspace(0, seq_len - 1, steps=self.target_length).long()
                obj_pose = obj_pose[indices]
                gaze = gaze[indices]
                right_hand_params = right_hand_params[indices]
                left_hand_params = left_hand_params[indices]

            step = int(30/self.fps)
            for i in range(step):
                gaze_ = gaze[i::step]
                num_frames_ = gaze_.shape[0]
                goal_index_ = int(goal_index/step)
                obj_pose_ = obj_pose[i::step]
                right_hand_params_lowfps = right_hand_params[i::step] # 30,61*2
                left_hand_params_lowfps = left_hand_params[i::step]
                hand_params_lowfps = np.concatenate((left_hand_params_lowfps, right_hand_params_lowfps),axis=-1)
                # print(hand_params_lowfps.shape)
                data = {"gaze":gaze_, #x
                        "obj_pose":obj_pose_, #x
                        "obj_verts": new_verts,
                        "seq":seq,
                        "active_obj":active_obj,
                        "num_frames":num_frames_,
                        "hand_params": hand_params_lowfps}#x
                        # "goal_index":goal_index_}#x
                
                self.datalist.append(data)
                if split == 'test':
                    break
        print(len(self.datalist))
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        flag = np.zeros(4)
        length = 30
        num_frames = data['num_frames']
        # goal_index = data['goal_index']
        seq = data['seq']
        # assert goal_index < num_frames,  f"spilt wrong {seq},{goal_index},{num_frames}"
        obj_pose = data['obj_pose']
        obj_verts = data['obj_verts']
        gaze = data['gaze']
        hand_params = data['hand_params']
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')
            hand_params = np.pad(hand_params, pad_width, mode='edge')

        obj_pose = torch.tensor(obj_pose)
        obj_pose_global_6d = obj_matrix2rot6d(obj_pose.unsqueeze(0)).squeeze(0).numpy()

        obj_pose_global_6d = (obj_pose_global_6d - self.obj_global_mean) / self.obj_global_std

        # print(obj_pose_global_6d.shape)
        hint = np.zeros((length,9))
        hint[0] = obj_pose_global_6d[0]
        # obj_verts = np.vstack(((obj_verts,self.table_plane)))
        # print(obj_pose_global_6d.shape)
        # print(hand_params.shape)
        return  obj_pose_global_6d, hint,gaze, obj_verts,hand_params, flag, num_frames,seq

def get_hand_joints_verts(hand_params,hand_flag):
    """
    hand params: N,61
    hand_flag : left or right
    """
    hand_trans = hand_params[:,:3]
    hand_rot = hand_params[:,3:6]
    hand_theta = hand_params[:,3:51]
    mano_beta = hand_params[:,51:]
    manolayer = ManoLayer(mano_assets_root='/root/code/CAMS/data/mano_assets/mano',side=hand_flag)
    hand_faces = manolayer.th_faces
    hand_output = manolayer(hand_theta, mano_beta)
    hand_verts = hand_output.verts - hand_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)
    hand_joints = hand_output.joints - hand_output.joints[:, 0].unsqueeze(1) + hand_trans.unsqueeze(1)
    return hand_joints, hand_verts, hand_faces

class GazeHOIDataset_eval(data.Dataset):
    def __init__(self, mode='stage0', datapath='/root/code/seqs/gazehoi_list_train_0718.txt', split='train',hint_type='goal_pose'):
        if split == 'test':
            datapath = '/root/code/seqs/gazehoi_list_test_0718.txt'
        print(datapath)
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        with open(datapath,'r') as f:
            info_list = f.readlines()
        self.seqs = []
        for info in info_list:
            seq = info.strip()
            self.seqs.append(seq)
        self.datalist = []
        self.fps = 6
        self.target_length = 150
        for seq in self.seqs:
            seq_path = join(self.root,seq)
            meta_path = join(seq_path,'meta.pkl')
            gaze_path = join(seq_path,'fake_goal.npy')
            gaze = np.load(gaze_path) # (num_frames, 3)
            num_frames = gaze.shape[0]
            with open(meta_path,'rb')as f:
                meta = pickle.load(f)
            
            active_obj = meta['active_obj']

            obj_verts = np.load(join(self.obj_path,active_obj,'resampled_500_trans.npy'))
            obj_pose = np.load(join(seq_path,active_obj+'_pose_trans.npy')).reshape(-1,3,4)

            new_verts = obj_verts @ obj_pose[0,:3,:3].T + obj_pose[0,:3,3].reshape(1,3)

            right_mano_path = join(seq_path, 'mano/poses_right.npy')
            left_mano_path = join(seq_path, 'mano/poses_left.npy')
            right_hand_params = torch.tensor(np.load(right_mano_path))
            right_joints,_,_ = get_hand_joints_verts(right_hand_params,'right')
            right_joints = right_joints.numpy()
            left_hand_params = torch.tensor(np.load(left_mano_path))
            left_joints,_,_ = get_hand_joints_verts(left_hand_params,'left')
            left_joints = left_joints.numpy()

            # 统一seq长度 150帧 -- 降低帧率 30fps--6fps
            seq_len = obj_pose.shape[0]
            if seq_len >= self.target_length:
                indices = torch.linspace(0, seq_len - 1, steps=self.target_length).long()
                obj_pose = obj_pose[indices]
                gaze = gaze[indices]
                right_joints = right_joints[indices]
                left_joints = left_joints[indices]

            step = int(30/self.fps)
            for i in range(step):
                gaze_ = gaze[i::step]
                num_frames_ = gaze_.shape[0]
                # goal_index_ = int(goal_index/step)
                obj_pose_ = obj_pose[i::step]
                right_joints_lowfps = right_joints[i::step] # 30,61*2
                left_joints_lowfps = left_joints[i::step]
                hand_params_lowfps = np.concatenate((left_joints_lowfps, right_joints_lowfps),axis=1) # 30,42,3
                # print(hand_params_lowfps.shape)
                data = {"gaze":gaze_, #x
                        "obj_pose":obj_pose_, #x
                        "obj_verts": new_verts,
                        "seq":seq,
                        "active_obj":active_obj,
                        "num_frames":num_frames_,
                        "hand_params": hand_params_lowfps}#x
                        # "goal_index":goal_index_}#x
                
                self.datalist.append(data)
                if split == 'test':
                    break
        print(len(self.datalist))
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        length = 30
        num_frames = data['num_frames']
        # goal_index = data['goal_index']
        seq = data['seq']
        # assert goal_index < num_frames,  f"spilt wrong {seq},{goal_index},{num_frames}"
        obj_pose = data['obj_pose']
        obj_verts = data['obj_verts']
        gaze = data['gaze']
        hand_params = data['hand_params']
        if num_frames < length:
            # 在后边补充
            pad_width = ((0,length-num_frames), (0, 0))  # 在第一维度上填充0行，使总行数变为10
            obj_pose = np.pad(obj_pose.reshape(-1,12), pad_width, mode='edge').reshape(-1,3,4)
            gaze = np.pad(gaze.reshape(-1,3), pad_width, mode='edge')
            hand_params = np.pad(hand_params.reshape(-1,42*3), pad_width, mode='edge').reshape(-1,42,3)
        obj_pose = obj_pose.reshape(-1,12)
        hand_params = hand_params.reshape(-1,42*3)
        return  obj_pose,gaze, obj_verts,hand_params,num_frames,seq







                
