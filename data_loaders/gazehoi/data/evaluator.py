import os
import numpy as np
from os.path import join as pjoin
from os.path import join
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm
from model.eval_model import Emb_model
from manotorch.manolayer import ManoLayer
import torch

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

# 已经保存了，直接load保存的结果来eval
# intergen原版是每次生成
class GazeHOI_eval_fid_div(Dataset):
    def __init__(self,gene_path='/nas/gazehoi-diffusion/final_result/0725_stage0_1obj_000100000_seed10',mode = 'dsg_guide'):
        print(mode,gene_path)
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        self.seqs = sorted(os.listdir(gene_path))
        self.gene_path = gene_path
        self.datalist = []
        for seq in tqdm(self.seqs):
            seq_path = join(gene_path,seq)
            gt_obj_pose = np.load(join(seq_path,'gt_obj_pose.npy'))
            pred_obj_pose = np.load(join(seq_path,'pred_obj_pose.npy'))
            gt_left_mano = torch.tensor(np.load(join(seq_path,'gt_left_mano.npy')))
            gt_right_mano = torch.tensor(np.load(join(seq_path,'gt_right_mano.npy')))
            gt_left_joint, _, _ = get_hand_joints_verts(gt_left_mano,'left')
            gt_right_joint, _, _ = get_hand_joints_verts(gt_right_mano,'right')
            gt_left_joint = gt_left_joint.numpy()
            gt_right_joint = gt_right_joint.numpy()
            # pred_left_mano = torch.tensor(np.load(join(seq_path,mode,'pred_left_mano.npy')))
            # pred_right_mano = torch.tensor(np.load(join(seq_path,mode,'pred_right_mano.npy')))
            pred_left_mano = torch.tensor(np.load(join(seq_path,'pred_left_mano.npy')))
            pred_right_mano = torch.tensor(np.load(join(seq_path,'pred_right_mano.npy')))
            pred_left_joint, _, _ = get_hand_joints_verts(pred_left_mano,'left')
            pred_right_joint, _, _ = get_hand_joints_verts(pred_right_mano,'right')
            pred_left_joint = pred_left_joint.numpy()
            pred_right_joint = pred_right_joint.numpy()
            # pred_left_joint = np.load(join(seq_path,mode,'pred_left_joint.npy'))
            # pred_right_joint = np.load(join(seq_path,mode,'pred_right_joint.npy'))
            # pred_obj_pose = np.load(join(seq_path,mode,'pred_obj_pose.npy'))
            data = {"gt_obj_pose":gt_obj_pose,
                    "pred_obj_pose": pred_obj_pose,
                    "gt_left_joint": gt_left_joint,
                    "gt_right_joint": gt_right_joint,
                    'pred_left_joint':pred_left_joint,
                    "pred_right_joint":pred_right_joint}
            self.datalist.append(data)

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = self.datalist[index]
        gt_obj_pose = torch.tensor(data['gt_obj_pose']).reshape(-1,12).float()
        gt_hand_params = torch.cat((torch.tensor(data['gt_left_joint']), torch.tensor(data['gt_right_joint'])),dim=1).reshape(-1,42*3).float()
        pred_obj_pose = torch.tensor(data['pred_obj_pose']).reshape(-1,12).float()
        pred_hand_params = torch.cat((torch.tensor(data['pred_left_joint']), torch.tensor(data['pred_right_joint'])),dim=1).reshape(-1,42*3).float()

        return gt_obj_pose, gt_hand_params, pred_obj_pose, pred_hand_params

# 已经保存了，直接load保存的结果来eval
# intergen原版是每次生成
class GazeHOI_eval_mm(Dataset):
    def __init__(self,gene_path='/nas/gazehoi-diffusion/final_result/0725_stage0_1obj_000100000_seed10',mode = 'repeat_2'):
        print(mode,gene_path)
        self.repeat_time = int(mode.split('_')[-1])
        print(self.repeat_time)
        self.root = '/root/code/seqs/0303_data/'
        self.obj_path = '/root/code/seqs/object/'
        self.seqs = sorted(os.listdir(gene_path))
        self.gene_path = gene_path
        self.datalist = []
        for seq in tqdm(self.seqs):
            seq_path = join(gene_path,seq)
            # pred_obj_pose = np.load(join(seq_path,'pred_obj_pose.npy'))
            pred_left_joints = []
            pred_right_joints = []
            pred_obj_poses = []
            for i in os.listdir(join(seq_path,mode)):
                # pred_left_joint = np.load(join(seq_path,mode,str(i),'pred_left_joint.npy')).reshape(1,-1,21,3)
                # pred_left_joints.append(pred_left_joint)
                # pred_right_joint = np.load(join(seq_path,mode,str(i),'pred_right_joint.npy')).reshape(1,-1,21,3)
                # pred_right_joints.append(pred_right_joint)
                # print(i)
                pred_left_mano = torch.tensor(np.load(join(seq_path,mode,i,'pred_left_mano.npy')))
                pred_right_mano = torch.tensor(np.load(join(seq_path,mode,i,'pred_right_mano.npy')))
                pred_left_joint, _, _ = get_hand_joints_verts(pred_left_mano,'left')
                pred_right_joint, _, _ = get_hand_joints_verts(pred_right_mano,'right')
                pred_left_joint = pred_left_joint.numpy()
                pred_right_joint = pred_right_joint.numpy()
                pred_left_joints.append(pred_left_joint)
                pred_right_joints.append(pred_right_joint)
                pred_obj_pose = np.load(join(seq_path,mode,i,'pred_obj_pose.npy'))
                pred_obj_poses.append(pred_obj_pose.reshape(1,-1,3,4))
            pred_left_joints = np.concatenate(pred_left_joints, axis=0)
            pred_right_joints = np.concatenate(pred_right_joints, axis=0)
            pred_obj_poses = np.concatenate(pred_obj_poses, axis=0)
            # print(pred_left_joints.shape, pred_right_joints.shape, pred_obj_poses.shape)
            data = {
                    "pred_obj_pose": pred_obj_poses,
                    'pred_left_joint':pred_left_joints,
                    "pred_right_joint":pred_right_joints,
                    'repeat_time': len(os.listdir(join(seq_path,mode)))}
            self.datalist.append(data)

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = self.datalist[index]
        repeat_time = data['repeat_time']
        pred_obj_pose = torch.tensor(data['pred_obj_pose']).reshape(-1,repeat_time,12).float()
        pred_hand_params = torch.cat((torch.tensor(data['pred_left_joint']), torch.tensor(data['pred_right_joint'])),dim=1).reshape(-1,repeat_time,42*3).float()
        # print(pred_obj_pose.shape, pred_hand_params.shape)
        return pred_obj_pose, pred_hand_params


def get_motion_loader_fid_div(batch_size,path,mode='dsg_guide'):
    print('get_motion_loader:',batch_size,mode)
    # Currently the configurations of two datasets are almost the same
    dataset = GazeHOI_eval_fid_div(path,mode)
    # print(len(dataset))
    torch.manual_seed(42)
    # motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=False)
    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    print('Generated Dataset Loading Completed!!!  GazeHOI_eval_mm')

    return motion_loader

def get_motion_loader_mm(batch_size,path,mode='dsg_guide'):
    print('get_motion_loader:',batch_size,mode)
    # Currently the configurations of two datasets are almost the same
    dataset = GazeHOI_eval_mm(path,mode)
    # print(len(dataset))
    torch.manual_seed(42)
    # motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=False)
    motion_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    print('Generated Dataset Loading Completed!!!  GazeHOI_eval_fid_div')

    return motion_loader



def build_models(cfg):
    model = Emb_model(cfg)

    checkpoint = torch.load('/root/code/gazehoi-diffusion/save/eval_train_kp_fix/model_750.pt',map_location="cpu")
    # print(checkpoint.keys())
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    # for k in list(checkpoint["state_dict"].keys()):
    #     if "model" in k:
    #         checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["model"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    print('Loading Evaluation Model Wrapper  Completed!!')
    return model



class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg) # bulid eval model
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            y = batch_data
            '''Motion Encoding'''
            motion_embedding = self.model.motion_encoder(y)

            '''Text Encoding'''
            cond_embedding = self.model.condition_encoder(y)

        return cond_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            y = batch_data

            '''Motion Encoding'''
            motion_embedding = self.model.motion_encoder(y)

        return motion_embedding
