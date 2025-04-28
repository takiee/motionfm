from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
# from model.rotation2xyz import Rotation2xyz
from model.base_cross_model import PerceiveEncoder
from model.base_cross_model import *
from model.pointnet_plus2 import *

class MDM_Flow_Gaze(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="amass",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        text_embed="clip",
        device="cuda",
        **kargs
    ):
        super().__init__()
        self.length = 30

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get("action_emb", None)

        self.device = device
        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.inputprocess_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.inputprocess_emb_dim, self.latent_dim
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == "trans_enc":
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )

            self.seqTransEncoder = nn.TransformerEncoder(
                seqTransEncoderLayer, num_layers=self.num_layers
            )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )
        
        if  self.dataset == 'gazehoi_stage0_1obj':
            self.encode_obj_pose = nn.Linear(9,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.fp_layer = MyFPModule()
            self.gaze_linear = nn.Linear(self.latent_dim, self.latent_dim)  
            self.encode_gaze = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
        elif self.dataset == 'gazehoi_o2h_mid':
            self.encode_obj_pose = nn.Linear(12,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.encode_obj = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.encode_hand_pose = nn.Sequential(nn.Linear(291,128), nn.ELU(),
                                                nn.Linear(128,self.latent_dim) )
        elif self.dataset == 'gazehoi_o2h_mid_2hand_assemobj':
            self.encode_obj_pose = nn.Linear(12,self.latent_dim)
            
            self.encode_obj_mesh = PointNet2SemSegSSGShape({'feat_dim': self.latent_dim})
            self.encode_obj = PerceiveEncoder(n_input_channels=self.latent_dim,
                                            n_latent=self.length,
                                            n_latent_channels=self.latent_dim,
                                            n_self_att_heads=4,
                                            n_self_att_layers=3,
                                            dropout=0.1)
            self.encode_hand_pose = nn.Sequential(nn.Linear(726+63,128), nn.ELU(), #TODO: fix dim
                                                nn.Linear(128,self.latent_dim) )
            self.encode_goal_pose = nn.Sequential(nn.Linear(726+63,128), nn.ELU(), #TODO: fix dim
                                                nn.Linear(128,self.latent_dim) )




        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )


    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = len(cond)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def mask_cond_3d(self, cond, force_mask=False):
        bs = len(cond)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (float)
        """
        bs, njoints, nfeats, nframes = x.shape
        if len(timesteps.shape) == 0:  # mainly in ODE sampling
            timesteps = repeat(timesteps.unsqueeze(0), "1 -> b", b=len(x))
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        emb_dim = 1

        force_mask = y.get("uncond", False)
        
        x = self.input_process(x)
        if self.dataset == 'gazehoi_stage0_1obj':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['gaze'].shape
            init_obj_pose = y['hint'][:,0]
            obj_pose_emb = self.encode_obj_pose(init_obj_pose).unsqueeze(0)
            points = y['obj_points']
            table = points[:,500:]
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))
            gaze = y['gaze']
            gaze_emb = self.fp_layer(gaze, points, points_feat).permute((0, 2, 1))
            gaze_feat = self.gaze_linear(gaze_emb)
            gaze_feat = self.encode_gaze(gaze_feat)
            # print(x.shape,obj_pose_emb.shape, global_obj_feat.shape, gaze_feat.shape)
            x = x + obj_pose_emb + global_obj_feat + gaze_feat.permute(1,0,2).contiguous()
        elif self.dataset == 'gazehoi_o2h_mid':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['obj_pose'].shape
            obj_pose = y['obj_pose']
            # print(obj_pose.shape)
            obj_pose_emb = self.encode_obj_pose(obj_pose) #[bs,30,256]

            points = y['obj_points']
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))

            obj_feat = self.encode_obj(obj_pose_emb)  #[bs,30,256]

            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D

            x = x  + global_obj_feat + obj_feat.permute(1,0,2).contiguous() + init_hand_emb 
        elif self.dataset == 'gazehoi_o2h_mid_2hand_assemobj':
            """
            提取物体pose和shape特征
            """
            bs, nf, _ = y['obj_pose'].shape
            obj_pose = y['obj_pose']
            # print(obj_pose.shape)
            obj_pose_emb = self.encode_obj_pose(obj_pose)

            points = y['obj_points']
            points_feat, global_obj_feat= self.encode_obj_mesh(points.repeat(1, 1, 2))

            obj_feat = self.encode_obj(obj_pose_emb)

            init_hand_emb = self.encode_hand_pose(y['init_hand_pose']) #b,D
            # goal_hand_emb = self.encode_goal_pose(y['goal_hand_pose']) #b,D

            x = x  + global_obj_feat + obj_feat.permute(1,0,2).contiguous() + init_hand_emb 
            # x = x  + global_obj_feat + obj_feat.permute(1,0,2).contiguous() + init_hand_emb + goal_hand_emb

        if self.arch == "trans_enc":
            # adding the timestep embed
            x_len_old = len(x)
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+emb_dim, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+emb_dim, bs, d]
            output = self.seqTransEncoder(xseq)[emb_dim:]
            assert len(output) == x_len_old  # 196=165+31
            # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]


        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    # def _apply(self, fn):
    #     super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)

    # def train(self, *args, **kwargs):
    #     super().train(*args, **kwargs)
    #     self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder, time_resolution=1000):
        super().__init__()
        self.time_resolution = time_resolution
        print("time_resolution: ", time_resolution)
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        timesteps = (timesteps * self.time_resolution).long()
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        x = rearrange(x, "b j f t -> t b (j f)")

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x



class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == "rot_vel":
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
      
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
