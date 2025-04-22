# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from einops import rearrange

import numpy as np
import torch
import torch as th
from tqdm import tqdm
from diffusion.nn import sum_flat
from data_loaders.humanml.scripts import motion_process
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_util import *
# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
import wandb

from utils.dist_util import is_rank_zero


class FlowMatching:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42


    """

    def __init__(
        self,
        args,
        lambda_rcxyz=0.0,
        lambda_vel=0.0,
        data_rep="rot6d",
        lambda_root_vel=0.0,
        lambda_vel_rcxyz=0.0,
        lambda_fc=0.0,
    ):
        self.data_rep = data_rep
        self.dataset = args.dataset
        if  self.dataset == 'gazehoi_stage0_1obj':
            self.obj_global_mean = torch.from_numpy(np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_global_obj_mean.npy'))
            self.obj_global_std = torch.from_numpy(np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_global_obj_std.npy'))
            self.obj_local_mean = torch.from_numpy(np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_local_obj_mean.npy'))
            self.obj_local_std = torch.from_numpy(np.load('/root/code/gazehoi-diffusion/dataset/gazehoi_local_obj_std.npy'))
   

        self.l2_loss = (
            lambda a, b: (a - b) ** 2
        )  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.
        
        
    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(
            loss * mask.float()
        )  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    @torch.no_grad()
    def sample_euler_raw(self, model, z_orig, N, model_kwargs, ode_kwargs):
        dt = 1.0 / N
        traj = []  # to store the trajectory

        z = z_orig.detach().clone()
        bs = len(z)

        est = []
        return_x_est = ode_kwargs["return_x_est"]
        if return_x_est:
            return_x_est_num = ode_kwargs["return_x_est_num"]
            est_ids = [int(i * N / return_x_est_num) for i in range(return_x_est_num)]

        traj.append(z.detach().clone())
        for i in range(0, N, 1):
            t = torch.ones(bs, device=z_orig.device) * i / N
            pred = model(z, t, **model_kwargs)

            _est_now = z + (1 - i * 1.0 / N) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]

    @torch.no_grad()
    def sample_euler_replacement_edit_till(
        self, model, z_orig, N, edit_till, model_kwargs=None, ode_kwargs=None
    ):
        inpainting_mask, inpainted_motion = (
            model_kwargs["y"]["inpainting_mask"],
            model_kwargs["y"]["inpainted_motion"],
        )

        dt = 1.0 / N
        traj = []  # to store the trajectory
        z = z_orig.detach().clone()
        batchsize = len(z)

        est = []
        return_x_est = ode_kwargs["return_x_est"]
        if return_x_est:
            return_x_est_num = ode_kwargs["return_x_est_num"]
            est_ids = [int(i * N / return_x_est_num) for i in range(return_x_est_num)]

        traj.append(z.detach().clone())
        for i in range(0, N, 1):
            t = torch.ones((batchsize), device=z_orig.device) * i / N

            _inpainted_motion = (z_orig * (N - i) + inpainted_motion * i) / N
            if i * 1.0 / N <= edit_till:
                z = (z * ~inpainting_mask) + (_inpainted_motion * inpainting_mask)

            pred = model(z, t, **model_kwargs)

            _est_now = z + (1 - i * 1.0 / N) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]

    @torch.no_grad()
    def cal_curveness(self, model, z_orig, N, model_kwargs):
        print(f"cal_curveness, N={N}")
        dt = 1.0 / N
        traj = []  # to store the trajectory
        preds = []
        z = z_orig.detach().clone()
        bs = len(z)

        func = lambda t, x: model(x, t, **model_kwargs)
        target = (
            odeint(
                func,
                z,
                # 0.0,
                torch.tensor([0.0, 1.0], device=z_orig.device, dtype=z_orig.dtype),
                # phi=self.parameters(),
                rtol=1e-5,
                atol=1e-5,
                method="dopri5",
                adjoint_params=(),
                # **ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]
            .detach()
            .clone()
        )
        # pre-compute the target, as it's too memory-consuming to save the intermediate results

        traj.append(z.detach().clone())
        for i in tqdm(range(0, N, 1), desc="cal_curveness", total=N):
            t = torch.ones(bs, device=z_orig.device) * i / N
            pred = model(z, t, **model_kwargs)
            pred = pred.detach().clone()
            preds.append((pred - target).pow(2).mean().item())
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        result = sum(preds) / len(preds)
        print("curveness: ", result)
        return result

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        ode_kwargs=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device="cuda",
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        sample_steps=None,  # backward compatibility, never use it
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        if noise is None:
            # print("noise is None, use randn instead")
            noise = torch.randn(*shape, device=device)

        func = lambda t, x: model(x, t, **model_kwargs)

        if ode_kwargs["method"] in ["euler", "dopri5"]:
            assert not ("return_x_est" in ode_kwargs and ode_kwargs["return_x_est"])
            if ode_kwargs["method"] == "euler":
                ode_kwargs = dict(
                    rtol=ode_kwargs["rtol"],
                    atol=ode_kwargs["atol"],
                    method="euler",
                    options=dict(step_size=ode_kwargs["step_size"]),
                )
            elif ode_kwargs["method"] == "dopri5":
                ode_kwargs = dict(
                    rtol=ode_kwargs["rtol"],
                    atol=ode_kwargs["atol"],
                    method="dopri5",
                )
            data = odeint(
                func,
                noise,
                # 0.0,
                torch.tensor([0.0, 1.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **ode_kwargs
                # options=dict(step_size=1/100),
            )
            data = data[-1]
        elif ode_kwargs["method"] == "euler_replacement_edit_till":
            data = self.sample_euler_replacement_edit_till(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                edit_till=ode_kwargs["edit_till"],
                model_kwargs=model_kwargs,
                ode_kwargs=ode_kwargs,
            )

        elif ode_kwargs["method"] == "odenoise_euler_replacement":
            inpainting_mask, inpainted_motion = (
                model_kwargs["y"]["inpainting_mask"],
                model_kwargs["y"]["inpainted_motion"],
            )
            partial_data = (0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            _ode_kwargs = dict(
                rtol=ode_kwargs["rtol"],
                atol=ode_kwargs["atol"],
                method="dopri5",
            )
            noise = odeint(
                func,
                partial_data,
                # 0.0,
                torch.tensor([1.0, 0.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **_ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]

            data = self.sample_euler_replacement_edit_till(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                edit_till=ode_kwargs["edit_till"],
                model_kwargs=model_kwargs,
                ode_kwargs=ode_kwargs,
            )
        elif ode_kwargs["method"] == "variation_euler_replacement":
            inpainting_mask, inpainted_motion = (
                model_kwargs["y"]["inpainting_mask"],
                model_kwargs["y"]["inpainted_motion"],
            )
            partial_data = inpainted_motion  # (0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            _ode_kwargs = dict(
                rtol=ode_kwargs["rtol"],
                atol=ode_kwargs["atol"],
                method="dopri5",
            )
            noise = odeint(
                func,
                partial_data,
                # 0.0,
                torch.tensor([1.0, 0.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **_ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]
            _noise_masked = (torch.randn_like(noise) * ~inpainting_mask) + (
                noise * inpainting_mask
            )

            data = self.sample_euler_replacement(
                model,
                z_orig=_noise_masked,
                N=int(1 / ode_kwargs["step_size"]),
                model_kwargs=model_kwargs,
            )
        elif ode_kwargs["method"] == "euler_raw":
            data = self.sample_euler_raw(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                ode_kwargs=ode_kwargs,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError

        is_return_est = isinstance(data, tuple)
        if is_return_est:
            data, x0_est = data
            assert model.training is not True, "x0_est is only for inference"

        data_range_dict = dict()
        # 0th-joint as an obversation
        data_range_dict["data_range_j0/gen_mean"] = data[:, 0].mean()
        data_range_dict["data_range_j0/gen_std"] = data[:, 0].std()
        data_range_dict["data_range_j0/gen_min"] = data[:, 0].min()
        data_range_dict["data_range_j0/gen_max"] = data[:, 0].max()
        if is_rank_zero() and model.training:
            wandb.log(data_range_dict)
        if is_return_est:
            return data, x0_est
        else:
            return data
    def get_1verts_global(self,o1_verts,target):
        bs = target.shape[0]
        frames = target.shape[1]
        tgt_R = rotation_6d_to_matrix(target[:,:,3:]) # bs,nframes,3,4
        tgt_T = target[:,:,:3].unsqueeze(2) #bs,nframes,1,3
        tgt_R = torch.einsum('...ij->...ji', [tgt_R]) # 对R的最后两个维度转置
        # print(o1_verts.shape,tgt_R.shape,tgt_T.shape)
        tgt_o1 = torch.einsum('bfpn,bfnk->bfpk',o1_verts,tgt_R) + tgt_T #    b frames 500 3
        tgt_obj_verts = tgt_o1.reshape(bs,frames,-1).permute(0,2,1).unsqueeze(2)
        return tgt_obj_verts
    def training_losses_stage0_1obj(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        dataset=None,
        sigma_min=1e-4,
    ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        mask = model_kwargs["y"]["mask"]

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert t is None
        t = torch.rand(len(x_start), device=x_start.device, dtype=x_start.dtype)
        t_1d = t[:,]  # [B, 1, 1, 1]
        t = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t = t * x_start + (1 - (1 - sigma_min) * t) * noise
        target = x_start - (1 - sigma_min) * noise
        bs = target.shape[0]
        nf = target.shape[-1]
        terms = {}
        model_output = model(x_t, t_1d, **model_kwargs)
        terms["rot_mse"] = self.masked_l2(
            target, model_output, mask
        )  # mean_flat(rot_mse)

        target = target.permute(0, 3, 2, 1).squeeze(2).contiguous()
        output = model_output.permute(0, 3, 2, 1).squeeze(2).contiguous()

        gt_T = target[:,:,:3].reshape(bs,nf,-1).permute(0,2,1).unsqueeze(2)
        out_T = output[:,:,:3].reshape(bs,nf,-1).permute(0,2,1).unsqueeze(2)
        # print(gt_T.shape)
        # print(mask.shape)
        terms['obj_T'] = self.masked_l2(gt_T, out_T, mask)


        # obj verts loss
        # print(model_kwargs['y']['obj_points'].shape)
        length = model.length
        # print(length)
        obj_verts = model_kwargs['y']['obj_points'][:,:500].reshape(-1,500,3).unsqueeze(1).repeat(1,length,1,1) # (b, nframs,npoints,3) 只截取物体的部分
        
        tgt_obj_verts = self.get_1verts_global(obj_verts,target)
        out_obj_verts = self.get_1verts_global(obj_verts,output)
        terms['obj_verts'] = self.masked_l2(tgt_obj_verts, out_obj_verts, mask)

        vel = model_output[:,:,:,1:] -  model_output[:,:,:,:-1]
        # print(vel.shape)
        terms['time_smooth'] = self.masked_l2(vel, torch.zeros_like(vel),mask[:,:,:,1:])

        terms["loss"] = 10*terms["rot_mse"] + 30* terms['obj_T'] + 10*terms['obj_verts'] + terms['time_smooth']
        return terms
    
    def training_losses_o2h_mid(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        dataset=None,
        sigma_min=1e-4,
    ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        mask = model_kwargs["y"]["mask"]

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert t is None
        t = torch.rand(len(x_start), device=x_start.device, dtype=x_start.dtype)
        t_1d = t[:,]  # [B, 1, 1, 1]
        t = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t = t * x_start + (1 - (1 - sigma_min) * t) * noise
        target = x_start - (1 - sigma_min) * noise
        bs = target.shape[0]
        nf = target.shape[-1]
        terms = {}
        model_output = model(x_t, t_1d, **model_kwargs)
        terms["rot_mse"] = self.masked_l2(
            target, model_output, mask
        )  # mean_flat(rot_mse)

        tips = [15,3,6,12,9,35,23,26,32,29]

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]
        # .permute(0, 3, 2, 1).squeeze(2).contiguous()
        pred_left_joints = model_output[:, :63, :, :].squeeze(2).permute(0,2,1).contiguous().reshape(-1,30,21,3) # b,t,21,3
        pred_right_joints = model_output[:, 63:126, :, :].squeeze(2).permute(0,2,1).contiguous().reshape(-1,30,21,3)
        tgt_left_joints = target[:, :63, :, :].squeeze(2).permute(0,2,1).contiguous().reshape(-1,30,21,3)
        tgt_right_joints = target[:, 63:126, :, :].squeeze(2).permute(0,2,1).contiguous().reshape(-1,30,21,3)

        def compute_bone_lengths(joints, connections):
            bones = []
            for connection in connections:
                joint_start, joint_end = connection
                bone = (joints[:,:,joint_start] - joints[:,:,joint_end]).norm(dim=-1, keepdim=True)
                bones.append(bone)
            # print(torch.cat(bones, dim=-1).shape)
            return torch.cat(bones, dim=-1)

        # 计算左右手的骨骼长度
        pred_left_bones = compute_bone_lengths(pred_left_joints, connections)
        tgt_left_bones = compute_bone_lengths(tgt_left_joints, connections)
        pred_right_bones = compute_bone_lengths(pred_right_joints, connections)
        tgt_right_bones = compute_bone_lengths(tgt_right_joints, connections)

        def mix_masked_mse(pred, tgt):
            mse_loss = F.mse_loss(pred, tgt, reduction='none')
            masked_mse_loss = mse_loss
            return masked_mse_loss.mean()

        # 计算左右手的骨骼损失
        left_bone_loss = mix_masked_mse(pred_left_bones, tgt_left_bones)
        right_bone_loss = mix_masked_mse(pred_right_bones, tgt_right_bones)


        # 合并损失
        terms['bone_length'] = left_bone_loss + right_bone_loss


        terms['loss'] =  terms["rot_mse"] * 100  + terms['bone_length'] * 100
        return terms

    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()

        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11
        """ Contact calculated by 'Kfir Method' Commented code)"""
        # contact_signal = torch.zeros((pose_xyz.shape[0], pose_xyz.shape[3], 2), device=pose_xyz.device) # [BatchSize, Frames, 2]
        # left_xyz = 0.5 * (pose_xyz[:, l_ankle_idx, :, :] + pose_xyz[:, l_foot_idx, :, :]) # [BatchSize, 3, Frames]
        # right_xyz = 0.5 * (pose_xyz[:, r_ankle_idx, :, :] + pose_xyz[:, r_foot_idx, :, :])
        # left_z, right_z = left_xyz[:, 2, :], right_xyz[:, 2, :] # [BatchSize, Frames]
        # left_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)  # [BatchSize, Frames]
        # right_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)
        #
        # left_z_mask = left_z <= torch.mean(torch.sort(left_z)[0][:, :left_z.shape[1] // 5], axis=-1)
        # left_z_mask = torch.stack([left_z_mask, left_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # left_z_mask[:, :, 1] = False  # Blank right side
        # contact_signal[left_z_mask] = 0.4
        #
        # right_z_mask = right_z <= torch.mean(torch.sort(right_z)[0][:, :right_z.shape[1] // 5], axis=-1)
        # right_z_mask = torch.stack([right_z_mask, right_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # right_z_mask[:, :, 0] = False  # Blank left side
        # contact_signal[right_z_mask] = 0.4
        # contact_signal[left_z <= (torch.mean(torch.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        # contact_signal[right_z <= (torch.mean(torch.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1

        # plt.plot(to_np_cpu(left_z[0]), label='left_z')
        # plt.plot(to_np_cpu(left_velocity[0]), label='left_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 0]), label='left_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.plot(to_np_cpu(right_z[0]), label='right_z')
        # plt.plot(to_np_cpu(right_velocity[0]), label='right_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 1]), label='right_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()

        gt_joint_xyz = gt_xyz[
            :, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :
        ]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(
            gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2
        )  # [BatchSize, 4, Frames]
        fc_mask = gt_joint_vel <= 0.01
        pred_joint_xyz = pred_xyz[
            :, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :
        ]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(
            pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2
        )  # [BatchSize, 4, Frames]
        pred_joint_vel[
            ~fc_mask
        ] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        """DEBUG CODE"""
        # print(f'mask: {mask.shape}')
        # print(f'pred_joint_vel: {pred_joint_vel.shape}')
        # plt.title(f'Joint: {joint_idx}')
        # plt.plot(to_np_cpu(gt_joint_vel[0]), label='velocity')
        # plt.plot(to_np_cpu(fc_mask[0]), label='fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        return self.masked_l2(
            pred_joint_vel,
            torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
            mask[:, :, :, 1:],
        )

    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def foot_contact_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259
        contact_mask_gt = (
            contact > 0.5
        )  # contact mask order for indexes are fid_l [7, 10], fid_r [8, 11]
        vel_lf_7 = local_velocity[:, 7 * 3 : 8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3 : 9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3 : 11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3 : 12 * 3, :, :]

        calc_vel_lf_7 = (
            ric_data[:, 6 * 3 : 7 * 3, :, 1:] - ric_data[:, 6 * 3 : 7 * 3, :, :-1]
        )
        calc_vel_rf_8 = (
            ric_data[:, 7 * 3 : 8 * 3, :, 1:] - ric_data[:, 7 * 3 : 8 * 3, :, :-1]
        )
        calc_vel_lf_10 = (
            ric_data[:, 9 * 3 : 10 * 3, :, 1:] - ric_data[:, 9 * 3 : 10 * 3, :, :-1]
        )
        calc_vel_rf_11 = (
            ric_data[:, 10 * 3 : 11 * 3, :, 1:] - ric_data[:, 10 * 3 : 11 * 3, :, :-1]
        )

        # vel_foots = torch.stack([vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], dim=1)
        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
            [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11],
            [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11],
            [7, 10, 8, 11],
            [0, 1, 2, 3],
        ):
            tmp_mask_gt = (
                contact_mask_gt[:, contact_mask_idx, :, :]
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)
                .astype(int)
            )
            chosen_vel_norm = np.linalg.norm(
                chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0
            )
            chosen_vel_calc_norm = np.linalg.norm(
                chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)), axis=0
            )

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt

            plt.plot(tmp_mask_gt, label="FC mask")
            plt.plot(chosen_vel_norm, label="Vel. XYZ norm (from vector)")
            plt.plot(chosen_vel_calc_norm, label="Vel. XYZ norm (calculated diff XYZ)")

            plt.title(f"FC idx {contact_mask_idx}, Joint Index {joint_idx}")
            plt.legend()
            plt.show()
        # print(vel_foots.shape)
        return 0

    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def velocity_consistency_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]  # Slicing out root
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(
            target.permute(0, 2, 3, 1).type(th.FloatTensor)
        )
        print(f"r_rot_quat: {r_rot_quat.shape}")
        print(f"calc_vel_from_xyz: {calc_vel_from_xyz.shape}")
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(
            th.FloatTensor
        )
        r_rot_quat_adapted = (
            r_rot_quat[..., :-1, None, :]
            .repeat((1, 1, 1, 21, 1))
            .to(calc_vel_from_xyz.device)
        )
        print(
            f"calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}"
        )
        print(
            f"r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}"
        )

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(
            f"calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}"
        )

        import matplotlib.pyplot as plt

        for i in range(21):
            plt.plot(
                np.linalg.norm(
                    calc_vel_from_xyz[:, i * 3 : (i + 1) * 3, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape((3, -1)),
                    axis=0,
                ),
                label="Calc Vel",
            )
            plt.plot(
                np.linalg.norm(
                    velocity_from_vector[:, i * 3 : (i + 1) * 3, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape((3, -1)),
                    axis=0,
                ),
                label="Vector Vel",
            )
            plt.title(f"Joint idx: {i}")
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz - velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0
