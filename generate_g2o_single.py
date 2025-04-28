# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.model_util import create_model_and_flow, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader

import shutil
from data_loaders.tensors import *
import hydra
from diffusion.nn import sum_flat
from diffusion import logger



@hydra.main(config_path="config", config_name="config_base", version_base=None)
def main(cfg):

    cfg.dataset = "gazehoi_stage0_1obj"
    cfg.dynamic = "flow"
    cfg.model_path = "/data/nas_24/motionfm/outputs/stage1/25-04-2025/10-05-06/model000007400.pt"
    # cfg.input_text = "assets/prompts_method_compare.txt"
    cfg.text_prompt = None
    cfg.guidance_param = 1.0
    cfg.num_samples = cfg.batch_size = 275
    cfg.seed = 130
   

    fixseed(cfg.seed)
    out_path = cfg.output_dir
    name = os.path.basename(os.path.dirname(cfg.model_path))
    niter = os.path.basename(cfg.model_path).replace("model", "").replace(".pt", "")
    max_frames = 30
    n_frames = max_frames
    print("dataset", cfg.dataset)
    print("cfg.motion_length", cfg.motion_length)
    print("max_frames", max_frames)
    print("n_frames", n_frames)

    dist_util.setup_dist()
    if out_path == "":
        out_path = os.path.join(
            os.path.dirname(cfg.model_path),
            "samples_{}_{}_seed{}".format(name, niter, cfg.seed),
        )

    assert (
        cfg.num_samples <= cfg.batch_size
    ), f"Please either increase batch_size({cfg.batch_size}) or reduce num_samples({cfg.num_samples})"
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    cfg.batch_size = (
        cfg.num_samples
    )  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Loading dataset...")
    data_loader = load_dataset(cfg, max_frames, n_frames)
    # total_num_samples = cfg.num_samples * cfg.num_repetitions

    if cfg.dynamic == "flow":
        model, dynamic = create_model_and_flow(cfg, data_loader)
    else:
        raise NotImplementedError
    print(f"Loading checkpoints from [{cfg.model_path}]...")
    state_dict = torch.load(cfg.model_path, map_location="cuda")
    load_model_wo_clip(model, state_dict)


    if cfg.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

  
    # iterator = iter(data_loader)
    # _, model_kwargs = next(iterator)
    all_motions = []
    all_lengths = []
    all_gt_motions = []
    all_seqs = []
    for batch_idx, (_, model_kwargs) in enumerate(data_loader):
        print(f"Sampling batch {batch_idx}...")
        
        for k, v in model_kwargs['y'].items():
            if torch.is_tensor(v):
                model_kwargs['y'][k] = v.to(dist_util.dev())


        # add CFG scale to batch
        if cfg.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(cfg.batch_size, device=dist_util.dev()) * cfg.guidance_param
            )

        sample_fn = dynamic.p_sample_loop

        sample = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (cfg.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            sample_steps=cfg.diffusion_steps_sample,
            ode_kwargs=cfg.ode_kwargs,
        )
        
        gt = model_kwargs['y']['gt'].permute(0,2,1).unsqueeze(2)
        mask = model_kwargs["y"]["mask"]
        loss = dynamic.masked_l2(
                    gt, sample, mask
                )
        print('val loss',loss.mean())
        logger.log('val loss',loss.mean())
        all_motions.append(sample.cpu().numpy())
        all_gt_motions.append(gt.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
        all_seqs.append(model_kwargs['y']['seq_name'])

        print(f"created {len(all_motions) * cfg.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_gt_motions = np.concatenate(all_gt_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)
    all_seqs =  [element for sublist in all_seqs for element in sublist]
    
    # all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    # all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    dict_2_save = {
        "motion": all_motions,
        "gt_motion": all_gt_motions,
        "lengths": all_lengths,
        "num_samples": cfg.num_samples,
    }
    np.save(
        npy_path,
        dict_2_save,
    )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")
    return loss.mean()





def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split="test"
    )
    return data


if __name__ == "__main__":
    main()
