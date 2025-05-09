from torch.utils.data import DataLoader
from data_loaders.tensors import *
from .gazehoi.data.dataset import *


def get_dataset_class(name):
    # if name == "amass":
    #     from .amass import AMASS

    #     return AMASS
    # elif name == "uestc":
    #     from .a2m.uestc import UESTC

    #     return UESTC
    # elif name == "humanact12":
    #     from .a2m.humanact12poses import HumanAct12Poses

    #     return HumanAct12Poses
    # elif name == "humanml":
    #     from data_loaders.humanml.data.dataset import HumanML3D

    #     return HumanML3D
    # elif name == "kit":
    #     from data_loaders.humanml.data.dataset import KIT

    #     return KIT
    if name == 'gazehoi_stage0_1obj':
        return GazeHOIDataset_stage0_1obj
    elif name == 'gazehoi_o2h_mid':
        return GazeHOIDataset_o2h_mid
    elif name == 'gazehoi_o2h_mid_2hand_assemobj':
        return GazeHOIDataset_o2h_mid_2hand_assemobj
    else:
        raise ValueError(f"Unsupported dataset name [{name}]")


def get_collate_fn(name, hml_mode='train'):

    if name == 'gazehoi_stage0_1obj' or name == 'gazehoi_stage0_norm' or name == 'gazehoi_stage0_point' or name == 'gazehoi_stage0_noatt':
        return g2m_stage0_1obj_collate
    if name == 'gazehoi_o2h_mid' or name == 'gazehoi_o2h_mid_2hand_assemobj':
        return o2h_mid_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split="train", hml_mode="train", is_debug=False):
    DATA = get_dataset_class(name)
    # if name in ["humanml", "kit"]:
    #     dataset = DATA(
    #         split=split, num_frames=num_frames, mode=hml_mode, is_debug=is_debug
    #     )
    # else:
    #     dataset = DATA(split=split, num_frames=num_frames, is_debug=is_debug)
    dataset = DATA(split=split)
    return dataset


def get_dataset_loader(
    name,
    batch_size,
    num_frames,
    num_workers=8,
    split="train",
    hml_mode="train",
    is_debug=False,
):
    print("creating data loader...")
    dataset = get_dataset(name, num_frames, split, hml_mode, is_debug=is_debug)
    collate = get_collate_fn(name, hml_mode)
    
    if split == 'test':
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate,
        )

    return loader
