import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run
        print("using ClassifierFreeSampleModel to wrap the model")

        assert (
            self.model.cond_mask_prob > 0
        ), "Cannot run a guided diffusion on a model that has not been trained with no conditions"

        # pointers to inner model
        # self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ["text", "action"]
        y_uncond = deepcopy(y)
        y_uncond["uncond"] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y["scale"].view(-1, 1, 1, 1) * (out - out_uncond))
