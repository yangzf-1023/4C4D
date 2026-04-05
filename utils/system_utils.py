#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torchvision
import torch
from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def check_optimizer_gradients(optimizer, iter, prefix="", args=None):
    
    for group_idx, param_group in enumerate(optimizer.param_groups):
        if (param_group['name'] == 'coefficient' and iter > args.densify_from_iter):# or param_group['name'] == 'opacity':
            for _, param in enumerate(param_group['params']):
                param_shape = list(param.shape)
                requires_grad = param.requires_grad
                # assert requires_grad and param.grad is not None and torch.norm(param.grad) > 0, f"Gradient issue in param group '{param_group['name']}'"
                grad_status = "No gradient (None)" if param.grad is None else f"Grad norm: {torch.norm(param.grad):.15f}"
                if requires_grad == False or param.grad is None or torch.norm(param.grad) == 0: 
                    print(f"{prefix} Checking optimizer gradients: Param {param_group['name']}: shape={param_shape}, requires_grad={requires_grad}, {grad_status}")
                    if requires_grad == False or param.grad is None:
                        raise ValueError(f"Gradient issue in param group '{param_group['name']}'")

def _save_tensor(img_cpu, out_path):
    # img_cpu: torch.Tensor on CPU, float [0,1], CxHxW
    torchvision.utils.save_image(img_cpu, out_path)
    
def _resize_chw(img, size_hw):
                    H, W = size_hw
                    return torchvision.transforms.functional.resize(
                        img, [H, W], 
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                        antialias=True
                    )