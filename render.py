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

import os

from flask import testing
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import random
import torch
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def validation(dataset, name, pipe, checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d,
               num_pts, num_pts_ratio, traj='ellipse', scale_factor=1.0, training_view=None, validate=False, testing_view=None,
               total_frames=300, fix_time=False, selected_frame=-1):

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, 
                              rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)

    assert checkpoint, "No checkpoint provided for validation"
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration, 
                  training_view=training_view, testing_view=testing_view)
    print("Loaded scene with initial gaussians")

    (model_params, first_iter) = torch.load(checkpoint)
    if validate:
        test_dir = os.path.join(dataset.model_path, 'test', "ours_{}".format(first_iter))
        os.makedirs(test_dir, exist_ok=True)
    if traj:
        traj_dir = os.path.join(dataset.model_path, 'traj', "ours_{}".format(first_iter))
        os.makedirs(traj_dir, exist_ok=True)
    gaussians.restore(model_params, None)
    print("Restored gaussians from checkpoint: {}".format(checkpoint))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)   

    #########    2. Render Trajectory       ############
    if traj:
        print("Rendering trajectory ...")
        n_frames = 480
        cam_traj = generate_path(scene.getAllCameras() , 
                                 n_frames=n_frames, traj=traj, scale_factor=scale_factor, 
                                 total_frames=total_frames, fix_time=fix_time, selected_frame=selected_frame)
        gaussExtractor.reconstruction(cam_traj, traj_dir, stage="trajectory")
        gaussExtractor.export_image(traj_dir, mode="trajectory")
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name=f'{name}_{traj}_scale{scale_factor}_{time.strftime("%m%d%H%M")}', 
                    num_frames=n_frames,)
        
    # #########   1. Validation and Rendering ############
    if validate:
        print("calculate rendered testing images ...")
        gaussExtractor.reconstruction_and_export(scene.getTestCameras(), test_dir, test_dir, stage="validation")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="")
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument('--total_frames', type=int, default=300)
    
    parser.add_argument("--training_view", type=str, default="",
                        help="Comma-separated list of cameras to use for validation, \
                            e.g. '0,1,2'. If not specified, all cameras will be used.")
    
    parser.add_argument("--testing_view", type=str, default="",
                        help="Comma-separated list of cameras to use for validation, \
                            e.g. '0,1,2'. If not specified, all cameras will be used.")
    
    parser.add_argument("--validate", default=False, action="store_true")
    parser.add_argument('--fix_time', default=False, action="store_true")
    
    parser.add_argument('--selected_frame', type=int, default=-1)
    parser.add_argument('--res', type=int, default=1)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.training_view:
        args.training_view = [f"cam{str(int(cam)).zfill(2)}" for cam in args.training_view.split(',')]
    if args.testing_view:
        args.testing_view = [f"cam{str(int(cam)).zfill(2)}" for cam in args.testing_view.split(',')]

    name = args.config.split("/")[-1].split(".")[0]
        
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.res is not None:
        args.resolution = args.res
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,op.iterations,500)]
        
    if args.output_dir:
        args.model_path = os.path.join(args.model_path, args.output_dir, 'render')
        os.makedirs(args.model_path, exist_ok=True)
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    assert args.traj or args.validate, "No validation or trajectory rendering requested"

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    validation(lp.extract(args), name, pp.extract(args), args.start_checkpoint,args.gaussian_dim, 
                   args.time_duration, args.rot_4d, args.force_sh_3d, args.num_pts, args.num_pts_ratio, 
                   traj=args.traj, training_view=args.training_view, scale_factor=args.scale, validate=args.validate, testing_view=args.testing_view,
                   total_frames=args.total_frames, fix_time=args.fix_time, selected_frame=args.selected_frame)
    # All done
    print("\nRendering complete.")
