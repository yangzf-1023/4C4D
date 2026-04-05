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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss #, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from module import Coefficient

import imageio
import math
from fused_ssim import fused_ssim as fast_ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from datetime import datetime


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    

    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    if args.opacity_decay:
        coefficient = Coefficient().cuda()
    else:
        coefficient = None
        
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, 
                              rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0, coefficient=coefficient)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, 
                  time_duration=time_duration, training_view=args.training_view, testing_view=args.testing_view,
                  redundant_ratio=args.redundant_ratio, downsample_method=args.downsample_method)
    gaussians.training_setup(opt)  
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print("Restored gaussians from checkpoint: {}".format(checkpoint))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training", ncols=110)
    first_iter += 1
        
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),
                                           dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        
    gaussians.env_map = env_map
    training_dataset = scene.getTrainCameras()
    if dataset.dataloader:
        print("\nUsing DataLoader for training dataset")
    else:
        print("\nNot using DataLoader for training dataset")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
    
    img_dir = os.path.join(scene.model_path, "rendered_images")
    os.makedirs(img_dir, exist_ok=True)
    
    iteration = first_iter
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            if iteration % 100 == 0:
                iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []
            
            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]
                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()
                
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, args=args, iteration=iteration)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # depth, alpha = render_pkg["depth"], render_pkg["alpha"]

                # Loss
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                    
                loss = loss / batch_size
                loss.backward()
                
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)
                
            if (iteration % 1500 == 0 or iteration == 2):  # Save every 100 iterations
                # Convert rendered image tensor to numpy and save
                image = torch.clamp(image, 0.0, 1.0)
                img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Convert ground truth image tensor to numpy and save
                gt_img_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
                gt_img_np = (gt_img_np * 255).astype(np.uint8)
                
                # Create filenames with iteration number and camera ID
                img_filename = f"iter_{iteration}_cam_{viewpoint_cam.image_name}.png"
                gt_img_filename = f"iter_{iteration}_cam_{viewpoint_cam.image_name}_gt.png"
                
                imageio.imwrite(os.path.join(img_dir, img_filename), img_np)
                imageio.imwrite(os.path.join(img_dir, gt_img_filename), gt_img_np)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()
            
            if iteration % 100 == 0:
                iter_end.record()
            loss_dict = {"Ll1": Ll1, "Lssim": Lssim}

            with torch.no_grad():
                if iteration % 10 == 0:
                    psnr_for_log = psnr(image, gt_image).mean().double()
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                    ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                            loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
                            
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                "PSNR": f"{psnr_for_log:.{2}f}",
                                "gs_num":f"{gaussians.get_xyz.shape[0]}"}
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                    
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log 
                if iteration % 100 == 0 or iteration in testing_iterations:
                    elapsed = 0.0
                    if iteration % 100 == 0:
                        torch.cuda.synchronize()
                        elapsed = iter_start.elapsed_time(iter_end)
                    
                    test_psnr = training_report(
                        tb_writer, iteration, Ll1, Lssim, loss, 
                        l1_loss, elapsed, 
                        testing_iterations, scene, render, 
                        (pipe, background), loss_dict, img_dir=img_dir)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, 
                                    visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, 
                                    visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                        
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval and args.add_size_threshold else None
                        if args.opacity_decay:
                            size_threshold = None
                        prune_only = opt.densify_until_num_points > 0 and gaussians.get_xyz.shape[0] >= opt.densify_until_num_points
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, 
                                                    size_threshold, opt.densify_grad_t_threshold, prune_only=prune_only)
                    
                    if ((iteration % opt.opacity_reset_interval == 0 and not args.opacity_decay) or (
                        dataset.white_background and iteration == opt.densify_from_iter)) and args.reset_opacity:
                        gaussians.reset_opacity()
                        
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    if gaussians.coefficient is not None:
                        gaussians.coef_optimizer.step()
                        gaussians.coef_optimizer.zero_grad(set_to_none = True)
                        
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)
                        
                # Save chkpnt
                if (iteration in testing_iterations):
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")
                        
                # Save Gaussians
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
        
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

def training_report(tb_writer, iteration, Ll1, Lssim, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None, img_dir=""):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Lssim.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = (
            {'name': 'train', 'cameras': scene.getValidationCameras(tag='train')},
            {'name': 'test', 'cameras': scene.getValidationCameras(tag='test')},
        )
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'], ncols=80)):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                         
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    
                    if config['name'] == 'test' and idx % 5 == 0:
                        # Convert rendered image tensor to numpy and save
                        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        # Convert ground truth image tensor to numpy and save
                        gt_img_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
                        gt_img_np = (gt_img_np * 255).astype(np.uint8)
                        
                        # Create filenames with iteration number and camera ID
                        img_filename = f"test_iter_{iteration}_cam_{viewpoint.image_name}.png"
                        gt_img_filename = f"test_iter_{iteration}_cam_{viewpoint.image_name}_gt.png"
                        
                        imageio.imwrite(os.path.join(img_dir, img_filename), img_np)
                        imageio.imwrite(os.path.join(img_dir, gt_img_filename), gt_img_np)
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0, 10.0])
    parser.add_argument('--initial_num_pts', type=int, default=-1)
    parser.add_argument('--num_pts', type=int, default=100000)
    parser.add_argument('--max_num_pts', type=int, default=None)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exhaust_test", action="store_true")
    
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--training_view", type=str, default="1,10,13,20",
                        help="Comma-separated list of cameras to use for validation, \
                            e.g. '0,1,2'. If not specified, all cameras will be used.")
    
    parser.add_argument("--testing_view", type=str, default="",
                        help="Comma-separated list of cameras to use for validation, \
                            e.g. '0,1,2'. If not specified, all cameras will be used.")
    
    # opacity decay
    parser.add_argument("--opacity_decay", action="store_true", default=True)
    parser.add_argument('--f_max', default=0.998, type=float, help='max factor')
    parser.add_argument("--f_min", type=float, default=0.996, help='min factor')
    
    parser.add_argument("--dropout_rate", type=float, default=0.1, help='dropout_rate')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay')
    parser.add_argument("--hidden_dim", type=int, default=32, help='dim of mlp')
    parser.add_argument("--decay_from_iter", type=int, default=500, help='decay from iter')
    
    parser.add_argument('--redundant_ratio', default=0.0, type=float)
    parser.add_argument('--res', default=1, type=int)
    parser.add_argument('--downsample_method', default='random', type=str, choices=['fps', 'random'])
    
    parser.add_argument('--test_per_iter', default=1500, type=int)
    
    parser.add_argument('--time_aware', action="store_true", default=True)
    parser.add_argument("--reset_opacity", action="store_true", default=False)
    parser.add_argument("--add_size_threshold", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
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
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0, op.iterations, args.test_per_iter)]
        
    if args.initial_num_pts is not None:
        args.num_pts = args.initial_num_pts
                
    if args.max_num_pts is not None:
        args.densify_until_num_points = args.max_num_pts
        
    if args.res is not None:
        args.resolution = args.res
        
    if args.output_dir:
        args.model_path = os.path.join(args.model_path, args.output_dir)
        
    if args.weight_decay:
        args.coefficient_weight_decay = args.weight_decay
    
    if os.path.exists(args.model_path):
        raise AssertionError(f"Output folder {args.model_path} already exists")
    os.makedirs(args.model_path, exist_ok=True)
        
    if args.training_view: 
        args.training_view = [f"cam{str(int(cam)).zfill(2)}" for cam in sorted(args.training_view.split(','))]
        
    if args.testing_view: 
        args.testing_view = [f"cam{str(int(cam)).zfill(2)}" for cam in sorted(args.testing_view.split(','))]
        
    if args.opacity_decay:
        args.densify_until_iter = args.iterations
    
    params_file = os.path.join(args.model_path, "training_params.txt")
    
    with open(params_file, "w") as f:
        f.write(str(args))
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, 
             args.rot_4d, args.force_sh_3d, args.batch_size)

    # All done
    print("\nTraining complete.")
