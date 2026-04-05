#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
from collections import defaultdict
from utils.loss_utils import  ssim
from utils.image_utils import psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import json
from lpipsPyTorch import lpips
from utils.general_utils import PILtoTorch
from torchvision.transforms import ToPILImage

from typing import Optional, Dict, List, Tuple

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS
        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []
        # self.flowmaps=[]
        
    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, model_path , stage="validation"):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        metrics = defaultdict(list)
            
        lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                    ).to("cuda")

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            if stage == "validation":
                render_pkg = self.render(viewpoint_cam[1].cuda(), self.gaussians)
                gt_image = viewpoint_cam[0].cuda()
            else:
                render_pkg = self.render(viewpoint_cam.cuda(), self.gaussians)

            rgb = render_pkg['render']
            rgb = rgb.clamp(0.0, 1.0) 
            # flow = render_pkg['flow']
            depth = render_pkg['depth']

            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.flowmaps.append(flow.cpu())

            if stage == "validation":
                metrics["psnr"].append(psnr(gt_image, rgb))
                metrics["ssim"].append(ssim(gt_image, rgb))
                metrics["lpips"].append(lpips(gt_image.unsqueeze(0), rgb.unsqueeze(0)))
   
        if stage == "validation":
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {"num_GS": self.gaussians.get_xyz.shape[0],}
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f} "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            stats_dir = os.path.join(model_path, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            with open(f"{stats_dir}/{stage}.json", "w") as f:
                json.dump(stats, f)


    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks
        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            # flow=self.flowmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh


    @torch.no_grad()
    def export_image(self, path, mode="validation"):
        render_path = os.path.join(path, "renders")
        os.makedirs(render_path, exist_ok=True)
        vis_path = os.path.join(path, "vis")
        os.makedirs(vis_path, exist_ok=True)
        
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))

    @torch.no_grad()
    def reconstruction_and_export(self, viewpoint_stack, path, model_path=None, stage="validation", skip_rendering=False):
        """
        Reconstruct radiance field and export images simultaneously to save memory
        
        Args:
            skip_rendering: If True, skip rendering and only compute metrics from existing images
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        metrics = defaultdict(list)
        
        # Setup render path
        render_path = os.path.join(path, "renders")
        os.makedirs(render_path, exist_ok=True)
        
        # Setup LPIPS for validation
        if stage == "validation":
            from skimage.metrics import structural_similarity as sk_ssim
            from lpipsPyTorch import lpips as lpips_fn

        # Helper function to load and validate image
        def load_image_safe(image_path):
            """Load image with error handling"""
            try:
                if not os.path.exists(image_path):
                    return None
                
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                
                # Check if image is valid
                if img is None or img.size[0] == 0 or img.size[1] == 0:
                    print(f"Warning: Invalid image size: {image_path}")
                    return None
                
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Check for invalid data
                if img_array.size == 0 or np.isnan(img_array).any() or np.isinf(img_array).any():
                    print(f"Warning: Invalid image data (NaN/Inf): {image_path}")
                    return None
                
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # H x W x C -> C x H x W
                return img_tensor
                
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                return None
        
        # Statistics
        skipped_count = 0
        rendered_count = 0
        loaded_count = 0
        
        # Process each view
        desc = "Loading and computing metrics" if skip_rendering else "Rendering and exporting"
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc=desc):
            img_path = os.path.join(render_path, '{0:05d}'.format(idx) + ".png")
            
            # Check if image already exists
            image_exists = os.path.exists(img_path)
            
            # Try to load existing image first
            rgb = None
            if image_exists:
                rgb = load_image_safe(img_path)
                if rgb is not None:
                    loaded_count += 1
                    rgb = rgb.cuda()
            
            # If image doesn't exist or loading failed, render it
            if rgb is None and not skip_rendering:
                try:
                    # Render
                    if stage == "validation":
                        render_pkg = self.render(viewpoint_cam[1].cuda(), self.gaussians)
                        gt_image = viewpoint_cam[0].cuda()
                    else:
                        render_pkg = self.render(viewpoint_cam.cuda(), self.gaussians)
                        gt_image = None

                    rgb = render_pkg['render']
                    
                    # Check if rendering is valid
                    if torch.isnan(rgb).any() or torch.isinf(rgb).any():
                        print(f"Warning: Rendered image {idx} contains NaN/Inf, skipping")
                        skipped_count += 1
                        del rgb, render_pkg
                        if gt_image is not None:
                            del gt_image
                        torch.cuda.empty_cache()
                        continue
                    
                    # Save image immediately
                    save_img_u8(
                        rgb.permute(1, 2, 0).cpu().numpy(), 
                        img_path
                    )
                    rendered_count += 1
                    
                    # Clean up render pkg
                    del render_pkg
                    
                except Exception as e:
                    print(f"Error rendering image {idx}: {str(e)}, skipping")
                    skipped_count += 1
                    if 'render_pkg' in locals():
                        del render_pkg
                    if 'gt_image' in locals() and gt_image is not None:
                        del gt_image
                    torch.cuda.empty_cache()
                    continue
            
            elif rgb is None and skip_rendering:
                # Skip rendering mode but image doesn't exist or invalid
                print(f"Warning: Image {idx} not found or invalid, skipping")
                skipped_count += 1
                continue
            
            # Calculate metrics if validation and we have valid rgb
            if stage == "validation" and rgb is not None:
                try:
                    # Load GT image if needed
                    if stage == "validation":
                        gt_image = viewpoint_cam[0].cuda()
                        gt_image = gt_image.contiguous()
                        rgb = rgb.contiguous()
                        gt_image = gt_image.clamp(0.0, 1.0)
                        rgb = rgb.clamp(0.0, 1.0)
                        
                        # Check if GT and rendered image have same shape
                        if gt_image.shape != rgb.shape:
                            print(f"Warning: Shape mismatch at {idx}: GT {gt_image.shape} vs Render {rgb.shape}, skipping")
                            skipped_count += 1
                            del rgb, gt_image
                            torch.cuda.empty_cache()
                            continue
                        
                        # Compute metrics
                        metrics["psnr"].append(psnr(gt_image.unsqueeze(0), rgb.unsqueeze(0)).cpu())
                        # metrics["ssim"].append(ssim(gt_image.unsqueeze(0), rgb.unsqueeze(0)).cpu())
                        metrics["lpips"].append(lpips_fn(gt_image, rgb, net_type='alex').cpu())
                        
                        # Convert to numpy for sk_ssim
                        rgb_np = rgb.detach().cpu().numpy()
                        gt_np = gt_image.detach().cpu().numpy()
                        
                        metrics['Dssim1'].append(torch.tensor((1.0 - sk_ssim(rgb_np, gt_np, data_range=1.0, multichannel=True, channel_axis=0)) / 2.0))
                        metrics['Dssim2'].append(torch.tensor((1.0 - sk_ssim(rgb_np, gt_np, data_range=2.0, multichannel=True, channel_axis=0)) / 2.0))
                        
                except Exception as e:
                    print(f"Error computing metrics for image {idx}: {str(e)}, skipping")
                    skipped_count += 1
            
            # Clear GPU memory
            if rgb is not None:
                del rgb
            if stage == "validation" and 'gt_image' in locals() and gt_image is not None:
                del gt_image
            torch.cuda.empty_cache()
        
        # Print statistics
        total_images = len(self.viewpoint_stack)
        print(f"\nProcessing Summary:")
        print(f"  Total images: {total_images}")
        print(f"  Loaded from disk: {loaded_count}")
        print(f"  Newly rendered: {rendered_count}")
        print(f"  Skipped (errors): {skipped_count}")
        print(f"  Valid for metrics: {len(metrics.get('psnr', []))}")
        
        # Save validation stats
        if stage == "validation" and model_path and len(metrics.get('psnr', [])) > 0:
            stats = {k: torch.stack(v).mean().item() if len(v) > 0 else 0.0 for k, v in metrics.items()}
            stats.update({
                "num_GS": self.gaussians.get_xyz.shape[0],
                "total_images": total_images,
                "valid_images": len(metrics.get('psnr', [])),
                "loaded_images": loaded_count,
                "rendered_images": rendered_count,
                "skipped_images": skipped_count
            })
            
            print(
                f"\nValidation Results (based on {stats['valid_images']}/{total_images} images):"
            )
            print(f"  PSNR: {stats['psnr']:.3f}")
            # print(f"  SSIM: {stats['ssim']:.4f}")
            if 'lpips' in stats:
                print(f"  LPIPS: {stats['lpips']:.4f}")
            print(f"  Number of GS: {stats['num_GS']}")
            
            # Save stats as json
            stats_dir = os.path.join(model_path, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            with open(f"{stats_dir}/{stage}.json", "w") as f:
                json.dump(stats, f, indent=2)
        elif stage == "validation" and len(metrics.get('psnr', [])) == 0:
            print("\nWarning: No valid images for validation metrics!")
        
        