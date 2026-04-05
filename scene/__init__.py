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
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, 
                 resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, 
                 training_view=['cam10', 'cam01', 'cam20', 'cam13'], redundant_ratio=0.2,
                 downsample_method='random', testing_view=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        else:
            print("Creating new model")
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio, 
                                                          training_cam=training_view, testing_cam=testing_view, num_pts=num_pts, time_duration=time_duration,
                                                          downsample_method=downsample_method)
            print(f"Found sparse folder in {args.source_path}, assuming Colmap data set!")
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print(f"Found transforms_train.json file in {args.source_path}, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                print(f"Copying input.ply from source path: {args.source_path} to model path: {self.model_path}.")
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                print(f"Writing cameras.json to model path: {self.model_path}.")
                json.dump(json_cams, file)

        if shuffle:
            print("Shuffling training and testing cameras")
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            print("Shuffling done.")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print(f"Loading cameras at resolution scale {resolution_scale}")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loaded Training Cameras with {len(self.train_cameras[resolution_scale])} frames with resolution scale {resolution_scale}")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Loaded Testing Cameras with {len(self.test_cameras[resolution_scale])} frames with resolution scale {resolution_scale}")
            
        if args.loaded_pth:
            print(f"Loading gaussians from {args.loaded_pth}")
            self.gaussians.create_from_pth(args.loaded_pth, self.cameras_extent)
        else:
            if self.loaded_iter:
                print(f"Loading gaussians from trained model's point cloud at iteration {self.loaded_iter}")
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                print(f"Creating gaussians from initial point cloud input.ply")
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,
                                               redundant_ratio=redundant_ratio)
                print(f"Created gaussians from input.ply")

    def save(self, iteration):
        torch.save((self.gaussians.capture(), iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)
    
    def getValidationCameras(self, scale=1.0, tag='train', num=100):
        if tag == 'train':
            return CameraDataset(self.train_cameras[scale][::num], self.white_background)
        elif tag == 'test':
            return CameraDataset(self.test_cameras[scale][::num], self.white_background)
        
    def getAllCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy() + self.test_cameras[scale].copy(), self.white_background)
    