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
import random
import json
from scene.dataset_readers import sceneLoadTypeCallbacks
# from scene.gaussian_model0 import GaussianModel
# from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from typing import List
from scene.cameras import Camera
from scene.gaussian_model import GaussianModelLite

class Args:
    def __init__(self):
        self.resolution = -1
        self.data_device = 'cuda'

class SceneLite:
    gaussians : GaussianModelLite

    def __init__(self, model_path: str, source_path: str, eval: bool, resolution_scales=None):
        if resolution_scales is None:
            resolution_scales = [1.0]
        self.model_path = model_path
        self.source_path = os.path.realpath(source_path)
        self.eval = eval
        self.train_cameras = {}
        self.test_cameras = {}
        self.gaussians = GaussianModelLite()
        scene_info = None
        self.pcd = None

        if os.path.exists(os.path.join(self.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path, 'images', eval)
            pass
        elif os.path.exists(os.path.join(self.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.source_path, False, eval)
        else:
            assert False, "Could not recognize scene type!"

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(model_path, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, Args())
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, Args())

        self.pcd = scene_info.point_cloud

    def save(self, iteration: int):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def get_train_cameras(self, scale=1.0) -> List[Camera]:
        return self.train_cameras[scale]

    def get_test_cameras(self, scale=1.0) -> List[Camera]:
        return self.test_cameras[scale]

    def write_cfg_file(self):
        # example:
        # Namespace(sh_degree=3, source_path='/home/lxl2024/projs/gaussian-splatting/data/USB100109/start',
        # model_path='./output/1db53ef5-2', images='images', resolution=-1, white_background=False,
        # data_device='cuda', eval=True)
        cfg_file_path = os.path.join(self.model_path, 'cfg_args')
        with open(cfg_file_path, 'w') as file:
            file.write(
                f"Namespace("
                f"sh_degree={self.gaussians.active_sh_degree}, source_path='{os.path.realpath(self.source_path)}', "
                f"model_path='{self.model_path}', images='images', resolution=-1, white_background=False, "
                f"data_device='cuda', eval={self.eval}"
                f")"
            )




# class Scene:
#
#     gaussians : GaussianModel
#
#     def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         self.model_path = args.model_path
#         self.loaded_iter = None
#         self.gaussians = gaussians
#
#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
#             else:
#                 self.loaded_iter = load_iteration
#             print("Loading trained model at iteration {}".format(self.loaded_iter))
#
#         self.train_cameras = {}
#         self.test_cameras = {}
#
#         if os.path.exists(os.path.join(args.source_path, "sparse")):
#             scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
#         elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
#             print("Found transforms_train.json file, assuming Blender data set!")
#             scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
#         else:
#             assert False, "Could not recognize scene type!"
#
#         if not self.loaded_iter:
#             with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
#                 dest_file.write(src_file.read())
#             json_cams = []
#             camlist = []
#             if scene_info.test_cameras:
#                 camlist.extend(scene_info.test_cameras)
#             if scene_info.train_cameras:
#                 camlist.extend(scene_info.train_cameras)
#             for id, cam in enumerate(camlist):
#                 json_cams.append(camera_to_JSON(id, cam))
#             with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
#                 json.dump(json_cams, file)
#
#         if shuffle:
#             random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
#
#         self.cameras_extent = scene_info.nerf_normalization["radius"]
#
#         for resolution_scale in resolution_scales:
#             print("Loading Training Cameras")
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
#             print("Loading Test Cameras")
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
#
#         if self.loaded_iter:
#             self.gaussians.load_ply(os.path.join(self.model_path,
#                                                            "point_cloud",
#                                                            "iteration_" + str(self.loaded_iter),
#                                                            "point_cloud.ply"))
#         else:
#             self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
#
#     def save(self, iteration):
#         point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
#         self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
#
#     def getTrainCameras(self, scale=1.0):
#         return self.train_cameras[scale]
#
#     def getTestCameras(self, scale=1.0):
#         return self.test_cameras[scale]
