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

import cv2
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        # torch.float32, [3, H, W]
        render_color = rendering["render"]
        render_depth = rendering["render_depth"]
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(render_color, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        # transform render_depth to visiable depth
        # render_depth = render_depth / render_depth.max()
        render_depth = render_depth / 10
        # 5 preset max vis depth
        # render_depth = render_depth / 5
        # render_depth = render_depth * 255
        # render_depth = cv2.applyColorMap(render_depth.squeeze().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
        # # save
        # cv2.imwrite(os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"), render_depth)
        
        torchvision.utils.save_image(render_depth, os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))
        
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
    # png2viewo_cmd = "ffmpeg -y -framerate 5 -i {}/%05d.png -c:v libx264 {}/video.mp4".format(render_path, render_path)
    # os.system(png2viewo_cmd)
    # png2viewo_cmd = "ffmpeg -y -framerate 5 -i {}/%05d.png -c:v libx264 {}/video_depth.mp4".format(render_depth_path, render_depth_path)
    # os.system(png2viewo_cmd)
    # png2viewo_cmd = "ffmpeg -y -framerate 5 -i {}/%05d.png -c:v libx264 {}/video_gt.mp4".format(gts_path, gts_path)
    # os.system(png2viewo_cmd)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet,args.device)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)