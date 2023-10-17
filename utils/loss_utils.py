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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True, stride=1):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel, stride=stride) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def s3im(src_vec, tar_vec, repeat_time=10,patch_height=64, patch_width=64, stride=4, window_size = 4, size_average = True):
    index_list = []
    print(len(tar_vec))
    print(tar_vec.size())
    for i in range(repeat_time):
        if i == 0:
            tmp_index = torch.arange(len(tar_vec))
            index_list.append(tmp_index)
        else:
            ran_idx = torch.randperm(len(tar_vec))
            index_list.append(ran_idx)
    res_index = torch.cat(index_list)
    tar_all = tar_vec[res_index]
    src_all = src_vec[res_index]
    print(tar_all.size())
    tar_patch = tar_all.reshape(1, 3, patch_height, patch_width * repeat_time)
    src_patch = src_all.reshape(1, 3, patch_height, patch_width * repeat_time)
    
    
    
    channel = src_patch.size(-3)
    window = create_window(window_size, channel)

    if src_patch.is_cuda:
        window = window.cuda(src_patch.get_device())
    window = window.type_as(src_patch)


    return _ssim(src_patch, tar_patch, window, window_size, channel, size_average, stride)