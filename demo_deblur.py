from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os

from networks.deblur_net import dinp_cen
import cv2
import torch
import torch.optim
from torch.autograd import Variable
import glob

import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM

import math
import re
import random

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)' ,text)]
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--kernel_size', type=int, default=[79, 79], help='size of blur kernel [height, width]')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--img_size1', type=int, default=[630, 518], help='size of each image dimension')
parser.add_argument('--img_size2', type=int, default=[630, 518], help='size of each image dimension')
parser.add_argument('--data_path', type=str, default="./datasets/lai", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="./results/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()

torch.cuda.set_device(1)
torch.set_num_threads(3)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort(key=natural_keys)

save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)
import numpy as np



def downsample_image(image, mode='topleft'):

    C, H, W = image.shape

    if mode == 'topleft':
        downsampled_image = image[:, :H // 2 * 2:2, :W // 2 * 2:2]
    elif mode == 'topright':
        downsampled_image = image[:, :H // 2 * 2:2, 1:W // 2 * 2:2]
    elif mode == 'bottomleft':
        downsampled_image = image[:, 1:H // 2 * 2:2, :W // 2 * 2:2]
    elif mode == 'bottomright':
        downsampled_image = image[:, 1:H // 2 * 2:2, 1:W // 2 * 2:2]
    else:
        raise ValueError("Unsupported mode. Choose from 'topleft', 'topright', 'bottomleft', 'bottomright'.")

    return downsampled_image


def downsample_image_twice(image, mode):

    images = [image]

    for _ in range(2):
        downsampled_image = downsample_image(images[-1], mode=mode)
        images.append(downsampled_image)

    return images


def downsample_tensor(tensor, mode='topleft'):

    _, _, H, W = tensor.shape

    if mode == 'topleft':
        downsampled_tensor = tensor[:, :, :H // 2 * 2:2, :W // 2 * 2:2]
    elif mode == 'topright':
        downsampled_tensor = tensor[:, :, :H // 2 * 2:2, 1:(W // 2 * 2) + 1:2]
    elif mode == 'bottomleft':
        downsampled_tensor = tensor[:, :, 1:(H // 2 * 2) + 1:2, :W // 2 * 2:2]
    elif mode == 'bottomright':
        downsampled_tensor = tensor[:, :, 1:(H // 2 * 2) + 1:2, 1:(W // 2 * 2) + 1:2]
    else:
        raise ValueError("Unsupported mode. Choose from 'topleft', 'topright', 'bottomleft', 'bottomright'.")

    return downsampled_tensor

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.type(dtype)) - torch.fft.fft2(y.type(dtype))
        loss = torch.mean(abs(diff))
        return loss

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.005
    num_iter = opt.num_iter

    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel_01') != -1:
        opt.kernel_size = [31, 31]
    if imgname.find('kernel_02') != -1:
        opt.kernel_size = [51, 51]
    if imgname.find('kernel_03') != -1:
        opt.kernel_size = [55, 55]
    if imgname.find('kernel_04') != -1:
        opt.kernel_size = [75, 75]
    _, imgs = get_image(path_to_image, -1)  # load image and convert to np.
    y_color = np_to_torch(imgs).type(dtype)

    img_gray = readimg_gray(path_to_image)

    img_gray = np.float32(img_gray / 255.0)
    y = np.expand_dims(img_gray, 0)

    img_size = y.shape
    n_scales = 3
    imgs_trans = imgs.transpose(1, 2, 0)

    pyramid = downsample_image_twice(imgs,'topleft')

    pyramid_size0 = pyramid[0].shape
    pyramid_size1 = pyramid[1].shape
    pyramid_size2 = pyramid[2].shape
    pyramid0 = np_to_torch(pyramid[0]).type(dtype)
    pyramid1 = np_to_torch(pyramid[1]).type(dtype)
    pyramid2 = np_to_torch(pyramid[2]).type(dtype)

    ker_size1 = math.ceil(opt.kernel_size[0] / 2)
    ker_size2 = math.ceil(ker_size1 / 2)

    padh1, padw1 = ker_size1 - 1, ker_size1 - 1
    padh2, padw2 = ker_size2 - 1, ker_size2 - 1

    opt.img_size1[0], opt.img_size1[1] = pyramid_size1[1] + padh1, pyramid_size1[2] + padw1
    opt.img_size2[0], opt.img_size2[1] = pyramid_size2[1] + padh2, pyramid_size2[2] + padw2

    padh, padw = opt.kernel_size[0] - 1, opt.kernel_size[1] - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padh, img_size[2] + padw

    img_size = imgs.shape

    print(imgname)
    # ######################################################################

    padw, padh = opt.kernel_size[0] - 1, opt.kernel_size[1] - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padw, img_size[2] + padh

    freq_dict = {
        'method': 'log',
        'cosine_only': False,
        'n_freqs': 16,
        'base': 2 ** (8 / (8 - 1)),
    }
    input_depth = freq_dict['n_freqs'] * 4
    net_input2 = get_input(input_depth, 'fourier', (opt.img_size2[0], opt.img_size2[1]), freq_dict=freq_dict).type(dtype)

    net_input1 = get_input(input_depth, 'fourier', (opt.img_size1[0], opt.img_size1[1]), freq_dict=freq_dict).type(
        dtype)


    net_input0 = get_input(input_depth, 'fourier', (opt.img_size[0], opt.img_size[1]), freq_dict=freq_dict).type(dtype)

    n_k = 2
    net_input_kernel2 = get_noise(n_k, 'grid', (ker_size2, ker_size2)).type(dtype).detach()
    net_input_kernel1 = get_noise(n_k, 'grid', (ker_size1, ker_size1)).type(dtype).detach()
    net_input_kernel0 = get_noise(n_k, 'grid', (opt.kernel_size[0], opt.kernel_size[1])).type(dtype).detach()

    net = dinp_cen(input_depth= input_depth, channel=3).type(dtype)
    lossL1 = nn.L1Loss().type(dtype)
    lossL2 = nn.MSELoss().type(dtype)

    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)
    tvloss = TVLoss().type(dtype)
    fft_loss =fftLoss().type(dtype)

    params_dict = [ {'params': net.skip.parameters(), 'lr': LR},
                   {'params': net.fcn1.parameters(), 'lr': 5e-5}
           ]

    optimizer = torch.optim.Adam(params_dict)
    scheduler = MultiStepLR(optimizer, milestones=[2000,4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input2.detach().clone()
    net_input_kernel_saved = net_input_kernel2.detach().clone()

    for step in tqdm(range(num_iter)):

        scheduler.step(step)
        optimizer.zero_grad()
        if step >=0:
            for m in net.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.p = 0.000
        if step >= 0 and step < 500:
            skip_out2, kernel2 = net(net_input2, net_input_kernel2, ker_size2)

            out_k_m2 = kernel2.repeat(3, 1, 1, 1)

            # out_y1_g3 = nn.functional.conv2d(skip_out1, out_k_m1_g3, padding=0, groups=3, bias=None)
            out_y2 = nn.functional.conv2d(skip_out2, out_k_m2, padding=0, groups=3, bias=None)

            total_loss =1- ssim(pyramid2, out_y2)
        elif step >= 500 and step < 1000:
            skip_out2, kernel2 = net(net_input2, net_input_kernel2,ker_size2)
            skip_out1, kernel1 = net(net_input1, net_input_kernel1, ker_size1)
            #out_k_m1 = kernel1.view(-1, 1, ker_size1, ker_size1)
            out_k_m1_g1 = kernel1.clone()
            out_k_m1_g1[:, :, 1::2, :] *= -1
            out_k_m1_g2 = kernel1.clone()
            out_k_m1_g2[:, :, :, 1::2] *= -1
            out_k_m1_g3 = kernel1.clone()
            out_k_m1_g3[:, :, :, 1::2] *= -1
            out_k_m1 = kernel1.repeat(3, 1, 1, 1)
            out_k_m1_g1 = out_k_m1_g1.repeat(3, 1, 1, 1)
            out_k_m1_g2 = out_k_m1_g2.repeat(3, 1, 1, 1)
            out_k_m1_g3 = out_k_m1_g3.repeat(3, 1, 1, 1)
            out_y1 = nn.functional.conv2d(skip_out1, out_k_m1, padding=0, groups=3, bias=None)
            out_k_m2 = kernel2.view(-1, 1, ker_size2, ker_size2)
            out_k_m2 = out_k_m2.repeat(3, 1, 1, 1)
            out_y1_g1 = nn.functional.conv2d(skip_out1, out_k_m1_g1, padding=0, groups=3, bias=None)
            out_y1_g2 = nn.functional.conv2d(skip_out1, out_k_m1_g2, padding=0, groups=3, bias=None)
            out_y1_g3 = nn.functional.conv2d(skip_out1, out_k_m1_g3, padding=0, groups=3, bias=None)
            out_y2 = nn.functional.conv2d(skip_out2, out_k_m2, padding=0, groups=3, bias=None)
            out_y1_down = downsample_tensor(out_y1,'topleft')
            out_y2_g1 = downsample_tensor(out_y1_g1,'topleft')
            out_y2_g2 = downsample_tensor(out_y1_g2, 'topleft')
            out_y2_g3 = downsample_tensor(out_y1_g3, 'topleft')
            sum=4 * out_y2 - (out_y2_g1 + out_y2_g2 + out_y2_g3)
            floss=fft_loss(pyramid2, sum)
            loss_main = 1-ssim(pyramid2, out_y1_down)
            total_loss =loss_main+0.001*floss

        elif step >=1000 and step < 1500:
            skip_out1, kernel1 = net(net_input1, net_input_kernel1, ker_size1)
            skip_out0, kernel0 = net(net_input0, net_input_kernel0, opt.kernel_size[1])
            out_k_m1 = kernel1.view(-1, 1, ker_size1, ker_size1)
            out_k_m1 = out_k_m1.repeat(3, 1, 1, 1)

            out_k_m0 = kernel0.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])
            out_k_m0_g1 = out_k_m0.clone()
            out_k_m0_g1[:, :, 1::2, :] *= -1
            out_k_m0_g2 = out_k_m0.clone()
            out_k_m0_g2[:, :, :, 1::2] *= -1
            out_k_m0_g3 = out_k_m0_g1.clone()
            out_k_m0_g3[:, :, :, 1::2] *= -1
            out_k_m0 = out_k_m0.repeat(3, 1, 1, 1)
            out_k_m0_g1 = out_k_m0_g1.repeat(3, 1, 1, 1)
            out_k_m0_g2 = out_k_m0_g2.repeat(3, 1, 1, 1)
            out_k_m0_g3 = out_k_m0_g3.repeat(3, 1, 1, 1)
            out_y1 = nn.functional.conv2d(skip_out1, out_k_m1, padding=0, groups=3, bias=None)
            out_y0 = nn.functional.conv2d(skip_out0, out_k_m0, padding=0, groups=3, bias=None)
            out_y0_g1 = nn.functional.conv2d(skip_out0, out_k_m0_g1, padding=0, groups=3, bias=None)
            out_y0_g2 = nn.functional.conv2d(skip_out0, out_k_m0_g2, padding=0, groups=3, bias=None)
            out_y0_g3 = nn.functional.conv2d(skip_out0, out_k_m0_g3, padding=0, groups=3, bias=None)
            out_y0_down = downsample_tensor(out_y0, 'topleft')
            out_y0_g1 = downsample_tensor(out_y0_g1, 'topleft')
            out_y0_g2 = downsample_tensor(out_y0_g2, 'topleft')
            out_y0_g3 = downsample_tensor(out_y0_g3, 'topleft')
            #total_loss = mse(pyramid1,out_y0_down)+mse(pyramid1, 4*out_y1-(out_y0_g1+out_y0_g2+out_y0_g3))
            sum = 4*out_y1-(out_y0_g1+out_y0_g2+out_y0_g3)
            floss = fft_loss(pyramid1, sum)
            loss_main=1-ssim(pyramid1, out_y0_down)
            total_loss =loss_main+0.001*floss
        else:
            skip_out0, kernel0 = net(net_input0, net_input_kernel0, opt.kernel_size[0])
            out_k_m0 = kernel0.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])
            out_k_m0 = out_k_m0.repeat(3, 1, 1, 1)
            out_y0 = nn.functional.conv2d(skip_out0, out_k_m0, padding=0, groups=3, bias=None)
            total_loss = 1 - ssim(pyramid0, out_y0)

        total_loss.backward()
        optimizer.step()

        if (step + 1) % opt.save_frequency == 0 and step < 500:
            # Your code to execute when both conditions are met

            save_path = os.path.join(opt.save_path, '%s_x.png' % (imgname))
            out_x_np = torch_to_np(skip_out2)
            out_x_np = out_x_np[:, padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2]]

            channel_out = out_x_np.transpose(1, 2, 0)
            channel_out = (channel_out * 255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
            save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            out_k_np = out_k_m2.permute(1, 0, 2, 3)
            out_k_np = torch_to_np(out_k_np)
            out_k_np /= np.max(out_k_np)

            channel_out = out_k_np.transpose(1, 2, 0)
            channel_out = (channel_out * 255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
            print(f"Epoch {step + 1}, Total Loss: {total_loss}, mse Loss: {mse(pyramid2, out_y2)}")
        elif (step + 1) % opt.save_frequency == 0 and step >= 500 and step < 1000:
            save_path = os.path.join(opt.save_path, '%s_x.png' % (imgname))
            out_x_np = torch_to_np(skip_out1)
            out_x_np = out_x_np[:, padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2]]

            channel_out = out_x_np.transpose(1, 2, 0)
            channel_out = (channel_out * 255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
            save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            out_k_np = out_k_m1.permute(1, 0, 2, 3)
            out_k_np = torch_to_np(out_k_np)
            out_k_np /= np.max(out_k_np)

            channel_out = out_k_np.transpose(1, 2, 0)
            channel_out = (channel_out * 255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
            print(f"Epoch {step + 1}, Total Loss: {total_loss}")
        elif (step + 1) % opt.save_frequency == 0:
            save_path = os.path.join(opt.save_path, '%s_x.png' % (imgname))
            out_x_np = torch_to_np(skip_out0)
            out_x_np = out_x_np[:, padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2]]

            channel_out = out_x_np.transpose(1, 2, 0)
            channel_out = (channel_out * 255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
            save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            out_k_np = out_k_m0.permute(1, 0, 2, 3)
            out_k_np = torch_to_np(out_k_np)
            out_k_np /= np.max(out_k_np)

            channel_out = out_k_np.transpose(1, 2, 0)
            channel_out = (channel_out * 255).astype(np.uint8)
            channel_out = Image.fromarray(channel_out).convert('RGB')
            channel_out.save(save_path)
            print(f"Epoch {step + 1}, Total Loss: {total_loss}")



