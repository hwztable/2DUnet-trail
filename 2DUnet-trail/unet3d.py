# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:43:55 2020

@author: windows
"""

import sys
sys.path.append("..")
import numpy as np
import nibabel as nib
import math
import torch
import torch.nn as nn

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTrans3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvTrans3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class UpBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock3d, self).__init__()
        self.up_conv = ConvTrans3d(in_ch, out_ch)
        self.conv = ConvBlock3d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


def maxpool():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model



class UNet3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, degree=64):
        super(UNet3D, self).__init__()

        chs = []
        for i in range(5):
            chs.append((2 ** i) * degree)

        self.downLayer1 = ConvBlock3d(in_ch, chs[0])
        self.downLayer2 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock3d(chs[0], chs[1]))

        self.downLayer3 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock3d(chs[1], chs[2]))

        self.downLayer4 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock3d(chs[2], chs[3]))

        self.bottomLayer = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock3d(chs[3], chs[4]))

        self.upLayer1 = UpBlock3d(chs[4], chs[3])
        self.upLayer2 = UpBlock3d(chs[3], chs[2])
        self.upLayer3 = UpBlock3d(chs[2], chs[1])
        self.upLayer4 = UpBlock3d(chs[1], chs[0])

        self.outLayer = nn.Conv3d(chs[0], out_ch, kernel_size=3, stride=1, padding=1)



    def forward(self, x):

        x1 = self.downLayer1(x)     # degree(32)   * 16    * W    * H
        x2 = self.downLayer2(x1)    # degree(64)   * 16/2  * W/2  * H/2
        x3 = self.downLayer3(x2)    # degree(128)  * 16/4  * W/4  * H/4
        x4 = self.downLayer4(x3)    # degree(256)  * 16/8  * W/8  * H/8

        x5 = self.bottomLayer(x4)   # degree(512)  * 16/16 * W/16 * H/16

        x = self.upLayer1(x5, x4)   # degree(256)  * 16/8 * W/8 * H/8
        x = self.upLayer2(x, x3)    # degree(128)  * 16/4 * W/4 * H/4
        x = self.upLayer3(x, x2)    # degree(64)   * 16/2 * W/2 * H/2
        x = self.upLayer4(x, x1)    # degree(32)   * 16   * W   * H
        x = self.outLayer(x)        # out_ch(2 )   * 16   * W   * H
        return x

def MultiOpera(stage_out,next_in):
    res = stage_out.get_data()*next_in.get_data()
    pair_img = nib.Nifti1Pair(res, np.eye(4)) 
    return pair_img


class Cascaded3DUnet(nn.module):
    def __init__(self, stg1in=2, stg1out=2, stg2in=2, stg2out=2, degree=64):
         super(UNet3D, self).__init__()
         self.stage1 = UNet3D(stg1in,stg1out)
         self.stage2 = UNet3D(stg2in,stg2out)
             
    def forward(self,x1,x2):
        stg1 = self.stage1(x1)
        mid_out = MultiOpera(x1,stg1)
        stg2 = self.stage2(mid_out,x2)
        
        return stg2

if __name__ == "__main__":
    net = UNet3D(4, 5, degree=64)

    batch_size = 4
    a = torch.randn(batch_size, 4, 192, 192)
    b = net(a)
    print(b.shape)

