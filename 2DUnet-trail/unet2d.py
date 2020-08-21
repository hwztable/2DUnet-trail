import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock2d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTrans2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvTrans2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class UpBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock2d, self).__init__()
        self.up_conv = ConvTrans2d(in_ch, out_ch)
        self.conv = ConvBlock2d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


class UNet2D(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, degree=64):
        super(UNet2D, self).__init__()

        chs = []
        for i in range(5):
            chs.append((2 ** i) * degree)

        self.downLayer1 = ConvBlock2d(in_ch, chs[0])
        self.downLayer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[0], chs[1]))

        self.downLayer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[1], chs[2]))

        self.downLayer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[2], chs[3]))

        self.bottomLayer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[3], chs[4]))

        self.upLayer1 = UpBlock2d(chs[4], chs[3])
        self.upLayer2 = UpBlock2d(chs[3], chs[2])
        self.upLayer3 = UpBlock2d(chs[2], chs[1])
        self.upLayer4 = UpBlock2d(chs[1], chs[0])

        self.outLayer = nn.Conv2d(chs[0], out_ch, kernel_size=3, stride=1, padding=1)



    def forward(self, x):

        x1 = self.downLayer1(x)     # degree(32)   * 16    * W    * H
        x2 = self.downLayer2(x1)    # degree(64)   * 16/2  * W/2  * H/2
        x3 = self.downLayer3(x2)    # degree(128)  * 16/4  * W/4  * H/4
        x4 = self.downLayer4(x3)    # degree(256)  * 16/8  * W/8  * H/8

        #x5 = self.bottomLayer(x4)   # degree(512)  * 16/16 * W/16 * H/16

        #x = self.upLayer1(x5, x4)   # degree(256)  * 16/8 * W/8 * H/8
        x = self.upLayer2(x4, x3)    # degree(128)  * 16/4 * W/4 * H/4
        x = self.upLayer3(x, x2)    # degree(64)   * 16/2 * W/2 * H/2
        x = self.upLayer4(x, x1)    # degree(32)   * 16   * W   * H
        x = self.outLayer(x)        # out_ch(2 )   * 16   * W   * H
        return x



if __name__ == "__main__":
    net = UNet2D(4, 5, degree=64)

    batch_size = 4
#    a = torch.randn(batch_size, 4, 192, 192)
#    b = net(a)
#    print(b.shape)
    ori_nii = 'F:/libtorch/Brats18_CBICA_ABE_1_flair.nii'
    nii_1 = nib.load(ori_nii).get_data()
    
#    net = GhostUNet2D(1,5,degree=32)
    net1 = UNet2D(1,5,degree=32)
    
    nii_1_tensor92 = torch.from_numpy(nii_1[:,:,92]).float()
    
    nii_1_tensor92 = nii_1_tensor92.unsqueeze(dim=0)
    nii_1_tensor92 = nii_1_tensor92.unsqueeze(dim=1)
    
    out_nii = net1(nii_1_tensor92)
    
    out_nii1=out_nii.detach().numpy()
    
    #plt.imshow(out_nii1.squeeze(),'gray')
    pair_img = nib.Nifti1Pair(out_nii1.squeeze().transpose(2,1,0), np.eye(4)) 
    nib.save(pair_img,'G:/a/test-92-unet.nii')

