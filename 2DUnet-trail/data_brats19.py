import SimpleITK as sitk
import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.misc
import torch.nn.functional as F

def load_nii_as_array(img_name):
    img = sitk.ReadImage(img_name)
    nda = sitk.GetArrayFromImage(img)
    return nda

def norm_vol(data):
    data = data.astype(np.float)
    index = data.nonzero()
    smax = np.max(data[index])
    smin = np.min(data[index])

    if smax - smin == 0:
        return data
    else:
        data[index] = (data[index] - smin * 1.0) / (smax - smin)
        return data
    
class DataLoader19(Dataset):
    def __init__(self, data_dir, conf='../config/train19.conf', train=True):
        img_lists = []
        train_config = open(conf).readlines()
        for data in train_config:
            img_lists.append(os.path.join(data_dir, data.strip('\n')))
        
        self.data = []
        self.freq = np.zeros(5)
        self.zero_vol = np.zeros((4, 240, 240))
        count = 0
        for subject in img_lists:
            count += 1
            if count % 10 == 0:
                print('loading imageSets %d' %count)
            volume, label = DataLoader19.get_subject(subject)   # 4 * 155 * 240 * 240,  155 * 240 * 240
            volume = norm_vol(volume)

            self.freq += self.get_freq(label)
            if train is True:
                length = volume.shape[1]
                for i in range(length):
                    name = subject + '=slice' + str(i)
                    if (volume[:, i, :, :] == self.zero_vol).all():  # when training, ignore zero data
                        continue
                    else:
                        self.data.append([volume[:, i, :, :], label[i, :, :], name])
            else:
                volume = np.transpose(volume, (1, 0, 2, 3))
                self.data.append([volume, label, subject])
        self.freq = self.freq / np.sum(self.freq)
        self.weight = np.median(self.freq) / self.freq
        print('********  Finish loading data  ********')
        print('********  Weight for all classes  ********')
        print(self.weight)
        if train is True:
            print('********  Total number of 2D images is ' + str(len(self.data)) + ' **********')
        else:
            print('********  Total number of subject is ' + str(len(self.data)) + ' **********')


    def __getitem__(self, index):
       
        [image, label, name] = self.data[index]  #获取单个数据和标签，包括文件名
        
        image = torch.from_numpy(image).float()  # Float Tensor 4, 240, 240
        label = torch.from_numpy(label).float()    # Float Tensor 240, 240
        return image, label, name

    def get_subject(subject):
        # **************** get file ****************
        files = os.listdir(subject)  #
        multi_mode_dir = []
        label_dir = ""
        for f in files:
            if 'flair' in f :    # if is data or 't1' in f or 't1ce' in f or 't2' in f，只加载flair
                multi_mode_dir.append(f)
            elif 'seg' in f:        # if is label
                label_dir = f

        # ********** load 4 mode images **********
        multi_mode_imgs = []  # list size :4      item size: 155 * 240 * 240
        for mod_dir in multi_mode_dir:
            path = os.path.join(subject, mod_dir)  # absolute directory
            img = load_nii_as_array(path)
            multi_mode_imgs.append(img)

        # ********** get label **********
        label_dir = os.path.join(subject, label_dir)# 
        label = load_nii_as_array(label_dir)  #
        volume = np.asarray(multi_mode_imgs)
        return volume, label

    def get_freq(self, label):
        class_count = np.zeros((5))
        for i in range(5):
            a = (label == i) + 0
            class_count[i] = np.sum(a)
        return class_count

if __name__ == "__main__":
    vol_num = 4
    data_dir = 'MICCAI_BraTS_2018_Data_Training/'#'../data_sample/'
    conf = 'MICCAI_BraTS_2018_Data_Training/config/valid18.config'
    # test for training data
    brats19 = DataLoader19(data_dir=data_dir, conf=conf, train=True)
    image2d, label2d, im_name = brats19[5]

    print('image size ......')
    print(image2d.shape)             # (4,  240, 240)

    print('label size ......')
    print(label2d.shape)             # (240, 240)
    print(im_name)
    name = im_name.split('/')[-1]
 
    test = DataLoader19(data_dir=data_dir, conf=conf, train=False)
    image_volume, label_volume, subject = test[0]
    print(image_volume.shape)
    print(label_volume.shape)
    print(subject)
