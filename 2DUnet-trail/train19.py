from unet2d import UNet2D
from data_brats19 import Brats19DataLoader
from test import evaluation

from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import numpy as np
import sys
import scipy.misc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ********** Hyper Parameter **********
data_dir = '/data/BraTS/MICCAI_BraTS_2019_Data_Training'
conf_train = '/code/Myunet/train19.conf'
conf_valid = '/code/Myunet/valid19.conf'
save_dir = '/output/ckpt'

learning_rate = 0.0001
batch_size = 32
epochs = 100

cuda_available = torch.cuda.is_available()


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ******************** build model ********************
net = UNet2D(in_ch=4, out_ch=5, degree=64)  # multi-modal =4, out binary classification one-hot
if cuda_available:
    net = net.cuda()

# ******************** data preparation  ********************
print('train data ....')
train_data = Brats19DataLoader(data_dir=data_dir, conf=conf_train,train=True)  # 224 subject, 34720 images
print('valid data .....')
valid_data = Brats19DataLoader(data_dir=data_dir,  conf=conf_valid,train=False)   #

# data loader
train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_dataset = DataLoader(dataset=valid_data, batch_size=1, shuffle=True)


def one_hot_reverse(predicts):
    predicts = F.softmax(predicts, dim=1)  # 4D float Tensor  bz * 5 * 240 * 240
    return torch.max(predicts, dim=1)[1]  # 3D long Tensor  bz * 240 * 240 (val 0-4)


def norm(data):
    data = np.asarray(data)
    smax = np.max(data)
    smin = np.min(data)
    if smax - smin == 0:
        return data
    else:
        return (data - smin) / ((smax - smin) * 1.0)


def norm4(data):
    data = np.asarray(data)
    smax = 4.0
    smin = 0.0
    return (data - smin) / ((smax - smin) * 1.0)

def save_train_images(images, predicts, labels, index, epoch, save_dir='ckpt/'):

    images = np.asarray(images.cpu().data)
    predicts = np.asarray(predicts.cpu().data)
    labels = np.asarray(labels.cpu())

    if not os.path.exists(save_dir + 'epoch' + str(epoch)):
        os.mkdir(save_dir + 'epoch' + str(epoch))
    for b in range(images.shape[0]):  # for each batch
        name = index[b].split('/')[-1]
        save_one_image_label_pre(images[b,:,:,:], labels[b,:,:], predicts[b,:,:],
                                 save_dir=save_dir + 'epoch' + str(epoch) + '/b_' +str(b) + name + '.jpg')

def save_one_image_label_pre(images, label, predict, save_dir):

    output = np.zeros((240, 250 * 3))  # H, W
    for m in range(1):  # for each modal
        output[:, 250 * m: 250 * m + 240] = norm(images[m, :, :])
    output[:, 250 * 1: 250 * 1 + 240] = norm4(predict)
    output[:, 250 * 2: 250 * 2 + 240] = norm4(label)
    scipy.misc.imsave(save_dir, output)


def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)


def run():
    score_max = -1.0
    best_epoch = 0
    weight = torch.from_numpy(train_data.weight).float()    # weight for all class
    weight = to_var(weight)                                 #

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(weight=weight)

    for epoch in range(1, epochs + 1):
        print('epoch....................................' + str(epoch))
        train_loss = []
        # *************** train model ***************
        print('train ....')
        net.train()
        for step, (image, label, index) in enumerate(train_dataset):
            image = to_var(image)    # 4D tensor   bz * 4(modal) * 240 * 240
            label = to_var(label)    # 3D tensor   bz * 240 * 240 (value 0-4)

            optimizer.zero_grad()       #
            predicts = net(image)    # 4D tensor   bz * 5(class) * 240 * 240
            loss_train = criterion(predicts, label.long())
            train_loss.append(float(loss_train))
            loss_train.backward()
            optimizer.step()

            # ****** save sample image for each epoch ******
            if step % 200 == 0:
                print('..step ....%d' % step)
                print('....loss....%f' %loss_train)
                predicts = one_hot_reverse(predicts)  # 3D long Tensor  bz * 240 * 240 (val 0-4)
                save_train_images(image, predicts, label, index, epoch, save_dir=save_dir)

        # ***************** calculate valid loss *****************
        print('valid ....')
        current_score, valid_loss = evaluation(net, valid_dataset, criterion, save_dir=None)

        # **************** save loss for one batch ****************
        print('train_epoch_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0)) )
        print('valid_epoch_loss ' + str(sum(valid_loss) / (len(valid_loss) * 1.0)) )

        # **************** save model ****************
        if current_score > score_max:
            best_epoch = epoch
            torch.save(net.state_dict(),
                       os.path.join(save_dir , 'best_epoch.pth'))
            score_max = current_score
        print('valid_meanIoU_max ' + str(score_max))
        print('Current Best epoch is %d' % best_epoch)

        if epoch == epochs:
            torch.save(net.state_dict(),
                       os.path.join(save_dir, 'final_epoch.pth'))

    print('Best epoch is %d' % best_epoch)
    print('done!')


if __name__ == '__main__':
    run()


