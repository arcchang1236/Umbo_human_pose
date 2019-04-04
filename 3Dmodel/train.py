import os
import torch
import glob
import time
import argparse
import numpy as np
from torch.autograd import Variable
from PIL import Image 
from torchvision import models, transforms, datasets
from tensorboardX import SummaryWriter 
import torch.nn as nn
import shutil

from dataset.panoptic import *
#from net.Arc import *
from net.Arc2 import *
from net.loss import *
from utils.data_parallel import DataParallel
from utils.utils import *

import matplotlib.pyplot as plt


train_txt_path = './data/train.txt'
val_txt_path = './data/val.txt'
test_txt_path = './data/test.txt'
data_dir = './data/panoptic'
joint_dir = './data/2d_joints'

BATCH_SIZE = 60
chunk_sizes = [30, 30]
NUM_WORKER = 24
LEARNING_RATE = 0.0001
EPOCHS = 100
RESUME_FROM_FILE = False
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
MODEL_WIDTH, MODEL_HEIGHT = 384 ,216
PRINT_FREQ_TRAIN = 200
PRINT_FREQ_VAL = 80

iddd = '_sigma11'
gt_dir = 'gt' + iddd
train_dir = 'train' + iddd
val_dir =  'val' + iddd
checkpoint_name = iddd + '.pth.tar'
resume_file = iddd + '.pth.tar'

def main():
    data_load_time = AverageMeter()
    predict_time = AverageMeter()
    end = time.time()
    
    output_path = 'output'
    if not os.path.isdir(os.path.join(output_path, gt_dir)):
        os.mkdir(os.path.join(output_path, gt_dir))
    if not os.path.isdir(os.path.join(output_path, train_dir)):
        os.mkdir(os.path.join(output_path, train_dir))
    if not os.path.isdir(os.path.join(output_path, val_dir)):
        os.mkdir(os.path.join(output_path, val_dir))
    
    '''
    Load Data (Train, Val, Test)
    '''
    ### Load Training and Validation data
    train_txt = open(train_txt_path, 'r')
    t = [line.strip() for line in train_txt]
    #print(t)
    train_loader = torch.utils.data.DataLoader(PanopticDataset(data_dir, joint_dir, t), batch_size=BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True, shuffle=True)                               
    print(len(train_loader))
    
    val_txt = open(val_txt_path, 'r')
    v = [line.strip() for line in val_txt]
    val_loader = torch.utils.data.DataLoader(PanopticDataset(data_dir, joint_dir, v), batch_size=BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True, shuffle=False)
    print(len(val_loader))

    print("Loading Data Time: {} sec".format(time.time()-end))

    '''
    Model Setting (Load, Set)
    '''
    #### TensorBoardX
    writer = SummaryWriter('runs/exp-1')
    writer = SummaryWriter()

    #### Initialize Model
    #model_no_parallel = Arc(BATCH_SIZE, (216, 384))
    model_no_parallel = Arc2(output_size=(216, 384), in_channels=3, pretrained=True)
    model = DataParallel(model_no_parallel, chunk_sizes=chunk_sizes)
    model = model.cuda()

    #### Load Model from Checkpoint
    start_epoch = 0
    if RESUME_FROM_FILE:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))
    '''
    Start Training
    '''
    print("Start Training...")
    #### Define Loss
    #criterion = JointsMSELoss().cuda()
    #criterion = torch.nn.L1Loss().cuda()
    criterion = torch.nn.MSELoss().cuda()

    lr = LEARNING_RATE
    best_loss = 100000
    train_iter = 0
    val_iter = 0
    for epoch in range(EPOCHS):
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loss = 0
        train_count = 0
        train_total = len(train_loader)*BATCH_SIZE
        val_loss = 0
        val_count = 0
        val_total = len(val_loader)*BATCH_SIZE
        #### Train Mode
        '''
        Image shape: (BatchSize, channel, H, W)                 e.g. torch.Size([4, 3, 216, 384])
        Joint shape: (BatchSize, people, joints, 2) 2: xy       e.g. torch.Size([4, 3, 15, 2])
        Target shape: (BatchSize, people, joints, 3) 3: xyz     e.g. torch.Size([4, 3, 15, 3])
        Output shape: (BatchSize, channel, H, W)                e.g. torch.Size([4, 1, 216, 384])
        Predicted shape: (BatchSize*people*joints)              e.g. torch.Size([228])
        lossTarget shape: (BatchSize*people*joints)             e.g. torch.Size([228])
        '''
        print("Training Mode")
        model.train()
        for idx_train, (image, joint, target, target_heatmap) in enumerate(train_loader):
            # Filter no person in the frame
            if joint.size(1) == 0 or target.size(1) == 0:
                continue
            
            data_load_time.update(time.time()-end)
            end = time.time()

            dtype = torch.cuda.FloatTensor
            image_var = Variable(image.type(dtype))
            joint_var = Variable(joint.type(dtype))
            target_heatmap_var = Variable(target_heatmap.type(dtype))

            #print(target_heatmap)

            output = model(image_var)

            #print(output)

            loss = criterion(output, target_heatmap_var)
            train_loss += loss.data.cpu().numpy()
            #print(loss, train_loss)

            predict_time.update(time.time()-end)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_count += BATCH_SIZE
            train_iter += BATCH_SIZE
            

            if (idx_train+1) % PRINT_FREQ_TRAIN == 0:
                print('Epoch[{3}][{0}/{1}]: [Time] {4:.4f} [Data] {5:.4f} [Loss] {2:.8f}'.format((idx_train+1)*BATCH_SIZE, train_total, train_loss/train_count, epoch+1, predict_time.avg, data_load_time.avg), end="  ||  ")
                writer.add_scalar('train_loss', train_loss/train_count, train_iter)
                target_heatmap_img = target_heatmap[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                #target_heatmap_img /= np.max(target_heatmap_img)
                plt.imsave('output/{0}/T{1}.png'.format(gt_dir, train_count), target_heatmap_img, cmap="viridis")
                
                output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                #output_heatmap /= np.max(output_heatmap)
                #print(np.amax(output_heatmap), np.amin(output_heatmap))
                plt.imsave('output/{2}/T{0}_{1}.png'.format(epoch+1, train_count, train_dir), output_heatmap, cmap="viridis")

                ooo = output.data.squeeze().cpu().numpy().astype(np.float32)
                ttt = target_heatmap.data.squeeze().cpu().numpy().astype(np.float32)
                print(np.amax(ooo), np.amin(ooo), np.amax(ttt), np.amin(ttt))
            
            
            
        
        #### Validation Mode
        print("Validataion Mode")
        model.eval()
        with torch.no_grad():
            for idx_val, (image, joint, target, target_heatmap) in enumerate(val_loader):
                # Filter no person in the frame
                if joint.size(1) == 0 or target.size(1) == 0:
                    continue

                dtype = torch.cuda.FloatTensor
                image_var = Variable(image.type(dtype))
                joint_var = Variable(joint.type(dtype))
                target_heatmap_var = Variable(target_heatmap.type(dtype))

                output = model(image_var)
                #print(output.shape)

                loss = criterion(output, target_heatmap_var)
                
                val_loss += loss.data.cpu().numpy()
                
                val_count += BATCH_SIZE
                val_iter += BATCH_SIZE
                
                if (idx_val+1) % PRINT_FREQ_VAL == 0:
                    print('Epoch[{3}][{0}/{1}] [Loss]: {2}'.format((idx_val+1)*BATCH_SIZE, val_total, val_loss/val_count, epoch+1))
                    writer.add_scalar('val_loss', val_loss/val_count, val_iter)
                    target_heatmap_img = target_heatmap[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                    #target_heatmap_img /= np.max(target_heatmap_img)
                    plt.imsave('output/{0}/V{1}.png'.format(gt_dir, val_count), target_heatmap_img, cmap="viridis")
                    
                    output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                    output_heatmap /= np.max(output_heatmap)
                    plt.imsave('output/{2}/V{0}_{1}.png'.format(epoch+1, val_count, val_dir), output_heatmap, cmap="viridis")

                    ### PLOT 3D

                    # predicted = []
                    # total_joints = joint.size(1)*joint.size(2)
                    # joint_t = joint.view(-1, total_joints, joint.size(3))
                    # output_np = output[0].data.squeeze().cpu().numpy().astype(np.float32)
                    # #print(joint.shape, joint_t.shape, total_joints, output_np.shape)
                    
                    # for js in range(total_joints):
                    #     x = joint_t.numpy().astype(int)[0][js][0]
                    #     y = joint_t.numpy().astype(int)[0][js][1]
                    #     if x < 0 or x >= MODEL_WIDTH or y < 0 or y >= MODEL_HEIGHT:
                    #         predicted.append([0, 0, 0])
                    #         continue
                    #     predicted.append([x, y, output_np[y][x]])
                    
                    # predicted = torch.from_numpy(np.array(predicted)).float()
                    # pv, _ = predicted.shape
                    # predicted = predicted.view((int(pv/15),15,3)).numpy()
                    # #print(predicted.shape)
                    # pvv, _, _ = predicted.shape
                    # print(np.amax(predicted), np.amin(predicted))
                    # plot3D(target[0], predicted, 0, 400)

                
                


        #### Print Epoch Log
        writer.add_scalar('train_loss', train_loss/train_count, train_iter)
        writer.add_scalar('val_loss', val_loss/val_count, val_iter)
        print("Epoch[{0}] [Train Loss]: {1:.8f} [Val Loss]: {2:.8f} \n".format(epoch+1, train_loss/train_count, val_loss/val_count))
        writer.add_scalar('train_loss_epoch', train_loss/train_count, epoch)
        writer.add_scalar('train_loss_epoch', val_loss/val_count, epoch)

        #### Update Parameters
        if epoch % 10 == 0:
            lr = lr * 0.6
        

        #### Save Checkpoint
        if val_loss/val_count < best_loss:
            best_loss = val_loss/val_count
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_name)

    print("Training End")



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
