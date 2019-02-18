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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot


train_txt_path = './data/train.txt'
val_txt_path = './data/val.txt'
test_txt_path = './data/test.txt'
data_dir = './data/panoptic'
joint_dir = './data/2d_joints'

BATCH_SIZE = 50
chunk_sizes = [24, 26]
LEARNING_RATE = 0.00001
EPOCHS = 100
RESUME_FROM_FILE = False
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
PRINT_FREQ = 50

def main():
    data_load_time = AverageMeter()
    end = time.time()
    
    '''
    Load Data (Train, Val, Test)
    '''
    ### Load Training and Validation data
    train_txt = open(train_txt_path, 'r')
    t = [line.strip() for line in train_txt]
    #print(t)
    train_loader = torch.utils.data.DataLoader(PanopticDataset(data_dir, joint_dir, t), batch_size=BATCH_SIZE, shuffle=False)                                    
    print(len(train_loader))
    
    val_txt = open(val_txt_path, 'r')
    v = [line.strip() for line in val_txt]
    val_loader = torch.utils.data.DataLoader(PanopticDataset(data_dir, joint_dir, v), batch_size=BATCH_SIZE, shuffle=False) 
    print(len(val_loader))

    data_load_time.update(time.time()-end)
    print("Loading Data Time: {} sec".format(data_load_time.val))

    '''
    Model Setting (Load, Set)
    '''
    #### TensorBoardX
    writer = SummaryWriter('runs/exp-1')
    writer = SummaryWriter()

    #### Initialize Model
    #model_no_parallel = Arc(BATCH_SIZE)
    model_no_parallel = Arc2(output_size=(216, 384), in_channels=3, pretrained=True)
    model = DataParallel(model_no_parallel, chunk_sizes=chunk_sizes)
    model = model.cuda()

    #### Load Model from Checkpoint
    resume_file = 'checkpoint.pth.tar'
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
    criterion = torch.nn.MSELoss().cuda()

    lr = LEARNING_RATE
    best_loss = 1000000
    train_iter = 0
    val_iter = 0
    for epoch in range(EPOCHS):
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
        Joint shape: (BatchSize, people, joints, 2) 2: xy       e.g. torch.Size([4, 3, 19, 2])
        Target shape: (BatchSize, people, joints, 3) 3: xyz     e.g. torch.Size([4, 3, 19, 3])
        Output shape: (BatchSize, channel, H, W)                e.g. torch.Size([4, 1, 216, 384])
        Predicted shape: (BatchSize*people*joints)              e.g. torch.Size([228])
        lossTarget shape: (BatchSize*people*joints)             e.g. torch.Size([228])
        '''
        model.train()
        for idx_train, (image, joint, target, target_heatmap) in enumerate(train_loader):
            #print(idx_train)
            # Filter no person in the frame
            if joint.size(1) == 0 or target.size(1) == 0:
                continue
            
            dtype = torch.cuda.FloatTensor
            image_var = Variable(image.type(dtype))
            joint_var = Variable(joint.type(dtype))
            target_heatmap_var = Variable(target_heatmap.type(dtype))

            output = model(image_var)

            loss = criterion(output, target_heatmap_var)

            train_loss += loss.data.cpu().numpy()
            #print(loss, train_loss)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if (idx_train+1) % PRINT_FREQ == 0:
                print('Batch[{0}/{1}] [Loss]: {2}'.format((idx_train+1)*BATCH_SIZE, train_total, train_loss/train_count))
                writer.add_scalar('train_loss', train_loss/train_count, train_iter)
                output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                output_heatmap /= np.max(output_heatmap)
                plot.imsave('output/train/T{}.png'.format(train_count), output_heatmap, cmap="viridis")
            
            train_count += BATCH_SIZE
            train_iter += BATCH_SIZE
            
        
        #### Validation Mode
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
                
                
                if (idx_val+1) % PRINT_FREQ == 0:
                    print('Batch[{0}/{1}] [Loss]: {2}'.format((idx_val+1)*BATCH_SIZE, val_total, val_loss/val_count))
                    writer.add_scalar('val_loss', val_loss/val_count, val_iter)
                    output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                    output_heatmap /= np.max(output_heatmap)
                    plot.imsave('output/val/V{}.png'.format(val_count), output_heatmap, cmap="viridis")

                val_count += BATCH_SIZE
                val_iter += BATCH_SIZE
                


        #### Print Epoch Log
        writer.add_scalar('train_loss', train_loss/train_count, train_iter)
        writer.add_scalar('val_loss', val_loss/val_count, val_iter)
        print("Epoch[{0}] [Train Loss]: {1} [Val Loss]: {2}".format(epoch+1, train_loss/train_count, val_loss/val_count))
        

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
            }, 'checkpoint2.pth.tar')

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
