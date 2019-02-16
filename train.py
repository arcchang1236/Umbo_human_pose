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


features_dir = './features'
#shutil.copytree(data_dir, os.path.join(features_dir, data_dir[2:]))

train_txt_path = './data/train.txt'
val_txt_path = './data/val.txt'
test_txt_path = './data/test.txt'
data_dir = './data/panoptic'
joint_dir = './data/2d_joints'

BATCH_SIZE = 14
LEARNING_RATE = 0.00001
EPOCHS = 100
RESUME_FROM_FILE = False
IMG_WIDTH, IMG_HEIGHT = 1920, 1080


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
    #print(len(train_loader))
    
    val_txt = open(val_txt_path, 'r')
    v = [line.strip() for line in val_txt]
    val_loader = torch.utils.data.DataLoader(PanopticDataset(data_dir, joint_dir, v), batch_size=BATCH_SIZE, shuffle=False) 
    #print(len(val_loader))

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
    #model = ArcNet(config, is_train=True)
    model = DataParallel(model_no_parallel, chunk_sizes=[6,8])
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
            
            #print(joint, target)
            #print(image.shape, joint.shape, target.shape)
            
            
            dtype = torch.cuda.FloatTensor
            image_var = Variable(image.type(dtype))
            joint_var = Variable(joint.type(dtype))
            target_heatmap_var = Variable(target_heatmap.type(dtype))

            output = model(image_var)
            
            # predicted = []
            # total_joints = joint.size(1)*joint.size(2)
            # joint_t = joint.view(-1, total_joints, joint.size(3))
            # output_np = output[0].data.squeeze().cpu().numpy().astype(np.float32)
            # for bs in range(BATCH_SIZE):
            #     for js in range(total_joints):
            #         x = joint_t.numpy().astype(int)[bs][js][0]
            #         y = joint_t.numpy().astype(int)[bs][js][1]
            #         if x < 0 or x >= IMG_WIDTH or y < 0 or y >= IMG_HEIGHT:
            #             predicted.append(0)
            #             continue
            #         predicted.append(output_np[y][x])
            
            # predicted = torch.from_numpy(np.array(predicted)).float()
            
            # t1, t2, t3, t4 = target.size()
            # lossTarget = target.view(t1*t2*t3, t4)[:,-1].float()
            #loss = criterion(predicted, lossTarget)
            #loss = Variable(loss, requires_grad=True)

            loss = criterion(output, target_heatmap_var)

            train_loss += loss.data.cpu().numpy()
            #print(loss, train_loss)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_count += BATCH_SIZE
            if idx_train % 500 == 0:
                print('Batch[{0}/{1}] [Loss]: {2}'.format(train_count, train_total, train_loss/train_count))
                output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                output_heatmap /= np.max(output_heatmap)
                plot.imsave('output/T{}.png'.format(train_count), output_heatmap, cmap="viridis")
            
        
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

                # predicted = []
                # total_joints = joint.size(1)*joint.size(2)
                # joint_t = joint.view(-1, total_joints, joint.size(3))
                # output_np = output[0].data.squeeze().cpu().numpy().astype(np.float32)
                # for bs in range(BATCH_SIZE):
                #     for js in range(total_joints):
                #         x = joint_t.numpy().astype(int)[bs][js][0]
                #         y = joint_t.numpy().astype(int)[bs][js][1]
                #         if x < 0 or x >= IMG_WIDTH or y < 0 or y >= IMG_HEIGHT:
                #             predicted.append(0)
                #             continue
                #         predicted.append(output_np[y][x])
                
                # predicted = torch.from_numpy(np.array(predicted)).float()
                
                # t1, t2, t3, t4 = target.size()
                # lossTarget = target.view(t1*t2*t3, t4)[:,-1].float()
                #loss = criterion(predicted, lossTarget)
                #loss = Variable(loss, requires_grad=True)

                loss = criterion(output, target_heatmap)
                
                val_loss += loss.data.cpu().numpy()
                
                val_count += image.size(0)
                if idx_val % 500 == 0:
                    print('Batch[{0}/{1}] [Loss]: {2}'.format(val_count, val_total, val_loss/val_count))
                    output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
                    output_heatmap /= np.max(output_heatmap)
                    plot.imsave('output/V{}.png'.format(val_count), output_heatmap, cmap="viridis")

            #print(output.shape)

        #### Print Epoch Log
        writer.add_scalar('train_loss', train_loss/train_count, epoch+1)
        writer.add_scalar('val_loss', val_loss/val_count, epoch+1)
        print("Epoch[{0}] [Train Loss]: {1} [Val Loss]: {2}".format(epoch+1, train_loss/train_count, val_loss/val_count))
        

        #### Update Parameters
        if epoch % 10 == 0:
            lr = lr * 0.6
        

        #### Save Checkpoint
        torch.save({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'checkpoint.pth.tar')



    #### ---------------Below is testing------------- ####


    #### Loading training data
    train_data = datasets.ImageFolder('./features', transform=transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor()
    ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)                                    
    print(len(train_loader))
    
    


    #### Construct Net
    resnet50 = models.resnet50(pretrained = True)
    #resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
    #print(*list(resnet50.children()))
    resnet50_feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])
    #torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False   
        
    use_gpu = torch.cuda.is_available()
    #print(use_gpu)
    
    #### Inference
    for i_batch, input in enumerate(train_loader):
        #print(i_batch, input[0].dtype, input[0].shape)
        #print(input[1])
        saved_path = os.path.join(features_dir, str(i_batch)+'.txt')

        x = Variable(input[0], requires_grad=False)
        if use_gpu:
            x = x.cuda()
            resnet50_feature_extractor = resnet50_feature_extractor.cuda()
        y = resnet50_feature_extractor(x).cpu()
        #print(y.dtype)
        y = y.data.numpy()
        #print(y.shape)
        #np.savetxt(saved_path, y, delimiter=',')



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
