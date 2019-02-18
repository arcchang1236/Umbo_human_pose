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

test_txt_path = './data/test.txt'
data_dir = './data/panoptic'
joint_dir = './data/2d_joints'

BATCH_SIZE = 1
RESUME_FROM_FILE = True
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
MODEL_WIDTH, MODEL_HEIGHT = 384 ,216
PRINT_FREQ = 50

def main():
    ### Load Testing data
    test_txt = open(test_txt_path, 'r')
    t = [line.strip() for line in test_txt]
    #print(t)
    test_loader = torch.utils.data.DataLoader(PanopticDataset(data_dir, joint_dir, t), batch_size=BATCH_SIZE, shuffle=False)                                    
    print(len(test_loader))
    

    #### Initialize Model
    #model_no_parallel = Arc(BATCH_SIZE)
    model_no_parallel = Arc2(output_size=(216, 384), in_channels=3, pretrained=True)
    model = DataParallel(model_no_parallel, chunk_sizes=[1])
    model = model.cuda()

    resume_file = 'checkpoint.pth.tar'

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
    Start Testing
    '''
    print("Start Testing...")
    #### Define Loss
    criterion = torch.nn.MSELoss().cuda()


    model.eval()
    test_count = 0
    test_iter = 0
    test_error = 0
    with torch.no_grad():
        for idx_test, (image, joint, target, target_heatmap) in enumerate(test_loader):
            # Filter no person in the frame
            if joint.size(1) == 0 or target.size(1) == 0:
                continue

            dtype = torch.cuda.FloatTensor
            image_var = Variable(image.type(dtype))
            joint_var = Variable(joint.type(dtype))
            target_heatmap_var = Variable(target_heatmap.type(dtype))

            output = model(image_var)            
            
            predicted = []
            total_joints = joint.size(1)*joint.size(2)
            joint_t = joint.view(-1, total_joints, joint.size(3))
            output_np = output[0].data.squeeze().cpu().numpy().astype(np.float32)
            for bs in range(BATCH_SIZE):
                for js in range(total_joints):
                    x = joint_t.numpy().astype(int)[bs][js][0]
                    y = joint_t.numpy().astype(int)[bs][js][1]
                    if x < 0 or x >= MODEL_WIDTH or y < 0 or y >= MODEL_HEIGHT:
                        predicted.append([0, 0, 0])
                        continue
                    predicted.append([x, y, output_np[y][x]])
            
            predicted = torch.from_numpy(np.array(predicted)).float()
            
            t1, t2, t3, t4 = target.shape
            target = target.reshape((t1*t2*t3, t4))
            dtype = torch.cuda.DoubleTensor
            target = Variable(target.type(dtype))
            predicted = Variable(predicted.type(dtype))

            print(predicted.shape, target.shape)

            loss = criterion(output, target_heatmap_var).data.cpu().numpy()
            test_error = MPJPE(predicted, target)
            
            print('[Loss]: {0}, [MPJPE]: {1}'.format(loss, test_error))
            

            output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
            output_heatmap /= np.max(output_heatmap)
            plot.imsave('output/test/T{}.png'.format(idx_test), output_heatmap, cmap="viridis")

            test_count += BATCH_SIZE
            test_iter += BATCH_SIZE



if __name__ == '__main__':
    main()
