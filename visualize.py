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
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
    "limbs":{
        {"head", [[0,1], [1,15], [1,17], [15,16], [17,18]]},
        {"body", [[0,2], [0,3], [3,4], [4,5], [0,9], [9,10], [10,11]]},
        {"bottom", [[2,6], [6,7], [7,8], [2,12], [12,13], [13,14]]}
    }
'''

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

def is_image(file_name):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext


def plot3D(joint):
    '''
    Visualize in 3D space
    Joint shape: (people, 19, 3)
    '''
    line_idx = [[0,1], [1,15], [1,17], [15,16], [17,18], [0,2], [0,3], [3,4], [4,5], 
                [0,9], [9,10], [10,11], [2,6], [6,7], [7,8], [2,12], [12,13], [13,14]]
    # Show Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    for num in range(len(all_people)):
        for i in range(len(line_idx)):
            if(C[line_idx[i][0]+num*17] > CONFIDENCE_THRESH and C[line_idx[i][1]+num*17] > CONFIDENCE_THRESH):
                plt.plot([X[line_idx[i][0]+num*17],X[line_idx[i][1]+num*17]], [Z[line_idx[i][0]+num*17],Z[line_idx[i][1]+num*17]], [Y[line_idx[i][0]+num*17],Y[line_idx[i][1]+num*17]], 'ro-')

    plt.savefig('3dpose/01_01_0000'+str(name_id)+'.jpg')
    plt.show()
    

def main(opt):
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
        if os.path.isdir(opt.demo):
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                if is_image(file_name):
                image_name = os.path.join(opt.demo, file_name)
                print('Running {} ...'.format(image_name))
                image = cv2.imread(image_name)
                demo_image(image, model, opt)
        elif is_image(opt.demo):
            print('Running {} ...'.format(opt.demo))
            image = cv2.imread(opt.demo)
            demo_image(image, model, opt)
    

if __name__ == '__main__':
    main()