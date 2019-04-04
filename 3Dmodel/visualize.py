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


'''
    "limbs":{
        {"head", [[0,1], [1,15], [1,17], [15,16], [17,18]]},
        {"body", [[0,2], [0,3], [3,4], [4,5], [0,9], [9,1], [1,11]]},
        {"bottom", [[2,6], [6,7], [7,8], [2,12], [12,13], [13,14]]}
    }
'''

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

def is_image(file_name):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext



    

test_txt_path = './data/test.txt'
data_dir = './data/panoptic'
joint_dir = './data/2d_joints'

BATCH_SIZE = 1
RESUME_FROM_FILE = True
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
MODEL_WIDTH, MODEL_HEIGHT = 384 ,216
PRINT_FREQ = 50
resume_file = '_sigma11.pth.tar'

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

            t1, t2, t3, t4 = target.shape
            target = target.view((t1*t2, t3, t4)).numpy()
            print(target.shape)
            #plot3D(target, 40, 100)

            output = model(image_var)            
            
            print(output.shape)
            output2 = output.unsqueeze(0)
            print(output2.shape)


            output_heatmap = output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
            #output_heatmap /= np.max(output_heatmap)
            plt.imsave('v1.png', output_heatmap, cmap="viridis")

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
                    sum_sq = 0.0
                    cnt_sq = 0.0
                    for dx in range(-3,4):
                        for dy in range(-3,4):
                            if x+dx < 0 or x+dx >= MODEL_WIDTH or y+dy < 0 or y+dy >= MODEL_HEIGHT:
                                continue
                            if output_np[y+dy][x+dx] == 0:
                                continue
                            sum_sq += output_np[y+dy][x+dx]
                            cnt_sq += 1
                    print(sum_sq/cnt_sq)
                    predicted.append([x, y, sum_sq/cnt_sq])
            
            predicted = torch.from_numpy(np.array(predicted)).float()
            pv, _ = predicted.shape
            predicted = predicted.view((int(pv/15),15,3)).numpy()
            print(predicted.shape)
            pvv, _, _ = predicted.shape
            plot3D(target, predicted, 0, 3)
            break



if __name__ == '__main__':
    main()