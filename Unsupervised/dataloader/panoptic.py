import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json_tricks
import json
import numpy as np
import cv2

import torch
import torch.utils.data as data

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

class PanopticDataset(data.Dataset):
   '''
   "keypoints":{
      {0,  "Neck"},
      {1,  "Nose"}, 
      {2,  "MidHip"},
      {3,  "LShoulder"},
      {4,  "LElbow"},
      {5,  "LWrist"},
      {6, "LHip"},
      {7, "LKnee"},
      {8, "LAnkle"},
      {9,  "RShoulder"},
      {10,  "RElbow"},
      {11,  "RWrist"},
      {12,  "RHip"},
      {13, "RKnee"},
      {14, "RAnkle"},
      //{15, "LEye"},
      //{16, "LEar"},
      //{17, "REye"},
      //{18, "REar"}
   }

   "limbs":{
      {"head", [[0,1], [1,15], [1,17], [15,16], [17,18]]},
      {"body", [[0,2], [0,3], [3,4], [4,5], [0,9], [9,10], [10,11]]},
      {"bottom", [[2,6], [6,7], [7,8], [2,12], [12,13], [13,14]]}
   }

   Path: ./data/panoptic/{activity}/{path_type}/{camera}/{camera}_{frame}

   activity: 160224_haggling1, ...
   path_type: hdImgs, hdPose3d_stage1_coco19
   camera: {camera_panel}_{camera_node} (00_09)

   data_dir:
   ./data/panoptic

   lists(train, val):
   160224_haggling1/hdImgs/00_00/00_00_00000000.jpg
   160224_haggling1/hdImgs/00_00/00_00_00000001.jpg
   ......

   2D joints:
   ./data/2d_joints/{activity}/{camera}/{camera}_{frame}.txt
   [Example]
   x1 y1
   x2 y2
   ...
   x19 y19
   end
   x1 y1
   x2 y2
   ...
   x19 y19
   end
   ...

   labels:
   (people)(joints)
   [
      [
         [proj_x, proj_y, depth, c]
         * joints(19)
      ]
      * people
   ]

   '''

   def __init__(self, cfg, train=True):
      self.data_dir = cfg.data_dir
      self.plane_dir = 'hdPose_image_plane'
      self.world_dir = 'hdPose3d_stage1_coco19'
      self.world_pose_prefix = 'body3DScene'
      
      if train:
         train_txt = open(cfg.train_txt_path, 'r')
         l = [line.strip() for line in train_txt]
      else:
         test_txt = open(cfg.test_txt_path, 'r')
         l = [line.strip() for line in val_txt]
      self.lists = l
      
      self.num_of_joints = cfg.num_of_joints
      self.img_res = cfg.img_shape
      self.inp_res = cfg.data_shape
      self.out_res = cfg.output_shape
      self.cfg = cfg
      

   def __len__(self):
      return len(self.lists)

   def __getitem__(self, idx):
      # e.g. 160906_pizza1/hdImgs/00_00/00_00_00005000.jpg
      activity, _, camera, wholename = self.lists[idx].split('/')
      camera_panel, camera_node = int(camera.split('_')[0]), int(camera.split('_')[1])
      _, _, frame = wholename.split('.')[0].split('_')

      targets_3d = np.zeros((self.out_res[0], self.out_res[1]))
      
      targets_2d = []
      target15 = np.zeros((self.num_of_joints, self.out_res[0], self.out_res[1]))
      target11 = np.zeros((self.num_of_joints, self.out_res[0], self.out_res[1]))
      target9 = np.zeros((self.num_of_joints, self.out_res[0], self.out_res[1]))
      target7 = np.zeros((self.num_of_joints, self.out_res[0], self.out_res[1]))

      joint_world = []
      joint_plane = []

      #### Image ####

      img_name = os.path.join(self.data_dir, self.lists[idx])
      image = Image.open(img_name)
      image = image.resize(self.inp_res, Image.BILINEAR)
      #print(img_name)
      transform_img = transforms.Compose([
         transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      ])
      image = transform_img(image) # CxHxW
      image = image.transpose(1,2)

      ##########

      #### Points in image plane ####
      
      joint_plane_name = '{0}/{1}.txt'.format(camera, wholename.split('.')[0])
      joint_plane_name = os.path.join(os.path.join(self.data_dir, activity), os.path.join(self.plane_dir, joint_plane_name))
      joint_plane_file = open(joint_plane_name, 'r')
      tmp = []
      for s in [l.strip() for l in joint_plane_file]:
         if s == 'end':
            joint_plane.append(tmp)
            tmp = []
         else:
            xyz = [float(i) for i in s.split(' ')]
            xyz[0] *= self.out_res[0] / self.img_res[0]
            xyz[1] *= self.out_res[1] / self.img_res[1]
            tmp.append(xyz)
      
      joint_plane = np.array(joint_plane)
      joint_plane_pad = self.padJoint(joint_plane, 3)
      
      if joint_plane.shape[0] == 0:
         joint_plane_pad = self.padJoint(joint_plane, 3)
         joint_world_pad = self.padJoint(joint_world, 3)
         targets_2d = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
         targets_3d = targets_3d.reshape((1, self.out_res[0], self.out_res[1]))
         targets_3d = torch.Tensor(targets_3d)
         #print(image.shape, joint_plane_pad.shape, joint_world_pad.shape, len(targets_2d), targets_3d.shape)
         return [image, joint_plane_pad, joint_world_pad, targets_2d, targets_3d]

      ##########

      #### Points in world coordinate ####

      joint_world_name = '{0}_{1}.json'.format(self.world_pose_prefix, frame)
      joint_dir = os.path.join(self.data_dir, activity)
      joint_dir = os.path.join(joint_dir, self.world_dir)
      joint_world_name = os.path.join(joint_dir, joint_world_name)
      #print(joint_world_name)
      joint_world_json = json.load(open(joint_world_name,'r'))
      people = len(joint_world_json['bodies'])
      for body in joint_world_json['bodies']:
         skel = np.array(body['joints19']).reshape((-1,4)).transpose()
         skel = skel[0:3,:]
         skel = np.transpose(skel) # shape: (people, 19, 3)
         joint_world.append(skel)
      
      joint_world = np.array(joint_world)
      joint_world_pad = self.padJoint(joint_world, 3)
      
      ##########
      
      #### 2D Target Heatmap ####
      
      
      for i in range(self.num_of_joints):
         tmp = []
         for j in range(people):
            tmp.append(joint_plane[j][i][:2])
         target15[i] = self.generateMultiHeatmap(target15[i], tmp, self.cfg.gk15)
         target11[i] = self.generateMultiHeatmap(target11[i], tmp, self.cfg.gk11)
         target9[i] = self.generateMultiHeatmap(target9[i], tmp, self.cfg.gk9)
         target7[i] = self.generateMultiHeatmap(target7[i], tmp, self.cfg.gk7)
      
      targets_2d = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]

      ##########

      #### 3D Target Heatmap ####
      
      
      targets_3d = self.generateMultiHeatmap3D(targets_3d, joint_plane, self.cfg.gk3d)
      targets_3d = targets_3d.reshape((1, self.out_res[0], self.out_res[1]))
      targets_3d = torch.Tensor(targets_3d)

      ##########

      #print(image.shape, joint_plane_pad.shape, joint_world_pad.shape, len(targets_2d), targets_3d.shape)

      return [image, joint_plane_pad, joint_world_pad, targets_2d, targets_3d]
      

   def padJoint(self, joints, dim, MAX_PEOPLE=10):
      # Padding the GT information in same dimension
      EXTRA_BODIES = MAX_PEOPLE - len(joints)
      if EXTRA_BODIES == MAX_PEOPLE:
         joints = np.zeros((EXTRA_BODIES, self.num_of_joints, dim))
      elif EXTRA_BODIES != 0:
         pad = np.zeros((EXTRA_BODIES, self.num_of_joints, dim))
         #print(joints.shape, pad.shape)
         joints = np.concatenate((joints, pad), axis=0)
      
      return joints

   def generateMultiHeatmap(self, heatmap, pts, gamma):
      for pt in pts:
         heatmap_tmp = np.zeros_like(heatmap)
         x = int(pt[0])
         y = int(pt[1])
         if x < heatmap.shape[0] and x >= 0 and y < heatmap.shape[1] and y >= 0:
            heatmap_tmp[x][y] = 1
            heatmap_tmp = cv2.GaussianBlur(heatmap_tmp, gamma, 0) # 0: color variance
         heatmap = np.add(heatmap_tmp, heatmap)

      return heatmap
   
   def generateMultiHeatmap3D(self, heatmap, pts, gamma):
      #print(pts.shape)
      p1, p2, p3 = pts.shape
      pts = pts.reshape((p1*p2, p3))
      for i in range(p1*p2):
         heatmap_tmp = np.zeros_like(heatmap)
         x = int(pts[i][0])
         y = int(pts[i][1])
         if x < heatmap.shape[0] and x >= 0 and y < heatmap.shape[1] and y >= 0:
            #print(x, y, ptb[2])
            heatmap_tmp[x][y] = pts[i][2]
            heatmap_tmp = cv2.GaussianBlur(heatmap_tmp, gamma, 0) # 0: color variance
         
         heatmap = np.add(heatmap_tmp, heatmap)
      
      return heatmap