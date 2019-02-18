import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json_tricks
import json
import numpy as np

from utils.utils import projectPoints

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
      {15, "LEye"},
      {16, "LEar"},
      {17, "REye"},
      {18, "REar"}
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

   def __init__(self, data_dir, joint_dir, lists):
      self.data_dir = data_dir
      self.joint_dir = joint_dir
      self.lists = lists
      self.pose_prefix = 'hdPose3d_stage1_coco19/body3DScene'

   def __len__(self):
      return len(self.lists)

   def __getitem__(self, idx):

      activity, _, camera, tmp = self.lists[idx].split('/')
      camera_panel, camera_node = int(camera.split('_')[0]), int(camera.split('_')[1])
      _, _, frame = tmp.split('.')[0].split('_')



      img_name = os.path.join(self.data_dir, self.lists[idx])
      image = Image.open(img_name)
      #print(img_name)
      transform_img = transforms.Compose([
         transforms.Resize(216),
         transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      ])
      image = transform_img(image)




      joint2d_name = '{0}/{1}/{2}.txt'.format(activity, camera, tmp.split('.')[0])
      joint2d_fullname = os.path.join(self.joint_dir, joint2d_name)
      joint2d = open(joint2d_fullname, 'r')
      joint = []
      tmp = []
      nj = 0
      for s in [l.strip() for l in joint2d]:
         if s == 'end':
            nj += 1
            joint.append(tmp)
            tmp = []
         else:
            xy = [float(i) for i in s.split(' ')]
            tmp.append(xy)

        
      joint = np.array(joint) * 216/1080
      EXTRA_BODIES = 7 - nj
      if EXTRA_BODIES == 7:
         joint = np.zeros((EXTRA_BODIES, 19, 2))
      elif EXTRA_BODIES != 0:
         pad = np.zeros((EXTRA_BODIES, 19, 2))
         joint = np.concatenate((joint, pad), axis=0)
      #print(EXTRA_BODIES, joint.shape)


      camera_name = '{0}/calibration_{0}.json'.format(activity)
      camera_fullname = os.path.join(self.data_dir, camera_name)
      camera_json = json.load(open(camera_fullname,'r'))
      # Cameras are identified by a tuple of (panel#,node#)
      cameras = {(cam['panel'],cam['node']):cam for cam in camera_json['cameras']}

      # Convert data into numpy arrays for convenience
      for k,cam in cameras.items():    
         cam['K'] = np.matrix(cam['K'])
         cam['distCoef'] = np.array(cam['distCoef'])
         cam['R'] = np.matrix(cam['R'])
         cam['t'] = np.array(cam['t']).reshape((3,1))
     
      cam = cameras[(camera_panel, camera_node)]

      joint_name = '{0}/{1}_{2}.json'.format(activity, self.pose_prefix, frame)
      joint_fullname = os.path.join(self.data_dir, joint_name)
      joint_json = json.load(open(joint_fullname,'r'))

      people = len(joint_json['bodies'])
      labels = []
      labels_mark = []
      for body in joint_json['bodies']:
         skel = np.array(body['joints19']).reshape((-1,4)).transpose()
         pt = projectPoints(skel[0:3,:], cam['K'], cam['R'], cam['t'], cam['distCoef'])
         pt_each_joint = np.transpose(pt) # shape: (people, 19, 3)
         
         # Sort from left to right according to Neck's joint
         flag = True
         neck_now = 0
         for i in range(len(labels_mark)):
               if pt_each_joint[0][0] < labels_mark[i]:
                  neck_now = i
                  flag = False
                  break
         if flag:
            neck_now = neck_now + 1
         labels_mark.insert(neck_now, pt_each_joint[0][0])
         labels.insert(neck_now, pt_each_joint)

      labels = np.array(labels) * 216/1080

      

      EXTRA_BODIES = 7 - people
      if EXTRA_BODIES == 7:
         labels = np.zeros((EXTRA_BODIES, 19, 3))
      elif EXTRA_BODIES != 0:
         pad = np.zeros((EXTRA_BODIES, 19, 3))
         labels = np.concatenate((labels, pad), axis=0)
      #print(EXTRA_BODIES, labels.shape)
      #print(image.shape, joint.shape, labels.shape)
      labels_heatmap = self.GenerateHeatmap(labels)
      
      return [image, joint, labels, labels_heatmap]

   def GenerateHeatmap(self, target):
      heatmap = np.zeros((216, 384))
      t1, t2, t3 = target.shape
      target = target.reshape((t1*t2, t3))
      for i in range(t1*t2):
         x = int(target[i][0])
         y = int(target[i][1])
         r = 2
         for dx in range(-r, r+1):
            dy1 = (r-dx)*(-1)
            dy2 = (r-dx) + 1
            for dy in range(dy1, dy2):
               de = abs(dx) + abs(dy)
               if x+dx < 384 and x+dx >= 0 and y+dy < 216 and y+dy >= 0:
                  heatmap[y+dy][x+dx] = target[i][2] - 0.5 * de

      heatmap = heatmap.reshape((1, 216, 384))

      return heatmap