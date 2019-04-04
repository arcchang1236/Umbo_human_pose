import cv2
import json
import time
import os
import numpy as np
from utils.utils import projectPoints

camera_panel, camera_node = 0, 0
camera_json = json.load(open('data/panoptic/160224_haggling1/calibration_160224_haggling1.json','r'))
# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'],cam['node']):cam for cam in camera_json['cameras']}

# Convert data into numpy arrays for convenience
for k,cam in cameras.items():    
	cam['K'] = np.matrix(cam['K'])
	cam['distCoef'] = np.array(cam['distCoef'])
	cam['R'] = np.matrix(cam['R'])
	cam['t'] = np.array(cam['t']).reshape((3,1))

cam = cameras[(camera_panel, camera_node)]

joint_json = json.load(open('data/panoptic/160224_haggling1/hdPose3d_stage1_coco19/body3DScene_00002600.json','r'))
people = len(joint_json['bodies'])
labels_2D = []
labels_3D = []
labels_mark = []
for body in joint_json['bodies']:
	skel = np.array(body['joints19']).reshape((-1,4)).transpose()
	skel = skel[:, :15]
	skel = skel[0:3,:]
	#print(skel.shape)
	pt = projectPoints(skel[0:3,:], cam['K'], cam['R'], cam['t'], cam['distCoef'])
	pt = pt[:-1]
	pt = np.transpose(pt) # shape: (people, 19, 3)
	skel = np.transpose(skel)
	labels_2D.append(pt)
	labels_3D.append(skel)

labels_2D = np.array(labels_2D)
labels_3D = np.array(labels_3D)
labels_2D = labels_2D.reshape((labels_2D.shape[0]*labels_2D.shape[1],2,1))
labels_3D = labels_3D.reshape((labels_3D.shape[0]*labels_3D.shape[1],3,1))

print(labels_2D.shape)
print("===")
print(labels_3D.shape)


objPoints = labels_3D[:15,:]
imgPoints = labels_2D[:15,:]


K = np.array([
	[1632.8, 0, 943.554],
	[0, 1628.61, 555.893],
	[0, 0, 1]
])
Kd = np.array([-0.217919,0.178319,0.000243348,0.000605667,0.0566046])

start = time.time()
retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K, Kd)
end = time.time()

print("Time: {}".format(end-start))

ost, jacobian = cv2.Rodrigues(rvec)
print("\nRotation Matrix: ")
print(ost)
print("\nTranslation Matrix: ")
print(tvec)

# print(a)
# print(b)
# print(c)

'''
"name": "00_00",
"type": "hd",
"resolution": [1920,1080],
"panel": 0,
"node": 0,
"K": [
	[1632.8,0,943.554],
	[0,1628.61,555.893],
	[0,0,1]
],
"distCoef": [-0.217919,0.178319,0.000243348,0.000605667,0.0566046],
"R": [
	[0.1335065269,0.03328299897,-0.9904888941],
	[-0.0922107759,0.9955175565,0.02102302156],
	[0.9867487928,0.08852703885,0.1359771426]
],
"t": [
	[22.59674467],
	[126.4923663],
	[283.6143792]
]
'''
