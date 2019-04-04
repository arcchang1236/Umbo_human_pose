import os
import json
import numpy as np

# Frame: 1000~8000
idx = ['00_00', '00_05', '00_09', '00_14', '00_15', '00_23', '00_27']

save_dir = '../data/2d_joints/'
joint_prefix = 'hdPose3d_stage1_coco19/body3DScene_'
img_dir = 'hdImgs'

def generate_2d_joint(path):
    
    for s1 in os.listdir(path):
        print(s1)
        f1 = os.path.join(s1,img_dir)
        for s2 in idx:
            f2 = os.path.join(f1, s2)
            #print(f2)
            a = sorted(os.listdir(os.path.join(path, f2)))[1000:6000]
            #print(a[0])
            cnt = 0
            for s3 in a:
                s4 = joint_prefix + s3.split('.')[0].split('_')[2] + '.json'
                f3 = os.path.join(s1,s4)
                camera_panel, camera_node = int(s3.split('.')[0].split('_')[0]), int(s3.split('.')[0].split('_')[1])
                camera_name = '{0}/calibration_{0}.json'.format(s1)
                camera_fullname = os.path.join(path, camera_name)
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

                joint_fullname = os.path.join(path, f3)
                joint_json = json.load(open(joint_fullname,'r'))

                people = len(joint_json['bodies'])
                labels = []
                labels_mark = []
                for body in joint_json['bodies']:
                    skel = np.array(body['joints19']).reshape((-1,4)).transpose()
                    skel = skel[:, :15] 
                    #print(skel.shape)
                    pt = projectPoints(skel[0:3,:], cam['K'], cam['R'], cam['t'], cam['distCoef'])
                    pt = pt[:-1]
                    pt_each_joint = np.transpose(pt) # shape: (people, 19, 3)
                    #print(pt_each_joint.shape)
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
                
                j2D_name = '{0}/{1}.txt'.format(s2, s3.split('.')[0])
                #print(j2D_name)
                save_dir_action = os.path.join(save_dir, s1)
                j2D_fullname = os.path.join(save_dir_action, j2D_name)
                j2D = open(j2D_fullname, 'w')
                for s in labels:
                    #print(s.shape, '!!')
                    for ss in s:
                        str1, str2 = ss[0], ss[1]
                        #print(str1, str2)
                        j2D.write('{0} {1}\n'.format(str1, str2))
                    j2D.write('end\n')
                
                j2D.close()
                cnt += 1
            
            print('   {0}: Generates {1} files'.format(s2, cnt))


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.asarray(R*X + t)

    x[0:2,:] = x[0:2,:]/x[2,:]

    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]

    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]

    return x

if __name__ == '__main__':
    print("Start generating 2d gt joints from gt 3d joints....")
    generate_2d_joint('../data/panoptic')
    print("Generated Done!")
