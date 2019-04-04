import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D(target, joint, yl, yr):
    '''
    Visualize in 3D space
    Joint shape: (people, 19, 3)
    '''
    line_idx = [[0,1], [0,2], [0,3], [3,4], [4,5], 
                [0,9], [9,10], [10,11], [2,6], [6,7], [7,8], [2,12], [12,13], [13,14]]
    gorup_idx = [[0, 1, 15, 16, 17, 18],
                 [2, 3, 4, 5, 9, 10, 11],
                 [6, 7, 8, 12, 13, 14]]
    # Show Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim3d(0,400)
    ax.set_ylim3d(yl,yr)
    ax.set_zlim3d(400,0)

    for id1, v in enumerate(target):
        for i in range(len(line_idx)):
            x0 = v[line_idx[i][0]][0]
            x1 = v[line_idx[i][1]][0]
            y0 = v[line_idx[i][0]][1]
            y1 = v[line_idx[i][1]][1]
            z0 = v[line_idx[i][0]][2]
            z1 = v[line_idx[i][1]][2]
            # if int(x0) != 0 or int(x1) != 0 or int(y0) != 0 or int(y1) != 0 or int(z0) != 0 or int(z1) != 0: 
            #    print(z0, z1)
            if (int(x0) == 0 and  int(y0) == 0) or (int(x1) == 0 and  int(y1) == 0):
                continue
            else:
                plt.plot([x0,x1], [z0/100, z1/100], [y0,y1], 'go-')


    for id1, v in enumerate(joint):
        for i in range(len(line_idx)):
            x0 = v[line_idx[i][0]][0]
            x1 = v[line_idx[i][1]][0]
            y0 = v[line_idx[i][0]][1]
            y1 = v[line_idx[i][1]][1]
            z0 = v[line_idx[i][0]][2]
            z1 = v[line_idx[i][1]][2]
            if (int(x0) == 0 and  int(y0) == 0) or (int(x1) == 0 and  int(y1) == 0):
                continue
            else:
                print(z0, z1)
                plt.plot([x0,x1], [z0,z1], [y0,y1], 'ro-')

    plt.savefig('test.jpg')
    plt.show()

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
