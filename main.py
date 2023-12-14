# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:38:19 2023

@author: anton
"""
import numpy as np
import RayCasting as RC
import DataPrep
import Manager_3DCoords
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG
#%% settings that are obligatory for usage!
method = ['real images','feature images','sobel','canny','mask','mask and sobel','mask and canny']
image_type = ['real images','segmentation']

#%% settings to be changed by the user
GNSS = np.loadtxt("./data/GNSS/9_Route3.txt")
chosenMethod = method[0]
chosenImageType = image_type[0]

curr_model = 'LoD3' #or 'LoD2'

folder_mask = r'D:/Bachelorthesis/9_Route3_seg_buildings'

if chosenImageType == 'segmentation':
    ImageFolder = "D:/Bachelorthesis/9_Route3_seg"
    
elif chosenImageType == 'real images':
    ImageFolder = "D:/Bachelorthesis/9_Route3"
    
else:
    print('Please select the image type to get the images from the right folder.')

# data preparation, settings of the camera for the real images
pixel_M = 3.4500e-6
c_9 = 0.0082198514/pixel_M
c_10 = 0.0081814370/pixel_M
width = 2464
height = 2056
dx_9 = 2.3104
dy_9 = -0.7484
dz_9 = 0.1080
roll = -101.2307
pitch = -0.0849
yaw = 85.3330

camera = [c_9,width,height,dx_9,dy_9,dz_9,roll,pitch,yaw,pixel_M]
GNSS = DataPrep.data_prep(GNSS, camera)



# ----------------------------------------------------------------------------
# START of the Actual Program
#%% Ray Casting and Coordinate Calculation
points_traj = np.array([0,0,0])
points_traj = points_traj[:,np.newaxis]
i = 1
for img in range(11,12):
    # step 1: create images of the LoD models
    raycast_results,mesh,path = RC.raycasting(camera,curr_model,ImageFolder,GNSS,img)

    # step 2: get the coordinates and calculate the camera position
    points_traj, std = Manager_3DCoords.main(camera,GNSS,mesh,ImageFolder,path,folder_mask,raycast_results,points_traj,chosenMethod,img)
    print("Point " + str(img))
    print('Standard deviation of the current point: ' + str(std[0:3]))
    traj_test = points_traj.T
    GNSS_m = np.sqrt(np.power((GNSS[i,0]-traj_test[i,0]),2) + np.power((GNSS[i,1]-traj_test[i,1]),2) + np.power((GNSS[i,2]-traj_test[i,2]),2))
    print('current deviation from GNSS: ' + str(GNSS_m) + 'm')
    i = i + 1
    
#%% Plotting of the Resulting Trajectory
points_traj = points_traj.T
points_traj = points_traj[1:,:]
fig = plt.figure("Trajectory")
ax = fig.add_subplot(projection='3d')
ax.scatter(GNSS[:,0],GNSS[:,1],GNSS[:,2])
ax.scatter(GNSS[img,0],GNSS[img,1],GNSS[img,2],c='g',marker='o')
ax.plot(GNSS[:,0],GNSS[:,1],GNSS[:,2])
ax.scatter(points_traj[:,0],points_traj[:,1],points_traj[:,2],c='r', marker='o')
ax.plot(points_traj[:,0],points_traj[:,1],points_traj[:,2],c='r')

GNSS_m = np.sqrt(np.power((GNSS[img:img+len(points_traj),0]-points_traj[:,0]),2) + np.power((GNSS[img:img+len(points_traj),1]-points_traj[:,1]),2) + np.power((GNSS[img:img+len(points_traj),2]-points_traj[:,2]),2))

print('current max deviation: ' + str(round(np.amax(GNSS_m),4)) + 'm')
print('current min deviation: ' + str(round(np.amin(GNSS_m),4)) + 'm')