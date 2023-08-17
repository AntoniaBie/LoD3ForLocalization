# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:38:19 2023

@author: anton
"""
import numpy as np
import open3d as o3d
import RayCasting as RC
import FeatureMatching #tmp
import os
from os import listdir
import DataPrep
import Manager_3DCoords
import spacialResection
import matplotlib.pyplot as plt
import math

#%% setting that are obligatory for usage!
method = ['real images','feature images','sobel','canny','mask','mask and sobel','mask and canny']
image_type = ['real images','segmentation']

#%% setting to be changed by the user
GNSS = np.loadtxt("./data/GNSS/9_Route3.txt")
LoD2 = o3d.io.read_point_cloud(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\xyz\LOD2_koord_neu_xyz/LOD2_selection.xyz")
LoD3 = o3d.io.read_point_cloud(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\xyz\LOD3_xyz\LOD3_selection.xyz")
LoD2_mesh = o3d.io.read_triangle_mesh("./data/Mesh/TriangleMesh_LoD2.ply", enable_post_processing=False, print_progress=True)
LoD3_mesh = o3d.io.read_triangle_mesh("./data/Mesh/TriangleMesh_LoD3.ply", enable_post_processing=False, print_progress=True)
LoD2_obj = o3d.io.read_triangle_mesh("./data/Mesh/LoD2.obj", enable_post_processing=False, print_progress=True)
LoD3_obj = o3d.io.read_triangle_mesh("./data/Mesh/LoD3.obj", enable_post_processing=False, print_progress=True)
combi = o3d.io.read_triangle_mesh("./data/Mesh/combi2.obj", enable_post_processing=False, print_progress=True)
LoD2_textured_23 = o3d.io.read_triangle_mesh(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\building_23.obj",print_progress=True)
LoD2u3 = o3d.io.read_triangle_mesh("./data/Mesh/LoD2u3.obj", enable_post_processing=False, print_progress=True)
LoD3_70 = o3d.io.read_triangle_mesh(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\building_70.obj",print_progress=True)
LoD3_72 = o3d.io.read_triangle_mesh(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\building_72.obj",print_progress=True)
LoD3_21 = o3d.io.read_triangle_mesh(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\building_21.obj",print_progress=True)
TUM_LoD2 = o3d.io.read_triangle_mesh("./data/Mesh/TUM_LoD2.obj", enable_post_processing=False, print_progress=True)
TUM_LoD3 = o3d.io.read_triangle_mesh("./data/Mesh/TUM_LoD3.obj", enable_post_processing=False, print_progress=True)
#test = o3d.io.read_triangle_mesh(r"E:\Bachelorthesis\GML models\DEBY_LOD3_4906972.gml")
#ImageFolder = "E:/Bachelorthesis/10_TUM_building34"

#11m deviation, iteration:11
#chosenMethod = method[0]
#chosenImageType = image_type[1]
#ImageFolder = "E:/Bachelorthesis/9_Route3_seg" #weiter unten eintragen
#curr_model = LoD3_70 

chosenMethod = method[3]
chosenImageType = image_type[0]
#curr_model = LoD3_70
#curr_model = TUM_LoD3

curr_model = 'LoD-2' #'LoD-2'

folder_mask = r'E:\Bachelorthesis\9_Route3_seg_buildings'

if chosenImageType == 'segmentation':
    ImageFolder = "E:/Bachelorthesis/9_Route3_seg"
    
elif chosenImageType == 'real images':
    ImageFolder = "E:/Bachelorthesis/9_Route3"
    
else:
    print('Please select the image type to get the images from the right folder.')

#%% data preparation
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
#vector, viewpoint cam:
viewpoint_cam = np.array([0.416341,0.890226,-0.184817])
camera = [c_9,width,height,dx_9,dy_9,dz_9,roll,pitch,yaw,pixel_M]
GNSS = DataPrep.data_prep(GNSS, camera)


#%% Ray Casting and Coordinate Calculation
points_traj = np.array([0,0,0])
points_traj = points_traj[:,np.newaxis]
for img in range(29,33):
    ans,mesh,path = RC.raycasting(camera,curr_model,ImageFolder,GNSS,viewpoint_cam,img)

    # Get coords and calculate camera position
    points_traj, std = Manager_3DCoords.main(camera,GNSS,mesh,ImageFolder,path,folder_mask,ans,points_traj,chosenMethod,img)
    print("Point " + str(img))
    print('Standard deviation of the current point: ' + str(std[0:3]))
    traj_test = points_traj.T
    #GNSS_m = np.sqrt(np.power((GNSS[img,0]-traj_test[img,0]),2) + np.power((GNSS[img,1]-traj_test[img,1]),2) + np.power((GNSS[img,2]-traj_test[img,2]),2))
    #print('current deviation from GNSS: ' + str(GNSS_m) + 'm')

#%% Test spacial resection
#point = spacialResection.main(a,b[1::3,:],c)
points_traj = points_traj.T
points_traj = points_traj[1:,:]
fig = plt.figure("Trajectory")
ax = fig.add_subplot(projection='3d')
ax.scatter(GNSS[:,0],GNSS[:,1],GNSS[:,2])
ax.scatter(GNSS[img,0],GNSS[img,1],GNSS[img,2],c='g',marker='o')
ax.plot(GNSS[:,0],GNSS[:,1],GNSS[:,2])
#for point in range(len(points_traj)):
#    if points_traj[point,0] < 1e+06:
#        ax.scatter(points_traj[point,0],points_traj[point,1],points_traj[point,2],c='r', marker='o')
#        ax.plot(points_traj[point,0],points_traj[point,1],points_traj[point,2],c='r')
ax.scatter(points_traj[:,0],points_traj[:,1],points_traj[:,2],c='r', marker='o')
ax.plot(points_traj[:,0],points_traj[:,1],points_traj[:,2],c='r')



GNSS_m = np.sqrt(np.power((GNSS[img:img+len(points_traj),0]-points_traj[:,0]),2) + np.power((GNSS[img:img+len(points_traj),1]-points_traj[:,1]),2) + np.power((GNSS[img:img+len(points_traj),2]-points_traj[:,2]),2))

print('current max deviation: ' + str(round(np.amax(GNSS_m),4)) + 'm')
print('current min deviation: ' + str(round(np.amin(GNSS_m),4)) + 'm')
    

#%% Test for combining LoD2 and LoD3 after raycasting
#import cv2

#pic1 = cv2.imread('./images/image13.jpeg')

#pic2 = cv2.imread('./images/image14.jpeg')

#pic3 = pic1 + pic2

#plt.imshow(pic3)

#%% Test for image-pairs of the same type
#print("__________________________________________")
#print("Now image matching between two images")
#k1 = FeatureMatching.get_coordinates("E:\Bachelorthesis\Python\LoD3ForLocalization\images\image13.jpeg","E:\Bachelorthesis\Python\LoD3ForLocalization\images\image13.jpeg",11)
#k3 = FeatureMatching.get_coordinates("E:/Bachelorthesis/9_TUM_Arcisstr/307t_2021_031121_296895_752_9.jpg","E:/Bachelorthesis/9_TUM_Arcisstr/307t_2021_031121_296896_251_9.jpg",12)
#k2 = FeatureMatching.get_coordinates("E:\Bachelorthesis\Python\LoD3ForLocalization\images\image13.jpeg","E:/Bachelorthesis/9_TUM_Arcisstr/307t_2021_031121_296896_251_9.jpg",13)
#k4 = FeatureMatching.get_coordinates("E:\Bachelorthesis\Python\images\image20.jpeg","E:\Bachelorthesis\9_TUM_Arcisstr_seg\9_TUM_Arcisstr_seg_TUM\image1.png",20)