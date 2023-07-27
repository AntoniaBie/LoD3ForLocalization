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
method = ['feature images','sobel']
image_type = ['corresponding image-pairs','segmentation']

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

#ImageFolder = "E:/Bachelorthesis/10_TUM_building34"


chosenMethod = method[0]
chosenImageType = image_type[1]
curr_model = LoD2_obj

if chosenImageType == 'segmentation':
    ImageFolder = "E:/Bachelorthesis/9_TUM_Arcisstr_seg/9_TUM_Arcisstr_seg_TUM"
    
elif chosenImageType == 'corresponding image-pairs':
    ImageFolder = "E:/Bachelorthesis/9_TUM_Arcisstr"
    
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

#%% Visualization
def visualize(mesh):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
        
visualize(curr_model) #for .obj

#%% Ray Casting and Coordinate Calculation
points_traj = np.array([0,0,0])
points_traj = points_traj[:,np.newaxis]
for i in range(20,21):
    ans,mesh,path = RC.raycasting(camera,curr_model,ImageFolder,GNSS,viewpoint_cam,i)

    # Get coords and calculate camera position
    points_traj = Manager_3DCoords.main(camera,GNSS,mesh,ImageFolder,path,ans,points_traj,chosenMethod,i)

#%% Test spacial resection
#point = spacialResection.main(a,b[1::3,:],c)
points_traj = points_traj.T
points_traj = points_traj[1:,:]
fig = plt.figure("Trajectory")
ax = fig.add_subplot(projection='3d')
ax.scatter(GNSS[:,0],GNSS[:,1],GNSS[:,2])
ax.scatter(GNSS[0,0],GNSS[0,1],GNSS[0,2],c='g',marker='o')
ax.scatter(points_traj[0:,0],points_traj[0:,1],points_traj[0:,2],c='r', marker='o')

GNSS_m = np.sqrt(np.power((GNSS[0:len(points_traj),0]-points_traj[:,0]),2) + np.power((GNSS[0:len(points_traj),1]-points_traj[:,1]),2) + np.power((GNSS[0:len(points_traj),2]-points_traj[:,2]),2))

print('current max deviation: ' + str(round(np.amax(GNSS_m),4)) + 'm')
print('current min deviation: ' + str(round(np.amin(GNSS_m),4)) + 'm')
    

#%% Test roll, pitch, yaw
GNSS1 = GNSS[0,:]
GNSS2 = GNSS[1,:]
GNSS_m = math.sqrt(pow((GNSS[1,0]-GNSS[2,0]),2) + pow((GNSS[1,1]-GNSS[2,1]),2) + pow((GNSS[1,2]-GNSS[2,2]),2))
roll = roll/180*math.pi
pitch = pitch/180*math.pi
yaw = yaw/180*math.pi
r11 =  math.cos(pitch) *  math.cos(yaw)
r12 = - math.cos(pitch) *  math.sin(yaw)
r13 =  math.sin(pitch)
r21 =  math.cos(roll) *  math.sin(yaw) +  math.sin(roll) *  math.sin(pitch) *  math.cos(yaw)
r22 =  math.cos(roll) *  math.cos(yaw) -  math.sin(roll) *  math.sin(pitch) *  math.sin(yaw)
r23 = - math.sin(roll) *  math.cos(pitch)
r31 =  math.sin(roll) *  math.sin(yaw) -  math.cos(roll) *  math.sin(pitch) *  math.cos(yaw)
r32 =  math.sin(roll) *  math.cos(yaw) +  math.cos(roll) *  math.sin(pitch) *  math.sin(yaw)
r33 =  math.cos(roll) *  math.cos(pitch)

R1 =  np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

X_test1 = GNSS2-GNSS1 + np.matmul(R1,GNSS1)

r11 = 1
r12 = 0
r13 = 0
r21 = 0
r22 = math.cos(roll) 
r23 = math.sin(roll) 
r31 = 0
r32 = -math.sin(roll)
r33 = math.cos(roll)

Rx =  np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

r11 = math.cos(pitch)
r12 = 0
r13 = -math.sin(pitch)
r21 = 0
r22 = 1
r23 = 0
r31 = math.sin(pitch)
r32 = 0
r33 = math.cos(pitch)

Ry =  np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

r11 = math.cos(yaw)
r12 = -math.sin(yaw)
r13 = 0
r21 = math.sin(yaw)
r22 = math.cos(yaw)
r23 = 0
r31 = 0
r32 = 0
r33 = 1

Rz =  np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

R = np.matmul(Rz,Ry)
R = np.matmul(R,Rx)

X_test1 = GNSS2-GNSS1 + np.matmul(R,GNSS2)

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