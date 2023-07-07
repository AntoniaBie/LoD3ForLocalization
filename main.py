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

GNSS = np.loadtxt("./data/GNSS/GNSS_cam9_TUM_Arcisstr.txt")
LoD2 = o3d.io.read_point_cloud(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\xyz\LOD2_koord_neu_xyz/LOD2_selection.xyz")
LoD3 = o3d.io.read_point_cloud(r"C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\xyz\LOD3_xyz\LOD3_selection.xyz")
LoD2_mesh = o3d.io.read_triangle_mesh("./data/Mesh/TriangleMesh_LoD2.ply", enable_post_processing=False, print_progress=True)
LoD3_mesh = o3d.io.read_triangle_mesh("./data/Mesh/TriangleMesh_LoD3.ply", enable_post_processing=False, print_progress=True)
LoD2_obj = o3d.io.read_triangle_mesh("./data/Mesh/LoD2.obj", enable_post_processing=False, print_progress=True)
LoD3_obj = o3d.io.read_triangle_mesh("./data/Mesh/LoD3.obj", enable_post_processing=False, print_progress=True)

#ImageFolder = "E:/Bachelorthesis/10_TUM_building34"
ImageFolder = "E:/Bachelorthesis/9_TUM_Arcisstr"
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
camera = [c_9,width,height,dx_9,dy_9,dz_9,roll,pitch,yaw]
GNSS = DataPrep.data_prep(GNSS, camera)

#%% Ray Casting
ans,mesh,path = RC.raycasting(camera,LoD2_obj,ImageFolder)

#%% Get coords and calculate camera position
points_traj = Manager_3DCoords.main(camera, mesh, ImageFolder, path, ans)

#%% Feature Matching for both image-parts (LoD and reality)

#%% Test for image-pairs of the same type
#k1,k2 = FeatureMatching.get_coordinates("E:\Bachelorthesis\Python\images\image11.jpeg","E:\Bachelorthesis\Python\images\image12.jpeg",11)
#k3,k4 = FeatureMatching.get_coordinates("E:/Bachelorthesis/9_TUM_Arcisstr/307t_2021_031121_296895_752_9.jpg","E:/Bachelorthesis/9_TUM_Arcisstr/307t_2021_031121_296896_251_9.jpg",11)
