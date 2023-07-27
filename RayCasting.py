# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 07:59:46 2023

@author: anton
"""

import open3d as o3d
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import math
import FeatureMatching
import os
from os import listdir
import time
import math

def raycasting(cam,rec_mesh,image_folder,GNSS,viewpoint_cam,i):
    
    #o3d.visualization.draw_geometries([pcd, rec_mesh]) #for .ply
    #print(rec_mesh)
    #print('Vertices:')
    #print(np.asarray(rec_mesh.vertices))
    #print('Triangles:')
    #print(np.asarray(rec_mesh.triangles))
    
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(rec_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    
    #%% Ray Casting
    
    # = np.loadtxt("./data/GNSS/GNSS_cam9_TUM_Arcisstr.txt")
   
    c = cam[0]
    width = cam[1]
    height = cam[2]
    roll = cam[6]
    pitch = cam[7]
    yaw = cam[8]
    roll = roll/180*math.pi
    pitch = pitch/180*math.pi
    yaw = yaw/180*math.pi
    
    fov = (np.arctan(((width/2)/c))*2)*180/math.pi

    # test with dy
    GNSS_m = math.sqrt(pow((GNSS[i,0]-GNSS[i+1,0]),2) + pow((GNSS[i,1]-GNSS[i+1,1]),2) + pow((GNSS[i,2]-GNSS[i+1,2]),2))
    GNSS_dz = cam[7]*GNSS_m*math.pi/180
    vp = viewpoint_cam*GNSS_m
    print(vp)
    print(GNSS[i,:] + vp)
    #vp = GNSS[i,:] + vp
    #print(vp)
    #print('pitch: ' + str(GNSS_dz))
    #GNSS_dy = cam[8]*GNSS_m
        
    # test with helmert transformation
    '''GNSS1 = GNSS[i,:]
    GNSS2 = GNSS[i+1,:]
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
    
    X = GNSS2-GNSS1 + np.matmul(R,GNSS2)
    print(X)'''
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov,
        # Calculation of pitch for the viewpoint
        center=[GNSS[i+1,0],GNSS[i+1,1],GNSS[i+1,2]+GNSS_dz], #Punkt, auf den man schaut
        #center=[X[0],X[1],X[2]],
        #center=[vp[0],vp[1],vp[2]],
        #center=[GNSS[i+1,0]+vp[0],GNSS[i+1,1]+vp[1],GNSS[i+1,2]+vp[2]],
        eye=[GNSS[i,0],GNSS[i,1],GNSS[i,2]], #Kameraposition
        up=[0, 0, -360], #Rotation
        width_px=width,
        height_px=height,
        )
    #print(GNSS[i,:])
    ans = scene.cast_rays(rays)

    path = './images/image' + str(i) + '.jpeg'
    # imsave does not save the colors?? plt.savefig does! But axes are visible then...
    #matplotlib.image.imsave(path, ans['t_hit'].numpy())
    
    # approximation with abs(), but the value is not important
    matplotlib.image.imsave(path, abs(ans['primitive_normals'].numpy()))
    # load images of the real world
    #list_images = os.listdir(image_folder)
    
    # save t_hit to save the colors
    #plt.savefig('./plots/t_hit_2.jpeg')
    #plt.savefig(path)
    
    path_to_images = r'./images/image'
    
    #return koord_LoD, koord_real, coords
    return ans, mesh, path_to_images
