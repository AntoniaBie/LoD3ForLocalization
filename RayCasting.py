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

#%% Visualization
def visualize(mesh):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
        

def raycasting(cam,rec_mesh,image_folder,GNSS,viewpoint_cam,i):
    
    if rec_mesh == 'LoD-2':
        print('LoD-2 model was chosen.')
        LoD2 = o3d.io.read_triangle_mesh("./data/Mesh/TUM_LoD2.obj", enable_post_processing=False, print_progress=True)
         
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(LoD2)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        
    elif rec_mesh == 'LoD-3': 
        print('LoD-3 model was chosen.')
        LoD2 = o3d.io.read_triangle_mesh("./data/Mesh/LoD2_selec_70_72_81.obj", enable_post_processing=False, print_progress=True)
        LoD3 = o3d.io.read_triangle_mesh("./data/Mesh/TUM_LoD3.obj", enable_post_processing=False, print_progress=True) # only for output and exact coordinate extraction
        windows = o3d.io.read_triangle_mesh(r'C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\LoD3_TUM_all\Window.obj')
        wall = o3d.io.read_triangle_mesh(r'C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\LoD3_TUM_all\RoofSurface.obj')
        roof = o3d.io.read_triangle_mesh(r'C:\Users\anton\OneDrive - TUM\Geodäsie und Geoinformation\A_Bachelorarbeit\Data\obj\newBuildings\LoD3_TUM_all\WallSurface.obj')
        
        LoD2_mesh = o3d.t.geometry.TriangleMesh.from_legacy(LoD2)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(LoD3)
        windows_mesh = o3d.t.geometry.TriangleMesh.from_legacy(windows)
        wall_mesh = o3d.t.geometry.TriangleMesh.from_legacy(wall)
        roof_mesh = o3d.t.geometry.TriangleMesh.from_legacy(roof)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(LoD2_mesh)
        _ = scene.add_triangles(windows_mesh)
        _ = scene.add_triangles(wall_mesh)
        _ = scene.add_triangles(roof_mesh)

        
    else:
        print('Please choose a model before running (LoD-2 or LoD-3)')
        
    #visualize(curr_model) #for .obj  
    
    
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
    print(GNSS_dz)
    #GNSS_dz = -0.3
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
    a = plt.figure(1000),plt.imshow(ans['geometry_ids'].numpy(), vmax=4)
    b = plt.figure(1001),plt.imshow(abs(ans['primitive_normals'].numpy()))
    matplotlib.image.imsave('./images/image_' + 'geometry_ids' + '.jpeg', ans['geometry_ids'].numpy(), vmax=3)
    matplotlib.image.imsave('./images/image_' + 'primitive_normals' + '.jpeg', abs(ans['primitive_normals'].numpy()))

    a = plt.imread('./images/image_' + 'geometry_ids' + '.jpeg')
    b = plt.imread('./images/image_' + 'primitive_normals' + '.jpeg')
    matplotlib.image.imsave(path, a+b)
    # load images of the real world
    #list_images = os.listdir(image_folder)
    
    # save t_hit to save the colors
    #plt.savefig('./plots/t_hit_2.jpeg')
    #plt.savefig(path)
    #plt.imshow(ans['primitive_ids'].numpy())
    
    path_to_images = r'./images/image'
    
    '''
    plt.figure(1000),plt.imshow(ans['primitive_ids'].numpy())
(<Figure size 640x480 with 1 Axes>, <matplotlib.image.AxesImage object at 0x0000016292258DC0>)

IPdb [4]: plt.figure(1000),plt.imshow(ans['geometry_ids'].numpy())
(<Figure size 640x480 with 1 Axes>, <matplotlib.image.AxesImage object at 0x0000016292270DF0>)

IPdb [5]: plt.figure(1000),plt.imshow(ans['geometry_ids'].numpy(), vmax=3)
(<Figure size 640x480 with 1 Axes>, <matplotlib.image.AxesImage object at 0x000001629229D7C0>)

IPdb [6]: plt.figure(1001),plt.imshow(abs(ans['primitive_normals'].numpy()))
(<Figure size 640x480 with 1 Axes>, <matplotlib.image.AxesImage object at 0x0000016292A92D30>)

IPdb [7]: a = plt.figure(1000),plt.imshow(ans['geometry_ids'].numpy(), vmax=3)
Warning: The exclamation mark option is enabled. Please use '!' as a prefix for Pdb commands.

IPdb [8]: b = plt.figure(1001),plt.imshow(abs(ans['primitive_normals'].numpy()))

IPdb [11]: plt.figure(1002),plt.imshow(ans['geometry_ids'].numpy()+abs(ans['primitive_normals'].numpy())[:,:,0])
(<Figure size 640x480 with 1 Axes>, <matplotlib.image.AxesImage object at 0x0000016292D5ED00>)

IPdb [13]: matplotlib.image.imsave('./images/image' + 'test1' + '.jpeg', abs(ans['primitive_normals'].numpy()))

IPdb [14]: matplotlib.image.imsave('./images/image' + 'test1' + '.jpeg', ans['geometry_ids'].numpy(), vmax=3)

IPdb [15]: matplotlib.image.imsave('./images/image' + 'test2' + '.jpeg', abs(ans['primitive_normals'].numpy()))

IPdb [16]: a = plt.imread('./images/image' + 'test2' + '.jpeg')

IPdb [17]: b = plt.imread('./images/image' + 'test1' + '.jpeg')

IPdb [18]: figure(5),plt.imshow(a+b)
*** NameError: name 'figure' is not defined

IPdb [19]: plt.figure(5),plt.imshow(a+b)
(<Figure size 640x480 with 1 Axes>, <matplotlib.image.AxesImage object at 0x0000016293730310>)
    '''
    
    #return koord_LoD, koord_real, coords
    return ans, mesh, path_to_images
