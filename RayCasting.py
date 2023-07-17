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

def raycasting(cam,rec_mesh,image_folder,GNSS,i):
    
    def visualize(mesh):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
        
    #visualize(rec_mesh) #for .obj
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
    print('pitch: ' + str(GNSS_dz))
    GNSS_dy = cam[8]*GNSS_m
        
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
    
    '''fig = plt.figure("GNSS Viewpoint")
    ax = fig.add_subplot(projection='3d')
    ax.scatter(GNSS[:,0],GNSS[:,1],GNSS[:,2])
    ax.scatter(X[0],X[1],X[2],c='r', marker='o')'''
    '''
    # Feature Matching for both image-parts (LoD and reality)
    koord_LoD, koord_real = FeatureMatching.get_coordinates(image_folder+"/"+list_images[i],path,i)
    #print(koord_LoD.size)
    #print(image_folder+"/"+list_images[i])
    if koord_LoD.size == 1:
        print("No good matches were found, skipping this image-pair")
        coords = []
        continue
    
    triangleIDs = ans['primitive_ids'].numpy()
    triangleIDs_hit = []
    vertices_1 = []
    vertices_2 = []
    vertices_3 = []
    coords = []
    triangle_vertices = mesh.triangle.indices.numpy()
    triangle_vertices_positions = mesh.vertex.positions.numpy()
    for i in range(0,len(koord_LoD)):
        tmp = triangleIDs[round(koord_LoD[i,0]),round(koord_LoD[i,1])] 
        if tmp < len(triangle_vertices):
            triangleIDs_hit.append(tmp)
            one,two,three = triangle_vertices[tmp,:] # tmp=row=ID
            # order: first all vertices from one triangle, then from the next
            coords.append(triangle_vertices_positions[one,:]) #3 coord per point
            coords.append(triangle_vertices_positions[two,:]) #3 coord per point
            coords.append(triangle_vertices_positions[three,:]) #3 coord per point
            vertices_1.append(one)
            vertices_2.append(two)
            vertices_3.append(three)
    triangleIDs_hit = np.asarray(triangleIDs_hit)
    vertices_1 = np.asarray(vertices_1)
    vertices_2 = np.asarray(vertices_2)
    vertices_3 = np.asarray(vertices_3)
    vertices = np.concatenate((np.array([vertices_1]).T,np.array([vertices_2]).T,np.array([vertices_3]).T),axis=1)

    #time.sleep(5)
    '''
    path_to_images = r'./images/image'
    
    #return koord_LoD, koord_real, coords
    return ans, mesh, path_to_images
