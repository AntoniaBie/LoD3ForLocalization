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

def raycasting(cam,rec_mesh,image_folder):
    
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
    
    GNSS = np.loadtxt("./data/GNSS/GNSS_cam9_TUM_Arcisstr.txt")
   
    c = cam[0]
    width = cam[1]
    height = cam[2]
    
    fov = (np.arctan(((width/2)/c))*2)*180/math.pi

    for i in range(13,14):
        GNSS_m = math.sqrt(pow((GNSS[i,0]-GNSS[i+1,0]),2) + pow((GNSS[i,1]-GNSS[i+1,1]),2) + pow((GNSS[i,2]-GNSS[i+1,2]),2))
        GNSS_dz = cam[7]*GNSS_m*math.pi/180
        #print(GNSS_dz)
        GNSS_dy = cam[8]*GNSS_m
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=fov,
            # Calculation of pitch for the viewpoint
            center=[GNSS[i+1,0],GNSS[i+1,1],GNSS[i+1,2]+GNSS_dz], #Punkt, auf den man schaut
            eye=[GNSS[i,0],GNSS[i,1],GNSS[i,2]], #Kameraposition
            up=[0, 0, -360], #Rotation
            width_px=width,
            height_px=height,
            )
        #print(GNSS[i,:])
        ans = scene.cast_rays(rays)
    
        path = './images/image' + str(i) + '.jpeg'
        matplotlib.image.imsave(path, ans['t_hit'].numpy())
        # load images of the real world
        #list_images = os.listdir(image_folder)

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
        path_to_images = r'E:/Bachelorthesis/Python/images/image'
        
    #return koord_LoD, koord_real, coords
    return ans, mesh, path_to_images
