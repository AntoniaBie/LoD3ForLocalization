# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 07:59:46 2023

@author: anton
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import math

#%% Visualization
def visualize(mesh):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
        

def raycasting(cam,rec_mesh,image_folder,GNSS,i):
    
    if rec_mesh == 'LoD2':
        print('LoD2 model was chosen.')
        LoD2 = o3d.io.read_triangle_mesh("./data/Mesh/TUM_LoD2.obj", 
                        enable_post_processing=False, print_progress=True)
         
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(LoD2)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        
    elif rec_mesh == 'LoD3': 
        print('LoD3 model was chosen.')
        # this LoD2 model is here, because a combination of LoD2 and LoD3 was used
        # for the buildings where no LoD3 model was available, a LoD2 model was used instead
        LoD2 = o3d.io.read_triangle_mesh("./data/Mesh/LoD2_selec_70_72_81_underpass.obj", 
                        enable_post_processing=False, print_progress=True)
        # only for output and exact coordinate extraction
        LoD3 = o3d.io.read_triangle_mesh("./data/Mesh/TUM_LoD3.obj", 
                        enable_post_processing=False, print_progress=True) 
        # for different colors the parts of the building have to be added seperately
        windows = o3d.io.read_triangle_mesh(r"./data/Mesh/Window.obj")
        roof = o3d.io.read_triangle_mesh(r"./data/Mesh/RoofSurface.obj")
        wall = o3d.io.read_triangle_mesh(r"./data/Mesh/WallSurface2.obj")
        
        #visualize(windows)
        #visualize(wall)
        #visualize(roof)
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
        print('Please choose a model before running (LoD2 or LoD3)')
        
    #visualize(curr_model) #for .obj  
    
    
    #%% Ray Casting   
    c = cam[0]
    width = cam[1]
    height = cam[2]
    roll = cam[6]/180*math.pi
    pitch = cam[7]/180*math.pi
    yaw = cam[8]/180*math.pi
    
    fov = (np.arctan(((width/2)/c))*2)*180/math.pi

    # take dy into account for the current dataset, depends on the camera 
    # orientation of the real images
    GNSS_m = math.sqrt(pow((GNSS[i,0]-GNSS[i+1,0]),2) + pow((GNSS[i,1]-GNSS[i+1,1]),2) + pow((GNSS[i,2]-GNSS[i+1,2]),2))
    GNSS_dz = pitch*GNSS_m*math.pi/180
        
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov,
        # point to which the virtual camera should be oriented to
        center=[GNSS[i+1,0],GNSS[i+1,1],GNSS[i+1,2]+GNSS_dz], 
        # current position of the camera
        eye=[GNSS[i,0],GNSS[i,1],GNSS[i,2]], 
        # rotation
        up=[0, 0, -360],
        width_px=width,
        height_px=height,
        )
    raycast_results = scene.cast_rays(rays)

    # save the created images to a folder
    path = './images/image' + str(i) + '.jpeg'
    # approximation with abs(), but the value is not important
    b = plt.figure(1001),plt.imshow(abs(raycast_results['primitive_normals'].numpy()))
    if rec_mesh == 'LoD2':
        a = plt.figure(1000),plt.imshow(raycast_results['geometry_ids'].numpy(), 
                                        vmax=1)
    else:
        a = plt.figure(1000),plt.imshow(raycast_results['geometry_ids'].numpy(), 
                                        vmax=4)
    matplotlib.image.imsave('./images/image_' + 'geometry_ids' + '.jpeg', 
                            raycast_results['geometry_ids'].numpy(), vmax=3)
    matplotlib.image.imsave('./images/image_' + 'primitive_normals' + '.jpeg', 
                            abs(raycast_results['primitive_normals'].numpy()))

    a = plt.imread('./images/image_' + 'geometry_ids' + '.jpeg')
    b = plt.imread('./images/image_' + 'primitive_normals' + '.jpeg')
    matplotlib.image.imsave(path, a+b)
    
    path_to_images = r'./images/image'
    
    return raycast_results, mesh, path_to_images
