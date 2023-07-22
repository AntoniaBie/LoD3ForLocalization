# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:15:05 2023

@author: anton
"""
import numpy as np
import FeatureMatching
import extractCoords
import os
from os import listdir
import matchCoords
import spacialResection
#import ausgleichung

def main(cam,mesh,image_folder_real,image_folder_LoD,ans,traj_points,i):
    traj_points_x = []
    traj_points_y = []
    traj_points_z = []
    # iterate through all (real and virtual) images and extract coords
    
    # extract LoD-model image coordinates
    #print(image_folder_LoD + str(i) + ".jpeg")
    print("__________________________________________")
    print("Now finding features in the single images (compared with themselves)")
    coord1_LoD_2D = FeatureMatching.get_coordinates(image_folder_LoD + str(i) + ".jpeg",image_folder_LoD + str(i) + ".jpeg",1000)
    #print(coord1_LoD)
    #print("Feature Matching für LoD funktioniert")
    
    # extract real world image coordinates
    list_images = os.listdir(image_folder_real)
    print(image_folder_real+"/"+list_images[i])
    coord1_real_2D = FeatureMatching.get_coordinates(image_folder_real+"/"+list_images[i],image_folder_real+"/"+list_images[i],1001)
    #print("Feature Matching für reality funktioniert")
        
    if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
        print("No good matches were found, skipping this image-pair")
        camera_pos = []
        
    else:
        
        # match real world and LoD-model 2D images coordinates
        # only for the one image-pair: these camera coords are calculated!
        print("__________________________________________")
        print("Now using images created on features")
        coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
            
        # extract the 3D coordinates for the features which are matched between real world and LoD-model
        coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
        
        # currently [1::3,:] because of approx to test if spacial resection works
        camera_pos = spacialResection.main(coord1_real_2D_selection,coord1_LoD_3D,cam)
      
    traj_points_x.append(camera_pos[0])
    traj_points_y.append(camera_pos[1])
    traj_points_z.append(camera_pos[2])
    
    traj_points_x = np.asarray([traj_points_x]).T
    traj_points_y = np.asarray([traj_points_y]).T
    traj_points_z = np.asarray([traj_points_z]).T
    
    traj = np.concatenate((traj_points_x,traj_points_y,traj_points_z),axis=0)
    traj = np.squeeze(traj)
    traj = traj[:,np.newaxis]
    
    traj = np.concatenate((traj_points,traj),axis = 1)
    
    return traj