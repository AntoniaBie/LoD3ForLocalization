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
import ausgleichung
#import ausgleichung

def main(cam,mesh,image_folder_real,image_folder_LoD,ans):
    traj_points = np.array([])
    # iterate through all (real and virtual) images and extract coords
    for i in range(13,15,2):
        # extract LoD-model image coordinates
        #print(image_folder_LoD + str(i) + ".jpeg")
        coord1_LoD_2D = FeatureMatching.get_coordinates(image_folder_LoD + str(i) + ".jpeg",image_folder_LoD + str(i) + ".jpeg",200)
        #print(coord1_LoD)
        #print("Feature Matching für LoD funktioniert")
    
        # extract real world image coordinates
        list_images = os.listdir(image_folder_real)
        #print(image_folder_real+"/"+list_images[i])
        coord1_real_2D = FeatureMatching.get_coordinates(image_folder_real+"/"+list_images[i],image_folder_real+"/"+list_images[i],201)
        #print("Feature Matching für reality funktioniert")
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            coords = []
            continue
        
        # match real world and LoD-model 2D images coordinates
        # only for the one image-pair: these camera coords are calculated!
        coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
        
        # extract the 3D coordinates for the features which are matched between real world and LoD-model
        coord1_LoD_3D = extractCoords.main(ans, mesh, coord1_LoD_2D_selection)
        
        #camera_pos = ausgleichung.main(coord1_LoD_3D[:,0],coord1_LoD_3D[:,1],coord1_LoD_3D[:,2],coord1_real_2D_selection[:,0],coord1_real_2D_selection[:,1])
       
    #traj_points = np.vstack((traj_points, camera_pos))  
    
    return traj_points