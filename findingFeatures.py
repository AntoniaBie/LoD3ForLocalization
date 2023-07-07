# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:45:23 2023

@author: anton
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates(im):
    image = cv2.imread(im)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    points = []
    for m, n in keypoints:
        points.append(m)
            
    final_points = np.array([0,0]) 
            
    for m in points:
        img1_idx = m.queryIdx
                
        x1, y1 = keypoints[img1_idx].pt
        final_points = np.vstack((final_points, [y1, x1]))  
        
    final_points = final_points[1:]
    
    return final_points
    