# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 07:37:04 2023

@author: anton
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates(im1,im2,i):
    image1 = cv2.imread(im1)
    image2 = cv2.imread(im2)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
   
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    des1 = np.float32(descriptors1)
    des2 = np.float32(descriptors2)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
            
    #print("Number of good matches: ", len(good_matches))
    koord_pic1 = np.array([0,0])  
    koord_pic2 = np.array([0,0]) 
            
    for match in good_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
                
        x1, y1 = keypoints1[img1_idx].pt
        koord_pic1 = np.vstack((koord_pic1, [y1, x1]))  
        
        x2, y2 = keypoints2[img2_idx].pt     
        koord_pic2 = np.vstack((koord_pic2, [y2, x2]))  
        
    koord_pic1 = koord_pic1[1:]
    koord_pic2 = koord_pic2[1:]
        
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
     
    text = "Figure " + str(i)
    fig = plt.figure(text)           
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.legend(["matched points 1", "matched points 2"])
    plt.title("ORB")
    plt.show()
    
    text = "Figure " + str(i) + "Plot Features 2"
    #image2_color = cv2.applyColorMap(image2, cv2.COLORMAP_BONE)
    fig = plt.figure(text)           
    plt.imshow(image2)
    plt.legend(["matched points"])
    plt.title("ORB")
    plt.plot(koord_pic2[:,1],koord_pic2[:,0],'ro')
    plt.show()
    
    text = "Figure " + str(i) + "Plot Features 1"
    #image1_color = cv2.applyColorMap(image1, cv2.COLORMAP_BONE)
    fig = plt.figure(text)           
    plt.imshow(image1)
    plt.legend(["matched points"])
    plt.title("ORB")
    plt.plot(koord_pic1[:,1],koord_pic1[:,0],'ro')
    plt.show()
    return koord_pic1,koord_pic2