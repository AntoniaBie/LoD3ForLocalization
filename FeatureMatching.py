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
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
            
    print("Number of good matches: ", len(good_matches))
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
    return koord_pic1