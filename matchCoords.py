# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:35:22 2023

@author: anton
"""

# match the coordinates of two different images to one feature image
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import cv2

def main(camera,coord_LoD,coord_real):
    print("Now creating feature images.")
    #images in path are deleted after each for-loop
    width = camera[1]
    height = camera[2]
    
    # extract the first virtual feature-image (LoD)
    virt_pic1 = np.array(np.zeros((height,width)), dtype=np.uint8)
    
    coord_LoD = np.around(coord_LoD,decimals=0).astype(np.int32)
    
    for i in range(0,len(coord_LoD)):
        virt_pic1[coord_LoD[i,0],coord_LoD[i,1]] = 255
    #plt.imshow(virt_pic1)
    
    path = './images_features/image1_LoD.jpeg'
    matplotlib.image.imsave(path, virt_pic1)
    #plt.imshow(virt_pic1)
    
    # extract the second virtual feature-image (reality)
    virt_pic2 = np.array(np.zeros((height,width)), dtype=np.uint8)
    
    coord_real = np.around(coord_real,decimals=0).astype(np.int32)
    
    for i in range(0,len(coord_real)):
        virt_pic2[coord_real[i,0],coord_real[i,1]] = 255
    #plt.imshow(virt_pic2)
    
    path = './images_features/image2_real.jpeg'
    matplotlib.image.imsave(path, virt_pic2)
    
    # Match features again on the feature-images
    
    path1 = r'./images_features/image1_LoD.jpeg'
    path2 = r'./images_features/image2_real.jpeg'
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
     
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
     
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
      
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
                
    print("Number of good matches in the feature-images: ", len(good_matches))
    koord_pic1 = np.array([0,0])  
    koord_pic2 = np.array([0,0]) 
               
    for match in good_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
                 
        x1, y1 = keypoints1[img1_idx].pt
        koord_pic1 = np.vstack((koord_pic1, [y1, x1]))  
         
        x2, y2 = keypoints2[img2_idx].pt     
        koord_pic2 = np.vstack((koord_pic2, [y2, x2]))  
            
    imageCoords_match_LoD = koord_pic1[1:]
    imageCoords_match_real = koord_pic2[1:]
            
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    path = './images_features/image3_matchingResult.jpeg'
    matplotlib.image.imsave(path, cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    
    return imageCoords_match_LoD,imageCoords_match_real