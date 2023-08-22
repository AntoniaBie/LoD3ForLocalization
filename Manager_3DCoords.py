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
import spatialResection
import matplotlib.image
import matplotlib.pyplot as plt
import cv2

def sobelFilterx_o_y(eingabebild):
    maskey = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    maskex = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    #xs = np.shape(eingabebild)
    xs, ys = eingabebild.shape
    xt, yt = maskex.shape  # weil beide Masken die gleiche Dimension haben
    ausgabebildx = np.zeros(eingabebild.shape)
    ausgabebildy = np.zeros(eingabebild.shape)
    B12 = np.zeros([xt, yt])
    # print(maske.shape)
    #for r in range(0, xs[0], 1):
    #    for c in range(0, xs[1], 1):
    for r in range(0, (xs), 1):
        for c in range(0, (ys), 1):
            B12 = eingabebild[r:r+xt, c:c+yt]
            # print(B12.size)
            if B12.size != maskex.size:
                ausgabebildx[r, c] = 0
                ausgabebildy[r, c] = 0
            else:
                ausgabebildx[r, c] = np.sum(maskex*B12)
                ausgabebildy[r, c] = np.sum(maskey*B12)
    return ausgabebildx, ausgabebildy

def main(cam,GNSS,mesh,image_folder_real,image_folder_LoD,folder_mask,ans,traj_points,method,i):
    traj_points_x = []
    traj_points_y = []
    traj_points_z = []
    
    # iterate through all (real and virtual) images and extract coords
        
    if method == 'feature images':
        
        print('Method: feature images')
    
        # extract LoD-model image coordinates
        #print(image_folder_LoD + str(i) + ".jpeg")
        print("__________________________________________")
        print("Now finding features in the single images (compared with themselves)")
        coord1_LoD_2D,_ = FeatureMatching.get_coordinates(image_folder_LoD + str(i) + ".jpeg",image_folder_LoD + str(i) + ".jpeg",1000)
        #print(coord1_LoD)
        #print("Feature Matching für LoD funktioniert")
        
        # extract real world image coordinates
        list_images = os.listdir(image_folder_real)
        #print(image_folder_real+"/"+list_images[i])
        coord1_real_2D,_ = FeatureMatching.get_coordinates(image_folder_real+"/"+list_images[i],image_folder_real+"/"+list_images[i],1001)
        #print("Feature Matching für reality funktioniert")
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using images created on features")
            coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # manual testing
            #a_real = plt.imread(r'./images_sobel/image_real.jpeg')
            #a_real = a_real[:,:,0]
            #a_LoD = plt.imread(r'./images_sobel/image_LoD.jpeg')
            #a_LoD = a_LoD[:,:,0]
            #coord1_LoD_2D_selection = np.array([[827.0,56.0],[873.0,174.0],[1267.0,262.0],[1434.0,103.0],[496.0,5.0],[520.0,27.0],[516.0,207.0],[482.0,182],[612.0,138.0],[631.0,157.0],[613.0,283.0],[599.0,273.0],[795.0,143.0],[1737.0,378]])
            #coord1_real_2D_selection = np.array([[1056.0,505.0],[1100.0,621.0],[1488.0,678.0],[1661.0,503.0],[615.0,306.0],[645.0,341.0],[648.0,536.0],[616.0,520.0],[784.0,509.0],[804.0,526.0],[802.0,669.0],[784.0,662],[1015.0,551.0],[2001.0,567.0]])
            #coord1_LoD_2D_selection = np.concatenate([coord1_LoD_2D_selection[:,1],coord1_LoD_2D_selection[:,0]])
            #coord1_real_2D_selection = np.concatenate([coord1_real_2D_selection[:,1],coord1_real_2D_selection[:,0]])
            
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            print('.')
            
    elif method == 'canny':
        print('Method: canny')
        
        list_images = os.listdir(image_folder_real)
        tmp = plt.imread(image_folder_real+"/"+list_images[i])
        #sobelx,sobely = sobelFilterx_o_y(tmp[:,:,0])
        
        #path = './images_sobel/image_real.jpeg'
        #matplotlib.image.imsave(path, sobelx)
        
        #path2 = './images_sobel/image_real_y.jpeg'
        #matplotlib.image.imsave(path2, sobely)
        
        img_real = cv2.imread(image_folder_real+"/"+list_images[i])
        img_LoD = cv2.imread(image_folder_LoD + str(i) + ".jpeg")
        # Convert to graycsale
        img_real_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
        img_LoD_gray = cv2.cvtColor(img_LoD, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_real_blur = cv2.GaussianBlur(img_real_gray, (3,3), 0) 
        img_LoD_blur = cv2.GaussianBlur(img_LoD_gray, (3,3), 0)
        edges_real = cv2.Canny(image=img_real_blur, threshold1=100, threshold2=100)
        edges_LoD = cv2.Canny(image=img_LoD_blur, threshold1=100, threshold2=100)
        
        #cv2.imshow('Canny Edge Detection', edges_real)
        #cv2.imshow('Canny Edge Detection', edges_LoD)
        
        path_real = './images_sobel/image_real.jpeg'
        path_LoD = './images_sobel/image_LoD.jpeg'
        cv2.imwrite(path_real, edges_real)
        cv2.imwrite(path_LoD, edges_LoD)  

        coord1_LoD_2D,coord1_real_2D = FeatureMatching.get_coordinates(path_real,path_LoD,1003)
        
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using images created by the canny edge detection")
            #coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            #coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D, coord1_real_2D)
       
    
    elif method == 'real images':
        print('Method: real images')
        
        list_images = os.listdir(image_folder_real)
        path_real = image_folder_real+"/"+list_images[i]
        
        path_LoD = image_folder_LoD + str(i) + ".jpeg"

        coord1_LoD_2D,coord1_real_2D = FeatureMatching.get_coordinates(path_real,path_LoD,1003)
        
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using real images")
            #coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            #coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D, coord1_real_2D)
       
    
    elif method == 'sobel':
        print('Method: sobel')
        
        list_images = os.listdir(image_folder_real)
        tmp = plt.imread(image_folder_real+"/"+list_images[i])
        #sobelx,sobely = sobelFilterx_o_y(tmp[:,:,0])
        
        #path = './images_sobel/image_real.jpeg'
        #matplotlib.image.imsave(path, sobelx)
        
        #path2 = './images_sobel/image_real_y.jpeg'
        #matplotlib.image.imsave(path2, sobely)
        
        img_real = cv2.imread(image_folder_real+"/"+list_images[i])
        img_LoD = cv2.imread(image_folder_LoD + str(i) + ".jpeg")
        # Convert to graycsale
        img_real_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
        img_LoD_gray = cv2.cvtColor(img_LoD, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_real_blur = cv2.GaussianBlur(img_real_gray, (3,3), 0) 
        img_LoD_blur = cv2.GaussianBlur(img_LoD_gray, (3,3), 0)
        sobelx_real = cv2.Sobel(src=img_real_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely_real = cv2.Sobel(src=img_real_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelx_LoD = cv2.Sobel(src=img_LoD_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely_LoD = cv2.Sobel(src=img_LoD_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

        sobel_real = sobelx_real + sobely_real
        sobel_LoD = sobelx_LoD + sobely_LoD
        
        #cv2.imshow('Canny Edge Detection', edges_real)
        #cv2.imshow('Canny Edge Detection', edges_LoD)
        
        path_real = './images_sobel/image_real.jpeg'
        path_LoD = './images_sobel/image_LoD.jpeg'
        cv2.imwrite(path_real, sobel_real)
        cv2.imwrite(path_LoD, sobely_LoD)  

        coord1_LoD_2D,coord1_real_2D = FeatureMatching.get_coordinates(path_real,path_LoD,1003)
        
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using images created by the sobel filter")
            #coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            #coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D, coord1_real_2D)
       
    elif method == 'mask':
        print('Method: mask')
        
        list_images = os.listdir(image_folder_real)
        list_mask = os.listdir(folder_mask)
        
        image_folder_real
        
        img1 = plt.imread(image_folder_real+"/"+list_images[i])[:,:,0] * plt.imread(folder_mask+"/"+list_mask[i])
        
        path_real = './images_mask/image_mask.jpeg'
        matplotlib.image.imsave(path_real, img1)
        
        coord1_LoD_2D,coord1_real_2D = FeatureMatching.get_coordinates(path_real,image_folder_LoD + str(i) + ".jpeg",1003)
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using masked images")
            #coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            #coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D, coord1_real_2D)
       
        
    elif method == 'mask and sobel':
        print('Method: mask and sobel')
        
        list_images = os.listdir(image_folder_real)
        list_mask = os.listdir(folder_mask)
        
        image_folder_real
        
        img1 = plt.imread(image_folder_real+"/"+list_images[i])[:,:,0] * plt.imread(folder_mask+"/"+list_mask[i])
        
        path_real = './images_mask/image_mask.jpeg'
        matplotlib.image.imsave(path_real, img1)
        
        img_real = cv2.imread(path_real)
        img_LoD = cv2.imread(image_folder_LoD + str(i) + ".jpeg")
        # Convert to graycsale
        img_real_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
        img_LoD_gray = cv2.cvtColor(img_LoD, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_real_blur = cv2.GaussianBlur(img_real_gray, (3,3), 0) 
        img_LoD_blur = cv2.GaussianBlur(img_LoD_gray, (3,3), 0)
        sobelx_real = cv2.Sobel(src=img_real_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely_real = cv2.Sobel(src=img_real_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelx_LoD = cv2.Sobel(src=img_LoD_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely_LoD = cv2.Sobel(src=img_LoD_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

        sobel_real = sobelx_real + sobely_real
        sobel_LoD = sobelx_LoD + sobely_LoD
        
        #cv2.imshow('Canny Edge Detection', edges_real)
        #cv2.imshow('Canny Edge Detection', edges_LoD)
        
        path_real = './images_sobel/image_real.jpeg'
        path_LoD = './images_sobel/image_LoD.jpeg'
        cv2.imwrite(path_real, sobel_real)
        cv2.imwrite(path_LoD, sobel_LoD)  
        
        coord1_LoD_2D,coord1_real_2D = FeatureMatching.get_coordinates(path_real,path_LoD,1003)
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using masked images and sobel")
            #coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            #coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D, coord1_real_2D)
       

    elif method == 'mask and canny':
        print('Method: mask and canny')
        
        list_images = os.listdir(image_folder_real)
        list_mask = os.listdir(folder_mask)
        
        image_folder_real
        
        img1 = plt.imread(image_folder_real+"/"+list_images[i])[:,:,0] * plt.imread(folder_mask+"/"+list_mask[i])
        
        path_real = './images_mask/image_mask.jpeg'
        matplotlib.image.imsave(path_real, img1)
        
        img_real = cv2.imread(path_real)
        img_LoD = cv2.imread(image_folder_LoD + str(i) + ".jpeg")
        # Convert to graycsale
        img_real_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
        img_LoD_gray = cv2.cvtColor(img_LoD, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_real_blur = cv2.GaussianBlur(img_real_gray, (3,3), 0) 
        img_LoD_blur = cv2.GaussianBlur(img_LoD_gray, (3,3), 0)
        canny_real = cv2.Canny(image=img_real_blur, threshold1=100, threshold2=200)
        canny_LoD = cv2.Canny(image=img_LoD_blur, threshold1=100, threshold2=200) 
        
        path_real = './images_sobel/image_real.jpeg'
        path_LoD = './images_sobel/image_LoD.jpeg'
        cv2.imwrite(path_real, canny_real)
        cv2.imwrite(path_LoD, canny_LoD)  
        
        coord1_LoD_2D,coord1_real_2D = FeatureMatching.get_coordinates(path_real,path_LoD,1003)
        
        if coord1_LoD_2D.size == 1 or coord1_real_2D.size == 1:
            print("No good matches were found, skipping this image-pair")
            traj = traj_points
            
        else:
            
            # match real world and LoD-model 2D images coordinates
            # only for the one image-pair: these camera coords are calculated!
            print("__________________________________________")
            print("Now using masked images and canny")
            #coord1_LoD_2D_selection,coord1_real_2D_selection = matchCoords.main(cam,coord1_LoD_2D,coord1_real_2D)
                
            # extract the 3D coordinates for the features which are matched between real world and LoD-model
            #coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D_selection, coord1_real_2D_selection)
            coord1_LoD_3D, coord1_real_2D_selection = extractCoords.main(ans, mesh, coord1_LoD_2D, coord1_real_2D)
       
    else:
        print('Please select a method to find features in the images.')
    
    camera_pos, std = spatialResection.main(coord1_real_2D_selection,coord1_LoD_3D,cam,GNSS[i,:])
        
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
    
    return traj, std