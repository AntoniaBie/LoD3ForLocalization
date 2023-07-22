# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:33:31 2023

@author: anton
"""

import torch
import torchvision.models as models

#torch.save(model, 'model.pth')

#use this snippet in case you want to save your weights and re-use them
model = torch.load("./data/Weights/model.pth",map_location=torch.device('cpu'))

#and this one, in case of a GPU/CPU clash use this snippet:
#model = torch.load("/content/drive/MyDrive/Facade_instance_segmentation/CMP_Facade_base_changed/weights/model.pth", map_location=torch.device('cpu'))

# pick one image from the test set
img, _ = "E:/Bachelorthesis/9_TUM_Arcisstr/307t_2021_031121_296890_251_9.jpg"
#img, _ = dataset_test[4]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])