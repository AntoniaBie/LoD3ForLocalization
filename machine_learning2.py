# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:01:01 2023

@author: anton
"""

from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
checkpoint_file = 'mit_b5_20220624-658746d9.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cpu')

# test a single image and show the results
img = mmcv.imread("./images_input/307t_2021_031121_296890_251_9.jpg")
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)