# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:44:41 2023

@author: anton
"""
#Changes: 
#   - GNSS position of the virtual camera
#   -

# GNSS Data
def main(GNSS,camera):
    GNSS[:,0] = GNSS[:,0] + camera[3]
    GNSS[:,1] = GNSS[:,1] + camera[4]
    GNSS[:,2] = GNSS[:,2] + camera[5]
    
    return GNSS