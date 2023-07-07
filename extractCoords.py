# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:24:43 2023

@author: anton
"""
import numpy as np

def main(ans,mesh,koord_LoD):
    triangleIDs = ans['primitive_ids'].numpy()
    triangleIDs_hit = []
    vertices_1 = []
    vertices_2 = []
    vertices_3 = []
    coords = []
    triangle_vertices = mesh.triangle.indices.numpy()
    triangle_vertices_positions = mesh.vertex.positions.numpy()
    #number of point which are not on the model, but were detected as features
    n = 0
    for i in range(0,len(koord_LoD)):
        tmp = triangleIDs[round(koord_LoD[i,0]),round(koord_LoD[i,1])] 
        if tmp < len(triangle_vertices): # otherwise =4294967295 which means that a feature was detected which is not on the model, but in the background
            triangleIDs_hit.append(tmp)
            one,two,three = triangle_vertices[tmp,:] # tmp=row=ID
            # order: first all vertices from one triangle, then from the next
            coords.append(triangle_vertices_positions[one,:]) #3 coord per point
            coords.append(triangle_vertices_positions[two,:]) #3 coord per point
            coords.append(triangle_vertices_positions[three,:]) #3 coord per point
            vertices_1.append(one)
            vertices_2.append(two)
            vertices_3.append(three)
        else:
            n += 1
            coords.append([0,0,0]) #3 coord per point
            coords.append([0,0,0]) #3 coord per point
            coords.append([0,0,0]) #3 coord per point
            
    triangleIDs_hit = np.asarray(triangleIDs_hit)
    vertices_1 = np.asarray(vertices_1)
    vertices_2 = np.asarray(vertices_2)
    vertices_3 = np.asarray(vertices_3)
    vertices = np.concatenate((np.array([vertices_1]).T,np.array([vertices_2]).T,np.array([vertices_3]).T),axis=1)

    print(int(n),"points are in the background and therefore not usable.")
    print(int(len(koord_LoD)-n), "points are on the LoD-model and therefore usable.")
    return coords