# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:24:43 2023

@author: anton
"""
import numpy as np

def main(ans,mesh,coord_LoD,coord_real):
    triangleIDs = ans['primitive_ids'].numpy()
    bary_coords = ans['primitive_uvs'].numpy()
    triangleIDs_hit = []
    vertices_1 = []
    vertices_2 = []
    vertices_3 = []
    tri_coords = []
    bary_coords_hit = []
    triangle_vertices = mesh.triangle.indices.numpy()
    triangle_vertices_positions = mesh.vertex.positions.numpy()
    coords_X = []
    coords_Y = []
    coords_Z = []
    coord_real_new_x = []
    coord_real_new_y = []
    #number of point which are not on the model, but were detected as features
    n = 0
    for i in range(0,len(coord_LoD)):
        tmp = triangleIDs[round(coord_LoD[i,0]),round(coord_LoD[i,1])] 
        if tmp < len(triangle_vertices): # otherwise =4294967295 which means that a feature was detected which is not on the model, but in the background
            triangleIDs_hit.append(tmp)
            one,two,three = triangle_vertices[tmp,:] # tmp=row=ID
            # order: first all vertices from one triangle, then from the next
            tri_coords.append(triangle_vertices_positions[one,:]) #3 coord per point
            tri_coords.append(triangle_vertices_positions[two,:]) #3 coord per point
            tri_coords.append(triangle_vertices_positions[three,:]) #3 coord per point
            vertices_1.append(one)
            vertices_2.append(two)
            vertices_3.append(three)
            u = bary_coords[round(coord_LoD[i,0]),round(coord_LoD[i,1]),0]
            v = bary_coords[round(coord_LoD[i,0]),round(coord_LoD[i,1]),1]
            s = 1-u-v  
            bary_coords_hit.append(u)
            bary_coords_hit.append(v) 
            bary_coords_hit.append(s) 
            
            # calculate the final coords of the hit points
            tmp2 = u*triangle_vertices_positions[one,:] + v*triangle_vertices_positions[two,:] + s*triangle_vertices_positions[three,:]
            coords_X.append(tmp2[0]) 
            coords_Y.append(tmp2[1]) 
            coords_Z.append(tmp2[2]) 
            
            #save only those image coords which have a corresponding 3D coord
            coord_real_new_x.append(coord_real[i,0])
            coord_real_new_y.append(coord_real[i,1])
        else:
            n += 1
            
    triangleIDs_hit = np.asarray(triangleIDs_hit)
    vertices_1 = np.asarray(vertices_1)
    vertices_2 = np.asarray(vertices_2)
    vertices_3 = np.asarray(vertices_3)
    vertices = np.concatenate((np.array([vertices_1]).T,np.array([vertices_2]).T,np.array([vertices_3]).T),axis=1)
    tri_coords = np.asarray(tri_coords)
    coords_X = np.asarray([coords_X]).T
    coords_Y = np.asarray([coords_Y]).T
    coords_Z = np.asarray([coords_Z]).T
    coords = np.concatenate((coords_X,coords_Y,coords_Z),axis=1)
    
    coord_real_new_x = np.asarray([coord_real_new_x]).T
    coord_real_new_y = np.asarray([coord_real_new_y]).T
    coord_real_new = np.concatenate((coord_real_new_x,coord_real_new_y),axis=1)
    
    print(int(n),"points are in the background and therefore not usable.")
    print(int(len(coord_LoD)-n), "points are on the LoD-model and therefore usable.")
    #print(np.shape(tri_coords))
    return coords,coord_real_new