#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get the 2d projection of 0000.png point cloud with ground true pose 

@author: g13
"""
import numpy as np
import os
from PIL import Image, ImageDraw
import yaml
import random


def main():
    
    vimg_dir        = 'verify_img'
    dataset_root    = 'datasets/linemod/Linemod_preprocessed'
    item            = 2   
    modify_scale    = 1000
    num_pt_mesh     = 5000
    refine_rotation = [0.1, -0.35, 0.1]
    #loading point cloud, image and pose info
    pt              = {}
    pt[item]        = ply_vtx('{0}/models/obj_{1}.ply'.format(dataset_root, '%02d' % item))
    model_points    = pt[item] / modify_scale
    img_filename    = '{0}/data/{1}/rgb/0000.png'.format(dataset_root, '%02d' % item)
    img = Image.open(img_filename)
    draw = ImageDraw.Draw(img)     
    meta_file = open('{0}/data/{1}/gt.yml'.format(dataset_root, '%02d' % item), 'r')
    meta = {}
    meta = yaml.load(meta_file)
    k_idx = 0
    png_id = 0
    while 1:
        if meta[png_id][k_idx]['obj_id'] == item:
            which_dict = k_idx
            break
        k_idx = k_idx+1

    zeropos_R = np.resize(np.array(meta[0][which_dict]['cam_R_m2c']), (3, 3))
    zeropos_t = np.array(meta[0][which_dict]['cam_t_m2c'])
    zeropos_t.resize([3,1])
    zeropos_t = zeropos_t/modify_scale
    cam_intrinsic = np.resize(np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]),(3,3))
    #reduce the amount of points
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - num_pt_mesh)
    model_points = np.delete(model_points, dellist, axis=0)
    
    #adjust with camera ground true pose R and t
    model_points_x = np.transpose(model_points)
    model_points_x = np.dot(zeropos_R, model_points_x)
    
    #modify the rotation angle
    r = rotation_matrix_3d(refine_rotation)
    model_points_x = np.dot(r, model_points_x)
    
    
    model_points_x = np.add(model_points_x, zeropos_t.repeat(model_points_x.shape[1],axis=1))
    model_points_x = np.transpose(model_points_x)
    #model_points_x = model_points
    for pti in model_points_x:
        pti.transpose()
        pti_2d = np.matmul(cam_intrinsic, pti)
        print('({0},{1})\n'.format(int(pti_2d[0]),int(pti_2d[1])))
        draw.point([int(pti_2d[0]),int(pti_2d[1])], fill=(255,0,0))
        #draw.rectangle([int(pti_2d[0])-1,int(pti_2d[1])-1, int(pti_2d[0]),int(pti_2d[1])], outline=(255,0,255),fill=(255,0,0))
    oimg_filename = '{0}/test_item_{1}_pts.png'.format(vimg_dir,item)
    img.save( oimg_filename, "PNG" )
    img.close()
    img = Image.open(img_filename)
    oimg_filename = '{0}/test_item_{1}_ori.png'.format(vimg_dir,item)
    img.save( oimg_filename, "PNG" )
    img.close()
    print('completed')
def rotation_matrix_3d(deg):
    #degree [x y z]
    k_x = np.resize(np.array([1, 0, 0, 0, np.cos(np.pi*deg[0]/2), -np.sin(np.pi*deg[0]/2), 0, np.sin(np.pi*deg[0]/2), np.cos(np.pi*deg[0]/2)]),(3,3))
    k_y = np.resize(np.array([np.cos(np.pi*deg[1]/2), 0, np.sin(np.pi*deg[1]/2), 0, 1, 0, -np.sin(np.pi*deg[1]/2), 0, np.cos(np.pi*deg[1]/2)]),(3,3))
    k_z = np.resize(np.array([np.cos(np.pi*deg[2]/2), -np.sin(np.pi*deg[2]/2), 0, np.sin(np.pi*deg[2]/2), np.cos(np.pi*deg[2]/2), 0, 0, 0, 1]),(3,3))
    k = np.dot(np.dot(k_x, k_y), k_z)
    return k
def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

if __name__ == '__main__':
    main()