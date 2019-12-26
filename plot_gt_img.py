
"""
Created on Thu Dec 19 15:22:50 2019

Utilize pose info to draw 3D axis on image

@author: g13
"""
import torch.utils.data as data
from PIL import Image
import yaml
import cv2
import os
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageDraw

# parm setting -----------------------------------
root ='./datasets/linemod/Linemod_preprocessed'
item = 1    
i_th = 22 #which photo
index = '%04d' % (i_th)
meta_file = open('{0}/data/{1}/gt.yml'.format(root, '%02d' % item), 'r')
axis_range = 0.1   # the length of X, Y, and Z axis in 3D
cam_cx = 325.26110 
cam_cy = 242.04899
cam_fx = 572.41140
cam_fy = 573.57043
    
meta = {}
meta = yaml.load(meta_file) #load yaml file in dict type
#file_name = "./datasets/linemod/Linemod_preprocessed/data/01/rgb/0000.png"
#meta[i-th][0]['cam_R_m2c'], the i-th cannnot array/list called by index
file_name = '{0}/data/{1}/rgb/{2}.png'.format(root, '%02d' % item, index)
im = Image.open(file_name)  
draw = ImageDraw.Draw(im) 
bbx = meta[i_th][0]['obj_bb']
#draw.line(x1,y1,x2,y2) while obj_bb(x,y,x_offest, y_offset)
#draw.line((bbx[0],bbx[0]+bbx[2], bbx[1], bbx[1]+bbx[3]), fill=128, width=3)
#draw.line((240,150,240+44,150+58), fill=128, width=3)

draw.line((bbx[0],bbx[1], bbx[0], bbx[1]+bbx[3]), fill=(255,0,0), width=5)
draw.line((bbx[0],bbx[1], bbx[0]+bbx[2], bbx[1]), fill=(255,0,0), width=5)
draw.line((bbx[0],bbx[1]+bbx[3], bbx[0]+bbx[2], bbx[1]+bbx[3]), fill=(255,0,0), width=5)
draw.line((bbx[0]+bbx[2],bbx[1], bbx[0]+bbx[2], bbx[1]+bbx[3]), fill=(255,0,0), width=5)

#draw.arc((bbx[0],bbx[1], bbx[0]+bbx[2], bbx[1]+bbx[3]), fill=(255,0,0), width=5)

#get center
c_x = bbx[0]+int(bbx[2]/2)
c_y = bbx[1]+int(bbx[3]/2)
draw.point((c_x,c_y), fill=(255,255,0))
#print('center is ({0},{1})'.format(c_x,c_y))

#im.show()


#read R, T
target_r = np.resize(np.array(meta[i_th][0]['cam_R_m2c']), (3, 3))
print('the shape of \n{0} is {1} \n'.format(target_r,target_r.shape))

target_t = np.array(meta[i_th][0]['cam_t_m2c'])
target_t = target_t[np.newaxis, :]
print('the shape of {0} is {1} \n'.format(target_t,target_t.shape))

cam_extrinsic = np.concatenate((target_r, target_t.T), axis=1)
print(cam_extrinsic)

cam_intrinsic = np.zeros((3,3))
cam_intrinsic.itemset(0, cam_fx)
cam_intrinsic.itemset(4, cam_fy)
cam_intrinsic.itemset(2, cam_cx)
cam_intrinsic.itemset(5, cam_cy)
cam_intrinsic.itemset(8, 1)
print(cam_intrinsic)

#get center 3D
cam2d_3d = np.matmul(cam_intrinsic, cam_extrinsic)
cen_3d = np.matmul(np.linalg.pinv(cam2d_3d), [[c_x],[c_y],[1]])
print(cen_3d)

#transpose three 3D axis point into 2D
x_3d = cen_3d + [[axis_range],[0],[0],[0]]
y_3d = cen_3d + [[0],[axis_range],[0],[0]]
z_3d = cen_3d + [[0],[0],[axis_range],[0]]

x_2d = np.matmul(cam2d_3d, x_3d)
y_2d = np.matmul(cam2d_3d, y_3d)
z_2d = np.matmul(cam2d_3d, z_3d)
print('x:{0}\ny:{1}\nz:{2}\n'.format(x_2d, y_2d, z_2d))
print('center:({0},{1})\nindex:{2}'.format(c_x, c_y, i_th))
#draw the axis on 2D
draw.line((c_x, c_y, x_2d[0], x_2d[1]), fill=(255,255,0), width=5)
draw.line((c_x, c_y, y_2d[0], y_2d[1]), fill=(0,255,0), width=5)
draw.line((c_x, c_y, z_2d[0], z_2d[1]), fill=(0,0,255), width=5)

im.show()


# unused--------------------------------------------------------------------
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax
