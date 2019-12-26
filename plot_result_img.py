#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Verify the predicted pose in image

@author: g13

1) load the test_dataset
2) import data into the trained model
3) get R and T
4) plot the 3D axis in image then show them

* locate the key part by searching keyword 'g13' 

"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageDraw
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
import matplotlib as plt
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import copy
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

def main():
    # g13: parameter setting -------------------
    opt.dataset ='linemod'
    opt.dataset_root = './datasets/linemod/Linemod_preprocessed'
    estimator_path = 'trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth'
    refiner_path = 'trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth'
    opt.resume_posenet = estimator_path
    opt.resume_posenet = refiner_path
    dataset_config_dir = 'datasets/linemod/dataset_config'
    output_result_dir = 'experiments/eval_result/linemod'
    bs = 1 #fixed because of the default setting in torch.utils.data.DataLoader
    opt.iteration = 2 #default is 4 in eval_linemod.py
    t1_idx = 0
    t1_total_eval_num = 3
    
    axis_range = 0.1   # the length of X, Y, and Z axis in 3D
    vimg_dir = 'verify_img'
    if not os.path.exists(vimg_dir):
        os.makedirs(vimg_dir)
    #-------------------------------------------
    
    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return
    
    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load(estimator_path))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load(refiner_path))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    print('complete loading testing loader\n')
    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\n\
        length of the testing set: {0}\nnumber of sample points on mesh: {1}\n\
        symmetry object list: {2}'\
        .format( len(test_dataset), opt.num_points_mesh, opt.sym_list))
    
    
    
    #load pytorch model
    estimator.eval()    
    refiner.eval()
    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)
    fw = open('{0}/t1_eval_result_logs.txt'.format(output_result_dir), 'w')

    #Pose estimation
    for j, data in enumerate(testdataloader, 0):
        # g13: modify this part for evaluation target--------------------
        if j == t1_total_eval_num:
            break
        #----------------------------------------------------------------
        points, choose, img, target, model_points, idx = data
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(j))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(j))
            continue
        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

        #if opt.refine_start: #iterative poserefinement
        #    for ite in range(0, opt.iteration):
        #        pred_r, pred_t = refiner(new_points, emb, idx)
        #        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
        
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, opt.num_points, 1)
        pred_c = pred_c.view(bs, opt.num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * opt.num_points, 1, 3)
    
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * opt.num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)
    
        for ite in range(0, opt.iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(opt.num_points, 1).contiguous().view(1, opt.num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2
    
            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
    
            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        # g13: start drawing pose on image------------------------------------
        # pick up image
        print("index {0}: {1}".format(j, test_dataset.list_rgb[j]))
        img = Image.open(test_dataset.list_rgb[j])
        
        # pick up center position by bbox
        meta_file = open('{0}/data/{1}/gt.yml'.format(opt.dataset_root, '%02d' % test_dataset.list_obj[j]), 'r')
        meta = {}
        meta = yaml.load(meta_file)
        which_item = test_dataset.list_rank[j]
        bbx = meta[which_item][0]['obj_bb']
        draw = ImageDraw.Draw(img) 
        
        # draw box (ensure this is the right object)
        draw.line((bbx[0],bbx[1], bbx[0], bbx[1]+bbx[3]), fill=(255,0,0), width=5)
        draw.line((bbx[0],bbx[1], bbx[0]+bbx[2], bbx[1]), fill=(255,0,0), width=5)
        draw.line((bbx[0],bbx[1]+bbx[3], bbx[0]+bbx[2], bbx[1]+bbx[3]), fill=(255,0,0), width=5)
        draw.line((bbx[0]+bbx[2],bbx[1], bbx[0]+bbx[2], bbx[1]+bbx[3]), fill=(255,0,0), width=5)
        
        #get center
        c_x = bbx[0]+int(bbx[2]/2)
        c_y = bbx[1]+int(bbx[3]/2)
        draw.point((c_x,c_y), fill=(255,255,0))
        
        #get the 3D position of center
        cam_intrinsic = np.zeros((3,3))
        cam_intrinsic.itemset(0, test_dataset.cam_fx)
        cam_intrinsic.itemset(4, test_dataset.cam_fy)
        cam_intrinsic.itemset(2, test_dataset.cam_cx)
        cam_intrinsic.itemset(5, test_dataset.cam_cy)
        cam_intrinsic.itemset(8, 1)
        cam_extrinsic = my_mat_final[0:3, :]
        cam2d_3d = np.matmul(cam_intrinsic, cam_extrinsic)
        cen_3d = np.matmul(np.linalg.pinv(cam2d_3d), [[c_x],[c_y],[1]])
        # replace img.show() with plt.imshow(img)
        
        #transpose three 3D axis point into 2D
        x_3d = cen_3d + [[axis_range],[0],[0],[0]]
        y_3d = cen_3d + [[0],[axis_range],[0],[0]]
        z_3d = cen_3d + [[0],[0],[axis_range],[0]]
        x_2d = np.matmul(cam2d_3d, x_3d)
        y_2d = np.matmul(cam2d_3d, y_3d)
        z_2d = np.matmul(cam2d_3d, z_3d)
        
        #draw the axis on 2D
        draw.line((c_x, c_y, x_2d[0], x_2d[1]), fill=(255,255,0), width=5)
        draw.line((c_x, c_y, y_2d[0], y_2d[1]), fill=(0,255,0), width=5)
        draw.line((c_x, c_y, z_2d[0], z_2d[1]), fill=(0,0,255), width=5)

        #g13: show image
        #img.show()
        
        #save file under file 
        img_file_name = '{0}/pred_obj{1}_pic{2}.png'.format(vimg_dir, test_dataset.list_obj[j], which_item)
        img.save( img_file_name, "PNG" )
        img.close()
if __name__ == '__main__':
    main()
    
