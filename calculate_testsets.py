#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:18:58 2019

@author: g13
"""
import os
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
def pickupset_create(fname, valid_list,ith_per_ten):
    with open(fname) as f:
        for i, l in enumerate(f):
            if i % 10 == ith_per_ten:
                valid_list.append(l)
    return valid_list

#param setting
first_id = 1
last_id = 40
dataset_dir = 'datasets/linemod/Linemod_preprocessed' 
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
total_num_test = 0
total_num_train = 0
for i, item in enumerate(objlist):
    input_file = '{0}/data/{1}/test.txt'.format(dataset_dir, '%02d' % item)
    num_files = file_len(input_file)
    floor_num = num_files//10
    print('obj {0}: num of total test images is {1}\n'.format(item, num_files))
    total_num_test += floor_num
print('total num of test files is {0}'.format(total_num_test))    

for i, item in enumerate(objlist):
    input_file = '{0}/data/{1}/train.txt'.format(dataset_dir, '%02d' % item)
    num_files = file_len(input_file)
    print('obj {0}: num of total train images is {1}\n'.format(item, num_files))
    total_num_train += num_files
print('total num of train files is {0}'.format(total_num_train))   




'''
x = np.arange(0, 10)
y = x**2
#k = y.round(2)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='epoch', ylabel='Avg dis', title='learning curve')
ax.grid()
#plt.ticklabel_format(style='sci', axis='y')
print(x)
print(y)
def callme():
    print("call in with callme()")
    
if __name__ == '__main__':
    callme()
    print("__name__ is ", __name__)
    print("this is main of test.py")
'''