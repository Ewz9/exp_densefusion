#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw learning the curve of Densefusion

@author: g13
"""

import os
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)

#
#parm setting
first_id = 1
last_id = 40
root_dir = 'experiments/logs/linemod/'
save_img_nm = 'learning_curve_testdata{0}-{1}.png'.format(first_id, last_id)
data_trend = []
xtick_num = 10
# 

for i in range(first_id, last_id+1):
    file_name = '{0}/epoch_{1}_test_log.txt'.format(root_dir, i)
    print(i)
    s = []
    with open(file_name) as f:
        s = f.readlines()[-1]
        x = s.split()
        #print(x[-1])
        f.close()
    data_trend.append(x[-1])
dis_trend = np.array(data_trend)
dis_trend_r = dis_trend.astype(np.float).round(5)
#dis_trend_r = np.arange(1, 21)
epoch_x = np.arange(first_id, last_id+1)
#max_y = float(max(dis_trend))
#min_y = float(min(dis_trend))
#space_size = (min_y+max_y)/10
#dis_trend_r = dis_trend.round(5)

fig, ax = plt.subplots()
ax.plot(epoch_x, dis_trend_r, '-o')
ax.set(xlabel='epoch', ylabel='Avg dis', title='learning curve')
ax.grid()
#gt_linex = np.arange(0, last_id+10)
ax.plot(epoch_x, np.zeros_like(epoch_x), 'r')
new_xticks = np.arange(first_id-1, last_id+1, (last_id-first_id+1)/xtick_num)
new_xticks[0] = 1
plt.xticks(new_xticks)
#plt.xticks(np.linspace(first_id,last_id,1))
#plt.ticklabel_format(style='sci', axis='y')
#plt.yticks(np.arange(0, max_y, step=0.04))
fig.savefig(save_img_nm)
plt.show()
'''
plt.plot(x, y, color='red', linewidth=2.0)
plt.xlabel('epoch')
plt.ylabel('Avg dis')
plt.title('learning curve')
#plt.grid()

plt.show()
plt.savefig("test.png")
'''