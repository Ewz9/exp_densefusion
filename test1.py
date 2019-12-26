#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw learning the curve of Densefusion

@author: g13
"""

import os
import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)

#
#parm setting
first_id = 1
last_id = 84
root_dir = 'experiments/logs/linemod/'
data_trend = []
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
y = np.array(data_trend)
x = np.arange(first_id, last_id+1)

plt.plot(x, y, color='red', linewidth=1.0, linestyle='--')
plt.xlabel('epoch')
plt.ylabel('Avg dis')
plt.title('learning curve')
plt.grid()

plt.show()
plt.savefig("test.png")
