#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

pick up the worst case and show the predicted pose in image  

@author: g13
"""
#def pick_scenes_id(fn, which_id_per_ten, ):
    
input_id = 1135

def main():
    # draw
    dataset_dir = 'datasets/linemod/Linemod_preprocessed' 
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    total_num_test = 0
    total_num_train = 0
    test_id_list = []
    test_item_list = []
    #load all id of test scenes
    for i, item in enumerate(objlist):
        input_file = '{0}/data/{1}/test.txt'.format(dataset_dir, '%02d' % item)
        with open(input_file,'r') as f:
            counter = 0
            line = f.readline()
            while 1:   
                counter = counter+1
                # dataset pick up each id through 10 lines
                if (counter % 10 == 0):
                    test_id_list.append(line[:-1]) #wipe out '\n'
                    test_item_list.append(item)
                line = f.readline()
                if not line:
                    break
                
    #print('the total length of test dataset is {0},\
    #      first three items are {1}'.format(len(test_id_list),test_id_list[0:3]))        
            
    #load model
    print('id {0} is obj #{1} pic {2}.png'.format(input_id, test_item_list[input_id], test_id_list[input_id]))
    
if __name__ == '__main__':
    main()
    