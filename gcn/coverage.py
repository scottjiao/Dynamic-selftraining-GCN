# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:52:08 2019

@author: admin
"""
from __future__ import division
from __future__ import print_function
from emailme import *
import time
import numpy as np
from utils_onetime import *
from models import GCN, MLP
import os

path=os.path.abspath(os.path.dirname(__file__))
os.chdir(path)

parameters={}




def neighbors(i,adj):
    """
    ----------------------------------------------------------------------
    return the neighborhood node
    ----------------------------------------------------------------------
    """
    return adj[i].indices
    
def blacken_neighbor(seed,color_list,layer,adj):
    """
    ----------------------------------------------------------------------
    blacken the neighborhood node
    ----------------------------------------------------------------------
    """
    if layer==0:
        return
    for neighbor in neighbors(seed,adj):
        if color_list[neighbor]<=layer:
            color_list[neighbor]=layer
            blacken_neighbor(neighbor,color_list,layer-1,adj)
    
    
    
def compute_coverage(layer,mode='uniform sampleing',x=0.01,class_sample_number=1):
    """
    ----------------------------------------------------------------------
    compute the converage of specified seeds
    ----------------------------------------------------------------------
    """
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask ,labels= \
            load_data(parameters['dataset'],parameters['train_size'] ,standard_split=parameters['standard_split'])

    size=adj.shape[0]
    class_number=labels.shape[1]
    if mode=='uniform sampleing':
        
        seeds_list=np.random.choice(range(size),int(x*size),replace=False)
    elif mode=='sampling per class':
        seeds_list=[]
        class_dict={}
        for label_class in range(class_number):
            class_dict[label_class]=[]
        for item_id in range(size):
            item_label=list(labels[item_id]).index(1)
            class_dict[item_label].append(item_id)
        for label_class in class_dict:
            if len(class_dict[label_class])+1>=class_sample_number:
                seeds_list.extend(np.random.choice(class_dict[label_class],class_sample_number,replace=False))
            else:
                seeds_list.extend(class_dict[label_class])
                
    if parameters['standard_split']==True:
        seeds_list=[]
        for i in range(len(train_mask)):
            if train_mask[i]:
                seeds_list.append(i)
        
        
    color_list=[0 for i in range(size)]
    for seed in seeds_list:
        blacken_neighbor(seed,color_list,layer,adj)
        
    count=0
    for i in color_list:
        if i!=0:
            count+=1
    return count/size
        
"""
----------------------------------------------------------------------
compute the converage
----------------------------------------------------------------------
"""
for dataset in ['cora','citeseer','pubmed']:
    for seed_number in [5,20]:
        for layer in [2,3]:
            parameters['dataset']=dataset
            parameters['train_size']=140
            parameters['standard_split']=False
            coverage=np.mean([compute_coverage(layer,mode='sampling per class',class_sample_number=seed_number) for i in range(100)])
            print('dataset {} with depth {} and seed number {} has {} coverage'.format(
                    dataset,layer,seed_number,coverage))
    
    

'''for dataset in ['cora','citeseer']:
    for layer in [2]:
        parameters['dataset']=dataset
        parameters['train_size']=60
        parameters['standard_split']=True
        print('{} {} label_rate with {} layer is {}'.format(dataset,x,layer,np.mean([compute_coverage(x,layer) for i in range(300)])))   ''' 
   
    
    
    
    
    


