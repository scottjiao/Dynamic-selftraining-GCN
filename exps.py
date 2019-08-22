# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:06:42 2019

@author: admin
"""

import pysnooper



import os
def locate():
    path=os.path.abspath(os.path.dirname(__file__))
    os.chdir(path)
    
locate()
import numpy as np    

import gcn.train_pipeline
locate()
from gcn.utils_onetime import *
locate()

from matplotlib import pyplot as plt




#{'dataset':'cora_full','train_size':1,'exp':random_val_acc_strategy,'epoch':600,},
                  
#@pysnooper.snoop()    
def excute_exp(parameters,repeat_times=100):
    #seed = 2018
    #np.random.seed(seed)
    #tf.set_random_seed(seed)
    
    parameter_information=[]
    counter=statistic_recorder()
    
    for i in range(repeat_times):
        print('{} task {} begins!'.format(parameters,i))
        results=parameters['exp'](inputParameters=parameters)
        print('task {} ends!'.format(i))
        locate()
        print('enhancedTestAccuracy:{} '.format(results['enhancedTestAccuracy']))
        
        counter.insert(results)
        
    #compute mean and variance
    counter.update()
    resultLog=open('result.txt','a')
    resultLog.write('\n'+str_dict(counter)+'of \n'+str_dict(parameters))
    resultLog.close()


    parameter_information.append(parameters)
    return results


'''#______________________________________________
#important: set the experiments and parameters here
#______________________________________________'''
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    visualize_parameters=[]
    for dataset in ['cora','citeseer','pubmed','cora_full']:
        for train_size in [1,3,5,10,20,50]:
            for epoch in [200,600]:
                results_list=[]
                
                for thresh in [0.45,0.6,0.75,0.9]:
                    for parameters in [{'dataset':dataset,'train_size':train_size,
                                        'exp':gcn.train_pipeline.solid_acc_strategy,'thresh':thresh,'epoch':epoch,
                                        'linear_weight':True,'feature_normalize':False},
                                       ]:
                        
                        results=excute_exp(parameters,repeat_times=50)
                        results_list.append(results)
                        
                if visualize_parameters!=[]:
                    visualize_exps(visualize_parameters,results_list)





