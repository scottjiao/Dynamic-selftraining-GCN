# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:39:49 2019

@author: admin
"""

import os



os.getcwd()
import numpy as np


x=np.load('cora.npz')



x.keys()

x['adj_indices'].shape
x['adj_indptr'].shape
x['adj_shape']


