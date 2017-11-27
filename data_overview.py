#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:53:13 2017

@author: vnguye04
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from netCDF4 import Dataset
import scipy.misc as misc

patch_size = 508   
def import_image(name):
    fh=Dataset(name,mode='r',format="NETCDF4")
#    a=deepcopy(fh.variables)
    a={key:value[:] for key,value in fh.variables.items()}
    fh.close()
    return a

def import_field(name,field='Nrcs'):
    fh=Dataset(name)
    a=fh.variables[field][:]
    fh.close()
    return a

netcdf_dir="/users/local/rfablet/Deep_learning_pollution/Database_netcdf/{}/{}"
output_dir = "/users/local/dnguyen/polluNet2/data_overview/{}/{}"
files_train=os.listdir(netcdf_dir.format("train",""))
files_test=os.listdir(netcdf_dir.format("test",""))
hdf_data = "/users/local/dnguyen/polluNet2/data.hdf5"

with h5py.File(hdf_data,'r') as f:
    for fn in files_train:
        print fn
        fh = import_image(netcdf_dir.format("train",fn))
        pts = f['train/patches/' + fn][:]
        
        x = 10
        im = 1 - 10**(-3*fh['Nrcs'])
        H,W = fh['Nrcs'].shape
        for p in pts:
            r_from = p[0]; c_from = p[1]
            r_to = min(r_from + patch_size -1, H)
            c_to = min(c_from + patch_size -1, W)
            im[r_from:r_from+x,c_from:c_to] = 1
            im[r_to:r_to+x,c_from:c_to] = 1
            im[r_from:r_to,c_from:c_from+x] = 1
            im[r_from:r_to,c_to:c_to+x] = 1
        plt.imsave(output_dir.format('training_set',fn[:-3] + '.png'),im)
        plt.imsave(output_dir.format('training_set',fn[:-3] + '_Mask.png'),fh['Mask']&6)
        

for i in xrange(250):
    x = pts[i,0]
    y = pts[i,1]
    gmf = gmf_list[i]
    dy,dx = gmf.shape
    im[x:x+dx,y:y+dy] = gmf
    
for i in xrange(52):
    x = pts[i,0]
    y = pts[i,1]
    seg = y_pred[i,:,:,2]
#    seg = tmp[i,:,:]
    dy,dx = seg.shape
    mask[x+o:x+o+dx,y+o:y+o+dy] = seg
    