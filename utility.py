#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:11:50 2017

@author: vnguye04

"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from skimage.measure import label, regionprops
from os import listdir
import h5py
from scipy import ndimage, signal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



#"""
#SAR-GMF Correlation GAN1 and other
################################################################################
#"""
#
#import matplotlib.pyplot as plt
#import numpy as np
#from keras.models import Model
#from keras.layers.core import Activation, Reshape, Permute
#from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
#from keras.layers.convolutional import Cropping2D
#from keras.layers.normalization import BatchNormalization
#from keras.optimizers import *
#from keras import backend as K
#K.set_image_dim_ordering('th') # Theano dimension ordering in this code
#from keras.layers.merge import concatenate
#from keras.utils import plot_model
#from keras.callbacks import CSVLogger
#import h5py
#import sys
#sys.path.append('../')
#import measures2 as m
#from sklearn.metrics import confusion_matrix
#import os
#
#from pix2pix.networks.generator import UNETGenerator
#from pix2pix.utils import patch_utils
#import log_manager
#import logging
#
#fl = 'GAN1'
#nb_shuffle=1 # Nombre de fois qu'on choisit training_size images
#epochs=10 # Nombre d'époques sur un choix d'images
#
##hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"
#hdf_data="/users/local/dnguyen/polluNet2/data.hdf5"
##hdf_data="/users/local/dnguyen/polluNet2/test.hdf5"
#hdf_result = "/users/local/dnguyen/polluNet2/results/" + fl + ".hdf5"
## Dossier de sauvegarde des poids des modèles
#weight_dir = "/users/local/dnguyen/polluNet2/weights/"
#gen_weight_dir = "/users/local/dnguyen/polluNet2/pix2pix/pix2pix_out/weights/gen_weights_epoch_40.h5"
#
#    
#"""
###########################
#GAN
###########################
#"""
#
#im_width, im_height = 512,512
## input/output channels in image
#input_channels = 1
#output_channels = 1
#
## image dims
#input_img_dim = (input_channels, im_width, im_height)
#output_img_dim = (output_channels, im_width, im_height)
#sub_patch_dim = (512, 512)
#nb_patch_patches, patch_gan_dim = patch_utils.num_patches(output_img_dim=output_img_dim, 
#                                                          sub_patch_dim=sub_patch_dim)
#
## GENERATOR
## Our generator is an AutoEncoder with U-NET skip connections
## ----------------------
#generator_nn = UNETGenerator(input_img_dim=input_img_dim, 
#                             num_output_channels=output_channels)
#generator_nn.load_weights(gen_weight_dir)
#
#def shape_modify(input_image):
#    # Convert (x,508,508) image to (x,512,512) image
#    output_image = np.concatenate((input_image[:,:2,:],
#                                   input_image,
#                                   input_image[:,-2:,:]),
#                                   axis = 1)
#    output_image = np.concatenate((output_image[:,:,:2],
#                                   output_image,
#                                   output_image[:,:,-2:]),
#                                   axis = 2)
#    return output_image
#
#
#fl = 'GAN1'
#hdf_result = '/users/local/dnguyen/polluNet2/results/' + fl + '.hdf5'
#hdf_data = '/users/local/dnguyen/polluNet2/data.hdf5'
#fl0 = 'sar_gmf_ship.py'
#size_out = 316
#o = 96
#N = 508*508
#def normcorrelation2d(in1,in2):
#    in1 = in1 - in1.mean()
#    in2 = in2 - in2.mean()
#    nume = signal.correlate2d(in1,in2,0)[0][0]
#    return nume/N/np.sqrt(in1.var()*in2.var())
#    
#ncorr = dict()
#ncorr['GMF'] = []
#ncorr['GAN'] = [] 
#with h5py.File(hdf_data,'r') as f:
#        sar_set = f['test/Nrcs/testing_images'][:]
#        gmf_set = f['test/GMF/testing_images'][:]
#
#    
#d= gmf_set.shape[0]
#gmf_set2 = shape_modify(gmf_set)
#input_gen = np.zeros((d,im_height,im_width))
#for i_d in xrange(d):
#    im_tmp = gmf_set2[i_d,:,:].reshape(-1,1,im_height,im_width)
#    
#    input_gen[i_d,:,:] = generator_nn.predict(im_tmp).reshape(im_height,im_width)
#input_gen = input_gen[:,2:-2,2:-2] # crop (:,512,512) -> (:,508,508)
#
#
#for i in range(sar_set.shape[0]):
#    print i
#    sar = np.copy(sar_set[i])
#    sar = ndimage.median_filter(sar,11)
#    gmf = np.copy(gmf_set[i])
#    ncorr['GMF'].append(normcorrelation2d(sar,gmf))
#    gen = np.copy(input_gen[i])
#    gen = ndimage.median_filter(gen,11)
#    ncorr['GAN'].append(normcorrelation2d(sar,gen))
#    
#plt.figure()
#plt.subplot(121)
#plt.hist(ncorr['GMF'],100)
#plt.title('GMF')
#plt.xlabel('Normalized 2-D cross-correlation SAR-SAR*')        
#plt.subplot(122)
#plt.hist(ncorr['GAN'],100)
#plt.title('GAN')
#plt.xlabel('Normalized 2-D cross-correlation SAR-SAR*')




"""
SAR-GMF Correlation Histogram 
###############################################################################
"""
#fl = 'GAN1'
#hdf_result = '/users/local/dnguyen/polluNet2/results/' + fl + '.hdf5'
#hdf_data = '/users/local/dnguyen/polluNet2/data.hdf5'
#fl0 = 'sar_gmf_ship.py'
#size_out = 316
#o = 96
#N = 508*508
#def normcorrelation2d(in1,in2):
#    in1 = in1 - in1.mean()
#    in2 = in2 - in2.mean()
#    nume = signal.correlate2d(in1,in2,0)[0][0]
#    return nume/N/np.sqrt(in1.var()*in2.var())
#    
#ncorr = dict()
#ncorr['testing_images'] = []
#ncorr['training_images'] = [] 
#with h5py.File(hdf_data,'r') as f:
#    for test_set in ['testing_images','training_images']:
#        print test_set
#        if test_set == 'testing_images':
#            sar_set = f['test/Nrcs/' + test_set][:]
#            gmf_set = f['test/GMF/' + test_set][:]
#        else:
#            sar_set = f['test/Nrcs/' + test_set][:]
#            gmf_set = f['test/GMF/' + test_set][:]
#        for i in range(sar_set.shape[0]):
#            print i
#            sar = np.copy(sar_set[i])
#            sar = ndimage.median_filter(sar,11)
#            gmf = np.copy(gmf_set[i])
#            ncorr[test_set].append(normcorrelation2d(sar,gmf))
#            
#plt.figure()
#plt.subplot(121)
#plt.hist(ncorr['training_images'],100)
#plt.title('Test set 1')
#plt.xlabel('Normalized 2-D cross-correlation SAR-SAR*')        
#plt.subplot(122)
#plt.hist(ncorr['testing_images'],100)
#plt.title('Test set 2')
#plt.xlabel('Normalized 2-D cross-correlation SAR-SAR*')


"""
# Comparision of GMF and SAR
###############################################################################
"""

#hdf_data = '/users/local/dnguyen/polluNet2/data.hdf5'
#
#with h5py.File(hdf_data,'r') as f:
#    ### TESTING
#    print 'testing_images'
#    gmf = f['test/GMF/testing_images'][:]
#    sar = f['test/Nrcs/testing_images'][:]
#    n_im = sar.shape[0]
#    for i in xrange(n_im):
#        print i
#        for n_ft in xrange(2):
#            sar[i,:,:] = ndimage.median_filter(sar[i,:,:],11)
#    np.save('2median11_testing_images', sar)
#    e = sar - gmf
#    
#    e = e.reshape(n_im,-1)
#    e_mean = e.mean(axis = -1)
#    e_std = e.std(axis = -1)
    
    




#    n_im = f['test/Nrcs/testing_images'].shape[0]
#    vessel_size = []
#    im2 = []
#    br = False
#    for i in xrange(n_im):
#        print i
#        im = f['test/Nrcs/testing_images'][i,:,:]        
#        if (im > 2).any():
#            mask = np.zeros(im.shape,dtype = np.integer)
#            mask[im > im.mean() + 3*im.std()] = 1
#            lb = label(mask)
#            # Region
#            regions = regionprops(lb)
#            for r in regions:
#                if r .area < 100:
#                    vessel_size.append(r.area)
#                if r.area > 100:
#                    im2.append(im)
##            if br:
##                break
#    vessel_size = np.array(vessel_size) #(1373,) 6027
    




"""
# Surface plot
###############################################################################
"""

#hdf_data = '/users/local/dnguyen/polluNet2/data.hdf5'
#
#with h5py.File(hdf_data,'r') as f:
#    ### TRAIN
#    print 'testing_images'
#    gmf = f['test/GMF/testing_images'][:]
#    sar = f['test/Nrcs/testing_images'][:]
#    im = sar[4,:,:]
#    H,W = im.shape
#    gmfim = gmf[4,:,:]
#    idx = np.where(im > 0.4)
#    im2 = ndimage.median_filter(im,11)
#    
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    X = np.arange(0, H, 1)
#    Y = np.arange(0, W, 1)
#    X, Y = np.meshgrid(X, Y)
#    # Plot the surface.
#    surf = ax.plot_surface(X, Y, im, cmap=cm.coolwarm, 
#                           linewidth=0, antialiased=False)
#    # Customize the axes.
##    plt.xlim([0,H])
##    plt.ylim([0,W])
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.show()
    

    



"""
# Statistic of the area of the the pollutions
###############################################################################
"""
netcdf_dir="/users/local/rfablet/Deep_learning_pollution/Database_netcdf/{}/{}"
def import_image(name):
    fh=Dataset(name,mode='r',format="NETCDF4")
#    a=deepcopy(fh.variables)
    a={key:value[:] for key,value in fh.variables.items()}
    fh.close()
    return a
#
#files_train=listdir(netcdf_dir.format("train",""))
#files_test=listdir(netcdf_dir.format("test",""))

p_area = []
count = 0
for fset in ['train','test']:
    flist = listdir(netcdf_dir.format(fset,""))
    for fn in flist: 
        count += 1
        print count, fn
        fh=import_image(netcdf_dir.format(fset,fn))
        # fh = import_image(netcdf_dir.format("train",fn))
        #    - Nrcs (19141, 6336)
        #    - Mask (19141, 6336)
        #    - Incidence angle (6336,)
        #    - Qualite estimation probabilite pollution (764, 253)
        #    - modelWindSpeed (764, 253)
        #    - Densite du traffic (764, 253)
        #    - Bathymetrie (764, 253)
        #    - sarWindDirection (764, 253)
        #    - sarWindSpeed (764, 253)
        #    - modelWindDirection (764, 253)
        #    - Probabilite pollution (764, 253)
        mask = fh['Mask']&32 # 2: LookALike, 4: OilSpill
        mask[mask != 0] = 1
        # Region
        regions = regionprops(mask)
        for r in regions:
             p_area.append(r.area)



#
#plt.figure()
#plt.subplot(121)
#plt.imshow(1-10**(-4*sar))
#plt.title('SAR patch')
#plt.subplot(122)
#plt.imshow(y_true[i,:,:])
#plt.title('Mask')
 






"""
SAR_GMF curve
###############################################################################
"""

#hdf_data = '/users/local/dnguyen/polluNet2/data.hdf5'
#f = h5py.File(hdf_data,'r')   
#training_sar = f['test/Nrcs/training_images'][:]
#training_gmf = f['test/GMF/training_images'][:]
#training_sar = training_sar.reshape(-1)
#training_gmf = training_gmf.reshape(-1)
#m_sar = training_sar.mean()
#std_sar = training_sar.std()
#idx = list(np.where(training_sar<m_sar + 3*std_sar)[0])
#training_sar = training_sar[idx]
#training_gmf = training_gmf[idx]
#
#idx2 = list(np.where(training_sar<0.2))
#training_sar = training_sar[idx2]
#training_gmf = training_gmf[idx2]
#plt.figure()
#plt.hist2d(training_gmf,training_sar,100)
#plt.hold(True)
#plt.plot([0,0.2],[0,0.2],'r')
#plt.title('Test set 1')
#
#
#f = h5py.File(hdf_data,'r')   
#training_sar = f['test/Nrcs/testing_images'][::2]
#training_gmf = f['test/GMF/testing_images'][::2]
#training_sar = training_sar.reshape(-1)
#training_gmf = training_gmf.reshape(-1)
#m_sar = training_sar.mean()
#std_sar = training_sar.std()
#idx = list(np.where(training_sar<m_sar + 3*std_sar)[0])
#training_sar = training_sar[idx]
#training_gmf = training_gmf[idx]
#
#idx2 = list(np.where(training_sar<0.2))
#training_sar = training_sar[idx2]
#training_gmf = training_gmf[idx2]
#plt.figure()
#plt.hist2d(training_gmf,training_sar,100)
#plt.hold(True)
#plt.plot([0,0.2],[0,0.2],'r')
#plt.title('Test set 2')





mask = np.copy(v)
mask = mask[:,:,2]
mask = mask.reshape(52,316,316)
#mask = mask.reshape(52,316,316,3)
#mask = mask[:,:,:,1]
o = 96
x = 10
for i in range(x,x+10):
    plt.figure()
    plt.subplot(131)
    plt.imshow(1-10**(-3*im[i,o:-o,o:-o]),cmap = 'gray')
    plt.title('SAR')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(1-10**(-3*mask[i,o:-o,o:-o]), cmap = 'gray')
    plt.title('Mask')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(1-10**(-0.1*proba[i,:,:]))
    plt.axis('off')
    plt.title('Output \n (in probability)')
    

