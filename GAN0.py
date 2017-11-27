#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:46:35 2017

@author: vnguye04
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers.core import Activation, Reshape, Permute
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras import backend as K
K.set_image_dim_ordering('th') # Theano dimension ordering in this code
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.callbacks import CSVLogger
import h5py
import sys
sys.path.append('../')
import measures as m
from sklearn.metrics import confusion_matrix
import os

from pix2pix.networks.generator import UNETGenerator
from pix2pix.utils import patch_utils
import log_manager
import logging

fl = 'GAN1_2'
fl0 = 'sar_gmf_ship.py'
log_manager.new_log(fl)
csv_logger = CSVLogger('./logs/' + fl + '.csv', append=True, separator=';')
nb_shuffle=1 # Nombre de fois qu'on choisit training_size images
epochs=10 # Nombre d'époques sur un choix d'images

#hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"
hdf_data="/users/local/dnguyen/polluNet2/data.hdf5"
#hdf_data="/users/local/dnguyen/polluNet2/test.hdf5"
hdf_result = "/users/local/dnguyen/polluNet2/results/" + fl + ".hdf5"
# Dossier de sauvegarde des poids des modèles
weight_dir = "/users/local/dnguyen/polluNet2/weights/"
gen_weight_dir = "/users/local/dnguyen/polluNet2/pix2pix/pix2pix_out/weights/gen_weights_epoch_40.h5"
"""
###########################
# Paramètres du modèle
###########################
"""

# Poids des différentes classes pour le loss
######
weight_pollution = 1000.
weight_land = 5.
weight_boats = 1000.
weight_sea = 1.
reload_weights = False

# Paramètres du réseau
######
width = 508 # Taille des images en entrée
height = 508
nbClass = 3 # Nombre de classes à distinguer
kernel = 5 # Taille du kernel utilisé pour les convolutions
depth = 4 # Profondeur du réseau U-net (en nombre de max-pooling/upsampling)
nb_conv = 1 # Nombre de convolutions après chaque max-pooling/upsampling
nb_conv_out = 2 # Nombre de convolutions en sortie
activation = "relu" # Activation
# Dropout dans la première et la deuxième moitié du réseau (résolution descendant et montante)
dropout_down = 0.3
dropout_up = 0.3
channels_max = 48 # Nombre max de channels utilisés dans le réseau
training_size = 2000 # Nombre de patch utilisés
batch_size=12 # Batch size (Préferer un grand batch_size, mais si il est trop grand le réseau ne rentre plus en mémoire)

# Fonction qui donne le nombre de channels à utiliser en fonction de la profondeur (profondeur décroissante: elle vaut 0 en bas du réseau)
def channels(depth, channels_max=channels_max):
    return channels_max
#    return channels_max / 2**depth
optimizer=adadelta() # Optimizer


"""
##########################
GAN
##########################
"""

im_width, im_height = 512,512
# input/output channels in image
input_channels = 1
output_channels = 1

# image dims
input_img_dim = (input_channels, im_width, im_height)
output_img_dim = (output_channels, im_width, im_height)
sub_patch_dim = (512, 512)
nb_patch_patches, patch_gan_dim = patch_utils.num_patches(output_img_dim=output_img_dim, 
                                                          sub_patch_dim=sub_patch_dim)

# GENERATOR
# Our generator is an AutoEncoder with U-NET skip connections
# ----------------------
generator_nn = UNETGenerator(input_img_dim=input_img_dim, 
                             num_output_channels=output_channels)
generator_nn.load_weights(gen_weight_dir)


"""
###########################
# Génération du réseau
###########################
"""

# Nombre de pixels à rogner sur les images
def crop():
    c=kernel-1
    for i in range(depth):
        yield c
        c+=(kernel-1)*nb_conv
        c*=2

size_out = width - 2 * list(crop())[-1] - (kernel-1) * (nb_conv_out + 2 * nb_conv)

def unet_layers(x,depth,channels_max):
    """
    Construit récursivement les couches U-net à partir de la couche x.

    Inputs: 
        x: keras layer
        depth: profondeur du réseau
        channels_max: 
    """
    if depth == 0:
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_down)(x)
        x=Conv2D(channels(depth,channels_max)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Dropout(dropout_down)(x)
    else:
        for i in range(nb_conv):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_down)(x)
    if depth>0:
        y=MaxPooling2D(pool_size=(2,2))(x)
        y=unet_layers(y,depth-1,channels_max)
        y=UpSampling2D(size=(2,2))(y)
        x=Cropping2D(c.next(),data_format='channels_first')(x)
        x=concatenate([x,y],axis=1)
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_up)(x)
        x=Conv2D(channels(depth,channels_max)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Dropout(dropout_up)(x)
    return x

"""
###########################
# Construction du réseau
###########################
"""

sar_input = Input(shape=(height, width))
gmf_input = Input(shape=(height, width))
ship_input = Input(shape=(height,width))

sar = Reshape((1,height,width))(sar_input) #<tf.Tensor shape=(?, 1, 508, 508) dtype=float32>
gmf = Reshape((1,height,width))(gmf_input) 
ship = Reshape((1,height,width))(ship_input) 


x = concatenate([sar,gmf,ship],axis=1)
c = crop() # call this function before any unet_layers() use.
x = unet_layers(x,depth,channels_max) # <tf.Tensor shape=(?, 24, 324, 324) dtype=float32>
for i in range(nb_conv_out):
    x = Conv2D(channels(depth),kernel,padding='valid',data_format='channels_first')(x)
    x = BatchNormalization(axis = 1)(x)
x = Conv2D(nbClass,1,padding='valid',data_format='channels_first')(x)
x = Reshape((nbClass,-1))(x)
x = BatchNormalization(axis = 1)(x)
x = Permute((2,1))(x)
x = Activation('softmax')(x)

unet = Model([sar_input,gmf_input,ship_input],x)

# print unet.summary()
# exit()
# print size_out
unet.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'],
                sample_weight_mode="temporal")
plot_model(unet, to_file= '/users/local/dnguyen/polluNet2/logs/' + fl + '.png')


"""
###########################
# Chargement des masques
###########################
"""
logging.info(fl)
logging.info('nb_shuffle: ' + str(nb_shuffle))
logging.info('dropout_down: ' + str(dropout_down))
logging.info('dropout_up: ' + str(dropout_up))
logging.info('nb_epoch: ' + str(epochs))
logging.info('kernel_size: ' + str(kernel))
logging.info('weight_pollution: ' + str(weight_pollution))
logging.info('weight_ship: ' + str(weight_boats))
logging.info('weight_sea: ' + str(weight_sea))
logging.info('Begin loading the masks...')


b=size_out*size_out
o=(width-size_out)/2
with h5py.File(hdf_data,"r") as f:
    nb_train = len(f["train/Nrcs"])
    nb_set1 = len(f["test/Nrcs/training_images"])
    nb_set2 = len(f["test/Nrcs/testing_images"])

logging.info('Finish loading the masks!')

"""
###########################
# Apprentissage
###########################
"""
logging.info('Begin learning...')
if os.path.isfile(weight_dir+fl):
    unet.load_weights(weight_dir+fl)

def shape_modify(input_image):
    # Convert (x,508,508) image to (x,512,512) image
    output_image = np.concatenate((input_image[:,:2,:],
                                   input_image,
                                   input_image[:,-2:,:]),
                                   axis = 1)
    output_image = np.concatenate((output_image[:,:,:2],
                                   output_image,
                                   output_image[:,:,-2:]),
                                   axis = 2)
    return output_image


with h5py.File(hdf_data,"r") as f:
    logging.info('nb_shuffle: ' + str(nb_shuffle))
    for i in range(nb_shuffle):
        l = m.randl(training_size,nb_train,m.l4_train) # list, len = 1898

        #    print len(l), nb_train,len(l), f["weights/"+fl].shape
        # print l[-1], nb_train
        # Weights
        w=f["weights/"+fl0][l] # np.array, shape = (1898, 99856)
        # Train nrcs
        train_nrcs=f["train/Nrcs"][l] # np.array, shape = (1898, 508, 508)
        # Input gmf
        input_gmf = shape_modify(f["train/GMF"][l])
        # GEN
        d,_,_ = input_gmf.shape
        input_gen = np.zeros((d,im_height,im_width))
        for i_d in xrange(d):
            im_tmp = input_gmf[i_d,:,:].reshape(-1,1,im_height,im_width)
            input_gen[i_d,:,:] = generator_nn.predict(im_tmp).reshape(im_height,im_width)
        input_gen = input_gen[:,2:-2,2:-2] # crop (:,512,512) -> (:,508,508)
        # Input ship
        input_ship = f["masks/train/"+fl0][l][...,1] #[...,1]: ship
        w[w==1.]=weight_boats
        w[w==0.]=weight_sea   
        w[w==2.]=weight_pollution

        # Train mask
        # train_mask is np.array, shape(1898, 316, 316, 3) -> (1898, 99856, 3)
        train_mask=f["masks/train/"+fl0][l][:,o:-o,o:-o,:].reshape((-1,size_out*size_out,nbClass))

        unet.fit([np.sqrt(train_nrcs),np.sqrt(input_gen), input_ship],
                  train_mask,shuffle=True,
                  verbose=1,batch_size=batch_size,
                  epochs=epochs,
                  sample_weight=w,
                  callbacks=[csv_logger])

        unet.save_weights(weight_dir+fl)

        del input_gmf
        del train_mask
        del train_nrcs


logging.info('Finish learning!')

"""
###########################
# Test sur les jeux de test
###########################
"""
logging.info('Begin testing...')

# Test with patches extracted from the different SAR images
with h5py.File(hdf_data,"r") as f:    
    test_nrcs=f["test/Nrcs/testing_images/"][:] # HDF5 dataset, shape = (5065, 508, 508)
    input_gmf = shape_modify(f["test/GMF/testing_images"][:]) # HDF5 dataset, shape = (5065, 508, 508)
    input_ship = f["masks/testing_images/"+fl0][:][...,1]  # np.array, shape = (5065, 508, 508)
    # GEN
    d,_,_ = input_gmf.shape
    input_gen = np.zeros((d,im_height,im_width))
    for i_d in xrange(d):
        im_tmp = input_gmf[i_d,:,:].reshape(-1,1,im_height,im_width)
        input_gen[i_d,:,:] = generator_nn.predict(im_tmp).reshape(im_height,im_width)
    input_gen = input_gen[:,2:-2,2:-2] # crop (:,512,512) -> (:,508,508)
    v=unet.predict([np.sqrt(test_nrcs),np.sqrt(input_gen),input_ship],verbose=1,batch_size=16) #np.array, shape = (5044, 99856, 3) 99856 = 316**2



with h5py.File(hdf_result,"a") as f:
    f.require_dataset("results/testing_images/"+fl,
                      shape=(nb_set2,size_out,size_out,nbClass),
                      dtype='f4',
                      exact=False)
    f["results/testing_images/"+fl][:]=v.reshape(-1,size_out,size_out,nbClass) #shape = (5044, 316, 316, 3)
    
    f.require_dataset("segmentation/testing_images/"+fl,
                      shape=(nb_set2,size_out,size_out),
                      dtype='i8',
                      exact=False)
    f["segmentation/testing_images/"+fl][:]=m.to_classes(v).reshape(-1,size_out,size_out) #(5044, 99856)

    del input_gmf
    del test_nrcs

# Test with patches extracted from the same SAR images
with h5py.File(hdf_data,"r") as f:
    test_nrcs=f["test/Nrcs/training_images/"][:]
    input_gmf = shape_modify(f["test/GMF/training_images"][:])
    input_ship = f["masks/training_images/"+fl0][:][...,1]#.reshape(-1,size_out,size_out,2)
    # GEN
    d,_,_ = input_gmf.shape
    input_gen = np.zeros((d,im_height,im_width))
    for i_d in xrange(d):
        im_tmp = input_gmf[i_d,:,:].reshape(-1,1,im_height,im_width)
        input_gen[i_d,:,:] = generator_nn.predict(im_tmp).reshape(im_height,im_width)
    input_gen = input_gen[:,2:-2,2:-2] # crop (:,512,512) -> (:,508,508)
    w=unet.predict([np.sqrt(test_nrcs),np.sqrt(input_gen),input_ship],verbose=1,batch_size=16)

with h5py.File(hdf_result,"a") as f:
    f.require_dataset("results/training_images/"+fl,
                      shape=(nb_set1,size_out,size_out,nbClass),
                      dtype='f4',
                      exact=False)
    f["results/training_images/"+fl][:]=w.reshape(-1,size_out,size_out,nbClass) 
    
    f.require_dataset("segmentation/training_images/"+fl,
                      shape=(nb_set1,size_out,size_out),
                      dtype='i8',
                      exact=False)
    f["segmentation/training_images/"+fl][:]=m.to_classes(w).reshape(-1,size_out,size_out)

    del input_gmf
    del test_nrcs
    
logging.info('End testing!')
logging.info(fl)
f = h5py.File(fl + 'finished.hdf5','a')
f.close()