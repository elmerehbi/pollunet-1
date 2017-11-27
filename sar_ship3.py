#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:45:39 2017

@author: vnguye04
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, load_model
from keras.layers.core import Activation, Reshape, Permute
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, merge
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras.utils import plot_model
import h5py
import h5py as h
import sys
sys.path.append('../')
import measures as m
import os.path 
import datetime
import logging

fl = 'sar_ship3'
#logging.basicConfig(filename= '/users/local/dnguyen/PolluNet2/log/' + fl + '.log',
#                    level=logging.DEBUG,
#                    format='[%(asctime)s] - [%(levelname)s] - %(message)s')


print fl
nb_shuffle=1 # Nombre de fois qu'on choisit training_size images
print 'nb_shuffle', nb_shuffle
epochs=30 # Nombre d'époques sur un choix d'images
print 'nb_epoch', epochs

#hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"
hdf_data="/users/local/dnguyen/polluNet2/data.hdf5"
hdf_result = "/users/local/dnguyen/polluNet2/results/" + fl + ".hdf5"
"""
###########################
# Paramètres du modèle
###########################
"""
# Dossier de sauvegarde des poids des modèles
weight_dir = "/users/local/dnguyen/polluNet2/weights/"

# Poids des différentes classes pour le loss
######
weight_pollution = 500.
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
dropout_down = 0.0
dropout_up = 0.0
channels_max = 48 # Nombre max de channels utilisés dans le réseau
training_size = 2000 # Nombre de patch utilisés
batch_size=12 # Batch size (Préferer un grand batch_size, mais si il est trop grand le réseau ne rentre plus en mémoire)

# Fonction qui donne le nombre de channels à utiliser en fonction de la profondeur (profondeur décroissante: elle vaut 0 en bas du réseau)
def channels(depth, channels_max=channels_max):
    return channels_max
#    return channels_max / 2**depth
optimizer=adadelta() # Optimizer


"""
###########################
# Génération du réseau
###########################
"""

K.set_image_dim_ordering('th') # Theano dimension ordering in this code

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
        x = BatchNormalization(axis = 1)(x)
        x=Dropout(dropout_down)(x)
    else:
        for i in range(nb_conv):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x = BatchNormalization(axis = 1)(x)
            x=Dropout(dropout_down)(x)
    if depth>0:
        y=MaxPooling2D(pool_size=(2,2))(x)
        y=unet_layers(y,depth-1,channels_max)
        y=UpSampling2D(size=(2,2))(y)
        x=Cropping2D(c.next(),data_format='channels_first')(x)
        x=concatenate([x,y],axis=1)
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x = BatchNormalization(axis = 1)(x)
            x=Dropout(dropout_up)(x)
        x=Conv2D(channels(depth,channels_max)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x = BatchNormalization(axis = 1)(x)
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
sar = BatchNormalization(axis = 1)(sar)
gmf = Reshape((1,height,width))(gmf_input) 
gmf = BatchNormalization(axis = 1)(gmf)
ship = Reshape((1,height,width))(ship_input) 

c = crop() # call this function before any unet_layers() use.
x = concatenate([sar,gmf,ship],axis=1)
x = unet_layers(x,depth,channels_max) # <tf.Tensor shape=(?, 24, 324, 324) dtype=float32>
x = BatchNormalization(axis = 1)(x)
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
plot_model(unet, to_file= '/users/local/dnguyen/polluNet2/log/' + fl + '.png')
"""
###########################
# Chargement des masques
###########################
"""

b=size_out*size_out
o=(width-size_out)/2
print datetime.datetime.utcnow(), 'begin loading the masks'
with h5py.File(hdf_data,"r") as f:
    nb_train = len(f["train/Nrcs"])
    nb_set1 = len(f["test/Nrcs/training_images"])
    nb_set2 = len(f["test/Nrcs/testing_images"])

print datetime.datetime.utcnow(), 'finish loading the masks!'
"""
###########################
# Apprentissage
###########################
"""
print datetime.datetime.utcnow(), 'begin learning...'
if os.path.isfile(weight_dir+fl):
    unet.load_weights(weight_dir+fl)

fl0 = 'sar_gmf_ship.py'
with h5py.File(hdf_data,"r") as f:
    print nb_shuffle
    for i in range(nb_shuffle):
        print i
        l = m.randl(training_size,nb_train,m.l4_train) # list, len = 1898

        #    print len(l), nb_train,len(l), f["weights/"+fl].shape
        # print l[-1], nb_train
        # Weights
        print "weights"
        w=f["weights/"+fl0][l] # np.array, shape = (1898, 99856)
        # Train nrcs
        print "nrcs"
        train_nrcs=f["train/Nrcs"][l] # np.array, shape = (1898, 508, 508)
        # Input gmf
        print "gmf"
        input_gmf = f["train/GMF"][l]
        # Input ship
        print "mask"
        input_ship = f["masks/train/"+fl0][l][...,1] #[...,1]: ship
        w[w==1.]=weight_boats
        w[w==0.]=weight_sea   
        w[w==2.]=weight_pollution

        # Train mask
        print "mask"
        # train_mask is np.array, shape(1898, 316, 316, 3) -> (1898, 99856, 3)
        train_mask=f["masks/train/"+fl0][l][:,o:-o,o:-o,:].reshape((-1,size_out*size_out,nbClass))
        
        print "fit"
        unet.fit([np.sqrt(train_nrcs),np.sqrt(input_gmf), input_ship],train_mask,shuffle=True,verbose=1,batch_size=batch_size,epochs=epochs,sample_weight=w)

        print "save"
        unet.save_weights(weight_dir+fl)

        del input_gmf
        del train_mask
        del train_nrcs


print datetime.datetime.utcnow(), 'finish learning!'

"""
###########################
# Test sur les jeux de test
###########################
"""
print datetime.datetime.utcnow(), 'begin testing...'

# Test with patches extracted from the different SAR images
with h5py.File(hdf_data,"r") as f:    
    test_nrcs=f["test/Nrcs/testing_images/"] # HDF5 dataset, shape = (5065, 508, 508)
    input_gmf = f["test/GMF/testing_images"] # HDF5 dataset, shape = (5065, 508, 508)
    input_ship = f["masks/testing_images/"+fl0][:][...,1]  # np.array, shape = (5065, 508, 508)
    v=unet.predict([np.sqrt(test_nrcs),np.sqrt(input_gmf),input_ship],verbose=1,batch_size=16) #np.array, shape = (5044, 99856, 3) 99856 = 316**2



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
    test_nrcs=f["test/Nrcs/training_images/"]
    input_gmf=f["test/GMF/training_images"]
    input_ship = f["masks/training_images/"+fl0][:][...,1]#.reshape(-1,size_out,size_out,2)
    w=unet.predict([np.sqrt(test_nrcs),np.sqrt(input_gmf),input_ship],verbose=1,batch_size=16)

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
    
print datetime.datetime.utcnow(), 'end testing!'
print fl