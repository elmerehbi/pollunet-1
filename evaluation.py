#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:55:57 2017

@author: vnguye04
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
sys.path.append('../')
import measures2 as m
import os.path 
from sklearn.metrics import confusion_matrix


def evaluate(yt,yp):
    # Confusion matrix
    cm = confusion_matrix(yt,yp)
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP.astype(np.float)/(TP+FN)
    # Specificity or true negative rate
    TNR = TN.astype(np.float)/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP.astype(np.float)/(TP+FP)
    # Negative predictive value
    NPV = TN.astype(np.float)/(TN+FN)
    # Fall out or false positive rate
    FPR = FP.astype(np.float)/(FP+TN)
    # False negative rate
    FNR = FN.astype(np.float)/(TP+FN)
    # False discovery rate
    FDR = FP.astype(np.float)/(TP+FP)
    
    # Overall accuracy
    ACC = (TP+TN).astype(np.float)/(TP+FP+FN+TN)
    
    print cm
    print 'TPR', TPR
    print 'FPR', FPR

fl = 'sar_ship2_2'
hdf_result = '/users/local/dnguyen/polluNet2/results/' + fl + '.hdf5'
hdf_data = '/users/local/dnguyen/polluNet2/data.hdf5'
fl0 = 'sar_gmf_ship.py'
size_out = 316
o = 96

ncorr = np.load('ncorr2.npy')
corr_thresh = 0.5

with h5py.File(hdf_result,'r') as g:
    with h5py.File(hdf_data,'r') as f:
        for test_set in ['testing_images','training_images']:
            corr_idx = list(np.where(np.abs(ncorr.item().get(test_set)) > corr_thresh)[0])
            y_pred = g['segmentation'][test_set][fl][corr_idx].reshape(-1,size_out*size_out)
            
            mask = f['masks'][test_set][fl0][corr_idx][:,o:-o,o:-o,:]
            y_true = m.to_classes(mask).reshape(-1,size_out*size_out)
            del mask
            
            print 'Pixelwise', hdf_result, test_set
            yp = y_pred.reshape(-1)
            yt = y_true.reshape(-1)
            evaluate(yt,yp)
            
            print 'Patchwise', hdf_result, test_set
            y_pred = y_pred&2
            y_true = y_true&2
            yp = y_pred.any(axis = 1)
            yt = y_true.any(axis = 1)
            evaluate(yt,yp)


#with h5py.File(hdf_result,'r') as g:
#    with h5py.File(hdf_data,'r') as f:
#        for test_set in ['testing_images','training_images']:
#            y_pred = g['segmentation'][test_set][fl][:].reshape(-1,size_out*size_out)
#            
#            mask = f['masks'][test_set][fl0][:][:,o:-o,o:-o,:]
#            y_true = m.to_classes(mask).reshape(-1,size_out*size_out)
#            del mask
#            
#            print 'Pixelwise', hdf_result, test_set
#            yp = y_pred.reshape(-1)
#            yt = y_true.reshape(-1)
#            evaluate(yt,yp)
#            
#            print 'Patchwise', hdf_result, test_set
#            y_pred = y_pred&2
#            y_true = y_true&2
#            yp = y_pred.any(axis = 1)
#            yt = y_true.any(axis = 1)
#            evaluate(yt,yp)



#with h5py.File(hdf_data,"r") as f:
#    wind_speed = f['train/modelWindSpeed'][:]
#    ws_idx_train = (wind_speed.reshape(-1,21*21) > 2).all(axis = 1)
#    ws_idx_train = list(np.where(ws_idx_train)[0])
#    nb_train = len(ws_idx_train)
#    
#    wind_speed = f['test/modelWindSpeed/training_images'][:]
#    ws_idx_test1 = (wind_speed.reshape(-1,21*21) > 2).all(axis = 1)
#    ws_idx_test1 = list(np.where(ws_idx_test1)[0])
#    nb_set1 = len(ws_idx_test1)
#    
#    wind_speed = f['test/modelWindSpeed/testing_images'][:]
#    ws_idx_test2 = (wind_speed.reshape(-1,21*21) > 2).all(axis = 1)
#    ws_idx_test2 = list(np.where(ws_idx_test2)[0])
#    nb_set2 = len(ws_idx_test2)
#
#with h5py.File(hdf_result,'r') as g:
#    with h5py.File(hdf_data,'r') as f:
#        for test_set,idx in zip(['testing_images','training_images'],[ws_idx_test2, ws_idx_test1]):
#           
#            y_pred = g['segmentation'][test_set][fl][:].reshape(-1,size_out*size_out)
#            
#            mask = f['masks'][test_set][fl0][idx][:,o:-o,o:-o,:]
#            y_true = m.to_classes(mask).reshape(-1,size_out*size_out)
#            del mask
#            
#            print 'Pixelwise', hdf_result, test_set
#            yp = y_pred.reshape(-1)
#            yt = y_true.reshape(-1)
#            evaluate(yt,yp)
#            
#            print 'Patchwise', hdf_result, test_set
#            y_pred = y_pred&2
#            y_true = y_true&2
#            yp = y_pred.any(axis = 1)
#            yt = y_true.any(axis = 1)
#            evaluate(yt,yp)

            
            









#y_pred = m.to_classes(w).reshape(-1,size_out,size_out)

#"""
#ROC
#"""
#labels = ['sar_gmf_ship', 'sar_gmf_ship2', 'sar_ship2_2', 'sar_ship3', 
#          'sar_ship4_2','sar_ship5','sar_ship6', 
#          'GAN1', 'density0',]
#
#labels = ['sar_gmf_ship', 'sar_gmf_ship2', 'sar_ship2_2', 'sar_ship3', 
#          'sar_ship4_2','sar_ship5','sar_ship6', 
#          'GAN1', 'density0',]

#
labels = ['gmf', 'gmf2', 'gmf2_2', 'gmf2_3', 
          'gmf2_4', 'gmf3','gmf4', 
          'GAN', 'density0',]

#PIXELWISE
TPR1 = np.array([0.44436167, 0.63795512, 0.73819723, 0.61467548, 
                0.3964289, 0.34698296, 0.22760499, 
                0.53780613, 0.10431604])
FPR1 = np.array([6.88945824e-03, 6.23971398e-03, 7.56624678e-03, 4.16092213e-03,
                4.99256359e-04, 1.58351955e-03, 2.81897643e-04, 
                2.53284810e-03, 3.21241766e-03])


fig = plt.figure()
ax = fig.add_subplot(121)
#ax.plot(FPR1,TPR1,'bo')
ax.hold(True)
ax.plot([0,0.008],[0,0.8])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Pixelwise')
for i in xrange(len(labels)):
    ax.plot(FPR1[i],TPR1[i],'o')
    ax.annotate(labels[i], xy = (FPR1[i],TPR1[i]))
    
#PATCHWISE
TPR2 = np.array([1., 0.96153846, 1., 0.96153846, 
                1., 0.88461538, 0.88461538, 
                0.96153846, 0.53846154])
FPR2 = np.array([0.2668545, 0.36566714, 0.29543168, 0.30106661,
                0.56671362, 0.20185148, 0.07868786, 
                0.25840209, 0.19742403])


ax = fig.add_subplot(122)
#ax.plot(FPR2,TPR2,'ro')
ax.hold(True)
ax.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Patchwise')
for i in xrange(len(labels)):
    
    ax.plot(FPR2[i],TPR2[i],'o')
    if i == 7:
        ax.annotate(labels[i], xy = (FPR2[i]-0.1,TPR2[i]-0.04))
    elif i == 3:
        ax.annotate(labels[i], xy = (FPR2[i]-0.02,TPR2[i]-0.05))
    elif i == 0:
        ax.annotate(labels[i], xy = (FPR2[i]-0.07,TPR2[i]-0.03))
    elif i == 2:
        ax.annotate(labels[i], xy = (FPR2[i]+0.02,TPR2[i]-0.03))
    elif i == 1:
        ax.annotate(labels[i], xy = (FPR2[i]+0.02,TPR2[i]-0.03))
    else:
        ax.annotate(labels[i], xy = (FPR2[i],TPR2[i]-0.05))

plt.show()
#
#
#
#
#
#
#
#
#
#"""
#Test set 1 and Test set 2
#"""
#
#labels = ['sar_gmf_ship', 'sar_gmf_ship2', 'sar_ship2_2', 'sar_ship3', 
#          'sar_ship4_2','sar_ship5','sar_ship6', 
#          'GAN1', 'density0',]

# Test set 1
#labels = ['gmf', 'gmf2', 'gmf2_2', 'gmf2_3', 
#          'gmf2_4', 'gmf3','gmf4', 
#          'GAN', 'GAN*','density0',]
#
#TPR0 = np.array([0.82519833, 0.96220802, 0.82960555, 0.93515866, 
#                0.72080212, 0.        , 0.48837594, 
#                0.8487219, 0.95703334, 0.36794844])
#FPR0 = np.array([1.38258149e-02, 1.06399459e-02, 1.31437717e-02, 6.77186279e-03,
#                6.98663028e-04, 7.53860171e-04, 5.06294890e-04, 
#                3.93629445e-03, 3.00560799e-03, 6.97443495e-03])
#
#
#fig = plt.figure()
#ax = fig.add_subplot(121)
##ax.plot(FPR1,TPR1,'bo')
#ax.hold(True)
#ax.plot([0,0.01],[0,1])
#plt.xlim([0,0.015])
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('Test set 1 (Pixelwise)')
#for i in xrange(len(labels)):
#    ax.plot(FPR0[i],TPR0[i],'o')
#    if i == 2:
#        ax.annotate(labels[i], xy = (FPR0[i]-0.001,TPR0[i]+0.03))
#    elif i == 0:
#        ax.annotate(labels[i], xy = (FPR0[i],TPR0[i]-0.05))
#    else:
#        ax.annotate(labels[i], xy = (FPR0[i],TPR0[i]))
#    
## Testset 2
#TPR1 = np.array([0.44436167, 0.63795512, 0.73819723, 0.61467548, 
#                0.3964289, 0.34698296, 0.22760499, 
#                0.53780613, 0.74749387, 0.10431604])
#FPR1 = np.array([6.88945824e-03, 6.23971398e-03, 7.56624678e-03, 4.16092213e-03,
#                4.99256359e-04, 1.58351955e-03, 2.81897643e-04, 
#                2.53284810e-03, 2.37057902e-03, 3.21241766e-03])
#
#ax = fig.add_subplot(122)
##ax.plot(FPR1,TPR1,'bo')
#ax.hold(True)
#ax.plot([0,0.01],[0,1])
#plt.xlim([0,0.015])
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('Test set 2 (Pixelwise)')
#for i in xrange(len(labels)):
#    ax.plot(FPR1[i],TPR1[i],'o')
#    ax.annotate(labels[i], xy = (FPR1[i],TPR1[i]))
#plt.show()







































#"""
#ROC
#"""
#labels = ['sar_gmf_ship', 'sar_gmf_ship2', 'sar_ship2_2', 'sar_ship3', 
#          'sar_ship4_2','sar_ship5','sar_ship6', 'sar_ship7',
#          'GAN1', 'density0',]

#labels = ['sar_gmf_ship', 'sar_gmf_ship2', 'sar_ship2_2', 'sar_ship3', 
#          'sar_ship4_2','sar_ship5','sar_ship6', 'sar_ship7',
#          'GAN1', 'density0',]


#labels = ['gmf', 'gmf2', 'gmf2_2', 'gmf2_3', 
#          'gmf2_4', 'gmf3','gmf4', 'gmf5',
#          'GAN', 'density0',]
#
##PIXELWISE
#TPR1 = np.array([0.44436167, 0.63795512, 0.73819723, 0.61467548, 
#                0.3964289, 0.34698296, 0.22760499, 0.0597821, 
#                0.53780613, 0.10431604])
#FPR1 = np.array([6.88945824e-03, 6.23971398e-03, 7.56624678e-03, 4.16092213e-03,
#                4.99256359e-04, 1.58351955e-03, 2.81897643e-04, 3.47938361e-05,
#                2.53284810e-03, 3.21241766e-03])
#
#
#fig = plt.figure()
#ax = fig.add_subplot(121)
#ax.plot(FPR1,TPR1,'bo')
#ax.hold(True)
#ax.plot([0,0.008],[0,0.8])
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('Pixelwise')
#for i in xrange(len(labels)):
#    ax.annotate(labels[i], xy = (FPR1[i],TPR1[i]))
#    
##PATCHWISE
#TPR2 = np.array([1., 0.63795512, 1., 0.96153846, 
#                1., 0.88461538, 0.88461538, 0.65384615, 
#                0.96153846, 0.53846154])
#FPR2 = np.array([0.2668545, 6.23971398e-03, 0.29543168, 0.30106661,
#                0.56671362, 0.20185148, 0.07868786, 0.17085933,
#                0.25840209, 0.19742403])
#
#
#ax = fig.add_subplot(122)
#ax.plot(FPR2,TPR2,'ro')
#ax.hold(True)
#ax.plot([0,1],[0,1])
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('Patchwise')
#for i in xrange(len(labels)):
#    ax.annotate(labels[i], xy = (FPR2[i],TPR2[i]))
#
#plt.show()