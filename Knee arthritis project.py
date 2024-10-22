# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:47:38 2023

@author: ranjan
"""
import os
from PIL import Image, ImageFilter
from numpy import asarray
import numpy as np
import random

entries = os.listdir('Expert1')

filepathlist=[]
for i in range(len(entries)):
    path = os.listdir('Expert1/'+entries[i])
    for j in range(len(path)):
        fullpath='Expert1/'+entries[i]+"/"+path[j]
        #print(fullpath)
        filepathlist.append(fullpath)

totalFile=len(filepathlist)
print("Total Files"+str(totalFile))    

noSample=totalFile
sampleData=np.zeros((noSample,162,300,3))
sampleClass=np.zeros((noSample,5))     
#=======================================================
for k in range(noSample):        
    img = asarray(Image.open(filepathlist[k]))
    #print(filepathlist[k][41])
    if len(img.shape)==3:
        sampleData[k,:,:,:]=img
    else:
        sampleData[k,:,:,0]=img
        sampleData[k,:,:,1]=img
        sampleData[k,:,:,2]=img
    #print(filepathlist[k])
    sampleClass[k,int(filepathlist[k][8])]=1
sampleData=sampleData/255.0
#======================================================
suffleList=list(range(0,noSample))
random.shuffle(suffleList)
trainValidDataSuff=np.zeros((noSample,162,300,3))
trainValidClassSuff=np.zeros((noSample,5))
for i in range(0,noSample):
    trainValidDataSuff[i,:,:,:]=sampleData[suffleList[i],:,:,:]
    trainValidClassSuff[i,:]=sampleClass[suffleList[i],:]
print('All Data and class suffeled')

#=====================================================
ntrain=int(noSample*0.8)   
nvalid=noSample-ntrain
print(nvalid)   
trainData=np.zeros((ntrain,162,300,3))
trainClass=np.zeros((ntrain,5))
validData=np.zeros((nvalid,162,300,3))
validClass=np.zeros((nvalid,5))
print(ntrain,noSample)
trainData[0:ntrain,:,:,:]=trainValidDataSuff[0:ntrain,:,:,:]
trainClass[0:ntrain,:]=trainValidClassSuff[0:ntrain,:]
validData[0:nvalid,:,:,:]=trainValidDataSuff[ntrain:noSample,:,:,:]
validClass[0:nvalid,:]= trainValidClassSuff[ntrain:noSample,:]
#================================================================

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def CNN2D(nrows, ncols, nchannels):
    model = Sequential()
    #================================== For 64x512x6 =================
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(nrows, ncols,nchannels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=48, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=80, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=96, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    print('---2D CNN for Image Size '+str(nrows)+'x'+str(ncols)+'--Developed by Ranjan-----')
    print(model.summary())
    return model

#===========================================================
model=CNN2D(162,300,3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results=model.fit(trainData, trainClass, batch_size=64, epochs=100, validation_data=(validData, validClass))

 
        
