import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import scipy
import scipy.signal

def get_kernel(kernel_width=5, sigma=0.5):

    kernel = np.zeros([kernel_width, kernel_width])
    center = (kernel_width + 1.)/2.
    sigma_sq =  sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center)/2.
            dj = (j - center)/2.
            kernel[i - 1][j - 1] = np.exp((-(di * di + dj * dj)/(2 * sigma_sq)))
    kernel /= kernel.sum()
    kernel=np.repeat(kernel[:,:,None],3,axis=2)
    return kernel

if __name__=='__main__':
    path='DIPDKP/data/datasets/data/HR1'
    output='DIPDKP/data/datasets/data/HR'
    imgs=os.listdir(path)
    
    # kernel=get_kernel(kernel_width=3)
    for img in imgs:
        # if img[-3:]!='png' or img[-3:]!='jpg':
            
        #     continue
        img_name=img
        
        print(img_name)
        img1=Image.open(path+'/'+img_name)
        img1=np.array(img1)
        img2=np.zeros_like(img1)
        img2[:,:,0]=img1[:,:,2]
        img2[:,:,1]=img1[:,:,1]
        img2[:,:,2]=img1[:,:,0]     
        
        w,h,_=img2.shape

        img2=img2[:w-w%32,:h-h%32,:]
        w,h,_=img2.shape
        l=480
        if w>l:
            d=int((w-l)/2)
            img2=img2[d:d+l,:,:]
        if h>l:
            d=int((h-l)/2)
            img2=img2[:,d:d+l,:]

        cv2.imwrite(output+'/'+img_name[:-4]+'.png',img2)