import cv2
import numpy as np
import os
import random as rd
from generate_xml import write_xml

tl_list = []
br_list = []
object_list = []
savedir='C:\\Users\\hp\\Desktop\\Datset\\annotations'
image_folder='C:\\Users\\hp\\Desktop\\Datset\\FinalImages'

Backlist = os.listdir('C:\\Users\\hp\\Desktop\\Datset\\BacgroundImages')
BackgroundCount = len(Backlist)
i=0
for n, CropfileName in enumerate(os.scandir('C:\\Users\\hp\\Desktop\\Datset\\CroppedImages')):
    Croplist = os.listdir('C:\\Users\\hp\\Desktop\\Datset\\CroppedImages\\'+CropfileName.name)
    CroppedImageCount = len(Croplist)
    while(i<CroppedImageCount):
        Croppedimage = cv2.imread('C:\\Users\\hp\\Desktop\\Datset\\CroppedImages\\'+CropfileName.name+'\\'+Croplist[i])
        Backgroundimage=cv2.imread('C:\\Users\\hp\\Desktop\\Datset\\BacgroundImages\\'+Backlist[rd.randint(0, BackgroundCount)])
        img2=Croppedimage
        img1=Backgroundimage
        rows, cols, channels = img2.shape
        roi = img1[0:rows,0:cols]
        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        try:
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        except:
            print(CropfileName.name+'\\'+Croplist[i])
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        rndRow=rd.randint(0,(Backgroundimage.shape[0]-Croppedimage.shape[0]))
        rndCol = rd.randint(0, (Backgroundimage.shape[1] - Croppedimage.shape[1]))
        img1[rndRow:rows+rndRow,rndCol:cols+rndCol] = dst
        #cv2.imshow('res', img1)
        cv2.imwrite('C:\\Users\\hp\\Desktop\\Datset\\FinalImages\\{:06}.png'.format(i),img1)
        br_list.append((int(rndRow+(Croppedimage.shape[0]/2)), (int(rndCol+(Croppedimage.shape[1]/2)))))
        tl_list.append((int(rndRow-(Croppedimage.shape[0]/2)), (int(rndCol-(Croppedimage.shape[1]/2)))))
        object_list.append(CropfileName.name)
        write_xml(image_folder, 'C:\\Users\\hp\\Desktop\\Datset\\FinalImages\\{:06}.png'.format(i),'{:06}.png'.format(i), object_list, tl_list, br_list, savedir)
        tl_list.clear()
        br_list.clear()
        object_list.clear()
        i=i+1
