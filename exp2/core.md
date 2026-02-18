import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("img.jpg")
from operator import add
from functools import reduce

def split4(image):
    half_split = np.array_split(image,2)
    res = map(lambda x: np.array_split(x,2,axis = 1),half_split)
    return reduce(add,res)

split_img = split4(img)
split_img[0].shape,split_img[1].shape
fig,axs = plt.subplots(2,2)
axs[0,0].imshow(split_img[0])
axs[0,1].imshow(split_img[1])
axs[1,0].imshow(split_img[2])
axs[1,1].imshow(split_img[3])

def concatenate4(north_west,north_east,south_west,south_east):
    top = np.concatenate((north_west,north_east),axis = 1)
    bottom = np.concatenate((south_west,south_east),axis = 1)
    return np.concatenate((top,bottom),axis = 0)

full_img = concatenate4(split_img[0],split_img[1],split_img[2],split_img[3])
plt.figure()
plt.imshow(full_img)
plt.show()

<img width="405" height="475" alt="Screenshot 2026-02-18 092154" src="https://github.com/user-attachments/assets/958154a1-fd1c-4bbf-97ce-398333565dbb" />
