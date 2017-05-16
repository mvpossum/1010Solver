import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from utils import *

def remove_borders(img, pixels=None):
    if pixels is None:
        tam=min(img.shape[0], img.shape[1])
        border_length=math.ceil(tam*0.02)
        pixels = (border_length, border_length)
    return img[pixels[0]:-pixels[0],pixels[1]:-pixels[1]]
    
def remove_header(img, pixels=None):
    if pixels is None:
        WIDTH_LINE=5
        best_s = None
        best_mean = None
        for s in range(20, int(img.shape[0]*0.2), 2):
            sep = img[s:s+WIDTH_LINE,:]
            if best_s is None or best_mean<sep.mean():
                best_s = s
                best_mean = sep.mean()
        pixels=best_s
    return img[pixels:,:]

def main():
    #~ find_screen_borders(cv2.imread('samples/sample6.jpg'))
    for f in get_files(SCREEN_SAMPLES_DIR):
        print(f)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = remove_borders(img)
        img = 1-remove_header(img)
        ret3,edges = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #~ edges = cv2.Canny(img,20,50)
        kernel = np.ones((5,5),np.uint8)
        #~ edges = cv2.dilate(edges,kernel,iterations = 1)
        #~ edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        #~ edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        plt.subplot(121), plt.imshow(img, 'gray')
        plt.subplot(122), plt.imshow(edges, 'gray')
        plt.show()

if __name__ == "__main__":
    main()
