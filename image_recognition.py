import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
SAMPLES_DIR='samples'

def get_files(input_dir):
    return [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir)]

files = get_files(SAMPLES_DIR)
print(files[0])
img = cv2.imread(files[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
