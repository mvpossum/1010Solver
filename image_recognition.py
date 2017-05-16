import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

SAMPLES_DIR='samples'

def get_files(input_dir):
    return [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir)]

def get_rect(contour, img_shape):
    if contour.shape[0]<4:
        return None
    if contour.shape[0]==4:
        return contour
    contour=contour[:,0]
    img_corners = np.array([[0, 0], [img_shape[0], 0], [img_shape[0], img_shape[1]], [0, img_shape[1]]])
    corners = [contour[0], contour[0], contour[0], contour[0]]
    dist = lambda p,q: np.linalg.norm(q-p)
    for p in contour:
        for i in range(4):
            if dist(img_corners[i], p)<dist(img_corners[i], corners[i]):
                corners[i]=p
    rect = np.array(corners).reshape(4, 1, 2)
    if not cv2.isContourConvex(rect):
        return None
    
    return rect

def sides_paralell(rect):
    rect=rect[:,0]
    TOLERANCE=0.3
    angle = lambda p,q: math.atan2(q[1]-p[1], q[0]-p[0])
    similar = lambda x,y: abs(x-y)<TOLERANCE
    return similar(angle(rect[0], rect[1]), angle(rect[3], rect[2])) and  similar(angle(rect[1], rect[2]), angle(rect[0], rect[3])) 
    
def contour_to_mask(contour, img_shape):
    mask = np.zeros(img_shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)
    return mask

def find_screen_borders(img):
    kernel = np.ones((21,21),np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(15,15),0)
    area = img.size
    best_fit = None
    for threshold_level in np.linspace(70,160,10):
        ret,th = cv2.threshold(gray,threshold_level,255,cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        img_count=img.copy()
        for i,cnt in enumerate(contours):
            epsilon = 0.02*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            rect = get_rect(approx, gray.shape)
            if cv2.isContourConvex(approx) and rect is not None and sides_paralell(rect) and cv2.contourArea(rect)>0.018*area:
                area_proportion_difference = abs(1-abs(cv2.contourArea(approx)/float(cv2.contourArea(rect))))
                if area_proportion_difference*100<15:
                    cv2.drawContours(img_count, [rect], 0, (0, 0, 255*float(i)/len(contours)), 3) 
                    if best_fit is None or cv2.contourArea(rect)>cv2.contourArea(best_fit):
                        best_fit = rect
    return best_fit
    
def main():
    #~ find_screen_borders(cv2.imread('samples/sample6.jpg'))
    for f in get_files(SAMPLES_DIR):
        print(f)
        img = cv2.imread(f)
        borders = find_screen_borders(img)
        cv2.drawContours(img, [borders], 0, (255, 0, 0), 3) 
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    main()
