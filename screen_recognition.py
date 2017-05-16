import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from utils import *

def get_rect(contour, img_shape):
    if contour.shape[0]<4:
        return None
    if contour.shape[0]==4:
        return contour
    contour=contour[:,0]
    img_corners = np.array([[0, 0], [img_shape[0], 0], [img_shape[0], img_shape[1]], [0, img_shape[1]]])
    corners = [contour[0], contour[0], contour[0], contour[0]]
    for p in contour:
        for i in range(4):
            if dist(img_corners[i], p)<dist(img_corners[i], corners[i]):
                corners[i]=p
    rect = np.array(corners).reshape(4, 1, 2)
    if not cv2.isContourConvex(rect):
        return None
    
    return rect

def sides_paralell(rect):
    TOLERANCE=0.2 # This limits tolerance in the angle difference between opposite edges of the rectangle
    rect=rect[:,0]
    angle = lambda p,q: math.atan2(q[1]-p[1], q[0]-p[0])
    similar = lambda x,y: abs(x-y)<TOLERANCE
    return similar(angle(rect[0], rect[1]), angle(rect[3], rect[2])) and  similar(angle(rect[1], rect[2]), angle(rect[0], rect[3])) 
    
def contour_to_mask(contour, img_shape):
    mask = np.zeros(img_shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)
    return mask

def find_screen_borders(img):
    kernel = np.ones((21,21),np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)>2 else img
    gray = cv2.GaussianBlur(gray,(15,15),0)
    area = img.size
    best_fit = None
    MIN_DETECTABLE_AREA = 0.018*area
    for threshold_level in np.linspace(70,140,2):
        ret,th = cv2.threshold(gray,threshold_level,255,cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>MIN_DETECTABLE_AREA:
                epsilon = 0.02*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                if cv2.isContourConvex(approx):
                    rect = get_rect(approx, gray.shape)
                    if rect is not None and sides_paralell(rect) and cv2.contourArea(rect)>MIN_DETECTABLE_AREA:
                        area_proportion_difference = abs(cv2.contourArea(approx)/float(cv2.contourArea(rect)))
                        if 0.95 <= area_proportion_difference and area_proportion_difference < 1.15:
                            if best_fit is None or cv2.contourArea(rect)>cv2.contourArea(best_fit):
                                best_fit = rect
    return best_fit

def apply_prespective_transform(img, borders):
    dst_shape = (int(max(dist(borders[0], borders[1]), dist(borders[2], borders[3]))),
                 int(max(dist(borders[1], borders[2]), dist(borders[3], borders[0]))))
    borders = borders.reshape(4, 2).astype(np.float32)
    img_corners = np.float32([[0,0], [dst_shape[0],0], [dst_shape[0],dst_shape[1]], [0,dst_shape[1]]])
    M = cv2.getPerspectiveTransform(borders,img_corners)
    persp = cv2.warpPerspective(img, M, dst_shape)
    if persp.shape[1]>persp.shape[0]:
        persp=cv2.transpose(persp)
    else:
        persp=cv2.flip(persp, flipCode=1)
    return persp
        
def main():
    #~ find_screen_borders(cv2.imread('samples/sample6.jpg'))
    ensure_dir(SCREEN_SAMPLES_DIR)
    for f in get_files(RAW_SAMPLES_DIR):
        print(f)
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        borders = find_screen_borders(img)
        img_with_borders = img.copy()
        if borders is not None:
            cv2.drawContours(img_with_borders, [borders], 0, (255, 0, 0), 3) 
            persp = apply_prespective_transform(gray, borders)
            plt.subplot(122),plt.imshow(persp, 'gray')
            cv2.imwrite(os.path.join(SCREEN_SAMPLES_DIR, os.path.basename(f)), persp)
        plt.subplot(121),plt.imshow(img_with_borders)
        plt.show()

if __name__ == "__main__":
    main()
