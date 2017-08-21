import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import game
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

def remove_blank_borders(img):
    for i in range(10):
        print(img[i,:].mean())
    return img

def remove_bottom_light(img):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill = img.copy()
    def turn_off(x,y):
        if im_floodfill[y][x]>127:
            cv2.floodFill(im_floodfill, mask, (x,y), 0) 
    for s in range(w):
        turn_off(s, 0)
        turn_off(s,h-1)
    for s in range(h):
       turn_off(0,s)
       turn_off(w-1,s)
    return im_floodfill

def apply_multi_threshold(img):
    kernel = np.ones((5,5),np.uint8)
    res=np.zeros(img.shape[:2], img.dtype)
    img=1-img
    #~ plt.subplot(121), plt.imshow(img)
    for i,gray in enumerate(cv2.split(img)):
        ret3,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        edges = remove_bottom_light(edges)
        #~ plt.subplot(221+i), plt.imshow(edges, 'gray')
        res=cv2.add(res, edges)
    kernel = np.ones((7,7),np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    #~ plt.show()
    return res

def fix_rotation(edges):
    h,w=edges.shape
    def first_blank(col):
        for y in range(h):
            if edges[y][col]>127:
                return y
        return w
    IGNORE = 3
    firsts = np.array([first_blank(col) for col in range(IGNORE, w-IGNORE)])
    block=int(len(firsts)/3)
    left = firsts[:block].argmin()
    right = firsts[2*block:].argmin()+2*block
    angle = math.atan2(firsts[right]-firsts[left], right-left)
    M = cv2.getRotationMatrix2D((w/2,h/2),angle*180/np.pi,1)
    edges = cv2.warpAffine(edges,M,(w,h))
    return edges

def calc_block_size(edges):
    h,w = edges.shape
    mnposib = int(w/12)
    mxposib = int(w/10)
    posib = [0 for _ in range(mxposib+1)]
    for y in range(h):
        cnt = 0
        ant_negro = False 
        for x in range(w):
            if ant_negro and edges[y][x]>127:
                if cnt<len(posib):
                    posib[cnt]+=1
                cnt=1
                ant_negro = False
            else:
                if edges[y][x]<127:
                    ant_negro = True
                cnt+=1
    for x in range(w):
        cnt = 0
        for y in range(h):
            if ant_negro and edges[y][x]>127:
                if cnt<len(posib):
                    posib[cnt]+=1
                cnt=1
                ant_negro = False
            else:
                if edges[y][x]<127:
                    ant_negro = True
                cnt+=1
    return np.array(posib[mnposib:]).argmax()+mnposib

#~ def draw_grid(edges, block_size):
    
    #~ for i in range(10):

def cut_top(edges):
    h,w = edges.shape
    for y in range(h):
        for x in range(w):
            if edges[y][x]>127:
                return edges[y:,:]
    return edges

TAM_GRID = 10
def draw_grid(edges, block_size):
    grid = edges.copy()
    h,w = grid.shape
    t = TAM_GRID
    offsetx = int((w-t*block_size)/2)
    for i in range(t+1):
        cv2.line(grid, (offsetx+block_size*i, 0), (offsetx+block_size*i, t*block_size), 40)
        cv2.line(grid, (offsetx, block_size*i), (offsetx+t*block_size, block_size*i), 40)
    return grid

def get_matrix(edges, block_size):
    t = TAM_GRID
    offsetx = int((edges.shape[1]-t*block_size)/2)
    mat = [[0 for _ in range(t)] for _ in range(t)]
    for y in range(t):
        for x in range(t):
            count = (edges[y*block_size:(y+1)*block_size,offsetx+x*block_size:offsetx+(x+1)*block_size]>127).sum()
            mat[y][x]=1 if count > block_size**2/2 else 0
    return mat

def get_lower(edges, block_size):
    s=(TAM_GRID+1)*block_size
    while s<edges.shape[0] and edges[s].max()>127:
        s+=1
    lower = edges[s:,:]
    kernel = np.ones((5,5),np.uint8)
    lower = cv2.dilate(lower, kernel, iterations=3)
    return lower

def get_contour(piece):
    piece = [(y,x) for x,y in piece.blocks]
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    bx = [-1, 0, 0, -1]
    by = [-1, -1, 0, 0]
    nbx = [0, 0, -1, -1]
    nby = [-1, 0, 0, -1]
    summ = lambda x,d: (x+d+4)%4
    def mano_derecha(y, x, d, poly):
        if (y, x) in poly:
            return poly
        for ds in [1, 0, -1]:
            nd =summ(d, ds)
            if (y+by[nd], x+bx[nd]) in piece and not (y+nby[nd], x+nbx[nd]) in piece:
                if d!=nd:
                    poly.append((y, x))
                return mano_derecha(y+dy[nd], x+dx[nd], nd, poly)
    def first_point():
        leftupper = piece[0]
        for y,x in piece:
            if x<leftupper[1] or (x==leftupper[1] and y<leftupper[0]):
                leftupper = (y,x)
        return leftupper
    sy,sx=first_point()
    poly = mano_derecha(sy, sx, 2, [])
    return np.array(poly).reshape(len(poly), 1, 2)

def get_pieces():
    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    pts = pts.reshape((-1,1,2))
    
def extract_game(img):
    edges = preprocess_screen(img)
    block_size = calc_block_size(edges)
    #~ grid = draw_grid(edges, block_size)
    mat = get_matrix(edges, block_size)
    lower = get_lower(edges, block_size)
    
    im2, contours, hierarchy = cv2.findContours(lower,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    pieces = [(piece.name, get_contour(piece)) for piece in game.piece_list]
    options = []
    for cnt in contours:
        epsilon = 0.02*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        comp = np.array([cv2.matchShapes(approx,piece_cnt,1,0.0) for _,piece_cnt in pieces])
        es = pieces[comp.argmin()][0]
        print(es)
        options.append(es)
    #~ plt.subplot(121)
    plt.imshow(lower, 'gray')
    plt.show()
    
    return mat
    
def preprocess_screen(img):
    img = remove_borders(img)
    img = remove_header(img)
    #~ plt.subplot(121), plt.imshow(img)
    edges = apply_multi_threshold(img)
    edges = fix_rotation(edges)
    edges = cut_top(edges)
    #~ plt.subplot(122), plt.imshow(edges, 'gray')
    #~ plt.show()
    return edges
    
def main():
    #~ extract_game(cv2.imread('samples/screen/sample2.jpg'))
    for f in get_files(SCREEN_SAMPLES_DIR):
        print(f)
        extract_game(cv2.imread(f))

if __name__ == "__main__":
    main()
