import numpy as np
from cv2 import imread,imwrite, dilate, erode
from cv2 import cvtColor, COLOR_BGR2HLS, calcHist
import cv2 as cv
import random
from matplotlib import pyplot as plt
from skimage.measure import label



def segment_util(img):
    """
    Given an input image, output the segmentation result
    Input:  
        img:        n x m x 3, values are within [0,255]
    Output:
        img_seg:    n x m
    """
    ## TODO
    mask = np.zeros(img.shape[:2], dtype="uint8") # define mask
    
    fgModel = np.zeros((1, 65), dtype="float") # for-ground
    bgModel = np.zeros((1, 65), dtype="float") # back-ground
    
    rect = (0, 0, img.shape[1] - 1, img.shape[0] - 1) # get whole image
    
    cv.grabCut(img,mask,rect,bgModel,fgModel,15,cv.GC_INIT_WITH_RECT) # read docs
    
    img_seg = np.where((mask != 0) & (mask != 2), 1, 0).astype('uint8') # apply mask where binary value = 1 keep it
    
    return img_seg

def close_hole_util(img):
    """
    Given the segmented image, use morphology techniques to close the holes
    Input:
        img:        n x m, values are within [0,1]
    Output:
        closed_img: n x m
    """
    ## TODO
    kernel = np.ones((3, 3), np.uint8) # define kernel for filtering
    
    img_dilation = cv.dilate(img, kernel, iterations=2) # dilation
    img_erosion = cv.erode(img_dilation, kernel, iterations=2) # eroding

    return img_erosion

def instance_segmentation_util(img):
    """
    Given the closed segmentation image, output the instance segmentation result
    Input:  
        img:        n x m, values are within [0,255]
    Output:
        instance_seg_img:    n x m x 3, different coin instances have different colors
    """
    ## TODO
    img  = close_hole_util(img)

    _, thresh = cv.threshold(img, 255//2, 255, cv.THRESH_BINARY)  
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 3)  

    thresh1 = dilate(thresh, np.ones((5, 5)), iterations=8)
    _, thresh2 = cv.threshold(dist_transform, np.max(dist_transform)/2 , 225, 0)  
    thresh2 = np.uint8(thresh2)

    unknown = cv.subtract(thresh1, thresh2) 
    _, marks = cv.connectedComponents(thresh2)  # assign a marker to each coin
    marks += 1
    marks[unknown == 255] = 0
    img = cv.merge((img, img, img))  # img needs to be 3 Channels

    marks = cv.watershed(img, marks)
    
    for segment in np.unique(marks):
        img[marks == segment] = [255 * random.random(), 255 * random.random(), 255 * random.random()]

    instance_seg_img = img * 255

    return instance_seg_img

def text_recog_util(text, letter_not):
    """
    Given the text and the character, recognise the character in the text
    Input:
        text:           n x m
        letter_not:     a x b
    Output:
        text_er_dil:    n x m
    """
    from scipy.ndimage import binary_erosion as erode
    from scipy.ndimage import binary_dilation as dilate
    ## TODO
    
    _, text_binary = cv.threshold(text, 0.8, 1, cv.THRESH_BINARY_INV)
    _, letter_binary = cv.threshold(letter_not, 0.8, 1, cv.THRESH_BINARY)

    # Perform erosion and dilation operations
    #kernel = np.ones((3, 3), np.uint8)
    img_eroded = erode(text_binary, letter_binary)
    img_dilated = dilate(img_eroded, letter_binary)

    return img_dilated