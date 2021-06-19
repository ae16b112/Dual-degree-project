#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:51:37 2021

@author: niwedita
"""
import cv2 
import numpy as np
from matplotlib import pyplot as plt

"""
MIN_MATCH_COUNT = 8
cap = cv2.VideoCapture('./input_images/jio.mp4')

#img = cv2.imread(filename = "img1.png", flags = cv2.IMREAD_GRAYSCALE)
input_img = cv2.imread("./input_images/img1.png")
img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

# creating the SIFT algorithm for detecting features
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SIFT_create()
#sift = cv2.ORB_create() 
#sift = cv2.AKAZE_create()
  
# find the keypoints and descriptors with SIFT 
kp_image, desc_image = sift.detectAndCompute(img, None) 
img_1 = cv2.drawKeypoints(img,kp_image,input_img)
plt.imshow(img_1)  
# intializing the dictionary 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
search_params = dict(checks=50) 
  
# by using Flann feature Matcher 
flann = cv2.FlannBasedMatcher(index_params, search_params) 
clusters = np.array([desc_image])
flann.add(clusters)
flann.train()
"""

def kaze_match():
    # load the image and convert it to grayscale
    im1 = cv2.imread("./input_images/bast_book1.png")
    im2 = cv2.imread("./input_images/bast_book2.png")
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)    
    #gray1 = rescale_frame(gray1, percent = 60)
    #gray2 = rescale_frame(gray2, percent = 60)
    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    #detector = cv2.xfeatures2d.SIFT_create()
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))    

    # Match the features
    index_params = dict(algorithm = 1, trees = 5) 
    search_params = dict(checks=50) 
  
# by using Flann feature Matcher 
    #bf = cv2.FlannBasedMatcher(index_params, search_params) 
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    clusters = np.array([descs1])
    bf.add(clusters)
    bf.train()
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3 = cv2.drawMatchesKnn(im1, kps1, im2, kps2, good[1:20], None, flags=2)
    cv2.imwrite("./images/bst_book/AKAZE.png", im3)
    cv2.imshow("AKAZE matching", im3)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)



def object_tracking():
    while True:  
        _, frame = cap.read() 
        frame = rescale_frame(frame, percent = 60)  
        # converting the frame into grayscale 
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
          
        # find the keypoints and descriptors with SIFT 
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None) 
          
        # finding nearest match with KNN algorithm 
        try:
        	matches= flann.knnMatch(np.float32(desc_image), np.float32(desc_grayframe), k=2) 
        except:
        	continue
          
        # initialize list to keep track of only good points 
        good_points=[] 
          
        for m, n in matches: 
            #append the points according 
            #to distance of descriptors 
            if(m.distance < 0.7*n.distance): 
                good_points.append(m)
                
        # maintaining list of index of descriptors 
        # in query descriptors
        if len(good_points)>MIN_MATCH_COUNT:
            query_pts = np.float32([kp_image[m.queryIdx] 
                         .pt for m in good_points]).reshape(-1, 1, 2) 
          
            # maintaining list of index of descriptors 
            # in train descriptors 
            train_pts = np.float32([kp_grayframe[m.trainIdx] 
                         .pt for m in good_points]).reshape(-1, 1, 2) 
          
            # finding  perspective transformation 
            # between two planes 
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0) 
          
            # ravel function returns  
            # contiguous flattened array 
            matchesMask = mask.ravel().tolist()
        
            # Perspective transform
            # initializing height and width of the image 
            h, w  = img.shape 
          
            # saving all points in pts 
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]) .reshape(-1, 1, 2) 
          
            # applying perspective algorithm 
            dst = cv2.perspectiveTransform(pts, matrix) 
        
            # using drawing function for the frame
            (x, y, w, h) = cv2.boundingRect(dst)
            x_medium = int((x + x + w) / 2)
            y_medium = int((y + y + h) / 2)
            cv2.line(frame, (x_medium, 0), (x_medium, 640), (0, 255, 0), 2)
            cv2.line(frame, (0, y_medium), (740,y_medium), (0, 255, 0), 2)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA) 
          
            # showing the final output  
            # with homography 
            cv2.imshow("Homography", homography)
            """
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
            img3 = cv2.drawMatches(img,kp_image,grayframe,kp_grayframe,good_points,None,**draw_params)
            plt.imshow(img3, 'gray'),plt.show()
            """
        else:
             print( "Not enough matches are found - {}/{}".format(len(good_points), MIN_MATCH_COUNT) )
             matchesMask = None
        key = cv2.waitKey(1)
        if key == 27:
            break



kaze_match()
#object_tracking()
    #cap.release()
    #cv2.destroyAllWindows()



# reading the frame


