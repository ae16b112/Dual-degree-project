#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:04:48 2021

@author: niwedita
"""


import cv2
import sys
import numpy as np
from flask import Flask, render_template, Response
from matplotlib import pyplot as plt
app = Flask(__name__)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 

def find_homography(img, frame):
    MIN_MATCH_COUNT = 10
    # creating the SIFT algorithm for detecting features
    sift = cv2.xfeatures2d.SIFT_create()
    
    # intializing the dictionary 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
    search_params = dict(checks=50) 
  
    # by using Flann feature Matcher 
    flann = cv2.FlannBasedMatcher(index_params, search_params) 
    #frame = rescale_frame(frame, percent=50)  
    # converting the frame into grayscale 
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      
    # find the keypoints and descriptors with SIFT 
    kp_image, desc_image = sift.detectAndCompute(img, None)
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None) 
    matches= flann.knnMatch(desc_image, desc_grayframe, k=2) 
      
    # initialize list to keep track of only good points 
    good_points=[] 
      
    for m, n in matches: 
        #append the points according to distance of descriptors 
        if(m.distance < 0.7*n.distance): 
            good_points.append(m)
            
    if len(good_points)>MIN_MATCH_COUNT:
        # maintaining list of index of descriptors in query descriptors
        query_pts = np.float32([kp_image[m.queryIdx] 
                     .pt for m in good_points]).reshape(-1, 1, 2) 
      
        # maintaining list of index of descriptors in train descriptors 
        train_pts = np.float32([kp_grayframe[m.trainIdx] 
                     .pt for m in good_points]).reshape(-1, 1, 2) 
      
        # finding  perspective transformation between two planes 
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0) 
      
        # ravel function returns contiguous flattened array 
        matchesMask = mask.ravel().tolist()
        # Perspective transform
        # initializing height and width of the image 
        h, w  = img.shape
        # saving all points in pts 
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]) .reshape(-1, 1, 2) 
        print("pts", pts)
        # applying perspective algorithm 
        dst = cv2.perspectiveTransform(pts, matrix) 
    
        # using drawing function for the frame
        (x, y, w, h) = cv2.boundingRect(dst)
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        cv2.line(frame, (x_medium, 0), (x_medium, 640), (0, 255, 0), 2)
        cv2.line(frame, (0, y_medium), (740,y_medium), (0, 255, 0), 2)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA) 
        print("homography", homography)
        # showing the final output with homography 
        #cv2.imshow("Homography", homography)
        return (x, y, w, h)
    
    
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def get_tracker():
        # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
    tracker_type = tracker_types[5]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
    return (tracker, tracker_type)


def object_tracking(img, video, tracker, tracker_type, selection_type):    
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
     
    if selection_type ==  "ROI":
        bbox = cv2.selectROI(frame, False)
    elif selection_type == "Homography_based":    
        bbox = find_homography(img, frame)
    print(bbox)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            (x, y, w, h) = bbox
            x_medium = int((x + x + w) / 2)
            y_medium = int((y + y + h) / 2)
            cv2.line(frame, (x_medium, 0), (x_medium, 740), (0, 255, 0), 2)
            cv2.line(frame, (0, y_medium), (740, y_medium), (0, 255, 0), 2)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        #cv2.imshow("Tracking", frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break


@app.route('/video_feed')
def video_feed():
    video = cv2.VideoCapture("./videos/test.mp4")
    img = cv2.imread("./videos/crop_image.png", cv2.IMREAD_GRAYSCALE) #or Img = ""  for ROI based selection type
    (tracker, tracker_type) = get_tracker()
    selection_type = "Homography_based"  #  "Homography_based" or "ROI"    
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(object_tracking(img, video, tracker, tracker_type, selection_type), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="localhost", port=4010, debug=True) 


    