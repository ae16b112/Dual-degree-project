# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:19:34 2020
https://www.geeksforgeeks.org/python-draw-rectangular-shape-and-extract-objects-using-opencv/
@author: Niharika
"""
"""
important
1) press 's' to take screenshot from the video
2) scroll mouse to capture the target i.e for cropping
3) press 'r' to undo step 2
4) press 'c' to copy the cropped image
5) Press 'esc' to exit video 
"""
 
# import the necessary packages 
import cv2 
# import argparse 
import numpy as np
import pyautogui

def rescale_frame(frame, percent=75):
    try:
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    except AttributeError:
        print("shape not found")
    return

def shape_selection(event, x, y, flags, param): 
   	# grab references to the global variables 
   	global ref_point, crop 
   
   	# if the left mouse button was clicked, record the starting 
   	# (x, y) coordinates and indicate that cropping is being performed 
   	if event == cv2.EVENT_LBUTTONDOWN: 
   		ref_point = [(x, y)] 
   
   	# check to see if the left mouse button was released 
   	elif event == cv2.EVENT_LBUTTONUP: 
   		# record the ending (x, y) coordinates and indicate that 
   		# the cropping operation is finished 
   		ref_point.append((x, y)) 
   
   		# draw a rectangle around the region of interest 
   		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2) 
   		cv2.imshow("image", image)



# now let's initialize the list of reference point 
ref_point = [] 
crop = False
net=cv2.dnn.readNet('yolov3_custom_last.weights','yolov3_custom.cfg')
classes=[]
with open('obj.names','r') as f:
    classes=f.read().splitlines()
cap=cv2.VideoCapture('test.mp4')
#img=cv2.imread('horses.jpg')
while True:
    _,img=cap.read()
    img = rescale_frame(img, percent=60)
    try:
        height,width,_ = img.shape
    
        blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
        
        net.setInput(blob)
        
        output_layers_names=net.getUnconnectedOutLayersNames()
        layerOutputs=net.forward(output_layers_names)
        
        boxes=[]
        confidences=[]
        class_ids=[]
        
        for output in layerOutputs:
            for detection in output:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]
                if confidence>0.5:
                    center_x=int(detection[0]*width)
                    center_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)
                    
                    x=int(center_x-w/2)
                    y=int(center_y-h/2)
                    
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        
        
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        
        font=cv2.FONT_HERSHEY_PLAIN
        colors=np.random.uniform(0,255,size=(len(boxes),1))
        if len(indexes)>0:
            for i in indexes.flatten():
                x,y,w,h=boxes[i]
                label=str(classes[class_ids[i]])
                confidence=str(round(confidences[i],2))
                color=colors[i]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label+" "+confidence,(x,y+20),font,1,(0,0,256),1)
                
        #print(indexes.flatten())            
        cv2.imshow('Image',img)
        key=cv2.waitKey(1)
        if key==ord("s"):
            image = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite("in_memory_to_disk.png", image)
            image = cv2.imread("in_memory_to_disk.png")
            
            #screenshot part begins
            
            # load the image, clone it, and setup the mouse callback function 
            #image = cv2.imread('img.jpeg') 
            clone = image.copy() 
            cv2.namedWindow("image") 
            cv2.setMouseCallback("image", shape_selection) 
            
            
            # keep looping until the 'q' key is pressed 
            while True: 
            	# display the image and wait for a keypress 
            	cv2.imshow("image", image) 
            	key = cv2.waitKey(1) & 0xFF
            
            	# press 'r' to reset the window 
            	if key == ord("r"): 
            		image = clone.copy() 
            
            	# if the 'c' key is pressed, break from the loop 
            	elif key == ord("c"): 
            		break
            
            if len(ref_point) == 2: 
            	crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]: 
            														ref_point[1][0]] 
            	cv2.imshow("crop_img", crop_img)
                
            	#cv2.waitKey(0)
            cv2.imwrite("crop_image.png", crop_img)
            cv2.imshow("crop_image.png", crop_img)
            cv2.waitKey(0)
            
        if key==27:
            break
    except AttributeError:
        print("shape not found")
# close all open windows 
#screenshot part ends here 

cap.release()
cv2.destroyAllWindows()