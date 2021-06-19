from flask import Flask, render_template, Response
import cv2 
import numpy as np
from matplotlib import pyplot as plt
app = Flask(__name__)
# import pyautogui
#import imutils

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

MIN_MATCH_COUNT = 10
cap = cv2.VideoCapture('jio.mp4')

img = cv2.imread(filename = "crop_image.png", flags = cv2.IMREAD_GRAYSCALE)
#input_img = cv2.imread("crop_image.png")
#img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

# creating the SIFT algorithm for detecting features
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SIFT_create()
#sift = cv2.ORB_create() 
#sift = cv2.KAZE_create()
  
# find the keypoints and descriptors with SIFT 
kp_image, desc_image = sift.detectAndCompute(img, None) 
  
# intializing the dictionary 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
search_params = dict(checks=50) 
  
# by using Flann feature Matcher 
flann = cv2.FlannBasedMatcher(index_params, search_params) 

clusters = np.array([desc_image])
flann.add(clusters)
flann.train()

# reading the frame
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
            
            # sending frames to webserver
            
            ret, buffer = cv2.imencode('.jpg', homography)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
        else:
             print( "Not enough matches are found - {}/{}".format(len(good_points), MIN_MATCH_COUNT) )
             matchesMask = None
        key = cv2.waitKey(1)
        if key == 27:
            break

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(object_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="localhost", port=3080, debug=True)        
#cap.release()
#cv2.destroyAllWindows()
