# Dual-degree-project

You can download pre_trained weights for object detection model from https://drive.google.com/file/d/1oau9UbG-4u0TIr_ZorJIlxZNk1foUdux/view?usp=sharing.
Download and keep the files in trained_yolov3

#installation steps:
1. Download and install python if not available in your system from https://www.python.org/downloads/
2. Install pip from https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/#installing-pip-for-python-3
3. Install opencv from source https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/
4. python3 -m pip install pyautogui

#Steps to run object detection
1. cd trained_yolov3
2. download pre_trained weights for object detection model from https://drive.google.com/file/d/1oau9UbG-4u0TIr_ZorJIlxZNk1foUdux/view?usp=sharing.
3. python3 object_detection.py

#Steps to run object tracking
1. cd cv2_flann_sift_homography_tracker
2. python3 object_tracking.py

#Steps to run attitude determination model
1. cd attitude_determination
2. cd attitude_determination_EKF
3. python3 BoardDisplay_simple.py



