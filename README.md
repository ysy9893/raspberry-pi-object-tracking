
# raspberry-pi-object-tracking
Implementation of object tracking on raspberry pi using Mobilenet SSD, NMS function, and traking library.


## This project is composed of three parts
### 1. Object Detection on Raspberry Pi using tensorflow lite Mobilenet SSD 
### 2. Non Maximum Suppression on outputs of Mobilenet SSD 
### 3. Object tracking using motpy library 

__Original project's result__
<img width="100%" scr='videos/raw_video.gif">



Credited by 
Object detection : https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
Non Maximum Suppression : https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
Tracking : using motpy library using the command "pip install motpy"

The purpose of this project: 
I am currently developing wearable devices for certain targets(people) to give them  very meaningful informations by analyzing their surroundings. To understand surroundings, visual informations are inevitable (visusal system is the most powerful  system among sensory systems) and there are a number of models imitating human visual system. I chose Mobilenet SSD for object detection because it is suitable for real time object detection on edge device(raspberry pi's picam). Object detection using Mobilenet ssd was referred from (https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md),
however, the result of original project seems unrefined with multiple bounding boxes on a single object and disconnected detections on a subsequent image frames. 
I improved the project by using Non max suppression function eliminating multiple boxes on a single object and motpy library based on kalman filter tracking object in a subsequent frames. 
There are subtle changes compared with the original code. A newly added or changed parts would be stated below. 

1. Object Detection on Raspberry Pi using tensorflow lite Mobilenet SSD 
You have to refer the original project's tutorial to setup an environment for object detection and to install the  TFLITE model provided by Google. 
--> https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

If you followed all the steps from the original project's tutorial, you would see the result like below. 



We can see the real time object detection in action, however, it's not what you've expected. 

It is because of untrimmed boxes which is duplicated and the flickering box on the detected object because of the disconnected detections in between two subsequent frames. 

To address such problems, I employed Non Maximum Suppression function and a tracking library. 

More details about Non Maximum Suppression are stated in this link :

More details about the kalman filter are stated in this link:

3. Non Maximum Suppression on outputs of Mobilenet SSD 

5. Object tracking using motpy library 

