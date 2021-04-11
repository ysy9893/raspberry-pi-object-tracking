
# raspberry-pi-object-tracking
Implementation of object tracking on raspberry pi using Mobilenet SSD, NMS function, and traking library.


## This project is composed of three parts
### 1. Object Detection on Raspberry Pi using tensorflow lite Mobilenet SSD 
### 2. Non Maximum Suppression on outputs of Mobilenet SSD 
### 3. Object tracking using motpy library 

__Original project's result__   

![sample](video/raw_video.gif)

__After applying Non Maximum Suppression__

![sample](video/nms_video.gif)

__After applying Tracking algorithm__

![sample](video/tracking_video.gif)




**Credited by**        

Object detection : https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md   
Non Maximum Suppression : https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/   
Tracking : using motpy library using the command "pip install motpy"

**The purpose of this project:**      

I am currently developing wearable devices for certain targets(people) to give them  very meaningful informations by analyzing their surroundings. To understand surroundings, visual informations are inevitable (visusal system is the most powerful system among sensory systems) and there are a number of models imitating human visual system.    I chose Mobilenet SSD for object detection because it is suitable for real time object detection on edge device(raspberry pi's picam). Object detection using Mobilenet ssd was referred from (https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md) ,however, the result of original project seems unrefined with multiple bounding boxes on a single object and disconnected detections on a subsequent image frames.    
I have improved the project by using Non max suppression function eliminating multiple boxes on a single object and motpy library based on kalman filter tracking object in a subsequent frames.    
There are subtle changes compared with the original code. A newly added or changed parts would be stated below.    

##### 1. Object Detection on Raspberry Pi using tensorflow lite Mobilenet SSD    
You have to refer the original project's tutorial to setup an environment for object detection and to install the  TFLITE model provided by Google.       
--> https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md   

If you followed all the steps from the original project's tutorial, you would see the result like below.    

![sample](video/raw_video.gif)   

We can see the real time object detection in action, however, it's not what you've expected. 

It is because of untrimmed boxes which is duplicated and the flickering box on the detected object.   

To address such problems, I employed Non Maximum Suppression function and a tracking library.    

More details about Non Maximum Suppression are stated in this link : https://whatdhack.medium.com/reflections-on-non-maximum-suppression-nms-d2fce148ef0a     

More details about the kalman filter are stated in this link: https://kr.mathworks.com/videos/introduction-to-kalman-filters-for-object-tracking-79674.html       

##### 2. Non Maximum Suppression on outputs of Mobilenet SSD    
I wrote a script nms.py that implement Non Maximum Suppression next to main.py! It could be used as module by simply importing NMS from nms.py!


```python
from nms import NMS

```
There are multiple modifications I made to adjust the script to be perfectly compatible with main codes. (For nms I would explain all details about codes)

- Additional parameters (modified code)
The original script's nms function only takes bounding boxes(on a single image), overlapThresh, and the size of image as parameters. In our case, however, we need classes and confidence scores info for each bounding boxes. 
```python
def NMS(boxes,classes, probs, overlapThresh,imH,imW) #nms.py
```
```python
boxes,scores,classes=NMS(boxes,classes,scores,0.5,imH,imW) #main.py
```
- Condition when there's no detection
```python
if len(boxes) == 0:
    return []
```
- Convert datatype of bbox from Integer to Float for the bunch of division 
```python
if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
```
- Denormalize bbox[x1,y1,x2,y2] (modified code)
```python
x1 = boxes[:, 1]*imW #Applied to all x1
y1 = boxes[:, 0]*imH #Applied to all y1
x2 = boxes[:, 3]*imW #Applied to all x2
y2 = boxes[:, 2]*imH #Applied to all y2
```
- Compute the area of bounding boxes and sort on confidence scores in ascending order (If no probabilities(conf scores) are provided, sort on y2 in ascending order)
```python
area = (x2 - x1+1) * (y2 - y1+1)#By adding 1, we can prevent multiplication on floating point numbers.
idxs=y2.copy()
#If probabilities are provided, sort on them instead
if probs is not None: 
    idxs=probs
#Sort in ascending order 
idxs=np.argsort(idxs) #take index of element and sort on them 
```
- Keep looping while some indices still remain in the index list 
```python
while len(idxs)>0: 
   ##grab the last index in the index list 
   ##and add the index value to the list of picked indices(pick=[])
   last=len(idxs)-1 # grab the last (It means picking up the index of box with the highest conf score)
   i=idxs[last] #i indicates the confidence score of a particular bbox
   pick.append(i)
   
   ### find the largest (x, y) coordinates for the start of the bounding box 
   ##and the smallest (x, y) coordinates for the end of the bounding box
   
   ###Explainable with the image below
   xx1=np.maximum(x1[i],x1[idxs[:last]])
   xx2=np.minimum(x2[i],x2[idxs[:last]])
   yy1=np.maximum(y1[i],y1[idxs[:last]])
   yy2=np.minimum(y2[i],y2[idxs[:last]])
   
   #Compute the width and height of the bbox
   w=np.maximum(0,xx2-xx1+1)
   h=np.maximum(0,yy2-yy1+1)
   
   #Compute the ratio of overlap 
   overlap=(w*h)/area[idxs[:last]]
   
   #Delete all indices from the index list having overlap ratio greater than 
   #specified overlap threhold
   idxs=np.delete(idxs,
    np.concatenate(([last],np.where(overlap>overlapThresh)[0])))
```

- Return chosen boxes, confidence scores, classes 
```python
return boxes[pick], probs[pick], classes[pick]
```



##### 3. Object tracking using motpy library 

