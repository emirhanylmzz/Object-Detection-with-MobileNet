# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 01:41:57 2019

@author: emirhanylmzz
"""

import cv2
import numpy as np
import time

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

def detect(frame):
    (w,h) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    for i in np.arange(0, detections.shape[2]):
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
  
        label = "{}: {:.2f}%".format(CLASSES[idx], detections[0, 0, i, 2] * 100)
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return frame
    
cap = cv2.VideoCapture(0)
start = time.time()
while True:
    _, frame = cap.read() 
    canvas = detect(frame)
    cv2.imshow('Video', canvas) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

end = time.time()
print("[INFO]  the program took {:.2f} seconds".format(end - start))
cap.release()
cv2.destroyAllWindows()
    
