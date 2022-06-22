import numpy as np
import os
import cv2
from email.mime import image
from flask import Flask, jsonify, request
#from yolo_detection_images import detectObjects
# from flask_cors import CORS
from werkzeug.utils import secure_filename
# import os
# from flask_restx import Api, Resource

def detectObjects(img_path):
    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    #modelConfiguration = 'cfg/yolov3.cfg'
    #modelWeights = 'yolov3.weights'

    #labelsPath = 'obj.names'
    #CUR_DIR = os.path.abspath('.')
    modelConfiguration = '/home/g2019sun0925/flask_yolo_api/YOLO-v3-Object-Detection/cfg/yolov3.cfg'
    #modelConfiguration = os.path.join(CUR_DIR,'cfg/yolov3.cfg')
    modelWeights = '/home/g2019sun0925/flask_yolo_api/YOLO-v3-Object-Detection/yolov3.weights'

    #modelWeights = os.path.join(CUR_DIR,'yolov3.weights')
    labelsPath = '/home/g2019sun0925/flask_yolo_api/YOLO-v3-Object-Detection/obj.names'
    #labelsPath = os.path.join(CUR_DIR, 'obj.names')
    
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    image = cv2.imread(img_path)
    #image = image.resize(int(image.width/2),int(image.height/2))
    (H, W) = image.shape[:2]
    print("h:w = ", H, W)
    #Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i - 1] for i in net.getUnconnectedOutLayers()] # layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

    outputs={}

    if len(detectionNMS)>0:
        #outputs['detections']={}
        #outputs['detections']['labels']=[]
        
        for i in detectionNMS.flatten():
            #detection={}
            #detection['Label']=labels[classIDs[i]]
            outputs['detections']=1
            outputs['label']=labels[classIDs[i]]

            #detection['confidence']=confidences[i]
            outputs['confidence']=confidences[i]
            #detection['X']=boxes[i][0]
            #detection['Y']=boxes[i][1]
            #detection['Width']=boxes[i][2] # width
            #detection['Height']=boxes[i][3] # height
            #outputs['detections']['labels'].append(detection)
    else:
        outputs['detections']=0
        outputs['label']=' '
        outputs['confidence']= 0

    return outputs
