#########################################################################################################
#                            How to use Darkflow from Python within RTMaps                            #
#                                                                                                       #
# Sample based on:                                                                                      #
#     https://github.com/tensorflow/models/tree/master/object_detection                                 #
# Using frozen model downloaded from:                                                                   #
#     http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz    #
#########################################################################################################

from darkflow.net.build import TFNet
import cv2
import numpy as np

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu":0.8}
tfnet = TFNet(options)
imgcv = cv2.imread("./sample_img/sample_horses.jpg")
results,json= tfnet.return_predict(imgcv)

print (results)
print (json)

# initializing
label = []
confidence = []
# Bounding box coordinates - top left corner
topleft_x=[]
topleft_y=[]
# Bounding box coordinates - bot right corner
botright_x=[]
botright_y=[]

# Format: label,conf,topleft_x,topleft_y,botright_x,botright_y
for index,result in enumerate(results):
    if results is None:
        continue
    # print (index)
    # print (list)
    label.append(result[0])
    confidence.append(result[1])
    topleft_x.append(result[2])
    topleft_y.append(result[3])
    botright_x.append(result[4])
    botright_y.append(result[5])

print(label)
print(confidence)
print(topleft_x)
print(topleft_y)
print(botright_x)
print(botright_y)




