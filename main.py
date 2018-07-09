from darkflow.net.build import TFNet
import cv2

options = {"pbLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.pb",
           "metaLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.meta",
           "threshold": 0.1,
           "gpu": 0.8}

tfnet = TFNet(options)
imgcv = cv2.imread("./sample_img/sample_horses.jpg")

# image input must be initialized outside of tfnet.return_predict
result, json = tfnet.return_predict(imgcv, 512, 773)
print(json)