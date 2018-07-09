from rtmaps.base_component import BaseComponent  # base class
import rtmaps.types
from darkflow.net.build import TFNet  # load the model from main.py
import numpy as np


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        self.add_input("img", rtmaps.types.ANY)  # original image without resiz    
        self.add_output("coord", rtmaps.types.INTEGER32, 15)  # define detection_result
        # Properties name cannot contain spaces
        self.add_property("threshold", 0.5)  # define threshold, max 1.0
        self.add_property("gpu_usage", 0.8)
        self.add_property("image_height", 480)
        self.add_property("image_width", 640)

    # Birth() will be called once at diagram execution startup
    def Birth(self):

        print('Python Birth')
        options = {"pbLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/built_graph/yolo.pb",
                   "metaLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/built_graph/yolo.meta",
                   "threshold": self.properties["threshold"].data,
                   "gpu": self.properties["gpu_usage"].data}

        self.tfnet = TFNet(options)

    # Core() is called every time you have a new input
    def Core(self):
        resized = self.inputs["img"].ioelt.data  # create an ioelt from resized image at 608 x 608 pixels
        h_original = self.properties["image_height"].data  # height and width from original image before resize
        w_original = self.properties["image_width"].data
        print(resized.image_data.shape)

        boxes = rtmaps.types.Ioelt()
        boxes.ts = self.inputs["img"].ioelt.ts
        boxes.data = []

        detection_result, json = self.tfnet.return_predict(resized.image_data, h_original, w_original)

        try:
            boxes.data = []
            if detection_result[0] is not None and detection_result[0][1] >= self.properties["threshold"].data:
                boxes.data.append(detection_result[0][2])
                boxes.data.append(detection_result[0][3])
                boxes.data.append(detection_result[0][4])
                boxes.data.append(detection_result[0][5])

        except IndexError:
            boxes.data.insert(0, 0)
            boxes.data.insert(1, 0)
            boxes.data.insert(2, 0)
            boxes.data.insert(3, 0)

        try:
            if detection_result[1] is not None and detection_result[1][1] >= self.properties["threshold"].data:
                boxes.data.append(detection_result[1][2])
                boxes.data.append(detection_result[1][3])
                boxes.data.append(detection_result[1][4])
                boxes.data.append(detection_result[1][5])

        except IndexError:
            boxes.data.insert(4, 0)
            boxes.data.insert(5, 0)
            boxes.data.insert(6, 0)
            boxes.data.insert(7, 0)

        try:
            if detection_result[2] is not None and detection_result[2][1] >= self.properties["threshold"].data:
                boxes.data.append(detection_result[2][2])
                boxes.data.append(detection_result[2][3])
                boxes.data.append(detection_result[2][4])
                boxes.data.append(detection_result[2][5])

        except IndexError:
            boxes.data.insert(8, 0)
            boxes.data.insert(9, 0)
            boxes.data.insert(10, 0)
            boxes.data.insert(11, 0)

        print(boxes.data)

        self.outputs["coord"].write(boxes)

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
