from rtmaps.base_component import BaseComponent  # base class
import rtmaps.types
from darkflow.net.build import TFNet  # load the model from main.py
import numpy as np


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        self.add_input("img", rtmaps.types.ANY)  # original image without resizing
        # Buffer size of 100 = maximum of 25 bounding boxes
        self.add_output("coord", rtmaps.types.INTEGER32, 100)  # define detection_result
        # Properties name cannot contain spaces
        self.add_property("threshold", 0.5)  # define threshold, max 1.0
        self.add_property("gpu_usage", 0.8)

    # Birth() will be called once at diagram execution startup
    def Birth(self):

        print('Python Birth')
        options = {"pbLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.pb",
                    "metaLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.meta",
                    "threshold": self.properties["threshold"].data,
                    "gpu": self.properties["gpu_usage"].data}

        self.tfnet = TFNet(options)


    # Core() is called every time you have a new input
    def Core(self):
        image = self.inputs["img"].ioelt.data  # create an ioelt from resized image at 608 x 608 pixels
        boxes = rtmaps.types.Ioelt()
        boxes.ts = self.inputs["img"].ioelt.ts
        boxes.data = []

        # Input image will be the resized image at 608,608 while the input dimensions will be that of the original.
        # Format of 'detection_result':
        # label, confidence, topleft_x / x1, topleft_y /  y1, botrightx / x2, botright_y / y2
        detection_result, json = self.tfnet.return_predict(image.image_data)

        for row, var in enumerate(detection_result):
            boxes.data.append(detection_result[row][2])
            boxes.data.append(detection_result[row][3])
            boxes.data.append(detection_result[row][4])
            boxes.data.append(detection_result[row][5])

        print(boxes.data)
        self.outputs["coord"].write(boxes)

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
