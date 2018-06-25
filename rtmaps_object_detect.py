from rtmaps.base_component import BaseComponent # base class
from darkflow.net.build import TFNet
import rtmaps.types
import cv2
import numpy as np


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        self.add_input("image_in", rtmaps.types.ANY) # define input
        self.add_output("result", rtmaps.types.AUTO) # define output
        self.add_output("box_out_1", rtmaps.types.DRAWING_OBJECT, 10)
        self.add_output("box_out_2", rtmaps.types.DRAWING_OBJECT, 10)
        # self.add_output("box_out_3", rtmaps.types.DRAWING_OBJECT, 10)

        # property names should not contain spaces
        self.add_property("threshold", 0.1) # define threshold, max 1.0
        self.add_property("gpu_usage", 0.8) # define gpu usage, max 0.8

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        print('Python Birth')
        # options = {"pbLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.pb",
        #            "metaLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.meta",
        #            "threshold": self.properties["threshold"].data,
        #            "gpu": self.properties["gpu_usage"].data}
        # tfnet = TFNet(options)

    # Core() is called every time you have a new input
    def Core(self):
        options = {"pbLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.pb",
                   "metaLoad": "C:/Program Files/Intempora/RTMaps 4/packages/rtmaps_object_detect/darkflow-master/built_graph/yolo.meta",
                   "threshold": self.properties["threshold"].data}
                   # ,"gpu": self.properties["gpu_usage"].data}
        tfnet = TFNet(options)  
        img = self.inputs["image_in"].ioelt.data # create an ioelt from the input
        print('hello')
        output, json = tfnet.return_predict(img.image_data) # output labels and pixel coordinates of bounding box
        print(output)

        # We create a Rectangle DrawingObject surrounding what we detected for 3 objects
        # output 1
        # output1 = rtmaps.types.Ioelt()  # create an Ioelt
        # output1.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        # output1.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        # output1.data.kind = 2  # set the kind as Rectangle
        # output1.data.color = 255
        # output1.data.width = 3  # set its width
        # output1.data.data = rtmaps.types.Rectangle()  # create a Rectangle and set the coordinate of its diagonal
        # output1.data.data.x1 = output[0][2] # * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        # print(output1.data.data.x1)
        # output1.data.data.y1 = output[0][3] # * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        # print(output1.data.data.y1)
        # output1.data.data.x2 = output[0][4] # * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        # print(output1.data.data.x2)
        # output1.data.data.y2 = output[0][5] # * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        # print(output1.data.data.y2)

        # # output 2
        # output2 = rtmaps.types.Ioelt()  # create an Ioelt
        # output2.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        # output2.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        # output2.data.kind = 2  # set the kind as Rectangle
        # output2.data.color = 255
        # output2.data.width = 3  # set its width
        # output2.data.data = rtmaps.types.Rectangle()  # create a Rectangle and set the coordinate of its diagonal
        # output2.data.data.x1 = output[1][2] # * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        # print(output2.data.data.x1)
        # output2.data.data.y1 = output[1][3] # * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        # print(output2.data.data.y1)
        # output2.data.data.x2 = output[1][4] # * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        # print(output2.data.data.x2)
        # output2.data.data.y2 = output[1][5] # * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        # print(output2.data.data.y2)

        # output 3
        # output3 = rtmaps.types.Ioelt()  # create an Ioelt
        # output3.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        # output3.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        # output3.data.kind = 2  # set the kind as Rectangle
        # output3.data.color = 255
        # output3.data.width = 3  # set its width
        # output3.data.data = rtmaps.types.Rectangle()  # create a Rectangle and set the coordinate of its diagonal
        # output3.data.data.x1 = output[2][2] # * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        # print(output3.data.data.x1)
        # output3.data.data.y1 = output[2][3] # * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        # print(output3.data.data.y1)
        # output3.data.data.x2 = output[2][4] # * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        # print(output3.data.data.x2)
        # output3.data.data.y2 = output[2][5] # * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        # print(output3.data.data.y2)

        self.outputs["result"].write(output) # output as array
        self.outputs["box_out_1"].write(output1) # output as drawing
        self.outputs["box_out_2"].write(output2)
        # self.outputs["box_out_3"].write(output3)

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass