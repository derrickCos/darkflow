# ---------------- SAMPLE -------------------
#########################################################################################################
#                            How to use TensorFlow from Python within RTMaps                            #
#                                                                                                       #
# Sample based on:                                                                                      #
#     https://github.com/tensorflow/models/tree/master/object_detection                                 #
# Using frozen model downloaded from:                                                                   #
#     http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz    #
#########################################################################################################

import rtmaps.types
import numpy as np
import tensorflow as tf
import os
import re
from rtmaps.base_component import BaseComponent  # base class
import time
from darkflow.net import help
from darkflow.net import flow
from darkflow.net.ops import op_create, identity
from darkflow.net.ops import HEADER, LINE
from darkflow.net.framework import create_framework
from darkflow.dark.darknet import Darknet
import json
import os


class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)  # call base class constructor
        self.add_input("image_in", rtmaps.types.ANY)
        self.add_output("max_score_out", rtmaps.types.AUTO, 10)
        self.add_output("max_score_box_out", rtmaps.types.DRAWING_OBJECT, 10)
        self.add_output("max_score_classe_out", rtmaps.types.AUTO, 10)
        self.add_output("max_score_description_out", rtmaps.types.DRAWING_OBJECT, 10)

    # Detection from TensorFlow returns indexes corresponding to each class.
    # This function returns a dictionnary to match classes names and indexes
    # {
    #    1: 'person',
    #    2: 'bicycle',
    #    3: 'car',
    #    ...
    # }
    def LoadClassesLabels(self, labels_path):
        labels = {}
        current_id = 0
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                match = re.match(r'id: ([0-9]+)', line)
                if match:
                    current_id = int(match.group(1))
                else:
                    match = re.match(r'display_name: \"([a-z:A-Z: ]+)\"', line)
                    if match:
                        labels[current_id] = match.group(1)
        # report_info(str(labels))
        return labels

    # Birth() will be called once at diagram execution startup
    def Birth(self):
        # Path to model data
        PATH_TO_MODEL_DIR = 'C:/Users/Administrator/Desktop/rtmaps_object_detect/darkflow-master/built_graph'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_MODEL = 'C:/Users/Administrator/Desktop/rtmaps_object_detect/darkflow-master/built_graph/yolo.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = 'C:/Users/Administrator/Desktop/rtmaps_object_detect/darkflow-master/built_graph/mscoco_label_map.pbtxt'

        # Load classes ids/labels dictionnary
        self.labels = self.LoadClassesLabels(PATH_TO_LABELS)

        # Load a frozen TensorFlow model into memory.
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
            # Create TensorFlow session
        self.sess = tf.Session(graph=self.graph)
        self.feed = dict()  # other placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

        self.setup_meta_ops()
        # Retrieve input image tensor
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        # Retrieve bouding box (output) tensor
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        # Retrieve score (output) tensor
        # Each score represent how level of confidence for each of the objects.
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        # Retrieve class index (output) tensor
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')
        # Retrieve number of detection (output) tensor
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    # Core() is called everytime you have a new input
    def Core(self):
        # We retrieve the input image as a numpy array from self.image_in (depends on the component defeinition)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(self.inputs["image_in"].ioelt.data.image_data.copy(), axis=0)

        # Run the TensorFlow model against our input image
        (boxes, scores, classes, num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                 feed_dict={self.image_tensor: image_np_expanded})

        # We retrieve boxes, scores, classes and num_detection from the model execution
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Let's select the higher score detection result
        max_score = 0.0
        max_score_box = [0.0, 0.0, 0.0, 0.0]
        max_score_classe = 0
        for i in range(boxes.shape[0]):
            if scores is not None and scores[i] > max_score:
                max_score = scores[i]
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                max_score_box = [xmin, ymin, xmax, ymax]
                max_score_classe = classes[i]

        # We create a Rectangle DrawingObject surrounding what we detected
        output2 = rtmaps.types.Ioelt()  # create an Ioelt
        output2.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        output2.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        output2.data.kind = 2  # set the kind as Rectangle
        output2.data.color = 255
        output2.data.width = 3  # set its width
        output2.data.data = rtmaps.types.Rectangle()  # create a Rectangle and set the coordinate of its diagonal
        output2.data.data.x1 = max_score_box[0] * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        output2.data.data.y1 = max_score_box[1] * self.inputs["image_in"].ioelt.data.image_data.shape[0]
        output2.data.data.x2 = max_score_box[2] * self.inputs["image_in"].ioelt.data.image_data.shape[1]
        output2.data.data.y2 = max_score_box[3] * self.inputs["image_in"].ioelt.data.image_data.shape[0]

        # We create a Text DrawingObject containing the name of what have been detected
        output4 = rtmaps.types.Ioelt()  # create an Ioelt
        output4.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        output4.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        output4.data.kind = 5  # set the kind as Text
        output4.data.color = 255
        output4.data.width = 3  # set its width
        output4.data.data = rtmaps.types.Text()  # create a Text and write the name of what we detect
        output4.data.data.x = 15
        output4.data.data.y = 15
        output4.data.data.cwidth = 15  # set its characters width
        output4.data.data.cheight = 15  # set its characters height
        output4.data.data.text = str(self.labels.get(int(max_score_classe), "UNKNOWN")) + " - " + str(max_score * 100)

        # We create a Rectangle DrawingObject equal to 0
        output2_2 = rtmaps.types.Ioelt()  # create an Ioelt
        output2_2.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        output2_2.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        output2_2.data.kind = 2  # set the kind as Rectangle
        output2_2.data.color = 255
        output2_2.data.width = 3  # set its width
        output2_2.data.data = rtmaps.types.Rectangle()  # create a null Rectangle
        output2_2.data.data.x1 = 0.0
        output2_2.data.data.y1 = 0.0
        output2_2.data.data.x2 = 0.0
        output2_2.data.data.y2 = 0.0

        # We create a Text DrawingObject containing "NO DETECTION RESULT"
        output4_2 = rtmaps.types.Ioelt()  # create an Ioelt
        output4_2.ts = self.inputs["image_in"].ioelt.ts  # get the ts from the input
        output4_2.data = rtmaps.types.DrawingObject()  # create a DrawingObject
        output4_2.data.kind = 5  # set the kind as Text
        output4_2.data.color = 255
        output4_2.data.width = 3  # set its width
        output4_2.data.data = rtmaps.types.Text()  # create a Text and write "NO DETECTION RESULT"
        output4_2.data.data.x = 15
        output4_2.data.data.y = 15
        output4_2.data.data.cwidth = 15  # set its characters width
        output4_2.data.data.cheight = 15  # set its characters height
        output4_2.data.data.text = str("NO DETECTION RESULT")

        # Let's write the selected result on the component outputs
        # If we detect something we write our output2 and our output4 containing respectively our rectangle and our text
        if max_score > 0.5:
            self.outputs["max_score_out"].write(float(max_score))
            self.outputs["max_score_box_out"].write(output2)
            self.outputs["max_score_classe_out"].write(str(self.labels.get(int(max_score_classe), "UNKNOWN")))
            self.outputs["max_score_description_out"].write(output4)

        # If we don't detect anything we write our output2_2 and our output4_2 containing respectively our null rectangle and our text reporting "NO DETECTION RESULT"
        else:
            self.outputs["max_score_out"].write(0.0)
            self.outputs["max_score_box_out"].write(output2_2)
            self.outputs["max_score_classe_out"].write(str("UNKNOWN"))
            self.outputs["max_score_description_out"].write(output4_2)

    # Death() will be called once at diagram execution shutdown
    def Death(self):
        self.sess.close()
