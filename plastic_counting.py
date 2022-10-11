# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

#input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"
input_video = "Input/plastic_test_sample.mp4"
# input_video = 0 # for live stream

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model(
    'plastic_bag_inference_graph', 'plastic_label_map.pbtxt')

roi = 640  # roi line position
deviation = 15  # the constant that represents the object counting area
camera_height = 0.0745  # m
camera_diameter = 0.25745  # mm
camera_focus = 0.26  # mm

object_counting_api.cumulative_object_counting_x_axis(input_video,
                                                      detection_graph,
                                                      category_index,
                                                      roi,
                                                      deviation,
                                                      camera_height,
                                                      camera_diameter,
                                                      camera_focus,
                                                      save_image=True)  # counting all the objects
