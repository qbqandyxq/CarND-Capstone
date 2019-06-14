from styx_msgs.msg import TrafficLight

import tensorflow as tf
import sys
import numpy as np
import os

THRESHOLD =0.5
NUM_CLASSES = 4
class TLClassifier(object):
    def __init__(self):
        self.path = '/home/lab/Desktop/gbq/test/Self-Driving-Car-ND-Capstone/ros/src/tl_detector/light_classification/tl_perception_models/tl_model.pb'
        self.build_network(self.path)
        
    
#uint8 UNKNOWN=4
#uint8 GREEN=2
#uint8 YELLOW=1
#uint8 RED=0

        print("network load ok")
    
    def build_network(self, model_path):
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                saved_graph = fid.read()
                od_graph_def.ParseFromString(saved_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.model_graph)
        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.classes= self.model_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.model_graph.get_tensor_by_name('num_detections:0')
        self.boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
            
            Args:
            image (cv::Mat): image containing the traffic light
            
            Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            
            """
        output = TrafficLight.UNKNOWN
        image_expanded = np.expand_dims(image, axis=0)
        with self.model_graph.as_default():
            _, scores, classes = self.sess.run([self.boxes, self.scores, self.classes], feed_dict={self.image_tensor: image_expanded})
        
        scores = np.squeeze(scores)
        #print(scores)
        classes = np.squeeze(classes).astype(np.int32)
        top_idx = scores.argmax()

        if scores[top_idx] >= THRESHOLD:
            print "get light"
            if classes[top_idx] == 0:
                output = TrafficLight.GREEN
            elif classes[top_idx] == 1:
                output = TrafficLight.GREEN
            elif classes[top_idx] == 2:
                output = TrafficLight.GREEN
            elif classes[top_idx] == 3:
                output = TrafficLight.GREEN
        else:
            print "can't get light status"
            output = TrafficLight.UNKNOWN
        return output


#def detect_traffic_light(self, scores, biggest_score_idx,
#                         classes,
#                         detected_light):
#    if scores[biggest_score_idx] > TL_THREHOLD:
#        rospy.logwarn("Current traffic light is: {}"
#                      .format(self.label_dict[classes[biggest_score_idx]]['name']))
#                      if classes[biggest_score_idx] == 1:
#                          detected_light = TrafficLight.GREEN
#                              elif classes[biggest_score_idx] == 2:
#                                  detected_light = TrafficLight.RED
#                                      elif classes[biggest_score_idx] == 3:
#                                          detected_light = TrafficLight.YELLOW
#                                  else:
#                                      rospy.logwarn("Not defined")
#                                          return detected_light



