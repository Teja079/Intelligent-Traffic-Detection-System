import os 
import cv2 
import numpy as np 
import tensorflow as tf 
import sys
from object_detection.utils import label_map_util 
from object_detection.utils import visualization_utils as vis_util
import random


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.98)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def detect_ambulance(label_path, checkpoint_path, image_path, num_classes = 1):
    label_map = label_map_util.load_labelmap(label_path) 
    categories = label_map_util.convert_label_map_to_categories( 
            label_map, max_num_classes
         = num_classes
        , use_display_name = True) 
    category_index = label_map_util.create_category_index(categories) 

    detection_graph = tf.compat.v1.Graph() 
    with detection_graph.as_default(): 
        od_graph_def = tf.compat.v1.GraphDef() 
        with tf.compat.v1.gfile.GFile(checkpoint_path, 'rb') as fid: 
            serialized_graph = fid.read() 
            od_graph_def.ParseFromString(serialized_graph) 
            tf.compat.v1.import_graph_def(od_graph_def, name ='') 
    
        sess = tf.compat.v1.Session(graph = detection_graph) 
        
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') 
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 
    num_detections = detection_graph.get_tensor_by_name('num_detections:0') 


    image = cv2.imread(image_path)
    image_expanded = np.expand_dims(image, axis = 0) 
    (boxes, scores, classes, num) = sess.run( 
        [detection_boxes, detection_scores, detection_classes, num_detections], 
        feed_dict ={image_tensor: image_expanded}) 
    
    
    vis_util.visualize_boxes_and_labels_on_image_array( 
        image, 
        np.squeeze(boxes), 
        np.squeeze(classes).astype(np.int32), 
        np.squeeze(scores), 
        category_index, 
        use_normalized_coordinates = True, 
        line_thickness = 2, 
        min_score_thresh = 0.95) 
    
    max_score = np.max(scores)
    detected_class = category_index[np.squeeze(classes)[np.argmax(scores)]]['name']
    
    if max_score > 0.90 and detected_class == 'ambulance':
        return True
    else:
        return False

    
