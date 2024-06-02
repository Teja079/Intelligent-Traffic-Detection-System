import cv2
import sys
import os
from collections import OrderedDict
from ambulance_detection import detect_ambulance
from ultralytics import YOLO

def get_white_pixels(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200)
    num_white_pixels = cv2.countNonZero(edges)
    return num_white_pixels


def get_match_score(pixel1, pixel2):
    return pixel1/pixel2 *100


def count_cars_in_images(image_paths, model_path='yolov8n.pt'):
    model = YOLO(model_path)
    car_counts = []
    for image_path in image_paths:
        results = model.predict(source=image_path)
        print("^^^^^^^^^^^^^^^^^^^^")
        for result in results:
            # Access the 'boxes' attribute, which contains detected objects
            # Filter boxes by class_id for cars, assuming '2' is the class ID for cars
            car_detections = [box for box in result.boxes if box.cls == 2]
            num_cars = len(car_detections)
            car_counts.append(num_cars)
    return car_counts


def get_duration_v1(match_score, min_duration=15, max_duration=60):
    normalized_score = 100 - match_score  
    duration = min_duration + (normalized_score / 100) * (max_duration - min_duration)
    return int(min(duration, max_duration))


def get_duration(num_cars, min_duration=15, max_duration=40, max_cars_threshold=20):
    if num_cars <= max_cars_threshold:
        duration = min_duration + (num_cars / max_cars_threshold) * (max_duration - min_duration)
    else:
        duration = max_duration
    return int(duration)


    
def generate_order(match_scores):
    pass_order = []
    temp = match_scores
    for i in range(len(match_scores)):
        max_idx = temp.index(max(temp))
        pass_order.append(max_idx)
        temp[max_idx] = -1

    return pass_order


def signal_order_v2(pass_order, signal_lights):
    org_set = set(pass_order)
    comp_set = set([0, 1, 2, 3])
    diff = list(comp_set.difference(org_set))[-1]
    print(diff)
    try:
        result = pass_order.index(next(num for num in pass_order if pass_order.count(num) > 1))
        pass_order[result] = diff
        return pass_order
    except:
        return pass_order
    

def signal_order(pass_order, signal_lights):
    green_idx = signal_lights.index(True)
    temp1 = pass_order.index(green_idx)
    temp2 = pass_order[0]
    pass_order[0] = temp1
    pass_order[green_idx] = temp2

    return pass_order

    

def lane_control(lane_images, reference_image, label_path, checkpoint_path):
    
    
    lane_dict = {}
    reference_pixel = get_white_pixels(reference_image)

    ambulance_states = []
    for x in range(len(lane_images)):
        state = detect_ambulance(label_path, checkpoint_path, lane_images[x])
        ambulance_states.append(state)

    amb_idx = ambulance_states.index(True) if any(ambulance_states) else None
    match_scores = []
    for i in range(len(lane_images)):
        pixel_value = get_white_pixels(lane_images[i])
        match_score = get_match_score(pixel_value, reference_pixel)
        # print(match_score, "Match Score")
        
        if match_score in match_scores:
            match_score += 0.5

        match_scores.append(match_score)

    car_counts = count_cars_in_images(lane_images)

    print("**************************************")
    print(ambulance_states)
    print(amb_idx)
    print(match_scores)
    if amb_idx is not None:
        car_counts[amb_idx] = max(car_counts)+1


    for y in range(len(car_counts)):
        lane_dict[car_counts[y]] = [lane_images[y], get_duration(car_counts[y])]
        

    # sorted_match_scores = list(sorted(lane_dict.keys()))
    # sorted_order = {}
    # for sr_match in sorted_match_scores:
    #     sorted_order[sr_match] = lane_dict[sr_match]
    sorted_order = OrderedDict(sorted(lane_dict.items()))
    desc_sorted_order = OrderedDict(reversed(list(sorted_order.items())))

    print(desc_sorted_order)
    
    return desc_sorted_order
    
    


# import glob
# image_path = glob.glob('./images/*.jpg')
# print(count_cars_in_images(image_path))


    
