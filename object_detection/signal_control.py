import cv2
import sys
import os

def get_white_pixels(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200)
    num_white_pixels = cv2.countNonZero(edges)
    return num_white_pixels


def get_match_score(pixel1, pixel2):
    return pixel1/pixel2 *100


def get_duration(match_score):
    if match_score<50:
        return 60
    elif match_score in range(51, 60):
        return 50
    elif match_score in range(61, 70):
        return 40
    elif match_score in range(71, 80):
        return 30
    else:
        return 20
    
    
def lane_control(sample1_image_path, sample2_image_path, reference_image_path):
    sample1_pixel_count = get_white_pixels(sample1_image_path)
    sample2_pixel_count = get_white_pixels(sample2_image_path)
    reference_pixel_count = get_white_pixels(reference_image_path)

    sample1_match = get_match_score(reference_pixel_count, sample1_pixel_count)
    sample2_match = get_match_score(reference_pixel_count, sample2_pixel_count)

    lane_1 = False
    lane_2 = False


        
    if sample1_match<sample2_match:
        elapsed_time = get_duration(sample1_match)
        lane_1 = True
    else:
        elapsed_time = get_duration(sample2_match)
        lane_2 = True
        
    if lane_1:
        return 1, elapsed_time
    else:
        return 2, elapsed_time
    



    
