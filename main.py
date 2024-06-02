from signal_control import *
from ambulance_detection import *



if __name__=='__main__':
    print(sys.argv)
    reference_image_path = os.path.join('images','reference', sys.argv[1])
    sample1_image_path = os.path.join('images','lane1', sys.argv[2])
    sample2_image_path = os.path.join('images','lane2', sys.argv[3])
    print(sample1_image_path, sample2_image_path, reference_image_path)
    checkpoint_path = 'frozen_inference_graph.pb'
    label_path = 'annotations/label_map.pbtxt'
    image_path =  'amb1.jpg'
    lane1_ambulance_state = detect_ambulance(label_path, checkpoint_path, sample1_image_path)
    # lane2_ambulance_state = detect_ambulance(label_path, checkpoint_path, sample2_image_path)
    # ambulance_state = lane1_ambulance_state or lane2_ambulance_state
    # if lane1_ambulance_state:
    #     return
    # lane_control(sample1_image_path, sample2_image_path, reference_image_path, ambulance_state)