from signal_control import *
from ambulance_detection import *
import streamlit as st
from PIL import Image
import glob
import streamlit.components.v1 as components
import time

def get_timer_cont(i, duration, bool_val):

    if bool_val:
        timer_html = f"""
                <div id="timer{i}" style="font-size: 20px;"> 00 : {duration}</div>
                <script>
                var timeLeft = {duration};
                var timerId = setInterval(countdown, 1000);
                function countdown() {{
                    if (timeLeft == -1) {{
                        clearTimeout(timerId);
                        document.getElementById("timer{i}").innerHTML = "00:00";
                    }} else {{
                        document.getElementById("timer{i}").innerHTML = "00: " + timeLeft;
                        timeLeft--;
                    }}
                }}
                </script>
                """
    else:
        timer_html = " "
    
    return timer_html


def display_columnar_images(sorted_order, images, lights):
    sorted_values = list(sorted_order.values())
    for count in range(4):
        col1, col2, col3, col4 = st.columns(4)
        columns = [col1, col2, col3, col4]
        print("*********************", sorted_values)
        for i in range(len(columns)):
            bool_val = False
            curr_light = lights[int(False)]
            with columns[i]:
                if i==count:
                    global sleep_duration
                    sleep_duration = sorted_values[i][1]
                    curr_light = lights[int(True)]
                    # st.write(f'Duration: {sorted_values[i][1]} Seconds')
                    bool_val = True
                
                st.write(sorted_values[i][0][:-9].split('.')[0])
                cont = "<h1 style='font-size: 85px;'>" + curr_light + "</h1>"
                st.write(cont, unsafe_allow_html=True)
                components.html(get_timer_cont(i, sorted_values[i][1], bool_val), height=100)
                st.image(sorted_values[i][0], use_column_width=True)

        time.sleep(sleep_duration)


def main_app():
    st.title('Traffic Control using CannyEdge Detection')

    IMG_DIR = './images'
    lane_img_names = ['lane1.jpg', 'lane2.jpg', 'lane3.jpg', 'lane4.jpg']
    
    ref_img = st.file_uploader('Upload a reference image', type=['jpg'])
    lane1_img = st.file_uploader('Upload Lane 1 image', type=['jpg'])
    lane2_img = st.file_uploader('Upload a Lane 2 image', type=['jpg'])
    lane3_img = st.file_uploader('Upload a Lane 3 image', type=['jpg'])
    lane4_img = st.file_uploader('Upload a Lane 4 image', type=['jpg'])

    lane_images = [lane1_img, lane2_img, lane3_img, lane4_img]

    red_signal = "🔴"
    green_signal = u"\U0001F7E2"

    lights = {
        int(True): green_signal, int(False): red_signal
    }

    if st.button('Submit') and all(lane_images):
        reference_image_path = os.path.join('images', 'reference', 'reference.jpg')
        with open(reference_image_path, 'wb') as f:
            f.write(ref_img.getbuffer())

        image_paths = []

        for i in range(len(lane_images)):
            img_path = os.path.join(IMG_DIR, lane_img_names[i])
            with open(img_path, 'wb') as f:
                f.write(lane_images[i].getbuffer())
            image_paths.append(img_path)
        



        img_objs = [Image.open(img_path).resize((150, 150)) for img_path in image_paths]

        # print(ambulance_states)
        # if any(ambulance_states):
        #     sorted_order = lane_control(image_paths, reference_image_path)
        #     signal_lights = ambulance_states
        #     pass_order = signal_order(pass_order, signal_lights)
        #     print("Pass Order 3: ", pass_order)
        #     st.success(f'Signal Order: {pass_order}')
        #     display_columnar_images(signal_lights, durations, img_objs, lights)
        # else:
        sorted_order = lane_control(image_paths, reference_image_path,  label_path, checkpoint_path)
        display_columnar_images(sorted_order, img_objs, lights)
        
            

if __name__=='__main__':
    checkpoint_path = 'frozen_inference_graph.pb'
    label_path = 'annotations/label_map.pbtxt'
    main_app()
    
    