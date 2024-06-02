from signal_control import *
from ambulance_detection import *
import streamlit as st
from PIL import Image

def main_app():
    st.title('Intelligent traffic Management System')
    
    ref_img = st.file_uploader('Upload a reference image', type=['jpg', 'png'])
    lane1_img = st.file_uploader('Upload Lane 1 image', type=['jpg', 'png'])
    lane2_img = st.file_uploader('Upload a Lane 2 image', type=['jpg', 'png'])
    if ref_img and lane1_img and lane2_img:
        
        with open(os.path.join('images','reference', ref_img.name), 'wb') as f:
            f.write(ref_img.getbuffer())
            
        with open(os.path.join('images','lane1', lane1_img.name), 'wb') as f:
            f.write(lane1_img.getbuffer())
            
        with open(os.path.join('images','lane2', lane2_img.name), 'wb') as f:
            f.write(lane2_img.getbuffer())
            
        ref_img_path = os.path.join('images','reference', ref_img.name)
        lane1_img_path = os.path.join('images','lane1', lane1_img.name)
        lane2_img_path = os.path.join('images','lane2', lane2_img.name)
        lane1_ambulance_state = detect_ambulance(label_path, checkpoint_path, lane1_img_path)
        lane2_ambulance_state = detect_ambulance(label_path, checkpoint_path, lane2_img_path)
        
        
        lane_no, elapsed_time = lane_control(lane1_img_path, lane2_img_path, ref_img_path)
        
        red_signal = "🔴"
        green_signal = u"\U0001F7E2"
        
        ambulance_state = lane1_ambulance_state or lane2_ambulance_state
        if ambulance_state:
            if lane1_ambulance_state:
                lane1_signal = green_signal
                lane2_signal = red_signal
                
            else:
                lane1_signal = red_signal
                lane2_signal = green_signal
                
        else:
            if lane_no==1:
                lane1_signal = green_signal
                lane2_signal = red_signal
                
            else:
                lane1_signal = red_signal
                lane2_signal = green_signal
            
            
        img1 = Image.open(lane1_img_path)
        img1 = img1.resize((300, 300))
        
        img2 = Image.open(lane2_img_path)
        img2 = img2.resize((300, 300))
        
        st.write(f'Elapsed_Time: {elapsed_time}')
        col1, col2 = st.columns(2)
        with col1:
            cont = "<h1 style='font-size: 85px;'>" + lane1_signal + "</h1>"
            st.write(cont, unsafe_allow_html=True)
            st.image(img1, use_column_width=True)
            
        with col2:
            cont = "<h1 style='font-size: 85px;'>" + lane2_signal + "</h1>"
            st.write(cont, unsafe_allow_html=True)
            st.image(img2, use_column_width=True)
        
            



if __name__=='__main__':
    checkpoint_path = 'frozen_inference_graph.pb'
    label_path = 'annotations/label_map.pbtxt'
    main_app()
    
    