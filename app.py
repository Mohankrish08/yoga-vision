# importing libraries
import streamlit as st
import streamlit_lottie as st_lottie
from streamlit_option_menu import option_menu
import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
import numpy as np 
import matplotlib.pyplot as plt
import requests
from PIL import Image
import os

# set page config 
st.set_page_config(page_title='Yoga vision', page_icon=":rocket:", layout='wide')

# loading animations
def loader_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# loading assets

front = loader_url('https://lottie.host/bf50b9df-5190-4b7e-b512-ab6465d35e23/zTgvT8DAOd.json')
image = loader_url('https://lottie.host/bf9ee849-b5f2-435a-896a-7b4e40d9c1b5/3LPWfn22Rl.json')
video = loader_url('https://lottie.host/0b1144a9-3356-411f-a821-c7e6f1ac354d/sOzGTpD7CM.json')
workout = loader_url('https://lottie.host/83220589-c58f-466d-914b-94e637c1ea29/jMcKpKj6VN.json')


# mediapipe 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# detections
def mediapipe_detection(image, model):
    result = model.process(image)
    return image, result

# landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )

# model
model = YOLO('best.pt')

# class names
classnames = ['Adho', 'Alanasana', 'Anjaneyasana', 'Ardha', 'Ashta', 'Baddha', 'Bakasana', 'Balasana', 'Bandha', 
              'Bhujangasana', 'Bitilasana', 'Camatkarasana', 'Chandrasana', 'Dhanurasana', 'Eka', 'Garudasana', 
              'Halasana', 'Hanumanasana', 'Hasta', 'Kapotasana', 'Konasana', 'Malasana', 'Marjaryasana', 'Matsyendrasana', 
              'Mayurasana', 'Mukha', 'Navasana', 'One', 'Pada', 'Padangusthasana', 'Padmasana', 'Parsva', 'Parsvakonasana', 
              'Parsvottanasana', 'Paschimottanasana', 'Phalakasana', 'Pincha', 'Rajakapotasana', 'Salamba', 'Sarvangasana',
              'Setu', 'Sivasana', 'Supta', 'Svanasana', 'Svsnssana', 'Three', 'Trikonasana', 'Two', 'Upavistha', 'Urdhva', 
              'Ustrasana', 'Utkatasana', 'Uttanasana', 'Utthita', 'Vasisthasana', 'Virabhadrasana', 'Vrksasana']

# functions

def yoga_image(img):
    image = Image.open(img)
    out = image.save('yoga.jpg')
    img_file = cv2.imread('yoga.jpg')
    results = model(img_file)[0]
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            cv2.rectangle(img_file, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
            
            new_img, out = mediapipe_detection(img_file,pose)
            
            draw_landmarks(new_img, out)
            st.image(new_img[:,:,::-1])
            word = classnames[int(class_id)]
    return word

def yoga_video(file):
    cap = cv2.VideoCapture(file)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1800)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break if no frame is read

            results = model(frame, show=False)[0]
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 4)
                
                cv2.rectangle(frame, (10,10), (300,80), (255,0,0), -1)
                cv2.putText(frame, classnames[int(class_id)], (40,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)

                # confidence
                cv2.rectangle(frame, (300,10), (500,80), (0,255,0), -1)
                cv2.putText(frame, str(f'{score:.2f}'), (350,45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)
                    
                image, out = mediapipe_detection(frame,pose)
            
                draw_landmarks(image, out)
                
                cv2.imshow('Output', frame)
                
            if cv2.waitKey(1) == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


# Home page sidebar
with st.sidebar:
    with st.container():
        i,j = st.columns((4,4))
        with i:
            st.empty()
        with j:
            st.empty()

    choose = option_menu(
        "Yoga vision",
        ['Home', 'Image', 'Video', 'Workout'],
        menu_icon='vision',
        default_index=0,
        orientation='vertical'
    )

if choose == 'Home':

    st.markdown("<h1 style='text-align: center;'>Welcome to Yoga vision</h1>", unsafe_allow_html=True)

    st.write('-------')

    st.markdown("""
            Experience seamless yoga pose detection and visualization with our innovative project using YOLOv8 and MediaPipe. 
            Our advanced system accurately identifies various yoga poses in real-time, providing users with instant feedback on their form. 
            Elevate your yoga practice with this cutting-edge technology for a more informed and effective workout.
            """, unsafe_allow_html=True)
    st.lottie(front, height=400, key='yoga')

elif choose == 'Image':

    st.title('For Images')
    st.lottie(image, height=200, key='image')
    upload_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'tiff'])

    if st.button('Convert') and upload_file is not None:
        
        out = yoga_image(upload_file)
        st.markdown(f"<h2 style='text-align: center;'>{out}</h2>", unsafe_allow_html=True)
        
        os.remove('yoga.jpg')

elif choose == 'Video':

    st.title("For Video")

    st.lottie(video, height=300, key='video')

    file = st.file_uploader(label='Upload your video', type=['.mp4', '.mkv', '.HEVC'])

    if st.button('Convert') and file is not None:
        yoga_video(file)


elif choose == 'Workout':

    st.title("Workout")

    st.lottie(workout, height=300, key='workout')

    selected_pose = st.selectbox(label='Select your Yoga', options=['Adho', 'Alanasana', 'Anjaneyasana', 'Ardha', 'Ashta', 'Baddha', 'Bakasana', 'Balasana', 'Bandha', 
              'Bhujangasana', 'Bitilasana', 'Camatkarasana', 'Chandrasana', 'Dhanurasana', 'Eka', 'Garudasana', 
              'Halasana', 'Hanumanasana', 'Hasta', 'Kapotasana', 'Konasana', 'Malasana', 'Marjaryasana', 'Matsyendrasana', 
              'Mayurasana', 'Mukha', 'Navasana', 'One', 'Pada', 'Padangusthasana', 'Padmasana', 'Parsva', 'Parsvakonasana', 
              'Parsvottanasana', 'Paschimottanasana', 'Phalakasana', 'Pincha', 'Rajakapotasana', 'Salamba', 'Sarvangasana',
              'Setu', 'Sivasana', 'Supta', 'Svanasana', 'Svsnssana', 'Three', 'Trikonasana', 'Two', 'Upavistha', 'Urdhva', 
              'Ustrasana', 'Utkatasana', 'Uttanasana', 'Utthita', 'Vasisthasana', 'Virabhadrasana', 'Vrksasana'])
    
    print('-----------------------')
    print(selected_pose)
    print('-----------------------')    

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1800)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while True:
            
            ret, frame = cap.read()
            
            results = model(frame, show=False, conf=0.7)[0]
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 4)
                
                cv2.rectangle(frame, (10,10), (300,80), (255,0,0), -1)
                cv2.putText(frame, classnames[int(class_id)], (40,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)

                # confidence
                cv2.rectangle(frame, (300,10), (500,80), (0,255,0), -1)
                cv2.putText(frame, str(f'{score:.2f}'), (350,45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)

                # feedback
                cv2.rectangle(frame, (800,10),(1500,80),(23,210,28),-1)
                try:

                    if selected_pose == classnames[int(class_id)]:
                        cv2.putText(frame, "You're doing Great!", (900,50),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Posture is irrelevant", (900,50),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                except:
                    pass
                
                    
                image, out = mediapipe_detection(frame,pose)
            
                draw_landmarks(image, out)
                
                cv2.imshow('Output', frame)
                
            if cv2.waitKey(1) == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
