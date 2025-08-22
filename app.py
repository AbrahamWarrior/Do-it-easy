<<<<<<< HEAD
import streamlit as st 
import copy
from multiprocessing import Queue, Process
import cv2 as cv
import numpy as np
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from utils import CvFpsCalc
# from main import draw_landmarks, draw_stick_figure

from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject
from functions import calculate_angle
from turn import get_ice_servers


_SENTINEL_ = "_SENTINEL_"


def main():
    # Creamos tres columnas: título, primera imagen, segunda imagen
    col1, col2, col3 = st.columns([1, 3, 1])

    # Columna 1: título
    col1.image("images/UTEQ.png", width=200)

    # Columna 2: primera imagen
    col2.image("images/logo2.png", width=400)

    # Columna 3: segunda imagen
    col3.image("images/UCR.png", width=100)

    # Sidebar
    st.sidebar.success("Select a workout above.")

    # Nombres del equipo debajo de todo
    st.markdown("""
    **Carlos Abraham Higuera Flores**  
    **Santiago Noriega Hernández**  
    **Angel Leon Herrera Rosales**
    """)

    # Información con texto
    st.info("If you would like to send us any comments to improve the app, please fill out the following form😁:")

    contact_form = """
    <form action="https://formsubmit.co/abrahamhf45@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="email" name="email" placeholder= "Your email:" required>
        <input type="text" name="feedback" placeholder= "Any feedback:" required>
        <input type="hidden" name="_next" value="https://huggingface.co/spaces/fatimahhussain/workoutwizard">
        <button type="submit"">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Imagen final al pie de la app con publicidad a la derecha
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.image("images/3foto.jpg", width=180)

    with col_right:
        st.image("images/publicidad.jpg", width=180)


def pose_process(
    in_queue: Queue, #queue where the function will read input items
    out_queue: Queue, #queue where function will put output items
):
    # Inicializa el objeto Pose de mediapipe
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = pose.process(input_item)

        # Check if landmarks data is present
        if results.pose_landmarks and results.pose_landmarks.landmark:
            landmarks = [
                FakeLandmarkObject(
                    x=pose_landmark.x,
                    y=pose_landmark.y,
                    z=pose_landmark.z,
                ) for pose_landmark in results.pose_landmarks.landmark
            ]
        else:
            landmarks = []

        # Crear objeto picklable_results
        picklable_results = FakeResultObject(
            pose_landmarks=FakeLandmarksObject(landmark=landmarks)
        )

        # Poner el resultado en la cola
        out_queue.put_nowait(picklable_results)


if __name__ == "__main__":
    main()
=======
import streamlit as st 
import copy
from multiprocessing import Queue, Process
import cv2 as cv
import numpy as np
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from utils import CvFpsCalc
# from main import draw_landmarks, draw_stick_figure

from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject
from functions import calculate_angle
from turn import get_ice_servers


_SENTINEL_ = "_SENTINEL_"


def main():
    # Creamos tres columnas: título, primera imagen, segunda imagen
    col1, col2, col3 = st.columns([1, 3, 1])

    # Columna 1: título
    col1.image("images/UTEQ.png", width=200)

    # Columna 2: primera imagen
    col2.image("images/logo2.png", width=400)

    # Columna 3: segunda imagen
    col3.image("images/UCR.png", width=100)

    # Sidebar
    st.sidebar.success("Select a workout above.")

    # Nombres del equipo debajo de todo
    st.markdown("""
    **Carlos Abraham Higuera Flores**  
    **Santiago Noriega Hernández**  
    **Angel Leon Herrera Rosales**
    """)

    # Información con texto
    st.info("If you would like to send us any comments to improve the app, please fill out the following form😁:")

    contact_form = """
    <form action="https://formsubmit.co/abrahamhf45@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="email" name="email" placeholder= "Your email:" required>
        <input type="text" name="feedback" placeholder= "Any feedback:" required>
        <input type="hidden" name="_next" value="https://huggingface.co/spaces/fatimahhussain/workoutwizard">
        <button type="submit"">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Imagen final al pie de la app con publicidad a la derecha
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.image("images/3foto.jpg", width=180)

    with col_right:
        st.image("images/publicidad.jpg", width=180)


def pose_process(
    in_queue: Queue, #queue where the function will read input items
    out_queue: Queue, #queue where function will put output items
):
    # Inicializa el objeto Pose de mediapipe
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = pose.process(input_item)

        # Check if landmarks data is present
        if results.pose_landmarks and results.pose_landmarks.landmark:
            landmarks = [
                FakeLandmarkObject(
                    x=pose_landmark.x,
                    y=pose_landmark.y,
                    z=pose_landmark.z,
                ) for pose_landmark in results.pose_landmarks.landmark
            ]
        else:
            landmarks = []

        # Crear objeto picklable_results
        picklable_results = FakeResultObject(
            pose_landmarks=FakeLandmarksObject(landmark=landmarks)
        )

        # Poner el resultado en la cola
        out_queue.put_nowait(picklable_results)


if __name__ == "__main__":
    main()
>>>>>>> 91a58af7dfa2f97fb9f88daa8baa95310bfabacd
