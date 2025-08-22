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
from app import pose_process
from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject
from functions import calculate_angle
from turn import get_ice_servers


_SENTINEL_ = "_SENTINEL_"


class Tokyo2020PictogramVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self.stage = 'up'
        self.counter = 0
        self.isrep = False
        self._pose_process = Process(target=pose_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
        })
        self._cvFpsCalc = CvFpsCalc(buffer_len=10)

        self._pose_process.start()

        # reason why stage doesn't make sense in recv function is bc stage is set to 'up' for every frame processed, so it resets.
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:



        # display_fps = self._cvFpsCalc.get()
        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)  # Mirrors image
        image = copy.deepcopy(image) #just copies the image so the original image will stay unaffected
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose

        results = self._infer_pose(image)

        if results.pose_landmarks:
            landmarks1 = results.pose_landmarks.landmark
            shoulder1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]),
                        int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])]
            elbow1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1]),
                    int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])]
            wrist1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1]),
                    int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])]
            hip1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1]),
                    int(landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0])]

            angle1 = calculate_angle(shoulder1, elbow1, wrist1)
            angle2 = calculate_angle(shoulder1, elbow1, hip1)
            # hip_angle = calculate_angle(shoulder1, hip1, elbow1)


            normal_range = (20,160)

            # this angle is tracking angle2, aka the angle between shoulder, elbow, and hip 
            cv.putText(image, f'Angle: {round(angle1, 2)}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(image, 'REPS:', (300,50), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


            cv.circle(image, tuple(shoulder1), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(elbow1), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(wrist1), 10, (255, 255, 255), -1)
            # cv.circle(image, tuple(hip1), 10, (255, 255, 255), -1)
            # print("Stage is!", self.stage)

            if angle1 > 100 and self.stage == "up":
                self.stage = "down"
            elif angle1 < 50 and self.stage =="down":
                # print("Will down print?!", self.stage)
                self.stage= "up"
                self.counter += 1

            cv.putText(image, str(self.counter), (450,50), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                        
            
            if 30 < angle1 < 180 and angle2 > 140:
                cv.putText(image, 'correct', (100,100), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
           

                
            else:
            
                if 30 > angle1:
                    set_text("arm too close", (255,0,0), image, coordinates= ( 100,100))
                elif angle2 < 160:
                    set_text("keep elbow inline", (255,0,0), image, coordinates=(100,100))
                else:
                    set_text("arm too close, line up elbow", (255,0,0), image, coordinates= (100,100))

                # print("In correct form! Here's what you need to fix")
            

        return av.VideoFrame.from_ndarray(image, format="rgb24")

    def _infer_pose(self, image):
        self._in_queue.put_nowait(image)
        return self._out_queue.get(timeout=10)

    def _stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

    def __del__(self):
        self._stop_pose_process()

    
def set_text(text, color, image, coordinates):
    coordinates = (50, 100)
    cv.putText(image, text, coordinates, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    st.header("Bicep Curls")
    st.markdown("""Keep your body fully in frame and slightly tilted, the webcam will track the right side of your body.""")
    def processor_factory():
        return Tokyo2020PictogramVideoProcessor()

    webrtc_ctx = webrtc_streamer(
        key="tokyo2020-Pictogram",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
    )
    st.session_state["started"] = webrtc_ctx.state.playing


    st.image("gif/curl.gif")

    st.video("videos/realcurls.mp4")

    



if __name__ == "__main__":
    main()



