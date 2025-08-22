import copy
from multiprocessing import Queue, Process
import cv2 as cv
import numpy as np
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from utils import CvFpsCalc
from app import pose_process
from functions import calculate_angle
from turn import get_ice_servers

_SENTINEL_ = "_SENTINEL_"

# ---------------------------
# Video Processor: PUSH-UPS
# ---------------------------
class PushUpVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self.stage = "up"      # "up" = top position | "down" = bottom position
        self.counter = 0

        self._pose_process = Process(
            target=pose_process,
            kwargs={"in_queue": self._in_queue, "out_queue": self._out_queue},
        )
        self._cvFpsCalc = CvFpsCalc(buffer_len=10)
        self._pose_process.start()

        # Thresholds to detect a push-up rep
        self.UP_ANGLE = 160     # arm extended (top position)
        self.DOWN_ANGLE = 90    # arm bent (bottom position)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)  # mirror effect
        image = copy.deepcopy(image)

        # Mediapipe expects RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose

        # Run inference in the background process
        results = self._infer_pose(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Key points: right shoulder, elbow, and wrist
            shoulder = [
                int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image.shape[1]),
                int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image.shape[0]),
            ]
            elbow = [
                int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image.shape[1]),
                int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image.shape[0]),
            ]
            wrist = [
                int(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image.shape[1]),
                int(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0]),
            ]

            # Elbow angle (shoulder–elbow–wrist)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Show angle
            cv.putText(
                image, f"Angle: {round(elbow_angle, 1)}",
                (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv.putText(
                image, "REPS:", (300, 50),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            # Draw key points
            cv.circle(image, tuple(shoulder), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(elbow), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(wrist), 10, (255, 255, 255), -1)

            # Counting logic
            if elbow_angle < self.DOWN_ANGLE and self.stage == "up":
                self.stage = "down"
            elif elbow_angle > self.UP_ANGLE and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            # Show counter
            cv.putText(
                image, str(self.counter),
                (450, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            # Simple feedback
            if self.stage == "down":
                if 80 <= elbow_angle <= 100:
                    set_text("Good depth ✅", (0, 255, 0), image, coordinates=(100, 100))
                elif elbow_angle > 100:
                    set_text("Go lower ⬇️", (0, 165, 255), image, coordinates=(100, 100))
                else:
                    set_text("Too low ❌", (0, 0, 255), image, coordinates=(100, 100))
            else:
                if elbow_angle < 150:
                    set_text("Extend arms ⬆️", (255, 255, 255), image, coordinates=(100, 100))
                else:
                    set_text("Ready", (200, 200, 200), image, coordinates=(100, 100))

        return av.VideoFrame.from_ndarray(image, format="rgb24")

    def _infer_pose(self, image_rgb):
        self._in_queue.put_nowait(image_rgb)
        return self._out_queue.get(timeout=10)

    def _stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

    def __del__(self):
        self._stop_pose_process()


def set_text(text, color, image, coordinates=(50, 100)):
    """Draws feedback text on the frame."""
    cv.putText(image, text, coordinates, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    st.header("Push-Ups")
    st.markdown(
        "Position yourself sideways in the camera frame so that your right shoulder, elbow, and wrist are visible."
    )

    def processor_factory():
        return PushUpVideoProcessor()

    webrtc_ctx = webrtc_streamer(
        key="pushups-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
    )
    st.session_state["started"] = webrtc_ctx.state.playing

    # Visual references
    st.image("gif/Push-ups.gif")
    # st.video("videos/pushup.mp4")  # Uncomment if you have a demo video

if __name__ == "__main__":
    main()