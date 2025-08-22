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
from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject
from functions import calculate_angle
from turn import get_ice_servers

_SENTINEL_ = "_SENTINEL_"

# ---------------------------
# Procesador de video: SQUATS
# ---------------------------
class SquatVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self.stage = "up"      # "up" = de pie | "down" = abajo
        self.counter = 0
        self.isrep = False

        self._pose_process = Process(
            target=pose_process,
            kwargs={"in_queue": self._in_queue, "out_queue": self._out_queue},
        )
        self._cvFpsCalc = CvFpsCalc(buffer_len=10)
        self._pose_process.start()

        # Umbrales para detectar una repetición de sentadilla
        self.UP_ANGLE = 165     # rodilla extendida (de pie)
        self.DOWN_ANGLE = 95    # rodilla flexionada (abajo)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Frame -> ndarray (BGR)
        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)  # espejo
        image = copy.deepcopy(image)

        # Mediapipe espera RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose

        # Inferencia en el proceso aparte (mediapipe)
        results = self._infer_pose(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Puntos clave pierna izquierda (en píxeles)
            hip = [
                int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1]),
                int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0]),
            ]
            knee = [
                int(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image.shape[1]),
                int(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image.shape[0]),
            ]
            ankle = [
                int(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image.shape[1]),
                int(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image.shape[0]),
            ]

            # Ángulo en la rodilla (cadera–rodilla–tobillo)
            knee_angle = calculate_angle(hip, knee, ankle)

            # UI básica en el frame
            cv.putText(
                image, f"Angle: {round(knee_angle, 1)}",
                (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv.putText(
                image, "REPS:", (300, 50),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            # Puntos
            cv.circle(image, tuple(hip), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(knee), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(ankle), 10, (255, 255, 255), -1)

            # Lógica de conteo:
            # - Baja cuando el ángulo < DOWN_ANGLE
            # - Sube y cuenta cuando pasa de nuevo > UP_ANGLE
            if knee_angle < self.DOWN_ANGLE and self.stage == "up":
                self.stage = "down"
            elif knee_angle > self.UP_ANGLE and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            # Mostrar contador
            cv.putText(
                image, str(self.counter),
                (450, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            # Feedback de forma simple
            # (Puedes ajustar los rangos según prefieras)
            if self.stage == "down":
                if 80 <= knee_angle <= 110:
                    set_text("Good depth ✅", (0, 255, 0), image, coordinates=(100, 100))
                elif knee_angle > 110:
                    set_text("Go deeper ⬇️", (0, 165, 255), image, coordinates=(100, 100))
                else:
                    set_text("Careful: too deep", (0, 0, 255), image, coordinates=(100, 100))
            else:
                if knee_angle < 160:
                    set_text("Stand tall ⬆️", (255, 255, 255), image, coordinates=(100, 100))
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
    """Dibuja texto en el frame en las coordenadas dadas (por defecto, 50,100)."""
    cv.putText(image, text, coordinates, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    st.header("Squats")
    st.markdown(
        "Stand fully in frame with a slight side view so the camera can track your hips, knees, and ankles during the squat."
    )

    def processor_factory():
        return SquatVideoProcessor()

    webrtc_ctx = webrtc_streamer(
        key="squats-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
    )
    st.session_state["started"] = webrtc_ctx.state.playing

    # Referencias visuales
    st.image("gif/Squat.gif")
    # st.video("videos/realsquats.mp4")  # Descomenta si tienes el video

if __name__ == "__main__":
    main()
