import copy
import csv
import argparse
import itertools
from collections import Counter
from collections import deque
from multiprocessing import Queue,Process
from typing import NamedTuple, List


import streamlit as st
from streamlit_webrtc import ClientSettings, RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

import av
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from app import draw_landmarks,calc_bounding_rect,calc_landmark_list,pre_process_landmark,draw_bounding_rect,draw_info_text

from model import KeyPointClassifier

_SENTINEL_ = "_SENTINEL_"

def sign_process(
    in_queue: Queue,
    out_queue: Queue,
    static_image_mode,
    model_complexity,
    min_detection_confidence,
    min_tracking_confidence,
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence, 
    )

    while True:
        input_item = in_queue.get(timeout=30)
        if isinstance(input_item , type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = hands.process(input_item)
        
        
        out_queue.put_nowait(results) 
    

#class VideoProcessor(VideoProcessorBase):
#    def __init__(self):
#        self.style = 'color'
#def recv(self, frame):
#        img = frame.to_ndarray(format="bgr24")
        
        # image processing code here
    
#        return av.VideoFrame.from_ndarray(img, format="bgr24")
#webrtc_streamer(key="sign-language-recognition", video_processor_factory=VideoProcessor)


class SignLanguageRecognitionProcess(VideoProcessorBase):
    def __init__(self,static_image_mode,
                    min_detection_confidence,
                    model_complexity,
                    min_tracking_confidence,
                    display,
                    show_fps
                ) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self._sign_process = Process(target=sign_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
            "static_image_mode": static_image_mode,
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        })

        self._cvFpsCalc = CvFpsCalc(buffer_len=30)
        self.show_fps = show_fps
        self.display = display
        self.use_brect = True 
        self.keypoint_classifier = KeyPointClassifier()
        
        with open('hand-gesture-recognition-mediapipe-main\model\keypoint_classifier\keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]

        self._sign_process.start()
        

    def _infer_sign(self,image):
        self._in_queue.put_nowait(image)
        return self._out_queue.get(timeout=30)

    def _stop_sign_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._sign_process.join(timeout=30)
    
    def recv(self, frame : av.VideoFrame) -> av.VideoFrame:
        display_fps = self._cvFpsCalc.get()

        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self._infer_sign(image)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                debug_image = draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id]
                )

        if self.show_fps:
            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            

        if self.display:
            return av.VideoFrame.from_ndarray(debug_image , format="bgr24")

    def __del__(self):
        print("Stop the inference process....")
        self._stop_sign_process()
        print("Stopped!")



def main():

    static_image_mode = False
    model_complexity = 1
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5
    
    display = True
    show_fps = True
    st.write("God Help me")

    def processing():
        return SignLanguageRecognitionProcess(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence= min_detection_confidence,
            min_tracking_confidence= min_tracking_confidence,
            display = display,
            show_fps=show_fps
        )
    
    web_display = webrtc_streamer(
        key= "sign-language-recognition",
        mode=WebRtcMode.SENDRECV,
        client_settings= ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints = {"video": True, "audio": False},
        ),
        video_processor_factory = processing,
        )
    
    st.session_state["started"] = web_display.state.playing

    
    if web_display.video_processor:
        web_display.video_processor.display = display
        web_display.video_processor.show_fps = show_fps








if __name__ == "__main__":
    main()