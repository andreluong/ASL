import mediapipe as mp
import time
import os
from abc import ABC, abstractmethod
import cv2

_hand_model_path = os.path.join(
    os.path.dirname(__file__), "hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = os.getenv(
    "HAND_LANDMARKER_MODEL_PATH", _hand_model_path
)

class Hand_cm(ABC):
    def __init__(self, result_callback_method=None):
        self.landmarker=None
        self.running_mode=None
        self.result_callback=result_callback_method

    def __enter__(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
            running_mode=self.running_mode, 
            result_callback=self.result_callback,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.landmarker.close()

    def detect(self, bgr_frame):
        print('Detect called')
        # Convert BGR (OpenCV) to RGB (MediaPipe expects)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
        return self._detect_impl(mp_image)

    @abstractmethod
    def _detect_impl(self, mp_image):
        pass
    
class Hand_Live_cm(Hand_cm):
    def __init__(self, result_callback_method):
        super().__init__(result_callback_method)
        self.running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM

    def _detect_impl(self, mp_image):
        print('detect impl called')
        timestamp = int(time.time() * 1000)
        if self.landmarker is None:
            print('landmarker is None')
        self.landmarker.detect_async(mp_image, timestamp)
        
# Pass in a result_callback that handles the detected result
# Takes in the object returned by self.landmarker.detect(mp_image)
class Hand_Image_cm(Hand_cm):
    def __init__(self, result_callback_method):
        super().__init__(None)
        self.callback=result_callback_method
        self.running_mode=mp.tasks.vision.RunningMode.IMAGE

    def _detect_impl(self, mp_image):
        landmark_result = self.landmarker.detect(mp_image)
        self.callback(landmark_result)
 
class Hand_Video_cm(Hand_cm):
    def __init__(self, result_callback_method):
        super().__init__(None)
        self.callback=result_callback_method
        self.running_mode=mp.tasks.vision.RunningMode.VIDEO

    def _detect_impl(self, mp_image):
        print('detect impl called')
        timestamp = int(time.time() * 1000)
        if self.landmarker is None:
            print('landmarker is None')
        landmark_result = self.landmarker.detect_for_video(mp_image, timestamp)
        if self.callback is not None:
            self.callback(landmark_result)
        return landmark_result
