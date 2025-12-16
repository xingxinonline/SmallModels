"""检测器模块"""
from .gesture_detector import GestureDetector
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .person_detector import PersonDetector

__all__ = [
    "GestureDetector",
    "FaceDetector", 
    "FaceRecognizer",
    "PersonDetector"
]
