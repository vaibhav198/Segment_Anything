import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

import matplotlib
# matplotlib.use('TkAgg')

class DetectionModel():
    def __init__(self, model_path = './models/yolov8n.pt'):
        self.model = YOLO(model_path)

    def pred(self, image):
        class_names = self.model.names
        results = self.model(image, save=False)
        return results, class_names

