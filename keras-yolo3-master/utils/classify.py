from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import os
import cv2
import numpy as np


class Beetle_Classifier:
    def __init__(self):
        self.model_path = os.path.abspath('./models/model-resnet50-0321-2.h5')
        self.net = load_model(self.model_path)
        print(">>init classifier by %s"%self.model_path)

    def predict_Beetle_id(self, img):
        img = cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis = 0)
        pred = self.net.predict(img)[0]
        pred_y = np.argmax(pred)
        return pred_y+1