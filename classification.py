import io
import os.path

import joblib
from PIL import Image
import numpy as np
import json


class WaterClassifier:
    def __init__(self):
        self.model = joblib.load('water_classifier.pkl')
        self.label_encoder = joblib.load('water_labels.pkl')

    def predict(self, image):

        imgJSONString = image.decode('utf-8')
        imgJSON = json.loads(imgJSONString)
        print(type(bytearray(imgJSON['data']['buffer']['data'])))
        imgAnotherByteArray = bytearray(imgJSON['data']['buffer']['data'])
        anotherImg = Image.open(io.BytesIO(imgAnotherByteArray))
        img = anotherImg.resize((128, 128))
        X = np.array(img).flatten()

        X = X / 255.0

        X = X.reshape(1, -1)

        pred_idx = self.model.predict(X)[0]

        return {
            'position': self.label_encoder.inverse_transform([pred_idx])[0],
        }

