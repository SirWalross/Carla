import numpy as np
import tensorflow as tf
from tensorflow import keras
from enum import Enum

from tensorflow.python.types.core import Value
import cv2

model: keras.Model = None
counter = 0


def load_model():
    global model
    model = keras.models.load_model('traffic_sign.h5')

class TrafficSignType(Enum):
    SPEED_30_SIGN = 1
    SPEED_50_SIGN = 2
    # SPEED_60_SIGN = 3
    SPEED_70_SIGN = 4
    # SPEED_80_SIGN = 5
    SPEED_100_SIGN = 7
    # SPEED_120_SIGN = 8
    # PRIORITY_ONCE_SIGN = 11
    # PRIORITY_SIGN = 12
    YIELD_SIGN = 13
    STOP_SIGN = 14
    INVALID_SIGN = 44

def detect_traffic_sign(image: np.ndarray) -> TrafficSignType:
    global counter
    image = tf.image.resize(image, (48, 48)) / 255.0
    cv2.imwrite(f"images/traffic{counter}.png", image.numpy()[:, :, ::-1] * 255)
    counter += 1
    traffic_sign = np.argmax(model.predict(image.numpy()[None, :, :, ::-1]))
    try:
        return TrafficSignType(traffic_sign)
    except ValueError:
        return TrafficSignType.INVALID_SIGN