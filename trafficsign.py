import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from enum import Enum

import cv2

model: keras.Model = None
counter = 0


def load_model():
    global model
    model = keras.models.load_model("traffic_sign.h5")


class TrafficSignType(Enum):
    SPEED_30_SIGN = 0
    # SPEED_50_SIGN = 2
    SPEED_60_SIGN = 1
    # SPEED_70_SIGN = 4
    # SPEED_80_SIGN = 5
    # SPEED_100_SIGN = 7
    # SPEED_120_SIGN = 8
    # PRIORITY_ONCE_SIGN = 11
    # PRIORITY_SIGN = 12
    INVALID_SIGN = 3
    SPEED_90_SIGN = 2


def detect_traffic_sign(image: np.ndarray) -> TrafficSignType:
    global counter
    image = tf.image.resize(image, (64, 64)) / 255.0
    counter += 1
    traffic_sign = model.predict(image.numpy()[None, :, :, ::-1])
    try:
        traffic_sign_type = TrafficSignType(np.argmax(traffic_sign))
        if np.min(traffic_sign) > 0.2:
            raise ValueError()
        elif traffic_sign_type == TrafficSignType.SPEED_30_SIGN and np.max(traffic_sign) < 0.92:
            traffic_sign_type = TrafficSignType.SPEED_90_SIGN
        cv2.imwrite(f"signs/traffic{counter}{traffic_sign_type.name}{traffic_sign[0].tolist()}.png", image.numpy()[:, :, ::-1] * 255)
        return traffic_sign_type
    except ValueError:
        # if traffic_sign < 10:
        #     return TrafficSignType.SPEED_90_SIGN
        # else:
        cv2.imwrite(f"signs/traffic{counter}invalid{traffic_sign[0].tolist()}.png", image.numpy()[:, :, ::-1] * 255)
        return TrafficSignType.INVALID_SIGN
