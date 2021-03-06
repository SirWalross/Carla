import numpy as np
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from enum import Enum

import cv2

model: keras.Model = None
counter = 0


def load_model():
    """Load the model for traffic sign detection."""
    global model
    model = keras.models.load_model("traffic_sign.h5")


class TrafficSignType(Enum):
    """The types of traffic signs."""

    SPEED_30_SIGN = 1
    SPEED_60_SIGN = 3
    INVALID_SIGN = 6
    SPEED_90_SIGN = 5


def detect_traffic_sign(image: np.ndarray) -> TrafficSignType:
    """Detect a traffic sign.

    Args:
        image (np.ndarray): The image of traffic sign.

    Returns:
        TrafficSignType: The type of traffic sign detected.
    """

    global counter
    image = tf.image.resize(image, (96, 96)) / 255.0
    counter += 1
    traffic_sign = model.predict(image.numpy()[None, :, :, ::-1])[0]
    vals = ""
    for val in traffic_sign.tolist():
        vals += f"{val:.2f},"
    try:
        traffic_sign_type = TrafficSignType(np.argmax(traffic_sign))
        if traffic_sign_type == TrafficSignType.SPEED_30_SIGN and traffic_sign[1] < 0.6:
            # sometimes missclassifies speed 60 sign as speed 30 sign
            traffic_sign_type = TrafficSignType.SPEED_60_SIGN
        cv2.imwrite(f"signs/traffic{counter}{traffic_sign_type.name}[{vals[:-2]}].png", image.numpy()[:, :, ::-1] * 255)
        return traffic_sign_type
    except ValueError:
        if np.argmax(traffic_sign) == 0 and traffic_sign[3] > 0.2:
            # sometimes missclassifies speed 60 sign as an invalid sign
            traffic_sign_type = TrafficSignType.SPEED_60_SIGN
            cv2.imwrite(f"signs/traffic{counter}{traffic_sign_type.name}[{vals[:-2]}].png", image.numpy()[:, :, ::-1] * 255)
            return traffic_sign_type
        cv2.imwrite(f"signs/traffic{counter}invalid[{vals[:-2]}].png", image.numpy()[:, :, ::-1] * 255)
        return TrafficSignType.INVALID_SIGN
