# %%
import matplotlib.pyplot as plt
import csv
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# %%

maxX = 64
maxY = 64

trainDatagen = ImageDataGenerator(
    rotation_range=30, rescale=1.0 / 255, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1
)

trainGenerator = trainDatagen.flow_from_directory(
    directory="../Schildererkennung/Train",
    target_size=(maxX, maxY),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    subset="training",
)


def get_model():
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=False, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint("traffic_sign.h5", monitor="loss", verbose=False, save_best_only=True)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(100, (7, 7), activation="relu", input_shape=(maxX, maxY, 3)))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(150, (4, 4), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(250, (4, 4), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model, [early_stop, checkpoint]


# %%
try:
    # model = keras.models.load_model('traffic_sign.h5')
    print("loaded model")
    raise OSError()
except OSError:
    model, callbacks = get_model()
    history = model.fit(
        trainGenerator,
        epochs=10,
        callbacks=callbacks,
    )

testDatagen = ImageDataGenerator(rescale=1.0 / 255)
testDatagen = testDatagen.flow_from_directory(
    directory="../Schildererkennung/Test", target_size=(maxX, maxY), color_mode="rgb", batch_size=64, class_mode=None, shuffle=False
)

import pandas as pd

# print(np.argmax(y_predict, axis=1))
y_test = pd.read_csv("../Schildererkennung/YTest.csv", delimiter=";")["ClassId"]
y_predict = model.predict(testDatagen)
y_test = y_test[y_test <= 5]

print(f"error: {np.count_nonzero(np.argmax(y_predict, axis=1) - y_test) / y_test.shape[0]*100:.2f}%")
