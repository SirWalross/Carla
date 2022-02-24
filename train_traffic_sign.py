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

maxX = 96
maxY = 96

trainDatagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1.0 / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.4,
    zoom_range=0.2,
    validation_split=0.1,
    brightness_range=[0.95, 1.1],
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

validationGenerator = trainDatagen.flow_from_directory(
    directory="../Schildererkennung/Train",
    target_size=(maxX, maxY),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    subset="validation",
)


def get_model():
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=False, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint("traffic_sign.h5", monitor="val_loss", verbose=False, save_best_only=True)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(50, (6, 6), activation="relu", padding="same", input_shape=(maxX, maxY, 3)))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(100, (4, 4), activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(150, (4, 4), activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(75, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(6, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model, [early_stop, checkpoint]


# %%
try:
    # model = keras.models.load_model('traffic_sign.h5')
    print("loaded model")
    raise OSError()
except OSError:
    model, callbacks = get_model()
    history = model.fit(trainGenerator, epochs=8, callbacks=callbacks, validation_data=validationGenerator)

testDatagen = ImageDataGenerator(rescale=1.0 / 255)
testDatagen = testDatagen.flow_from_directory(
    directory="../Schildererkennung/Test2", target_size=(maxX, maxY), color_mode="rgb", batch_size=64, shuffle=False
)


print(model.evaluate(testDatagen))
