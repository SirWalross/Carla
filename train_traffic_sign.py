# %%
import matplotlib.pyplot as plt
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# %%

maxX = 64
maxY = 64

trainDatagen = ImageDataGenerator(rotation_range=30, horizontal_flip=True, rescale=1.0 / 255)
trainGenerator = trainDatagen.flow_from_directory(
    directory="../Schilderkennung/Train", target_size=(maxX, maxY), color_mode="rgb", batch_size=64, class_mode="categorical", shuffle=True
)

# %%
def get_model():
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, verbose=False, restore_best_weights=True
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        "bestW", monitor="val_loss", verbose=False, save_weights_only=True, save_best_only=True
    )

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,(3,3), padding='same', activation='relu', input_shape=(maxX,maxY,3)))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32,(5,5), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64,(3,3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64,(3,3), padding='same', activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100,activation='relu'))
    model.add(keras.layers.Dense(50,activation='relu'))
    model.add(keras.layers.Dense(43,activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model, [early_stop, checkpoint]

# %%
model, callbacks = get_model()
history = model.fit(
    trainGenerator,
    epochs=10,
    callbacks=callbacks,
)
model.save('traffic_sign.h5')