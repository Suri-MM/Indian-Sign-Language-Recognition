import tensorflow as tf
import os 
import numpy as np
import pandas as pd

classes = ["food", "good", "hello", "meet", "phone", "protect", "student", "time", "up", "what"]

def dataset_extraction(file):
    X = []
    temp_Y = []
    Y = []

    df = pd.read_csv(file)

    X = df.to_numpy()

    temp_Y = X[:, [1]]
    X = X[:, 2:]

    for c in temp_Y:
        y = [0] * len(classes)
        y[classes.index(c)] = 1
        Y.append(y)

    X = np.asarray(X).astype('float32')
    Y = np.array(Y)
    return X, Y

train_X, train_Y = dataset_extraction("train_data.csv")
test_X, test_Y = dataset_extraction("test_data.csv")

train_X = np.reshape(train_X, (750, 75, -1))
test_X = np.reshape(test_X, (151, 75, -1))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(256, return_sequences = True, input_shape = (75, 150)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = "relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation = "relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
    ])

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(
        train_X,
        train_Y,
        epochs = 40,
        batch_size = 8, 
        shuffle = True 
        )

loss = model.evaluate(test_X, test_Y, steps = 8)
print("Loss:", loss)
model.save("video_model1.model")

