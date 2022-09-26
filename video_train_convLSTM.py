import tensorflow as tf
import os 
import numpy as np
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt

classes = ["food", "good", "hello", "meet", "phone", "protect", "student", "time", "up", "what"]
    
def load_data(file):
    df = pd.read_csv(file)
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    
    x = np.array(x)
    for i in range(len(y)):
        y[i] = classes.index(y[i])
        
    print(x.shape)
    x = np.reshape(x, (749, 75, 28, 28, 3))
    return (x, np_utils.to_categorical(y, num_classes = 10))

X, Y = load_data("train_data_video_28.csv")
test_X, test_Y = load_data("test_data_video_28.csv")

model = tf.keras.models.Sequential([
    tf.keras.layers.ConvLSTM2D(48, (3, 3), return_sequences = True, padding = "same", data_format = "channels_last", input_shape = (75, 28, 28, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation = "softmax")
    ])
 
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate = 0.00001)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(
    X,
    Y,
    epochs = 5,
    batch_size = 8, 
    shuffle = True,
    validation_data = (test_X, test_Y)
    )

model.save('convLSTM_video.model')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
