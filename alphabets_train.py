import tensorflow as tf
import mediapipe

datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_data = datagen.flow_from_directory("data", target_size = (64, 64), class_mode = "categorical")

print(train_data)
