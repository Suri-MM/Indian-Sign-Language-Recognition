import cv2
import os
import numpy as np
import pandas as pd
from csv import writer

path = "data/test/"
tData = []

file = open("test_data_video_28.csv", 'a')
writer_object = writer(file)

for d in os.listdir(path):
    print(d)
    for i in range(len(os.listdir(path + d))):
        print(i)
        cap = cv2.VideoCapture(path + d + "/" + str(i) + ".mp4")
        frames = []
        count = 0

        while True:
            ret, img = cap.read()

            if count == 75:
                break
            elif ret:
                img = cv2.resize(img, (28, 28))
                frame = np.asarray(img)                
                frame = np.reshape(frame, (1, -1))
                prev_frame = frame
                frames.append(frame)
            else:
                frames.append(prev_frame)
            count += 1

        data = []
        data.append(d)
        data = np.asarray(data)
        data = np.reshape(data, (1, 1))

        frames = np.asarray(frames)
        frames = np.reshape(frames, (1, -1))

        frames = np.concatenate((data, frames), axis = 1)
        frames = frames[0]


        tData.append(frames)
        print(frames)
        writer_object.writerow(frames)
        cap.release()

file.close()






            
            
