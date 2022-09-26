import os
import cv2

if not os.path.exists("data/train"):
    os.makedirs("data/train")

if not os.path.exists("data/test"):
    os.makedirs("data/test")

path = "data"

data_list = os.listdir(path)

for directory in data_list:
    if directory != "test" and directory != "train":
        os.makedirs("data/train/" + directory)
        os.makedirs("data/test/" + directory)
        for count in range(1200):
            if count < 960:
                img = cv2.imread(path + "/" + directory + "/" + str(count) + ".jpg")
                cv2.imwrite(path + "/train/" + directory + "/" + str(count) + ".jpg", img) 
            else:
                img = cv2.imread(path + "/" + directory + "/" + str(count) + ".jpg")
                cv2.imwrite(path + "/test/" + directory + "/" + str(count - 960) + ".jpg", img)
