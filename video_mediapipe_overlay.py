import os
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 2, min_detection_confidence = 0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils

path = "data/test/"
dirs = os.listdir(path)
tData = []

for d in dirs:
    for i in range(len(os.listdir(path+d))):
        cap = cv2.VideoCapture(path + d + "/" + str(i) + ".mp4") 
        landmarks = []
        count = 0 
        while True: 
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            if count == 75:
                break
                    
            if ret == False: 
                if count < 75:
                    while count < 75:
                        lms = []
                        for i in range(75):
                            lms.append([0, 0])
                        landmarks.append(lms)
                        count = count + 1
                break

            feed = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            lms = []
            flag = 0
            hand_results = hands.process(img)
            if hand_results.multi_handedness:
                if len(hand_results.multi_handedness) == 1:
                    if hand_results.multi_handedness[0].classification[0].label == "Left":
                        for i in range(21):
                            lms.append([0, 0])
                        flag = 0
                    else:
                        flag = 1
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        lms.append([lm.x, lm.y])
                    mp_drawing.draw_landmarks(feed, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if flag == 1:
                    for i in range(21):
                        lms.append([0, 0])
            else:
                for i in range(42):
                    lms.append([0,0])

            pose_results = pose.process(img)
            if pose_results.pose_landmarks:
                for lm in pose_results.pose_landmarks.landmark:
                    lms.append([lm.x, lm.y]) 
                mp_drawing.draw_landmarks(feed, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                for i in range(33):
                    lms.append([0,0])

            landmarks.append(lms)
                
            cv2.imshow("Image", feed)
            count = count + 1
            cv2.waitKey(1)

        landmarks = np.array(landmarks)
        landmarks = np.reshape(landmarks, (-1, 11250))

        data = []
        data.append(d)
        data = np.array(data)
        data = np.reshape(data, (1, 1))
        data = np.concatenate((data, landmarks), axis = 1)

        tData.append(data[0])
        print(data[0])
        cap.release()

    print(d + " :", len(tData))

tData = np.array(tData)
print(len(tData))
print(len(tData[0]))
DF = pd.DataFrame(tData)
DF.to_csv("test_data.csv")
cv2.destroyAllWindows()

