import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)
frame_count = 0
flag = 0
frames = []
classes = ["food", "good", "hello", "meet", "phone", "protect", "student", "time", "up", "what"]
prediction = ""

model = tf.keras.models.load_model("video_model1.model")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 2, min_detection_confidence = 0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frameVid = vid.read()
    frameVid = cv2.flip(frameVid, 1)
    img = frameVid.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(img)
    lms = []
    f = 0
    if hand_results.multi_handedness:
        if len(hand_results.multi_handedness) == 1:
            if hand_results.multi_handedness[0].classification[0].label == "Left":
                for i in range(21):
                    lms.append([0, 0])
                f = 0
            else:
                f = 1
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lms.append([lm.x, lm.y])
            mp_drawing.draw_landmarks(frameVid, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if f == 1:
            for i in range(21):
                lms.append([0, 0])
        flag = 1
    else:
        for i in range(42):
            flag = 0
            lms.append([0, 0])

    pose_results = pose.process(img)
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            lms.append([lm.x, lm.y])
        mp_drawing.draw_landmarks(frameVid, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        for i in range(33):
            lms.append([0, 0])

    keyPress = cv2.waitKey(1) & 0xFF
    
    '''
    if keyPress == ord('s'):
        flag = 1
        print("Recording")
    elif keyPress == ord('q'):
        break
    '''

    if flag == 0:
        frames.append(lms)
        frame_count += 1
        while frame_count != 0 and frame_count > 10:
            frames.pop(0)
            frame_count -= 1

    print(flag, frame_count)
    if frame_count < 75 and flag == 1:
        frames.append(lms)
        frame_count = frame_count + 1
    elif frame_count == 75 and flag == 1:
        flag = 0
        frame_count = 0
        modelFrames = np.asarray(frames).astype("float32")
        frames = []
        modelFrames = modelFrames.reshape((1, 75, 150))
        pred = model.predict(modelFrames)
        predictionNum = np.argmax(pred[0])
        prediction = classes[predictionNum]
        print(prediction)

    cv2.putText(frameVid, prediction, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Feed", frameVid)
    if keyPress == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
