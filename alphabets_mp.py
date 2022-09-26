import cv2
import os
import mediapipe as mp

path = "data/train"
flag = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 2, min_detection_confidence = 0.5)

mp_drawing = mp.solutions.drawing_utils

'''
for directory in os.listdir(path):
    if flag == 1:
        break
    for i in range(len(os.listdir(path + "/" + directory))):
            img = cv2.imread(path + "/" + directory + "/" + str(i) + ".jpg")
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                print("Nope")
                continue

            for hand_landmarks in results.multi_hand_landmarks:
                xList = []
                yList = []
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                    
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
            
            cv2.imshow("Image", img)

            print(i)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                flag = 1
                break
'''

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:      
        for hand_landmarks in results.multi_hand_landmarks:
            xList = []
            yList = []
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
                    
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
            
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
