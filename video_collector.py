import cv2
import os

if not os.path.exists("data/train/meet"):
    os.makedirs("data/train/meet")

if not os.path.exists("data/test/meet"):
    os.makedirs("data/test/meet")

vid = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

flag = 0
frame_count = 0
vid_count = len(os.listdir("data/train/meet"))
path = "data/train/meet/"
if vid_count >= 75:
    vid_count = len(os.listdir("data/test/meet"))
    path = "data/test/meet/"

while True:
    ret, frame = vid.read()
    keyPress = cv2.waitKey(1) & 0xFF

    if keyPress == ord('s'):
        flag = 1
        out = cv2.VideoWriter(path + str(vid_count) + ".mp4", fourcc, 30, (640, 480))
        print("Writing")
    elif keyPress == ord('x'):
        flag = 0
        vid_count = vid_count + 1
        if vid_count >= 75:
            vid_count = len(os.listdir("data/test/meet"))
            path = "data/test/meet/"
        out.release()
        print("Stopping")
        print(vid_count)
    elif keyPress == ord('q'):
        break

    if flag == 1:
        out.write(frame)

    cv2.imshow("Feed", frame)

vid.release()
cv2.destroyAllWindows()

