import cv2

cap = cv2.VideoCapture("D:/stepout/videos/raw/match30.mp4")

print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height :", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS :", cap.get(cv2.CAP_PROP_FPS))
print("Frames :", cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()