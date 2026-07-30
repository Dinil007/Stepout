import cv2
import numpy as np

video = "outputs/preprocessed/preprocessed_video.mp4"

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Cannot read video")
    exit()

points = []

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked: ({x}, {y})")

cv2.namedWindow("Pitch ROI")
cv2.setMouseCallback("Pitch ROI", mouse_callback)

while True:

    temp = frame.copy()

    # Draw clicked points
    for p in points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    # Draw polygon
    if len(points) > 1:
        cv2.polylines(temp, [np.array(points)], False, (0, 255, 0), 2)

    cv2.imshow("Pitch ROI", temp)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("\nFinal Points:")
print(points)