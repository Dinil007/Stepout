import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

input_video = "D:/stepout/videos/raw/match30.mp4"
output_video = "outputs/preprocessed/preprocessed_video.mp4"

os.makedirs("outputs/preprocessed", exist_ok=True)

cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (1280,720)
)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # OpenCV → Pillow
    img = Image.fromarray(rgb)

    # Resize
    img = img.resize((1280,720))

    # Contrast
    img = ImageEnhance.Contrast(img).enhance(1.2)

    # Sharpness
    img = ImageEnhance.Sharpness(img).enhance(2)

    # Pillow → OpenCV
    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    writer.write(img)

    frame_number += 1

    print(f"Processed Frame {frame_number}", end="\r")

cap.release()
writer.release()

print("\nVideo preprocessing completed.")