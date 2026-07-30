import cv2
from PIL import Image, ImageEnhance
import os

os.makedirs("outputs/preprocessed", exist_ok=True)

video = cv2.VideoCapture("D:/stepout/videos/raw/match30.mp4")

ret, frame = video.read()

video.release()

if not ret:
    print("Error reading video")
    exit()

# Convert BGR → RGB
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert to Pillow image
img = Image.fromarray(frame)

# Resize
img = img.resize((1280, 720))

# Increase contrast
contrast = ImageEnhance.Contrast(img)
img = contrast.enhance(1.2)

# Increase sharpness
sharp = ImageEnhance.Sharpness(img)
img = sharp.enhance(2)

# Save
img.save("outputs/preprocessed/frame1.jpg")

print("Preprocessing completed.")