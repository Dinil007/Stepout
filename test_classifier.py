from app.classification.person_classifier import PersonClassifier
import cv2
import numpy as np

# Test the person classifier
print("Testing Person Classifier with fine-tuned weights...")
pc = PersonClassifier(device='cpu')
print("Loading model...")
success = pc.load()
print(f"Loaded: {success}")
print(f"Classes: {pc.class_names}")
print(f"Referee color: {pc.get_class_color('Referee')}")
print(f"Referee label: {pc.get_class_label('Referee')}")
print(f"Coach color: {pc.get_class_color('Coach')}")
print(f"Coach label: {pc.get_class_label('Coach')}")
print(f"Player color: {pc.get_class_color('Player')}")
print(f"Player label: {pc.get_class_label('Player')}")

# Test with a dummy image
dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
result, conf = pc.classify(dummy_image)
print(f"Test classification: {result}, confidence: {conf}")
