"""
Person Type Classifier using EfficientNet-B0

Classifies detected persons into categories:
- Player: Football players (for team classification)
- Referee: Match officials (black dress)
- Coach: Team staff (sideline personnel)
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)


class PersonClassifier:
    """
    EfficientNet-B0 based person type classifier.
    
    Uses transfer learning with pre-trained EfficientNet-B0 to classify
    person crops into Player/Referee/Coach categories.
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.7):
        """
        Initialize the person classifier.
        
        Args:
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for classification
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.transform = None
        self.class_names = ["Player", "Referee", "Coach"]
        self.is_loaded = False

    def load(self) -> bool:
        """
        Load the EfficientNet-B0 model with fine-tuned weights.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info("Loading EfficientNet-B0 for person classification...")
            
            # Load pre-trained EfficientNet-B0
            weights = EfficientNet_B0_Weights.DEFAULT
            self.model = efficientnet_b0(weights=weights)
            
            # Modify the final layer for 3 classes (Player, Referee, Coach)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 3)  # 3 output classes
            )
            
            # Load fine-tuned weights
            model_path = "models/classifier/efficientnet_b0_best.pth"
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded fine-tuned weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load fine-tuned weights from {model_path}: {e}")
                logger.warning("Using ImageNet weights (model not trained for referee detection)")
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.is_loaded = True
            logger.info("EfficientNet-B0 loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B0: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for EfficientNet.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(rgb_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor

    def classify(self, person_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classify a person crop into Player/Referee/Coach.
        
        Args:
            person_crop: Cropped person image (BGR format)
            
        Returns:
            Tuple of (class_name, confidence) or (None, 0.0) if classification fails
        """
        if not self.is_loaded:
            logger.warning("Person classifier not loaded")
            return None, 0.0
        
        if person_crop is None or person_crop.size == 0:
            return None, 0.0
        
        try:
            # Preprocess
            input_tensor = self.preprocess(person_crop)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_class = predicted.item()
                
                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    return None, 0.0
                
                class_name = self.class_names[predicted_class]
                return class_name, confidence
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None, 0.0

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get visualization color for a person class.
        
        Args:
            class_name: "Player", "Referee", or "Coach"
            
        Returns:
            RGB color tuple
        """
        colors = {
            "Player": (0, 255, 0),    # Green (will be overridden by team colors)
            "Referee": (0, 255, 255), # Yellow
            "Coach": (255, 0, 255),   # Magenta
        }
        return colors.get(class_name, (128, 128, 128))  # Gray default

    def get_class_label(self, class_name: str) -> str:
        """
        Get display label for a person class.
        
        Args:
            class_name: "Player", "Referee", or "Coach"
            
        Returns:
            Display label string
        """
        labels = {
            "Player": "P",
            "Referee": "REF",
            "Coach": "COACH",
        }
        return labels.get(class_name, "?")
