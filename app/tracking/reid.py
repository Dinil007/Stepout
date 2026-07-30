"""
Appearance Re-Identification (ReID) Module using OSNet.

Uses torchreid's OSNet (osnet_x0_25) to extract appearance embeddings
from player crops for identity consistency across frames.

Architecture:
    Player Crop → Resize → Normalize → OSNet → L2 Normalize → Feature Vector

Key design decisions:
- Inference-only mode (no training, no fine-tuning)
- CUDA acceleration with FP16 support
- Batch processing for efficiency
- Minimal pre/post-processing
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.tracking.tracker_config import ReIDConfig

logger = logging.getLogger(__name__)

# Try importing torchreid (optional dependency)
try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logger.warning(
        "torchreid not installed. ReID disabled. "
        "Install with: pip install torchreid"
    )


class OSNetReID:
    """
    OSNet-based appearance ReID extractor.

    Produces L2-normalized 512-d embedding vectors from player crops.
    Runs in inference-only mode on CUDA or CPU.
    """

    def __init__(self, config: ReIDConfig) -> None:
        """
        Args:
            config: ReID configuration with model path, device, input size
        """
        self.config = config
        self.device = self._resolve_device()
        self.model = None
        self.input_width, self.input_height = config.input_size  # (W, H) = (128, 256)
        self._is_loaded = False
        self.total_inferences: int = 0
        self.total_inference_ms: float = 0.0

    def _resolve_device(self) -> torch.device:
        """Resolve the target device for inference."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        elif self.config.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.config.device)
        return torch.device("cpu")

    def load(self) -> None:
        """Load pretrained OSNet model. Raises RuntimeError if torchreid missing."""
        if not TORCHREID_AVAILABLE:
            raise RuntimeError(
                "torchreid is required for ReID. "
                "Install with: pip install torchreid"
            )

        logger.info(
            f"Loading OSNet ReID model: {self.config.model} "
            f"on {self.device}"
        )

        try:
            self.model = torchreid.models.build_model(
                name=self.config.model,
                num_classes=1,  # Not used (inference only)
                pretrained=True,
            )
            self.model.eval()
            self.model.to(self.device)

            # Enable FP16 for faster inference if CUDA available
            if self.device.type == "cuda":
                self.model.half()

            self._is_loaded = True
            logger.info(f"OSNet ReID model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load OSNet model: {e}")

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess a player crop for OSNet inference.

        Steps:
        1. Resize to (W=128, H=256)
        2. Convert BGR → RGB
        3. Normalize to [0, 1]
        4. Apply ImageNet mean/std normalization
        5. Add batch dimension
        6. Convert to FP16 if CUDA

        Args:
            crop: BGR image crop of a player

        Returns:
            Preprocessed tensor ready for OSNet inference
        """
        # Resize
        resized = cv2.resize(crop, (self.input_width, self.input_height))

        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC → CHW and normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Move to device and convert to half if CUDA
        tensor = tensor.to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()

        return tensor

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a single L2-normalized embedding from one player crop.

        Args:
            crop: BGR image crop of a player (any size)

        Returns:
            512-d L2-normalized feature vector as float32 numpy array
        """
        if self.model is None:
            raise RuntimeError("OSNetReID.load() must be called before extract_embedding()")

        tensor = self.preprocess_crop(crop)

        t0 = time.perf_counter()
        with torch.no_grad():
            embedding = self.model(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Convert to numpy float32
        result = embedding.cpu().numpy().flatten().astype(np.float32)

        self.total_inferences += 1
        self.total_inference_ms += elapsed_ms

        return result

    def extract_embeddings_batch(
        self, crops: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Extract L2-normalized embeddings from multiple player crops in batch.

        Batch processing is significantly more efficient than single-crop
        inference due to GPU parallelism.

        Args:
            crops: List of BGR image crops

        Returns:
            List of 512-d L2-normalized feature vectors as float32 arrays
        """
        if self.model is None:
            raise RuntimeError("OSNetReID.load() must be called before extract_embeddings_batch()")

        if not crops:
            return []

        # Preprocess all crops
        tensors = []
        for crop in crops:
            tensor = self.preprocess_crop(crop)
            tensors.append(tensor)

        # Stack into batch
        batch = torch.cat(tensors, dim=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            embeddings = self.model(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of numpy arrays
        results = [emb.cpu().numpy().flatten().astype(np.float32) for emb in embeddings]

        self.total_inferences += len(crops)
        self.total_inference_ms += elapsed_ms

        return results

    def compute_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.

        Since embeddings are L2-normalized, cosine similarity = dot product.

        Args:
            emb1: First L2-normalized embedding
            emb2: Second L2-normalized embedding

        Returns:
            Cosine similarity in [-1, 1] range (higher = more similar)
        """
        return float(np.dot(emb1, emb2))

    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._is_loaded

    def get_avg_inference_time_ms(self) -> float:
        """Get average inference time per embedding in milliseconds."""
        if self.total_inferences == 0:
            return 0.0
        return self.total_inference_ms / self.total_inferences

    def get_metrics(self) -> dict:
        """Get ReID inference metrics."""
        return {
            "model": self.config.model,
            "device": str(self.device),
            "total_embeddings": self.total_inferences,
            "avg_inference_ms": round(self.get_avg_inference_time_ms(), 2),
            "total_inference_ms": round(self.total_inference_ms, 2),
            "input_size": self.config.input_size,
        }