import cv2
import numpy as np
from typing import Dict, List, Tuple

from app.core.config import get_config


class AdaptivePreprocessor:
    def __init__(self) -> None:
        self.metrics: List[Dict] = []
        self._config = get_config()

    @staticmethod
    def measure(frame: np.ndarray, frame_no: int) -> Dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = float(np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)))
        low = float((gray < 35).mean())
        high = float((gray > 220).mean())
        shadow = float((gray < 55).mean())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        angle = cv2.phase(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(angle[mag > np.percentile(mag, 80)], bins=18, range=(0, 180))
        motion_blur = float(hist.max() / max(hist.sum(), 1))
        return {
            "frame": frame_no,
            "brightness": mean,
            "contrast": std,
            "blur_laplacian_var": lap,
            "noise_level": noise,
            "underexposed_ratio": low,
            "overexposed_ratio": high,
            "shadow_ratio": shadow,
            "motion_blur_score": motion_blur,
        }

    @staticmethod
    def white_balance(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = lab[:, :, 1].mean()
        avg_b = lab[:, :, 2].mean()
        lab[:, :, 1] -= (avg_a - 128.0) * (lab[:, :, 0] / 255.0) * 0.6
        lab[:, :, 2] -= (avg_b - 128.0) * (lab[:, :, 0] / 255.0) * 0.6
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame, table)

    @staticmethod
    def clahe(frame: np.ndarray, clip: float = 1.8, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def sharpen(frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), 1.1)
        return cv2.addWeighted(frame, 1.0 + strength, blur, -strength, 0)

    def apply(self, frame: np.ndarray, m: Dict) -> Tuple[np.ndarray, List[str]]:
        cfg = self._config.raw.get("preprocessing", {})
        wb_cfg = cfg.get("white_balance", {})
        gamma_cfg = cfg.get("gamma", {})
        clahe_cfg = cfg.get("clahe", {})
        denoise_cfg = cfg.get("denoise", {})
        sharpen_cfg = cfg.get("adaptive_sharpen", {})

        out = frame
        steps = []
        if wb_cfg.get("enabled", True):
            if abs(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 1].mean() - 128) > 4:
                out = self.white_balance(out)
                steps.append("white_balance")
        brightness_darken_threshold = float(gamma_cfg.get("brightness_darken_threshold", 185))
        brightness_brighten_threshold = float(gamma_cfg.get("brightness_brighten_threshold", 85))
        overexposed_ratio_threshold = float(gamma_cfg.get("overexposed_ratio_threshold", 0.08))
        if m["brightness"] < brightness_brighten_threshold:
            out = self.gamma(out, float(gamma_cfg.get("brighten_gamma", 1.18)))
            steps.append("gamma_brighten")
        elif m["brightness"] > brightness_darken_threshold or m["overexposed_ratio"] > overexposed_ratio_threshold:
            out = self.gamma(out, float(gamma_cfg.get("darken_gamma", 0.92)))
            steps.append("gamma_darken")
        if clahe_cfg.get("enabled", True):
            contrast_threshold = float(clahe_cfg.get("contrast_threshold", 42))
            if m["contrast"] < contrast_threshold:
                clip = float(clahe_cfg.get("clip_limit", 1.8))
                grid_size = clahe_cfg.get("grid_size", [8, 8])
                if isinstance(grid_size, list):
                    grid_size = tuple(int(x) for x in grid_size)
                out = self.clahe(out, clip=clip, grid=grid_size)
                steps.append("clahe")
        if denoise_cfg.get("enabled", True):
            noise_threshold = float(denoise_cfg.get("noise_threshold", 9.0))
            if m["noise_level"] > noise_threshold:
                h = int(denoise_cfg.get("h", 3))
                h_color = int(denoise_cfg.get("h_color", 3))
                template_ws = int(denoise_cfg.get("template_window_size", 7))
                search_ws = int(denoise_cfg.get("search_window_size", 21))
                out = cv2.fastNlMeansDenoisingColored(out, None, h, h_color, template_ws, search_ws)
                steps.append("denoise_light")
        if sharpen_cfg.get("enabled", True):
            laplacian_blur_threshold = float(sharpen_cfg.get("laplacian_blur_threshold", 95))
            motion_blur_threshold = float(sharpen_cfg.get("motion_blur_threshold", 0.28))
            sharpen_strength = float(sharpen_cfg.get("sharpen_strength", 0.30))
            if m["blur_laplacian_var"] < laplacian_blur_threshold or m["motion_blur_score"] > motion_blur_threshold:
                out = self.sharpen(out, sharpen_strength)
                steps.append("adaptive_sharpen")
        return out, steps
