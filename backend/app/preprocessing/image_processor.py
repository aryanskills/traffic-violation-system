"""
Image Preprocessing Pipeline — v2.0

Phases:
  1. Quality assessment  (brightness, contrast, blur, noise, resolution)
  2. Enhancement         (CLAHE, gamma correction, denoising, sharpening)
  3. Environmental fixes (low-light, rain, shadow, motion blur)
  4. Normalisation       (resize to target, pad)

Output: PreprocessingResult with enhanced image + quality metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    brightness: float = 0.0        # 0–255 mean luminance
    contrast: float = 0.0          # std of luminance
    blur_score: float = 0.0        # Laplacian variance (higher = sharper)
    noise_score: float = 0.0       # estimated noise sigma
    resolution_ok: bool = True
    is_low_light: bool = False
    is_blurry: bool = False
    is_rainy: bool = False
    has_shadows: bool = False
    overall_quality: float = 1.0   # 0–1 composite score


@dataclass
class PreprocessingResult:
    image: np.ndarray
    original_shape: Tuple[int, int, int]
    applied_steps: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)


class ImagePreprocessor:
    TARGET_SIZE: Tuple[int, int] = (640, 640)
    LOW_LIGHT_THRESHOLD = 75.0
    BLUR_THRESHOLD = 80.0
    MIN_RESOLUTION = (320, 240)

    def __init__(
        self,
        enable_low_light: bool = True,
        enable_deblur: bool = True,
        enable_rain_removal: bool = True,
        enable_shadow_removal: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.enable_low_light = enable_low_light
        self.enable_deblur = enable_deblur
        self.enable_rain_removal = enable_rain_removal
        self.enable_shadow_removal = enable_shadow_removal
        self.target_size = target_size or self.TARGET_SIZE
        logger.info("ImagePreprocessor v2 initialised")

    def process(self, image: np.ndarray) -> PreprocessingResult:
        t0 = time.perf_counter()
        img = image.copy()
        steps: List[str] = []
        original_shape = img.shape

        # ── Phase 1: Quality assessment ───────────────────────────────────────
        metrics = self._assess(img)

        # ── Phase 2: Resolution check ─────────────────────────────────────────
        h, w = img.shape[:2]
        if h < self.MIN_RESOLUTION[1] or w < self.MIN_RESOLUTION[0]:
            img = cv2.resize(img, (max(w * 2, self.MIN_RESOLUTION[0]),
                                   max(h * 2, self.MIN_RESOLUTION[1])),
                             interpolation=cv2.INTER_CUBIC)
            steps.append("upscale_low_res")

        # ── Phase 3: Enhancement ──────────────────────────────────────────────
        if metrics.is_low_light and self.enable_low_light:
            img = self._enhance_low_light(img)
            steps.append("low_light_clahe_gamma")

        if metrics.is_blurry and self.enable_deblur:
            img = self._deblur(img)
            steps.append("deblur_unsharp")

        if metrics.is_rainy and self.enable_rain_removal:
            img = self._remove_rain(img)
            steps.append("rain_removal")

        if metrics.has_shadows and self.enable_shadow_removal:
            img = self._remove_shadows(img)
            steps.append("shadow_normalisation")

        # ── Phase 4: CLAHE on all images (always improves detection) ──────────
        img = self._clahe_enhance(img)
        steps.append("clahe")

        # ── Phase 5: Normalise + pad ──────────────────────────────────────────
        img = self._resize_pad(img)
        steps.append("resize_pad")

        # ── Recompute quality after enhancement ───────────────────────────────
        metrics = self._assess(img)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"Preprocessed {original_shape} → {img.shape} in {elapsed:.1f}ms | {steps}")

        return PreprocessingResult(
            image=img,
            original_shape=original_shape,
            applied_steps=steps,
            processing_time_ms=elapsed,
            quality_metrics=metrics,
        )

    def process_from_bytes(self, image_bytes: bytes) -> PreprocessingResult:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image bytes")
        return self.process(img)

    # ── Quality Assessment ────────────────────────────────────────────────────

    def _assess(self, img: np.ndarray) -> QualityMetrics:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray.astype(np.float32)))

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        blur_score = float(lap.var())

        # Noise: difference between image and Gaussian smoothed version
        smooth = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        noise_score = float(np.std(gray.astype(np.float32) - smooth))

        # Rain heuristic: high vertical gradient ratio
        sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        vert_ratio = np.mean(np.abs(sob_y)) / (np.mean(np.abs(sob_x)) + 1e-6)

        is_low_light = brightness < self.LOW_LIGHT_THRESHOLD
        is_blurry = blur_score < self.BLUR_THRESHOLD
        is_rainy = vert_ratio > 1.15
        has_shadows = contrast > 60 and brightness > 60  # broad shadow check

        # Overall quality 0–1
        sharpness_score = min(blur_score / 500.0, 1.0)
        brightness_score = 1.0 - abs(brightness/255.0 - 0.45) * 2
        quality = float(0.5 * sharpness_score + 0.3 * max(0, brightness_score) + 0.2 * min(contrast/80.0, 1.0))

        return QualityMetrics(
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            blur_score=round(blur_score, 2),
            noise_score=round(noise_score, 2),
            is_low_light=is_low_light,
            is_blurry=is_blurry,
            is_rainy=is_rainy,
            has_shadows=has_shadows,
            overall_quality=round(min(1.0, max(0.0, quality)), 3),
        )

    # ── Enhancement Methods ───────────────────────────────────────────────────

    def _enhance_low_light(self, img: np.ndarray) -> np.ndarray:
        """CLAHE in LAB + gamma correction."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # Gamma correction: boost dark pixels
        l_f = l.astype(np.float32) / 255.0
        gamma = 0.55
        l_f = np.power(l_f, gamma) * 255.0
        l = np.clip(l_f, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _clahe_enhance(self, img: np.ndarray) -> np.ndarray:
        """Standard CLAHE for consistent contrast."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _deblur(self, img: np.ndarray) -> np.ndarray:
        """Unsharp masking for motion/focus blur."""
        gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        sharp = cv2.addWeighted(img, 1.6, gaussian, -0.6, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _remove_rain(self, img: np.ndarray) -> np.ndarray:
        """Guided filter approximation to suppress rain streaks."""
        guide = cv2.medianBlur(img, 3)
        result = np.zeros_like(img, dtype=np.float32)
        radius, eps = 4, 0.01 * 255 ** 2
        for c in range(3):
            p = img[:,:,c].astype(np.float32)
            g = guide[:,:,c].astype(np.float32)
            ks = (2*radius+1, 2*radius+1)
            mp = cv2.boxFilter(p, -1, ks)
            mg = cv2.boxFilter(g, -1, ks)
            mgp = cv2.boxFilter(g*p, -1, ks)
            mgg = cv2.boxFilter(g*g, -1, ks)
            a_ = (mgp - mg*mp) / (mgg - mg*mg + eps)
            b_ = mp - a_*mg
            result[:,:,c] = cv2.boxFilter(a_,-1,ks)*g + cv2.boxFilter(b_,-1,ks)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """HSV value-channel normalisation to flatten shadows."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:,:,2]
        ks = max(img.shape[0], img.shape[1]) // 8
        ks = ks if ks % 2 == 1 else ks + 1
        illum = cv2.morphologyEx(v, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        hsv[:,:,2] = np.clip(v / (illum + 1e-6) * 128, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _resize_pad(self, img: np.ndarray) -> np.ndarray:
        """Letterbox resize to target size."""
        th, tw = self.target_size
        h, w = img.shape[:2]
        scale = min(tw/w, th/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (th-nh)//2; bot = th-nh-top
        left = (tw-nw)//2; right = tw-nw-left
        return cv2.copyMakeBorder(resized, top, bot, left, right,
                                  cv2.BORDER_CONSTANT, value=(114,114,114))

    # Backward compat
    def _compute_quality_score(self, img):
        return self._assess(img).overall_quality

    def _is_low_light(self, img, threshold=75.0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < threshold

    def _is_blurry(self, img, threshold=80.0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_32F).var()) < threshold
