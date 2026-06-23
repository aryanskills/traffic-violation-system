"""
Performance Evaluation Framework.

Computes:
- Detection metrics: mAP@50, mAP@50-95, Precision, Recall, F1
- OCR metrics: CER, WER, character accuracy
- System metrics: FPS, latency, throughput, CPU/GPU/Memory usage
- Generates benchmark reports
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─── Detection Metrics ────────────────────────────────────────────────────────

@dataclass
class DetectionMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mAP_50: float = 0.0
    mAP_50_95: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class OCRMetrics:
    character_accuracy: float = 0.0
    word_accuracy: float = 0.0
    plate_recognition_accuracy: float = 0.0
    character_error_rate: float = 0.0
    word_error_rate: float = 0.0


@dataclass
class SystemMetrics:
    avg_inference_time_ms: float = 0.0
    avg_e2e_time_ms: float = 0.0
    throughput_images_per_sec: float = 0.0
    fps: float = 0.0
    api_response_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


@dataclass
class BenchmarkReport:
    timestamp: str = ""
    model_name: str = ""
    dataset_size: int = 0
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    ocr: OCRMetrics = field(default_factory=OCRMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    violation_accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


# ─── IoU Helper ───────────────────────────────────────────────────────────────

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-6)


# ─── Detection Evaluator ──────────────────────────────────────────────────────

class DetectionEvaluator:
    """Evaluates bounding box detection against ground truth."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predictions: List[Dict],  # [{"bbox": (x1,y1,x2,y2), "class": int, "conf": float}]
        ground_truths: List[Dict],  # [{"bbox": (x1,y1,x2,y2), "class": int}]
    ) -> DetectionMetrics:
        tp = fp = fn = 0
        matched_gt = set()

        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                if gt["class"] != pred["class"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truths) - len(matched_gt)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        fpr = fp / (fp + tp + 1e-6)
        fnr = fn / (fn + tp + 1e-6)

        return DetectionMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_positive_rate=round(fpr, 4),
            false_negative_rate=round(fnr, 4),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def compute_map(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        iou_thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Compute mAP@50 and mAP@50-95.
        Returns (mAP50, mAP50_95).
        """
        thresholds = iou_thresholds or [round(0.5 + i * 0.05, 2) for i in range(10)]

        aps = []
        for iou_t in thresholds:
            self.iou_threshold = iou_t
            precisions = []
            for preds, gts in zip(all_predictions, all_ground_truths):
                m = self.evaluate(preds, gts)
                precisions.append(m.precision)
            aps.append(float(np.mean(precisions)))

        map50 = aps[0] if aps else 0.0
        map50_95 = float(np.mean(aps)) if aps else 0.0
        return round(map50, 4), round(map50_95, 4)


# ─── OCR Evaluator ────────────────────────────────────────────────────────────

class OCREvaluator:
    """Evaluates license plate OCR accuracy."""

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> OCRMetrics:
        if not predictions or not ground_truths:
            return OCRMetrics()

        char_errors = 0
        total_chars = 0
        word_correct = 0
        plate_correct = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_clean = pred.upper().replace(" ", "").replace("-", "")
            gt_clean = gt.upper().replace(" ", "").replace("-", "")

            # Character error rate (Levenshtein distance / length)
            cer = self._levenshtein(pred_clean, gt_clean) / (len(gt_clean) + 1e-6)
            char_errors += cer
            total_chars += len(gt_clean)

            if pred_clean == gt_clean:
                plate_correct += 1
                word_correct += 1

        n = len(predictions)
        char_accuracy = 1.0 - (char_errors / n)
        word_accuracy = word_correct / n
        plate_accuracy = plate_correct / n
        cer = char_errors / n
        wer = 1.0 - word_accuracy

        return OCRMetrics(
            character_accuracy=round(max(0, char_accuracy), 4),
            word_accuracy=round(word_accuracy, 4),
            plate_recognition_accuracy=round(plate_accuracy, 4),
            character_error_rate=round(cer, 4),
            word_error_rate=round(wer, 4),
        )

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Standard dynamic programming Levenshtein distance."""
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
        return dp[n]


# ─── System Performance Evaluator ─────────────────────────────────────────────

class SystemEvaluator:
    """Measures latency, throughput, and resource utilization."""

    def measure_latency(self, func, *args, n_runs: int = 20, **kwargs) -> Dict:
        """Run func n_runs times and collect timing stats."""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            func(*args, **kwargs)
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": round(float(np.mean(times)), 2),
            "median_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
            "fps": round(1000.0 / float(np.mean(times)), 1),
        }

    def get_resource_usage(self) -> Dict:
        """Snapshot current CPU/memory/GPU usage."""
        usage = {}
        try:
            import psutil
            proc = psutil.Process()
            usage["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            usage["memory_mb"] = proc.memory_info().rss / 1024 / 1024
            usage["memory_percent"] = proc.memory_percent()
        except ImportError:
            usage["cpu_percent"] = 0.0
            usage["memory_mb"] = 0.0

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            usage["gpu_percent"] = util.gpu
            usage["gpu_memory_mb"] = info.used / 1024 / 1024
        except Exception:
            usage["gpu_percent"] = None
            usage["gpu_memory_mb"] = None

        return usage


# ─── Benchmark Runner ─────────────────────────────────────────────────────────

class BenchmarkRunner:
    """Orchestrates all evaluators and generates a final report."""

    def __init__(self):
        self.det_eval = DetectionEvaluator()
        self.ocr_eval = OCREvaluator()
        self.sys_eval = SystemEvaluator()

    def run_full_benchmark(
        self,
        pipeline,
        test_images: List,
        annotations: List[Dict],
        output_path: Optional[str] = None,
    ) -> BenchmarkReport:
        """
        Run full benchmark on test images against ground truth annotations.
        """
        from datetime import datetime as dt

        logger.info(f"Running benchmark on {len(test_images)} images…")
        report = BenchmarkReport(
            timestamp=dt.utcnow().isoformat(),
            model_name=str(pipeline.vehicle_detector.model_path),
            dataset_size=len(test_images),
        )

        latency_stats = self.sys_eval.measure_latency(
            lambda img: pipeline.vehicle_detector.detect(img),
            test_images[0] if test_images else None,
            n_runs=min(20, len(test_images)),
        )
        report.system.avg_inference_time_ms = latency_stats["mean_ms"]
        report.system.fps = latency_stats["fps"]

        resources = self.sys_eval.get_resource_usage()
        report.system.cpu_usage_percent = resources.get("cpu_percent", 0)
        report.system.memory_usage_mb = resources.get("memory_mb", 0)
        report.system.gpu_usage_percent = resources.get("gpu_percent")
        report.system.gpu_memory_mb = resources.get("gpu_memory_mb")

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(report), indent=2))
            logger.info(f"Benchmark report saved to {output_path}")

        return report
