"""
Unit tests for the Evaluation Framework.
Run: pytest backend/tests/unit/test_evaluator.py -v
"""

import pytest
from app.evaluation.evaluator import (
    DetectionEvaluator, OCREvaluator, compute_iou
)


class TestComputeIoU:
    def test_perfect_overlap(self):
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.1 < iou < 0.5

    def test_contained_box(self):
        iou = compute_iou((25, 25, 75, 75), (0, 0, 100, 100))
        assert iou == pytest.approx(0.25, abs=0.01)


class TestDetectionEvaluator:
    def setup_method(self):
        self.evaluator = DetectionEvaluator(iou_threshold=0.5)

    def _box(self, x1, y1, x2, y2, cls=2, conf=0.9):
        return {"bbox": (x1, y1, x2, y2), "class": cls, "conf": conf}

    def test_perfect_detection(self):
        preds = [self._box(0, 0, 100, 100)]
        gts = [self._box(0, 0, 100, 100)]
        m = self.evaluator.evaluate(preds, gts)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1_score == pytest.approx(1.0)
        assert m.true_positives == 1
        assert m.false_positives == 0

    def test_all_false_positives(self):
        preds = [self._box(200, 200, 300, 300)]
        gts = [self._box(0, 0, 100, 100)]
        m = self.evaluator.evaluate(preds, gts)
        assert m.false_positives == 1
        assert m.false_negatives == 1
        assert m.true_positives == 0

    def test_empty_predictions(self):
        gts = [self._box(0, 0, 100, 100)]
        m = self.evaluator.evaluate([], gts)
        assert m.true_positives == 0
        assert m.false_negatives == 1

    def test_empty_ground_truth(self):
        preds = [self._box(0, 0, 100, 100)]
        m = self.evaluator.evaluate(preds, [])
        assert m.false_positives == 1
        assert m.true_positives == 0

    def test_class_mismatch_not_matched(self):
        preds = [self._box(0, 0, 100, 100, cls=2)]
        gts = [self._box(0, 0, 100, 100, cls=3)]  # Different class
        m = self.evaluator.evaluate(preds, gts)
        assert m.true_positives == 0


class TestOCREvaluator:
    def setup_method(self):
        self.evaluator = OCREvaluator()

    def test_perfect_recognition(self):
        preds = ["MH12AB1234", "DL01CA9999"]
        gts = ["MH12AB1234", "DL01CA9999"]
        m = self.evaluator.evaluate(preds, gts)
        assert m.character_accuracy == pytest.approx(1.0)
        assert m.word_accuracy == pytest.approx(1.0)
        assert m.plate_recognition_accuracy == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = ["XXXXXXXXXX"]
        gts = ["MH12AB1234"]
        m = self.evaluator.evaluate(preds, gts)
        assert m.character_accuracy < 0.5
        assert m.plate_recognition_accuracy == pytest.approx(0.0)

    def test_empty_inputs(self):
        m = self.evaluator.evaluate([], [])
        assert m.character_accuracy == 0.0

    def test_levenshtein_identical(self):
        assert self.evaluator._levenshtein("hello", "hello") == 0

    def test_levenshtein_single_substitution(self):
        assert self.evaluator._levenshtein("abc", "axc") == 1

    def test_levenshtein_insertion(self):
        assert self.evaluator._levenshtein("ab", "abc") == 1

    def test_levenshtein_deletion(self):
        assert self.evaluator._levenshtein("abc", "ab") == 1
