"""
Unit tests for License Plate Recognition.
Run: pytest backend/tests/unit/test_ocr.py -v
"""

import pytest
from app.services.ocr.plate_recognizer import normalize_plate, validate_plate, PlateResult


class TestNormalizePlate:
    def test_removes_spaces(self):
        assert normalize_plate("MH 12 AB 1234") == "MH12AB1234"

    def test_removes_hyphens(self):
        assert normalize_plate("DL-01-CA-9999") == "DL01CA9999"

    def test_uppercase(self):
        assert normalize_plate("ka03mj4567") == "KA03MJ4567"

    def test_strips_whitespace(self):
        assert normalize_plate("  GJ05BT1234  ") == "GJ05BT1234"


class TestValidatePlate:
    def test_valid_standard_plate(self):
        assert validate_plate("MH12AB1234") is True

    def test_valid_with_spaces(self):
        assert validate_plate("DL 01 CA 9999") is True

    def test_invalid_too_short(self):
        assert validate_plate("AB1") is False

    def test_invalid_random_string(self):
        assert validate_plate("XXXXXXXXXX") is False

    def test_bh_series(self):
        # BH series: 22BH0001AA
        assert validate_plate("22BH0001AA") is True


class TestPlateResult:
    def test_display_text_uses_normalized(self):
        p = PlateResult(
            raw_text="MH 12 AB 1234",
            normalized_text="MH12AB1234",
            is_valid_format=True,
            detection_confidence=0.9,
            ocr_confidence=0.85,
            bbox=None,
        )
        assert p.display_text == "MH12AB1234"

    def test_display_text_fallback_raw(self):
        p = PlateResult(
            raw_text="SOME RAW",
            normalized_text=None,
            is_valid_format=False,
            detection_confidence=0.5,
            ocr_confidence=0.4,
            bbox=None,
        )
        assert p.display_text == "SOME RAW"

    def test_display_text_unknown_fallback(self):
        p = PlateResult(
            raw_text=None,
            normalized_text=None,
            is_valid_format=False,
            detection_confidence=0.0,
            ocr_confidence=0.0,
            bbox=None,
        )
        assert p.display_text == "UNKNOWN"
