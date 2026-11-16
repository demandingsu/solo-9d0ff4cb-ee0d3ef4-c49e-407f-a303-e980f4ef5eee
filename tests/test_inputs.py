"""Tests for CSV validation."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from app.inputs import InputValidationError, parse_baseline, parse_samples

GOOD_BASE = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

GOOD_SAMPLES = """sample_id,Fe,SiO2,Al2O3,CaO
S1,62.74,3.71,2.02,1.13
S2,61,4,2,1
"""


def test_parse_baseline_ok():
    parsed = parse_baseline(GOOD_BASE.encode())
    assert parsed.ores == ("A", "B", "C")
    assert parsed.elements == ("Fe", "SiO2", "Al2O3", "CaO")
    assert parsed.rows[0][0] == F("62.3")


def test_parse_samples_ok():
    parsed = parse_samples(GOOD_SAMPLES.encode())
    assert parsed.sample_ids == ("S1", "S2")
    assert parsed.rows[1][0] == 61


def test_utf8_bom_and_blank_lines():
    raw = ("﻿" + GOOD_BASE).replace("\n", "\n\n")
    assert parse_baseline(raw.encode("utf-8")) == parse_baseline(GOOD_BASE.encode())


@pytest.mark.parametrize(
    "value",
    ["-1", "nan", "NaN", "inf", "-inf", "", "abc", "1,2", "1e999999", "0x10"],
)
def test_rejects_invalid_amounts(value):
    raw = GOOD_BASE.replace("62.3", value, 1)
    with pytest.raises(InputValidationError):
        parse_baseline(raw.encode())


def test_accepts_scientific_notation():
    raw = GOOD_BASE.replace("62.3", "6.23e1", 1)
    assert parse_baseline(raw.encode()).rows[0][0] == F("62.3")


def test_negative_zero_is_accepted():
    raw = GOOD_BASE.replace("62.3", "-0.0", 1)
    assert parse_baseline(raw.encode()).rows[0][0] == 0


def test_rejects_wrong_ore_count():
    raw = GOOD_BASE + "D,50,5,5,5\n"
    with pytest.raises(InputValidationError, match="恰好包含 3"):
        parse_baseline(raw.encode())


def test_rejects_duplicate_ore_name():
    raw = GOOD_BASE.replace("\nB,", "\nA,", 1)
    with pytest.raises(InputValidationError, match="重复"):
        parse_baseline(raw.encode())


def test_rejects_duplicate_sample_id():
    raw = GOOD_SAMPLES.replace("\nS2,", "\nS1,", 1)
    with pytest.raises(InputValidationError, match="重复"):
        parse_samples(raw.encode())


def test_rejects_blank_sample_id():
    raw = GOOD_SAMPLES.replace("\nS2,", "\n,", 1)
    with pytest.raises(InputValidationError, match="样本编号不能为空"):
        parse_samples(raw.encode())


def test_rejects_wrong_column_count():
    raw = GOOD_SAMPLES.replace("S2,61,4,2,1", "S2,61,4,2")
    with pytest.raises(InputValidationError, match="应有 5 列"):
        parse_samples(raw.encode())


def test_rejects_duplicate_header_names():
    raw = GOOD_SAMPLES.replace("SiO2", "Fe")
    with pytest.raises(InputValidationError, match="不能重复"):
        parse_samples(raw.encode())


def test_rejects_empty_and_binary_files():
    with pytest.raises(InputValidationError):
        parse_samples(b"")
    with pytest.raises(InputValidationError):
        parse_baseline(b"\x00\x01\x02")
    with pytest.raises(InputValidationError):
        parse_baseline("矿石\n".encode("gb18030"))


def test_rejects_header_only_file():
    with pytest.raises(InputValidationError, match="没有数据行"):
        parse_baseline(b"ore,Fe,SiO2,Al2O3,CaO\n")
