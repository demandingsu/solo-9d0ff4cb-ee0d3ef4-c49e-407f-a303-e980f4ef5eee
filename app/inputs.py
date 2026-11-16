"""CSV parsing and validation for the two inputs.

Both uploads are *whole-file replacements*: parsing completely succeeds
before the database is touched, so a bad file can never leave a
half-replaced input behind.

Formats (header names of the first column are free-form):

baseline.csv::

    ore,Fe,SiO2,Al2O3,CaO
    A,62.30,3.90,2.10,1.20
    B,68.50,2.50,1.10,0.80
    C,55.20,6.20,3.40,2.10

samples.csv::

    sample_id,Fe,SiO2,Al2O3,CaO
    S-1001,62.74,3.71,2.02,1.13

Exactly three ore rows / four element columns are required for the
baseline.  Sample element labels are matched to the baseline by name at
calculation time.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
# Bounds the size of the exact integers.  Composition values are normally
# well under 10^5; this limit stops e.g. ``1e999999999`` (a few characters
# in the file) from exploding memory during exact rational arithmetic and
# report serialisation.
MAX_ADJUSTED_EXPONENT = 100
MAX_SAMPLES = 2000
ELEMENT_COUNT = 4
ORE_COUNT = 3

_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


class InputValidationError(ValueError):
    """A user-fixable problem with an uploaded CSV."""


@dataclass(frozen=True)
class BaselineInput:
    elements: tuple[str, ...]
    ores: tuple[str, ...]
    rows: tuple[tuple[Fraction, ...], ...]  # one row per ore

    def to_payload(self) -> dict:
        return {
            "elements": list(self.elements),
            "ores": list(self.ores),
            "rows": [
                [fraction_to_text(value) for value in row] for row in self.rows
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "BaselineInput":
        return cls(
            elements=tuple(payload["elements"]),
            ores=tuple(payload["ores"]),
            rows=tuple(
                tuple(Fraction(value) for value in row)
                for row in payload["rows"]
            ),
        )


@dataclass(frozen=True)
class SamplesInput:
    elements: tuple[str, ...]
    sample_ids: tuple[str, ...]
    rows: tuple[tuple[Fraction, ...], ...]  # one row per sample

    def to_payload(self) -> dict:
        return {
            "elements": list(self.elements),
            "sample_ids": list(self.sample_ids),
            "rows": [
                [fraction_to_text(value) for value in row] for row in self.rows
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SamplesInput":
        return cls(
            elements=tuple(payload["elements"]),
            sample_ids=tuple(payload["sample_ids"]),
            rows=tuple(
                tuple(Fraction(value) for value in row)
                for row in payload["rows"]
            ),
        )


def fraction_to_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _shown(raw: str) -> str:
    text = raw.strip()
    return repr(text if len(text) <= 40 else text[:40] + "…")


def parse_content(raw: str, *, line_no: int, column: str) -> Fraction:
    text = raw.strip()
    if not text:
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：含量不能为空"
        )
    shown = _shown(raw)
    if not _NUMBER_RE.fullmatch(text):
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：含量必须是有限非负数，实际为 {shown}"
        )
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：含量必须是有限非负数，实际为 {shown}"
        ) from None
    if not decimal.is_finite():
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：含量必须是有限非负数，实际为 {shown}"
        )
    if decimal < 0:
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：含量不能为负数（{shown}）"
        )
    if decimal != 0 and abs(decimal.adjusted()) > MAX_ADJUSTED_EXPONENT:
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：数值量级超出支持范围"
            f"（绝对值须小于 10^{MAX_ADJUSTED_EXPONENT + 1} 且不小于"
            f" 10^-{MAX_ADJUSTED_EXPONENT}）"
        )
    return Fraction(decimal)


def _read_rows(raw: bytes, label: str) -> list[tuple[int, list[str]]]:
    if not raw:
        raise InputValidationError(f"{label}文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InputValidationError(
            f"{label}文件过大（上限 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB）"
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputValidationError(f"{label}文件必须是 UTF-8 编码的 CSV") from None
    if "\x00" in text:
        raise InputValidationError(f"{label}文件包含 NUL 字符，不是有效的 CSV")

    rows: list[tuple[int, list[str]]] = []
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            rows.append((line_no, row))
    except csv.Error as exc:
        raise InputValidationError(f"{label} CSV 解析失败：{exc}") from None
    if not rows:
        raise InputValidationError(f"{label}文件没有表头和数据行")
    return rows


def _header(rows: list[tuple[int, list[str]]], label: str) -> list[str]:
    line_no, raw_header = rows[0]
    if len(raw_header) != ELEMENT_COUNT + 1:
        raise InputValidationError(
            f"{label}表头必须有 {ELEMENT_COUNT + 1} 列"
            f"（标识列 + {ELEMENT_COUNT} 个元素列），实际 {len(raw_header)} 列"
        )
    header = [cell.strip() for cell in raw_header]
    if any(not cell for cell in header):
        raise InputValidationError(f"第 {line_no} 行：表头存在空列名")
    if len(set(header)) != len(header):
        raise InputValidationError(f"第 {line_no} 行：表头列名不能重复")
    return header


def _data_rows(
    rows: list[tuple[int, list[str]]], label: str
) -> list[tuple[int, list[str]]]:
    result: list[tuple[int, list[str]]] = []
    for line_no, row in rows[1:]:
        if len(row) != ELEMENT_COUNT + 1:
            raise InputValidationError(
                f"第 {line_no} 行：应有 {ELEMENT_COUNT + 1} 列，实际 {len(row)} 列"
            )
        result.append((line_no, [cell.strip() for cell in row]))
    if not result:
        raise InputValidationError(f"{label}文件没有数据行")
    return result


def parse_baseline(raw: bytes) -> BaselineInput:
    rows = _read_rows(raw, "基准")
    header = _header(rows, "基准")
    elements = tuple(header[1:])
    data = _data_rows(rows, "基准")

    if len(data) != ORE_COUNT:
        raise InputValidationError(
            f"基准 CSV 必须恰好包含 {ORE_COUNT} 种原矿，实际 {len(data)} 行"
        )

    ores: list[str] = []
    parsed_rows: list[tuple[Fraction, ...]] = []
    for line_no, row in data:
        ore = row[0]
        if not ore:
            raise InputValidationError(f"第 {line_no} 行：原矿名称不能为空")
        if ore in ores:
            raise InputValidationError(
                f"第 {line_no} 行：原矿名称“{ore}”重复"
            )
        ores.append(ore)
        parsed_rows.append(
            tuple(
                parse_content(value, line_no=line_no, column=elements[index])
                for index, value in enumerate(row[1:])
            )
        )

    return BaselineInput(
        elements=elements, ores=tuple(ores), rows=tuple(parsed_rows)
    )


def parse_samples(raw: bytes) -> SamplesInput:
    rows = _read_rows(raw, "成品")
    header = _header(rows, "成品")
    elements = tuple(header[1:])
    data = _data_rows(rows, "成品")
    if len(data) > MAX_SAMPLES:
        raise InputValidationError(
            f"成品 CSV 最多支持 {MAX_SAMPLES} 个样本，实际 {len(data)} 个"
        )

    sample_ids: list[str] = []
    parsed_rows: list[tuple[Fraction, ...]] = []
    seen: set[str] = set()
    for line_no, row in data:
        sample_id = row[0]
        if not sample_id:
            raise InputValidationError(f"第 {line_no} 行：样本编号不能为空")
        if sample_id in seen:
            raise InputValidationError(
                f"第 {line_no} 行：样本编号“{sample_id}”重复"
            )
        seen.add(sample_id)
        sample_ids.append(sample_id)
        parsed_rows.append(
            tuple(
                parse_content(value, line_no=line_no, column=elements[index])
                for index, value in enumerate(row[1:])
            )
        )

    return SamplesInput(
        elements=elements,
        sample_ids=tuple(sample_ids),
        rows=tuple(parsed_rows),
    )
