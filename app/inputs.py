"""CSV parsing and validation for the inputs.

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

assay.csv (the instrument's original sheet; column labels are arbitrary
and are mapped to the baseline elements by the user)::

    sample_no,Fe_pct,Si,Al,Ca
    S-1001,62.74,37100,20200,11300

An assay sheet carries four element columns and one sample-id column.
Each element column is mapped to a distinct current baseline element and
declared to be either ``%`` or ``ppm``; conversion to percent is exact
(``1 ppm = 1/10000 %``).  Exactly three ore rows / four element columns
are required for the baseline.
"""

from __future__ import annotations

import base64
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
MAX_REPORTED_ERRORS = 25
ASSAY_UNITS = ("%", "ppm")
PPM_PER_PERCENT = 10000

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


def _decode_upload(raw: bytes, label: str) -> str:
    """Decode one uploaded CSV after the size/encoding checks."""

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
    return text


def _records_from_text(
    text: str, label: str
) -> list[tuple[int, list[str]]]:
    """Parse CSV records, returning each record's physical starting line.

    ``csv.reader.line_num`` counts source lines (including newlines inside
    quoted cells), so line numbers in validation messages point at the
    actual row in the uploaded file rather than the number of parsed
    records.
    """

    rows: list[tuple[int, list[str]]] = []
    previous_end = 0
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        for row in reader:
            start = previous_end + 1
            end = reader.line_num
            previous_end = end
            if not row or all(not cell.strip() for cell in row):
                continue
            rows.append((start, row))
    except csv.Error as exc:
        raise InputValidationError(f"{label} CSV 解析失败：{exc}") from None
    if not rows:
        raise InputValidationError(f"{label}文件没有表头和数据行")
    return rows


def _read_rows(raw: bytes, label: str) -> list[tuple[int, list[str]]]:
    return _records_from_text(_decode_upload(raw, label), label)


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


# -- assay sheet (the instrument's original file) ------------------------


@dataclass(frozen=True)
class AssayRow:
    line_no: int
    sample_id: str
    values: tuple[str, ...]  # four raw element cells, in file order


@dataclass(frozen=True)
class AssayInput:
    """Structurally parsed original sheet.

    Values stay as raw text, and the exact uploaded bytes are kept, so the
    original can be stored unchanged (detection-limit markers included) and
    re-converted after the mapping is edited.
    """

    columns: tuple[str, ...]  # four element-column labels
    rows: tuple[AssayRow, ...]
    raw_bytes: bytes

    def to_payload(self) -> dict:
        return {
            "columns": list(self.columns),
            "rows": [
                {
                    "line": row.line_no,
                    "sample_id": row.sample_id,
                    "values": list(row.values),
                }
                for row in self.rows
            ],
            "raw_bytes_b64": base64.b64encode(self.raw_bytes).decode("ascii"),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AssayInput":
        raw_bytes = base64.b64decode(payload["raw_bytes_b64"])
        return cls(
            columns=tuple(payload["columns"]),
            rows=tuple(
                AssayRow(
                    line_no=int(row["line"]),
                    sample_id=row["sample_id"],
                    values=tuple(row["values"]),
                )
                for row in payload["rows"]
            ),
            raw_bytes=raw_bytes,
        )


@dataclass(frozen=True)
class MappingEntry:
    column: int  # 0..3, position among the element columns
    label: str   # the original sheet's header for that column
    element: str
    unit: str    # "%" or "ppm"


@dataclass(frozen=True)
class AssayMapping:
    entries: tuple[MappingEntry, ...]

    def entry_for(self, element: str) -> MappingEntry:
        for entry in self.entries:
            if entry.element == element:
                return entry
        raise KeyError(element)

    def to_payload(self) -> dict:
        return {
            "entries": [
                {
                    "column": entry.column,
                    "label": entry.label,
                    "element": entry.element,
                    "unit": entry.unit,
                }
                for entry in self.entries
            ]
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AssayMapping":
        return cls(
            entries=tuple(
                MappingEntry(
                    column=int(entry["column"]),
                    label=entry["label"],
                    element=entry["element"],
                    unit=entry["unit"],
                )
                for entry in payload["entries"]
            )
        )


class AssayConversionError(ValueError):
    """One or more rows of the original sheet cannot be converted.

    ``errors`` contains human-readable messages that always begin with the
    physical line number in the original CSV.  The caller must reject the
    whole batch: there is no partial conversion.
    """

    def __init__(self, errors: list[str], total: int) -> None:
        self.errors = errors
        self.total = total
        super().__init__(
            f"原件共有 {total} 处问题，整批未换算；首批问题："
            + "；".join(errors)
        )


def parse_assay(raw: bytes) -> AssayInput:
    """Parse the original assay sheet.

    Only structure is validated here (five columns, unique header,
    non-empty identifier column, row limit).  Values, duplicate sample
    numbers and detection-limit markers are checked during conversion so
    every such error can be reported with its original line number.
    """

    text = _decode_upload(raw, "原件")
    rows = _records_from_text(text, "原件")
    line_no, raw_header = rows[0]
    if len(raw_header) != ELEMENT_COUNT + 1:
        raise InputValidationError(
            f"原件表头必须有 {ELEMENT_COUNT + 1} 列"
            f"（样本编号列 + {ELEMENT_COUNT} 个元素列），实际 {len(raw_header)} 列"
        )
    header = [cell.strip() for cell in raw_header]
    if any(not cell for cell in header):
        raise InputValidationError(f"第 {line_no} 行：表头存在空列名")
    if len(set(header)) != len(header):
        raise InputValidationError(f"第 {line_no} 行：表头列名不能重复")
    if not header[0]:
        raise InputValidationError(f"第 {line_no} 行：样本编号列名不能为空")

    data: list[AssayRow] = []
    for row_line, row in rows[1:]:
        if len(row) != ELEMENT_COUNT + 1:
            raise InputValidationError(
                f"第 {row_line} 行：应有 {ELEMENT_COUNT + 1} 列，"
                f"实际 {len(row)} 列"
            )
        data.append(
            AssayRow(
                line_no=row_line,
                sample_id=row[0].strip(),
                values=tuple(cell.strip() for cell in row[1:]),
            )
        )
    if not data:
        raise InputValidationError("原件文件没有数据行")
    if len(data) > MAX_SAMPLES:
        raise InputValidationError(
            f"原件最多支持 {MAX_SAMPLES} 个样本，实际 {len(data)} 个"
        )
    return AssayInput(columns=tuple(header[1:]), rows=tuple(data), raw_bytes=raw)


def build_mapping(
    assay: AssayInput,
    baseline: BaselineInput,
    element_choices: list[str],
    unit_choices: list[str],
) -> AssayMapping:
    """Validate the per-column mapping submitted by the user.

    The four original element columns must map one-to-one onto the four
    current baseline elements: every element chosen once, no duplicates,
    no unknown element, and every unit must be ``%`` or ``ppm``.
    """

    if len(assay.columns) != ELEMENT_COUNT:
        raise InputValidationError(
            f"原件必须有 {ELEMENT_COUNT} 个元素列，实际 {len(assay.columns)} 个"
        )
    if len(element_choices) != ELEMENT_COUNT or len(unit_choices) != ELEMENT_COUNT:
        raise InputValidationError(
            f"必须为 {ELEMENT_COUNT} 个原件元素列各指定一个基准元素和单位"
        )

    entries: list[MappingEntry] = []
    chosen: dict[str, int] = {}
    for column, (element, unit) in enumerate(
        zip(element_choices, unit_choices)
    ):
        label = assay.columns[column]
        element = element.strip()
        unit = unit.strip()
        if not element:
            raise InputValidationError(
                f"原件“{label}”列：必须指定对应的当前基准元素"
            )
        if element not in baseline.elements:
            raise InputValidationError(
                f"原件“{label}”列：元素“{element}”不在当前基准中"
                f"（基准元素为 {'、'.join(baseline.elements)}）"
            )
        if element in chosen:
            first = assay.columns[chosen[element]]
            raise InputValidationError(
                f"映射必须一一对应：元素“{element}”被原件“{first}”列和"
                f"“{label}”列同时指定，每个基准元素只能使用一次"
            )
        chosen[element] = column
        if unit not in ASSAY_UNITS:
            raise InputValidationError(
                f"原件“{label}”列：单位必须是 % 或 ppm，实际为“{unit or '未填写'}”"
            )
        entries.append(
            MappingEntry(
                column=column, label=label, element=element, unit=unit
            )
        )

    missing = [name for name in baseline.elements if name not in chosen]
    if missing:
        raise InputValidationError(
            "映射必须一一对应：当前基准元素没有对应的原件列："
            + "、".join(missing)
        )
    return AssayMapping(entries=tuple(entries))


def _is_detection_limit_marker(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if stripped[0] in ("<", "≤", "≦", "＜", "⩽"):
        return True
    normalised = re.sub(r"[\s.]", "", lowered)
    if normalised in {
        "nd", "n.d", "lod", "bdl", "trace", "tr", "未检出", "低于检出限",
        "检出限", "小于检出限",
    }:
        return True
    return "检出限" in stripped or normalised.startswith("lod")


def assay_value_to_percent(
    raw: str, *, line_no: int, column: str, unit: str
) -> Fraction:
    """Convert one original value to an exact percentage.

    ``ppm`` is divided by 10 000; ``%`` is used unchanged.  Detection-limit
    markers get their own message, distinct from generic invalid values.
    """

    text = raw.strip()
    if not text:
        raise InputValidationError(f"第 {line_no} 行“{column}”列：含量不能为空")
    if _is_detection_limit_marker(text):
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：检出限标记（{_shown(raw)}）"
            "不能换算，请提供高于检出限的实测值"
        )
    try:
        value = parse_content(text, line_no=line_no, column=column)
    except InputValidationError:
        raise InputValidationError(
            f"第 {line_no} 行“{column}”列：非法数值（{_shown(raw)}），"
            "含量必须是有限非负数"
        ) from None
    if unit == "ppm":
        return value / PPM_PER_PERCENT
    return value


def convert_assay(
    assay: AssayInput,
    mapping: AssayMapping,
    baseline: BaselineInput,
) -> SamplesInput:
    """Apply ``mapping`` to every row, producing a validated samples input.

    Every problem is collected (up to ``MAX_REPORTED_ERRORS``) with the
    physical line number of the original file.  Any problem rejects the
    whole batch; no partial samples object is ever returned.
    """

    entries = sorted(mapping.entries, key=lambda entry: entry.column)
    if len(entries) != ELEMENT_COUNT or [
        entry.column for entry in entries
    ] != list(range(ELEMENT_COUNT)):
        raise AssayConversionError(
            ["映射与原件列数不符：必须覆盖原件的四个元素列"], 1
        )
    mapped_elements = [entry.element for entry in entries]
    if len(set(mapped_elements)) != len(mapped_elements) or set(
        mapped_elements
    ) != set(baseline.elements):
        missing = [name for name in baseline.elements if name not in mapped_elements]
        duplicated = sorted(
            {name for name in mapped_elements if mapped_elements.count(name) > 1}
        )
        details = []
        if duplicated:
            details.append("重复元素：" + "、".join(duplicated))
        if missing:
            details.append("缺少元素：" + "、".join(missing))
        raise AssayConversionError(
            ["映射必须一一对应；" + "；".join(details)], 1
        )

    errors: list[str] = []
    total = 0

    def add(message: str) -> None:
        nonlocal total
        total += 1
        if len(errors) < MAX_REPORTED_ERRORS:
            errors.append(message)

    # Convert values and detect duplicate sample numbers simultaneously.
    first_seen: dict[str, int] = {}
    converted_rows: list[tuple[Fraction, ...]] = []
    sample_ids: list[str] = []
    for row in assay.rows:
        sample_id = row.sample_id
        if not sample_id:
            add(f"第 {row.line_no} 行：样本编号不能为空")
        elif sample_id in first_seen:
            add(
                f"第 {row.line_no} 行：样本编号“{sample_id}”重复"
                f"（首次出现在第 {first_seen[sample_id]} 行）"
            )
        else:
            first_seen[sample_id] = row.line_no
        sample_ids.append(sample_id)

        by_element: dict[str, Fraction] = {}
        if len(row.values) != ELEMENT_COUNT:
            add(
                f"第 {row.line_no} 行：应有 {ELEMENT_COUNT} 个元素值，"
                f"实际 {len(row.values)} 个"
            )
        else:
            for entry in entries:
                try:
                    by_element[entry.element] = assay_value_to_percent(
                        row.values[entry.column],
                        line_no=row.line_no,
                        column=entry.label,
                        unit=entry.unit,
                    )
                except InputValidationError as exc:
                    add(str(exc))
        converted_rows.append(
            tuple(by_element.get(name, Fraction(0)) for name in baseline.elements)
        )

    if total:
        raise AssayConversionError(errors, total)

    return SamplesInput(
        elements=tuple(baseline.elements),
        sample_ids=tuple(sample_ids),
        rows=tuple(converted_rows),
    )
