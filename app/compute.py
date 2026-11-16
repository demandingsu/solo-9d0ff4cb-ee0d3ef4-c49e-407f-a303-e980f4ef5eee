"""Report construction.

A report is a *pure function* of one baseline snapshot and one sample
snapshot.  Nothing is read from live input tables while computing, so an
upload landing midway cannot produce a mixed report; the version ids are
carried in the payload and re-checked atomically at publish time.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .inputs import BaselineInput, SamplesInput, fraction_to_text
from .solver import SampleSolution, solve_sample


class ComputeError(ValueError):
    """The two inputs are individually valid but incompatible."""


def _ordered_samples(
    baseline: BaselineInput, samples: SamplesInput
) -> list[tuple[Any, ...]]:
    """Match sample element columns to baseline elements by name."""

    missing = [name for name in baseline.elements if name not in samples.elements]
    extra = [name for name in samples.elements if name not in baseline.elements]
    if missing or extra:
        details = []
        if missing:
            details.append("成品缺少元素列：" + "、".join(missing))
        if extra:
            details.append("成品多出元素列：" + "、".join(extra))
        raise ComputeError(
            "两份输入的元素列必须一一对应；" + "；".join(details)
            + "。基准元素为：" + "、".join(baseline.elements)
        )
    index = [samples.elements.index(name) for name in baseline.elements]
    return [tuple(row[i] for i in index) for row in samples.rows]


def _solution_payload(solution: SampleSolution) -> dict[str, Any]:
    return {
        "sse": fraction_to_text(solution.sse),
        "unique": solution.unique,
        "weights": (
            [fraction_to_text(value) for value in solution.weights]
            if solution.weights is not None
            else None
        ),
        "weight_ranges": [
            [fraction_to_text(low), fraction_to_text(high)]
            for low, high in solution.weight_ranges
        ],
        "predicted": (
            [fraction_to_text(value) for value in solution.predicted]
            if solution.predicted is not None
            else None
        ),
        "residuals": (
            [fraction_to_text(value) for value in solution.residuals]
            if solution.residuals is not None
            else None
        ),
        "predicted_ranges": [
            [fraction_to_text(low), fraction_to_text(high)]
            for low, high in solution.predicted_ranges
        ],
        "residual_ranges": [
            [fraction_to_text(low), fraction_to_text(high)]
            for low, high in solution.residual_ranges
        ],
    }


def compute_report(
    baseline: BaselineInput, samples: SamplesInput
) -> dict[str, Any]:
    """Compute every sample.  Either the whole report is produced or an error
    is raised; there is no partial report."""

    ordered_rows = _ordered_samples(baseline, samples)

    a = [[baseline.rows[ore][e] for ore in range(3)] for e in range(4)]
    results: list[dict[str, Any]] = []
    non_unique = 0
    for sample_id, row in zip(samples.sample_ids, ordered_rows):
        solution = solve_sample(a, row)
        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "measured": [fraction_to_text(value) for value in row],
        }
        entry.update(_solution_payload(solution))
        results.append(entry)
        if not solution.unique:
            non_unique += 1

    warnings: list[str] = []
    if non_unique:
        warnings.append(
            f"{non_unique} 个样本的最优占比不唯一；报告只给出可达范围，"
            "不会任取其中一组。"
        )
    rank = _matrix_rank(a)
    if rank < 3:
        warnings.append(
            "基准元素矩阵的秩小于 3，无法区分的配料组合会表现为最优解不唯一。"
        )

    return {
        "elements": list(baseline.elements),
        "ores": list(baseline.ores),
        "baseline_rows": [
            [fraction_to_text(value) for value in row] for row in baseline.rows
        ],
        "sample_count": len(results),
        "non_unique_count": non_unique,
        "matrix_rank": rank,
        "warnings": warnings,
        "results": results,
    }


def _matrix_rank(a: list[list[Fraction]]) -> int:
    rows = [list(row) for row in a]
    rank = 0
    ncols = 3
    for col in range(ncols):
        pivot = next(
            (r for r in range(rank, len(rows)) if rows[r][col] != 0), None
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [value / scale for value in rows[rank]]
        for r in range(len(rows)):
            if r != rank and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [
                    rows[r][c] - factor * rows[rank][c] for c in range(ncols)
                ]
        rank += 1
    return rank
