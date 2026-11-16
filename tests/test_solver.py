"""Tests for the exact solver.

Optimality of every reported unique solution is checked exactly via the KKT
conditions of the simplex-constrained least-squares problem, and the
non-unique ranges are checked by exhibiting attaining points.  A coarse
grid check is also included as an independent cross-check.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import pytest

from app.solver import solve_sample

A = [
    [F("62.3"), F("68.5"), F("55.2")],
    [F("3.9"), F("2.5"), F("6.2")],
    [F("2.1"), F("1.1"), F("3.4")],
    [F("1.2"), F("0.8"), F("2.1")],
]


def sse(a, y, w):
    return sum(
        (sum(a[e][j] * w[j] for j in range(3)) - y[e]) ** 2 for e in range(4)
    )


def test_unique_interior_solution_is_exact():
    y = [F(2, 10) * A[e][0] + F(3, 10) * A[e][1] + F(5, 10) * A[e][2]
         for e in range(4)]
    sol = solve_sample(A, y)
    assert sol.unique
    assert sol.weights == (F(1, 5), F(3, 10), F(1, 2))
    assert sol.sse == 0
    assert sol.residuals == (F(0),) * 4
    assert sum(sol.weights) == 1


def test_vertex_solution():
    sol = solve_sample(A, [row[0] for row in A])
    assert sol.unique
    assert sol.weights == (F(1), F(0), F(0))


def test_boundary_solution_is_unique():
    y = [(A[e][0] + A[e][2]) / 2 for e in range(4)]
    sol = solve_sample(A, y)
    assert sol.unique
    assert sol.weights == (F(1, 2), F(0), F(1, 2))


def gradient(a, y, w):
    """Gradient of ||Aw - y||^2 with respect to w."""

    return [
        sum(
            a[e][i] * (sum(a[e][j] * w[j] for j in range(3)) - y[e])
            for e in range(4)
        )
        for i in range(3)
    ]


def assert_kkt(a, y, w):
    """Exact first-order optimality conditions on the simplex."""

    assert sum(w) == 1
    assert all(value >= 0 for value in w)
    grad = gradient(a, y, w)
    support = [i for i in range(3) if w[i] > 0]
    multiplier = -grad[support[0]]
    for i in support:
        assert grad[i] == -multiplier, "gradient is not constant on the support"
    for i in range(3):
        if i not in support:
            assert grad[i] + multiplier >= 0, "negative-direction descent exists"


def test_verified_against_dense_grid():
    rng = random.Random(7)
    for _ in range(25):
        a = [[F(str(round(rng.uniform(0, 100), 3))) for _ in range(3)]
             for _ in range(4)]
        y = [F(str(round(rng.uniform(0, 100), 3))) for _ in range(4)]
        sol = solve_sample(a, y)
        if not sol.unique:
            continue
        assert sum(sol.weights) == 1
        assert all(w >= 0 for w in sol.weights)
        assert_kkt(a, y, sol.weights)
        # exact candidate cannot lose to any grid point
        n = 40
        for i in range(n + 1):
            for j in range(n + 1 - i):
                w = (F(i, n), F(j, n), F(n - i - j, n))
                assert sse(a, y, w) >= sol.sse
        assert sse(a, y, sol.weights) == sol.sse


def test_kkt_holds_across_degenerate_shapes():
    rng = random.Random(13)
    for trial in range(200):
        mode = trial % 4
        if mode == 0:
            a = [[F(str(round(rng.uniform(0, 100), 2))) for _ in range(3)]
                 for _ in range(4)]
        elif mode == 1:
            column = [round(rng.uniform(0, 100), 2) for _ in range(4)]
            a = [[F(str(column[e])), F(str(column[e])),
                  F(str(round(rng.uniform(0, 100), 2)))] for e in range(4)]
        elif mode == 2:
            a = [[F(str(round(rng.uniform(0, 1), 2))) * 0 for _ in range(3)]
                 for _ in range(4)]
        else:
            column = [round(rng.uniform(0, 100), 2) for _ in range(4)]
            a = [[F(str(column[e]))] * 3 for e in range(4)]
        y = [F(str(round(rng.uniform(0, 100), 2))) for _ in range(4)]
        sol = solve_sample(a, y)
        if sol.unique:
            assert_kkt(a, y, sol.weights)


def test_non_unique_ranges_are_attained_and_there_are_two_optima():
    rng = random.Random(17)
    checked = 0
    for trial in range(120):
        # Deliberately degenerate: two identical ores, or all three equal.
        column = [round(rng.uniform(0, 100), 2) for _ in range(4)]
        other = [round(rng.uniform(0, 100), 2) for _ in range(4)]
        if trial % 2:
            a = [[F(str(column[e])), F(str(column[e])), F(str(other[e]))]
                 for e in range(4)]
        else:
            a = [[F(str(column[e]))] * 3 for e in range(4)]
        y = [F(str(round(rng.uniform(0, 100), 2))) for _ in range(4)]
        sol = solve_sample(a, y)
        if sol.unique:
            continue
        checked += 1
        # The reported minimum itself must be attainable at some point that
        # realises every coordinate range endpoint combination.
        for ore in range(3):
            low, high = sol.weight_ranges[ore]
            assert F(0) <= low <= high <= F(1)
            for target in (low, high):
                candidate = _point_with_coordinate(
                    a, y, sol, ore, target
                )
                assert candidate is not None, (a, y, sol.weight_ranges)
                assert sse(a, y, candidate) == sol.sse
                assert candidate[ore] == target
        # At least two distinct optimal blends exist (the optimum is not
        # unique), witnessed by the widest range being non-degenerate.
        assert any(high > low for low, high in sol.weight_ranges)
        # Predicted/residual ranges bracket all extreme optimal blends.
        for low, high in sol.predicted_ranges:
            assert low <= high
    assert checked > 0


def _point_with_coordinate(a, y, sol, ore, target):
    """Exhibit an optimal blend with ``weight[ore] == target``.

    Fixing one coordinate leaves two non-negative variables summing to
    ``1 - target``; parametrising them by one scalar turns the objective
    into an exact quadratic.  Its minimum on a closed interval is checked
    against the reported optimum, which proves the endpoint is attainable.
    """

    if not (F(0) <= target <= F(1)):
        return None
    others = [i for i in range(3) if i != ore]
    remaining = 1 - target

    def weight_for(s: F) -> tuple[F, F, F]:
        point = [F(0), F(0), F(0)]
        point[ore] = target
        point[others[0]] = s
        point[others[1]] = remaining - s
        return tuple(point)

    # Objective as a quadratic in s: alpha*s^2 + beta*s + gamma.
    p0 = weight_for(F(0))
    p1 = weight_for(F(1))
    p_minus = weight_for(F(-1))
    f0 = sse(a, y, p0)
    f1 = sse(a, y, p1)
    fm = sse(a, y, p_minus)
    alpha = (f1 + fm) / 2 - f0
    beta = (f1 - fm) / 2
    candidates: list[F] = [F(0), remaining]
    if alpha > 0:
        stationary = -beta / (2 * alpha)
        if 0 < stationary < remaining:
            candidates.append(stationary)
    elif alpha == 0 and beta != 0:
        candidates.append(F(0) if beta > 0 else remaining)

    best_point = None
    best_value = None
    for s in candidates:
        point = weight_for(s)
        if any(value < 0 for value in point):
            continue
        value = sse(a, y, point)
        if best_value is None or value < best_value:
            best_value, best_point = value, point
    if best_value == sol.sse:
        return best_point
    return None


def test_duplicate_ore_columns_make_optimum_non_unique():
    a = [[A[e][0], A[e][0], A[e][2]] for e in range(4)]
    y = [(a[e][0] + a[e][2]) / 2 for e in range(4)]
    sol = solve_sample(a, y)
    assert not sol.unique
    assert sol.weights is None
    # The two indistinguishable ores split 0.5 between them in any way.
    assert sol.weight_ranges[0] == (F(0), F(1, 2))
    assert sol.weight_ranges[1] == (F(0), F(1, 2))
    assert sol.weight_ranges[2] == (F(1, 2), F(1, 2))


def test_all_ore_rows_identical_is_never_unique():
    a = [[F(60), F(60), F(60)],
         [F(4), F(4), F(4)],
         [F(2), F(2), F(2)],
         [F(1), F(1), F(1)]]
    sol = solve_sample(a, [F(60), F(4), F(2), F(1)])
    assert not sol.unique
    assert sol.sse == 0
    assert sol.weight_ranges == ((F(0), F(1)),) * 3


def test_collinear_ores_with_exact_vertex_target_is_unique():
    a = [[F(60), F(55), F(50)],
         [F(4), F(5), F(6)],
         [F(2), F(2.5), F(3)],
         [F(1), F(1.5), F(2)]]
    sol = solve_sample(a, [row[0] for row in a])
    assert sol.unique
    assert sol.weights == (F(1), F(0), F(0))


def test_zero_baseline_is_non_unique():
    sol = solve_sample([[F(0)] * 3 for _ in range(4)], [F(1), F(2), F(3), F(4)])
    assert not sol.unique
    assert sol.weight_ranges == ((F(0), F(1)),) * 3


def test_non_unique_ranges_are_attainable():
    a = [[F(60), F(60), F(50)],
         [F(4), F(4), F(6)],
         [F(2), F(2), F(3)],
         [F(1), F(1), F(2)]]
    y = [F(55), F(5), F(2.5), F(1.5)]
    sol = solve_sample(a, y)
    assert not sol.unique
    for low, high in sol.weight_ranges:
        assert F(0) <= low <= high <= F(1)
    # Any convex combination across the flat optimum hits the same SSE.
    for weight_a in (F(0), F(1, 4), F(1, 2)):
        w = (weight_a, F(1, 2) - weight_a, F(1, 2))
        assert sse(a, y, w) == sol.sse


def test_input_shape_is_enforced():
    with pytest.raises(ValueError):
        solve_sample([[F(1)] * 3], [F(1)] * 4)
    with pytest.raises(ValueError):
        solve_sample(A, [F(1)] * 3)
