"""Exact simplex least-squares solver for ore-blend proportions.

The three raw ores form the columns of a 4x3 matrix ``A`` and a finished
sample is a 4-vector ``y``.  We minimise ``||A w - y||^2`` over the
probability simplex ``w >= 0, sum(w) = 1``.

All arithmetic is exact (:class:`fractions.Fraction`).  This is important
because the task requires an exact "is the optimum unique?" answer: when
several blends attain the same minimal residual we must say so instead of
silently returning one of them.

The solver enumerates the seven non-empty faces of the 3-simplex
(3 vertices, 3 edges, 1 triangle).  On each face the objective is a
convex quadratic; its stationary set is an affine subspace.  A face only
contributes a candidate when its stationary set meets the *relative
interior* of that face.  Boundary minimisers are therefore found exactly
once, through their own (smaller) face.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FaceCandidate:
    """A stationary family on one face of the simplex.

    ``dim`` is the dimension of the stationary family inside the face.
    For ``dim == 0`` exactly ``point`` is set.  For ``dim == 1`` the
    family is ``lam0 + t * direction`` for ``t`` strictly between
    ``t_low`` and ``t_high``.  For ``dim == 2`` the whole face is
    stationary (only possible for the full triangle).
    """

    support: tuple[int, ...]
    sse: Fraction
    dim: int
    point: tuple[Fraction, ...] | None = None
    lam0: tuple[Fraction, ...] | None = None
    direction: tuple[Fraction, ...] | None = None
    t_low: Fraction | None = None
    t_high: Fraction | None = None


@dataclass(frozen=True)
class SampleSolution:
    """Result of one sample.

    ``weights`` is only populated when the optimal blend is unique.  For a
    non-unique optimum it is ``None`` and the ``*_ranges`` fields describe
    the exact projection of the optimal set onto each ore / predicted
    content / residual.  Those ranges are derived from the extreme points
    of the optimal set, not from an arbitrary chosen optimum.
    """

    sse: Fraction
    unique: bool
    weights: tuple[Fraction, ...] | None
    weight_ranges: tuple[tuple[Fraction, Fraction], ...]
    predicted: tuple[Fraction, ...] | None
    residuals: tuple[Fraction, ...] | None
    predicted_ranges: tuple[tuple[Fraction, Fraction], ...]
    residual_ranges: tuple[tuple[Fraction, Fraction], ...]


# ---------------------------------------------------------------------------
# Small exact linear algebra
# ---------------------------------------------------------------------------


def _rref(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> tuple[list[list[Fraction]], list[int], bool] | None:
    """Reduced row echelon form of ``[matrix | rhs]``.

    Returns ``(rows, pivot_columns, consistent)``.  ``rows`` includes the
    augmented column as its last entry.  A pivot in the augmented column
    means the system is inconsistent.
    """

    if not matrix:
        return [], [], True
    ncols = len(matrix[0])
    rows = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    nrows = len(rows)
    pivot_columns: list[int] = []
    r = 0
    for c in range(ncols):
        pivot = next((i for i in range(r, nrows) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = Fraction(1, 1) / rows[r][c]
        rows[r] = [x * inv for x in rows[r]]
        for i in range(nrows):
            if i == r:
                continue
            factor = rows[i][c]
            if factor:
                rows[i] = [rows[i][j] - factor * rows[r][j] for j in range(ncols + 1)]
        pivot_columns.append(c)
        r += 1
        if r == nrows:
            break

    # Any all-zero coefficient row with a non-zero RHS is inconsistent.
    for i in range(r, nrows):
        if all(rows[i][c] == 0 for c in range(ncols)) and rows[i][ncols] != 0:
            return None
    return rows[:r], pivot_columns, True


def _affine_solution(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> tuple[list[Fraction], list[list[Fraction]]] | None:
    """Solve ``matrix x = rhs``.

    Returns a particular solution and a basis of the nullspace, or ``None``
    when the system is inconsistent.
    """

    reduced = _rref(matrix, rhs)
    if reduced is None:
        return None
    rows, pivots, _ = reduced
    ncols = len(matrix[0]) if matrix else 0
    free = [c for c in range(ncols) if c not in pivots]

    particular = [Fraction(0) for _ in range(ncols)]
    for row, pivot in enumerate(pivots):
        particular[pivot] = rows[row][ncols]

    null_basis: list[list[Fraction]] = []
    for free_col in free:
        vec = [Fraction(0) for _ in range(ncols)]
        vec[free_col] = Fraction(1)
        for row, pivot in enumerate(pivots):
            vec[pivot] = -rows[row][free_col]
        null_basis.append(vec)
    return particular, null_basis


def _independent_vectors(vectors: Iterable[Sequence[Fraction]]) -> list[list[Fraction]]:
    """Row-echelon basis for the span of a few short rational vectors."""

    basis: list[list[Fraction]] = []
    for original in vectors:
        vec = list(original)
        for b in basis:
            pivot = next(i for i, x in enumerate(b) if x != 0)
            factor = vec[pivot]
            if factor:
                vec = [vec[i] - factor * b[i] for i in range(len(vec))]
        pivot = next((i for i, x in enumerate(vec) if x != 0), None)
        if pivot is not None:
            scale = vec[pivot]
            vec = [x / scale for x in vec]
            basis.append(vec)
    return basis


# ---------------------------------------------------------------------------
# Face stationary sets
# ---------------------------------------------------------------------------


def _strict_interval(
    lam0: Sequence[Fraction], direction: Sequence[Fraction]
) -> tuple[Fraction, Fraction, Fraction] | None:
    """Interval of ``t`` with ``lam0 + t*direction > 0`` component-wise.

    ``direction`` comes from the nullspace of the KKT system, so it sums to
    zero and is non-zero; therefore both bounds are finite and the
    strictly-positive set is the open interval between them.  Returns
    ``(low, high, witness_t)`` or ``None`` when nothing is strictly
    positive.
    """

    if sum(direction) != 0:  # pragma: no cover - invariant of the KKT system
        raise AssertionError("stationary direction must preserve sum(w)=1")

    low: Fraction | None = None
    high: Fraction | None = None
    for a, v in zip(lam0, direction):
        if v == 0:
            if a <= 0:
                return None
        elif v > 0:
            bound = -a / v
            if low is None or bound > low:
                low = bound
        else:
            bound = -a / v
            if high is None or bound < high:
                high = bound

    # Non-zero direction summing to zero has both signs, so both bounds
    # are set.  This is an internal invariant rather than user data.
    if low is None or high is None:  # pragma: no cover
        raise AssertionError("direction must have both positive and negative entries")
    if low >= high:
        return None

    witness = (low + high) / 2
    if any(a + witness * v <= 0 for a, v in zip(lam0, direction)):
        # pragma: no cover - the midpoint of an open interval is interior
        return None
    return low, high, witness


def _face_candidate(
    q: Sequence[Sequence[Fraction]],
    b: Sequence[Fraction],
    a: Sequence[Sequence[Fraction]],
    y: Sequence[Fraction],
    support: tuple[int, ...],
) -> FaceCandidate | None:
    """Stationary set of the objective on ``support``'s relative interior."""

    m = len(support)
    matrix: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for i in support:
        row = [q[i][j] for j in support]
        row.append(Fraction(-1))  # Lagrange multiplier for sum(w)=1
        matrix.append(row)
        rhs.append(b[i])
    matrix.append([Fraction(1)] * m + [Fraction(0)])
    rhs.append(Fraction(1))

    solution = _affine_solution(matrix, rhs)
    if solution is None:
        return None
    particular, null_basis = solution

    lam0 = particular[:m]
    # The projected stationary set is lam0 + span(projected nullspace).
    # Project out the Lagrange multiplier before deciding its dimension.
    lambda_dirs = _independent_vectors(vec[:m] for vec in null_basis)
    dim = len(lambda_dirs)

    if dim > 2:  # pragma: no cover - impossible for a 2-simplex
        raise AssertionError("unexpected stationary-set dimension")

    full = [Fraction(0), Fraction(0), Fraction(0)]
    for idx, value in zip(support, lam0):
        full[idx] = value

    def sse_of(weights: Sequence[Fraction]) -> Fraction:
        total = Fraction(0)
        for element_index, target in enumerate(y):
            predicted = sum(
                a[element_index][ore_index] * weights[ore_index]
                for ore_index in range(len(weights))
            )
            total += (predicted - target) ** 2
        return total

    if dim == 0:
        if any(x <= 0 for x in lam0):
            # The stationary point is on a boundary; that point is handled
            # by one of the smaller faces.
            return None
        return FaceCandidate(
            support=support, sse=sse_of(full), dim=0, point=tuple(full)
        )

    if dim == 1:
        direction = lambda_dirs[0]
        interval = _strict_interval(lam0, direction)
        if interval is None:
            return None
        low, high, t = interval
        witness = [lam0[i] + t * direction[i] for i in range(m)]
        if any(x <= 0 for x in witness):  # pragma: no cover - guarded above
            return None
        full_witness = [Fraction(0), Fraction(0), Fraction(0)]
        for idx, value in zip(support, witness):
            full_witness[idx] = value
        return FaceCandidate(
            support=support,
            sse=sse_of(full_witness),
            dim=1,
            lam0=tuple(lam0),
            direction=tuple(direction),
            t_low=low,
            t_high=high,
        )

    # dim == 2: the stationary set is the whole plane sum(w)=1, hence it
    # meets the relative interior of the full triangle.  The uniform blend
    # is a convenient exact witness.
    if len(support) != 3:  # pragma: no cover - impossible for m <= 2
        raise AssertionError("a 2-dimensional stationary set needs the full face")
    uniform = [Fraction(1, m)] * m
    for value in uniform:
        if value <= 0:  # pragma: no cover - m <= 3
            return None
    full_uniform = [Fraction(0), Fraction(0), Fraction(0)]
    for idx, value in zip(support, uniform):
        full_uniform[idx] = value
    return FaceCandidate(
        support=support, sse=sse_of(full_uniform), dim=2
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_sample(
    a: Sequence[Sequence[Fraction]], y: Sequence[Fraction]
) -> SampleSolution:
    """Solve one sample exactly.

    ``a`` has shape 4 x 3 (rows are elements, columns are ores) and ``y``
    has length 4.
    """

    if len(a) != 4 or any(len(row) != 3 for row in a):
        raise ValueError("baseline matrix must be 4x3")
    if len(y) != 4:
        raise ValueError("sample vector must have length 4")

    q = [
        [sum(a[e][i] * a[e][j] for e in range(4)) for j in range(3)]
        for i in range(3)
    ]
    b = [sum(a[e][i] * y[e] for e in range(4)) for i in range(3)]

    candidates: list[FaceCandidate] = []
    for m in (1, 2, 3):
        for support in combinations(range(3), m):
            candidate = _face_candidate(q, b, a, y, support)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:  # pragma: no cover - vertices always qualify
        raise AssertionError("no stationary candidate found")

    best = min(candidate.sse for candidate in candidates)
    optimal = [candidate for candidate in candidates if candidate.sse == best]

    # --- unique point ---------------------------------------------------
    if len(optimal) == 1 and optimal[0].dim == 0:
        point = optimal[0].point
        assert point is not None
        predicted = _predict(a, point)
        residuals = tuple(y[e] - predicted[e] for e in range(4))
        return SampleSolution(
            sse=best,
            unique=True,
            weights=point,
            weight_ranges=tuple((value, value) for value in point),
            predicted=predicted,
            residuals=residuals,
            predicted_ranges=tuple((value, value) for value in predicted),
            residual_ranges=tuple((value, value) for value in residuals),
        )

    # --- non-unique optimum --------------------------------------------
    # Collect the extreme points of the optimal set.  A face candidate is
    # a point, an open segment, or the whole triangle; the extreme points
    # are the endpoints.  Bounding each linear functional over these
    # extremes gives the exact interval of attainable values.
    extremes: list[tuple[Fraction, ...]] = []
    for candidate in optimal:
        if candidate.dim == 0:
            assert candidate.point is not None
            extremes.append(candidate.point)
        elif candidate.dim == 2:
            for vertex in range(3):
                point = [Fraction(0), Fraction(0), Fraction(0)]
                point[vertex] = Fraction(1)
                extremes.append(tuple(point))
        else:
            assert (
                candidate.lam0 is not None
                and candidate.direction is not None
                and candidate.t_low is not None
                and candidate.t_high is not None
            )
            for t in (candidate.t_low, candidate.t_high):
                point = [Fraction(0), Fraction(0), Fraction(0)]
                for idx, lagrange_index in enumerate(candidate.support):
                    point[lagrange_index] = (
                        candidate.lam0[idx] + t * candidate.direction[idx]
                    )
                extremes.append(tuple(point))

    def bounds(function) -> tuple[Fraction, Fraction]:
        values = [function(point) for point in extremes]
        return min(values), max(values)

    weight_ranges = tuple(bounds(lambda point, i=i: point[i]) for i in range(3))

    predicted_extremes = [_predict(a, point) for point in extremes]

    def predicted_bound(i: int) -> tuple[Fraction, Fraction]:
        values = [item[i] for item in predicted_extremes]
        return min(values), max(values)

    predicted_ranges = tuple(predicted_bound(i) for i in range(4))
    residual_ranges = tuple(
        (y[i] - predicted_ranges[i][1], y[i] - predicted_ranges[i][0])
        for i in range(4)
    )

    return SampleSolution(
        sse=best,
        unique=False,
        weights=None,
        weight_ranges=weight_ranges,
        predicted=None,
        residuals=None,
        predicted_ranges=predicted_ranges,
        residual_ranges=residual_ranges,
    )


def _predict(
    a: Sequence[Sequence[Fraction]], weights: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    return tuple(
        sum(a[e][j] * weights[j] for j in range(3)) for e in range(4)
    )
