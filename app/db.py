"""SQLite persistence.

Four tables carry the whole state:

* ``input_versions`` – append-only versions of the two calculation inputs.
  Exactly one row per ``kind`` is flagged ``is_current``.  A report names
  the version ids it was computed from, so "is this report stale?" is
  always a comparison against the current versions rather than a flag that
  could drift.
* ``assay_versions`` – append-only versions of the instrument's original
  assay sheet (raw bytes and parsed values).  Uploading a new original
  never deletes an old one: historical reports still point at it.
* ``mapping_versions`` – append-only column mappings, each tied to the
  assay version it applies to.  Editing the mapping creates a new version.
* ``reports`` – only fully computed reports are ever inserted.  The insert
  re-checks that the snapshot versions are still current *inside* the same
  ``BEGIN IMMEDIATE`` transaction that writes the report, which is what
  makes publishing safe when an upload interleaves with a calculation.

Conversion is likewise transactional: a successful conversion inserts a
mapping version and a new current ``samples`` version in one transaction,
but only after re-checking that the baseline, original and mapping seen at
the start are still current.  A failed conversion writes nothing.

All writes from this process are serialised by a re-entrant lock; SQLite's
own locking covers other processes/workers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

KINDS = ("baseline", "samples")


class StoreError(RuntimeError):
    """A persistence problem surfaced to the UI."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class VersionedInput:
    version_id: int
    kind: str
    filename: str | None
    created_at: str
    payload: dict[str, Any]
    assay_version: int | None = None
    mapping_version: int | None = None


@dataclass(frozen=True)
class AssayVersion:
    version_id: int
    filename: str | None
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class MappingVersion:
    version_id: int
    assay_version: int
    filename: str | None
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AssayState:
    """The current original sheet together with its latest mapping.

    ``mapping`` is ``None`` while the sheet has never been mapped.
    """

    assay: AssayVersion
    mapping: MappingVersion | None


@dataclass(frozen=True)
class Snapshot:
    baseline: VersionedInput
    samples: VersionedInput


@dataclass(frozen=True)
class ReportRecord:
    report_id: int
    baseline_version: int
    samples_version: int
    created_at: str
    payload: dict[str, Any]
    baseline_is_current: bool
    samples_is_current: bool
    assay_version: int | None = None
    mapping_version: int | None = None

    @property
    def is_stale(self) -> bool:
        return not (self.baseline_is_current and self.samples_is_current)


@dataclass(frozen=True)
class ReportProvenance:
    """The original sheet and mapping a report was derived from."""

    assay: AssayVersion | None
    mapping: MappingVersion | None


@dataclass(frozen=True)
class ConversionResult:
    samples_version: int
    mapping_version: int


@dataclass(frozen=True)
class ConversionSnapshot:
    """One consistent view of everything a conversion depends on."""

    baseline: VersionedInput | None
    assay: AssayState | None


@dataclass(frozen=True)
class ReportSummary:
    report_id: int
    created_at: str
    baseline_version: int
    samples_version: int
    assay_version: int | None
    mapping_version: int | None
    sample_count: int
    non_unique_count: int
    baseline_is_current: bool
    samples_is_current: bool

    @property
    def is_stale(self) -> bool:
        return not (self.baseline_is_current and self.samples_is_current)


class Store:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self._write_lock = threading.RLock()
        self._initialized = False

    # -- connections -----------------------------------------------------

    def _ensure_ready(self) -> None:
        """Create the directory and schema on first use.

        Deferring this keeps merely importing the app (or constructing a
        store) free of filesystem side effects.
        """

        if self._initialized:
            return
        with self._write_lock:
            if self._initialized:
                return
            if self.path.parent and str(self.path.parent) not in ("", "."):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS input_versions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind        TEXT    NOT NULL
                                    CHECK (kind IN ('baseline', 'samples')),
                        filename    TEXT,
                        payload     TEXT    NOT NULL,
                        created_at  TEXT    NOT NULL,
                        is_current  INTEGER NOT NULL DEFAULT 0
                                    CHECK (is_current IN (0, 1))
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS
                        input_versions_one_current
                    ON input_versions (kind)
                    WHERE is_current = 1;

                    CREATE TABLE IF NOT EXISTS reports (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        baseline_version  INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        samples_version   INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        payload           TEXT    NOT NULL,
                        created_at        TEXT    NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS assay_versions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename    TEXT,
                        payload     TEXT    NOT NULL,
                        created_at  TEXT    NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS mapping_versions (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        assay_version   INTEGER NOT NULL
                                        REFERENCES assay_versions (id),
                        filename        TEXT,
                        payload         TEXT    NOT NULL,
                        created_at      TEXT    NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS
                        mapping_versions_by_assay
                    ON mapping_versions (assay_version, id);
                    """
                )
                # Columns added after the first release.  ``ALTER TABLE ...
                # ADD COLUMN`` is idempotent only when guarded, so probe the
                # table info instead of relying on ``IF NOT EXISTS``.
                report_columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(reports)")
                }
                if "assay_version" not in report_columns:
                    conn.execute(
                        "ALTER TABLE reports ADD COLUMN assay_version INTEGER "
                        "REFERENCES assay_versions (id)"
                    )
                if "mapping_version" not in report_columns:
                    conn.execute(
                        "ALTER TABLE reports ADD COLUMN mapping_version INTEGER "
                        "REFERENCES mapping_versions (id)"
                    )
                input_columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(input_versions)")
                }
                if "assay_version" not in input_columns:
                    conn.execute(
                        "ALTER TABLE input_versions "
                        "ADD COLUMN assay_version INTEGER "
                        "REFERENCES assay_versions (id)"
                    )
                if "mapping_version" not in input_columns:
                    conn.execute(
                        "ALTER TABLE input_versions "
                        "ADD COLUMN mapping_version INTEGER "
                        "REFERENCES mapping_versions (id)"
                    )
            except sqlite3.Error as exc:
                raise StoreError(f"初始化数据库失败：{exc}") from exc
            finally:
                conn.close()
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        self._ensure_ready()
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            self._ensure_ready()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # -- inputs ----------------------------------------------------------

    def save_inputs(
        self, updates: dict[str, tuple[dict[str, Any], str | None]]
    ) -> dict[str, int]:
        """Replace one or both calculation inputs in a single transaction.

        Whole-file replacement plus atomicity: either every supplied input
        gets a new current version, or none of them change.  A directly
        uploaded samples input carries no assay provenance; converting an
        original sheet uses :meth:`publish_conversion` instead.
        """

        for kind in updates:
            if kind not in KINDS:
                raise ValueError(f"unknown input kind: {kind!r}")
        encoded = {
            kind: json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            for kind, (payload, _filename) in updates.items()
        }
        now = _utcnow()
        new_ids: dict[str, int] = {}
        try:
            with self._write() as conn:
                for kind in sorted(encoded):
                    _payload, filename = updates[kind]
                    new_ids[kind] = self._replace_current(
                        conn,
                        kind,
                        encoded[kind],
                        filename,
                        now,
                        assay_version=None,
                        mapping_version=None,
                    )
        except sqlite3.Error as exc:
            raise StoreError(f"保存输入失败：{exc}") from exc
        return new_ids

    @staticmethod
    def _replace_current(
        conn: sqlite3.Connection,
        kind: str,
        encoded: str,
        filename: str | None,
        now: str,
        *,
        assay_version: int | None,
        mapping_version: int | None,
    ) -> int:
        conn.execute(
            "UPDATE input_versions SET is_current = 0 "
            "WHERE kind = ? AND is_current = 1",
            (kind,),
        )
        cursor = conn.execute(
            "INSERT INTO input_versions "
            "(kind, filename, payload, created_at, is_current, "
            " assay_version, mapping_version) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (kind, filename, encoded, now, assay_version, mapping_version),
        )
        return int(cursor.lastrowid)

    def save_input(
        self, kind: str, payload: dict[str, Any], filename: str | None
    ) -> int:
        return self.save_inputs({kind: (payload, filename)})[kind]

    def _current_row(self, conn: sqlite3.Connection, kind: str):
        return conn.execute(
            "SELECT id, kind, filename, payload, created_at, "
            "       assay_version, mapping_version "
            "FROM input_versions WHERE kind = ? AND is_current = 1",
            (kind,),
        ).fetchone()

    def _row_to_input(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> VersionedInput:
        return VersionedInput(
            version_id=row["id"],
            kind=row["kind"],
            filename=row["filename"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            assay_version=row["assay_version"],
            mapping_version=row["mapping_version"],
        )

    def current_inputs(self) -> dict[str, VersionedInput | None]:
        with self._read() as conn:
            result: dict[str, VersionedInput | None] = {}
            for kind in KINDS:
                row = self._current_row(conn, kind)
                result[kind] = self._row_to_input(conn, row) if row else None
            return result

    def snapshot(self) -> Snapshot | None:
        """Consistent read of both current inputs, or ``None`` if incomplete."""

        with self._read() as conn:
            rows = {kind: self._current_row(conn, kind) for kind in KINDS}
            if rows["baseline"] is None or rows["samples"] is None:
                return None
            return Snapshot(
                baseline=self._row_to_input(conn, rows["baseline"]),
                samples=self._row_to_input(conn, rows["samples"]),
            )

    # -- originals and mappings ------------------------------------------

    def save_assay(
        self, payload: dict[str, Any], filename: str | None
    ) -> int:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                cursor = conn.execute(
                    "INSERT INTO assay_versions (filename, payload, created_at) "
                    "VALUES (?, ?, ?)",
                    (filename, encoded, _utcnow()),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存原件失败：{exc}") from exc

    def save_mapping(
        self,
        assay_version: int,
        payload: dict[str, Any],
        filename: str | None = None,
    ) -> int | None:
        """Store a mapping version for ``assay_version``.

        Returns ``None`` when a newer original has been uploaded meanwhile:
        the mapping would apply to a superseded sheet, so it is not stored.
        """

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                row = conn.execute(
                    "SELECT CASE WHEN id = (SELECT MAX(id) FROM assay_versions) "
                    "THEN 1 ELSE 0 END AS is_latest "
                    "FROM assay_versions WHERE id = ?",
                    (assay_version,),
                ).fetchone()
                if row is None:
                    raise StoreError(f"原件 v{assay_version} 不存在")
                if not row["is_latest"]:
                    return None
                latest_mapping = self._latest_mapping_row(conn, assay_version)
                if (
                    latest_mapping is not None
                    and latest_mapping["payload"] == encoded
                ):
                    return int(latest_mapping["id"])
                cursor = conn.execute(
                    "INSERT INTO mapping_versions "
                    "(assay_version, filename, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (assay_version, filename, encoded, _utcnow()),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存映射失败：{exc}") from exc

    @staticmethod
    def _row_to_assay(row: sqlite3.Row) -> AssayVersion:
        return AssayVersion(
            version_id=row["id"],
            filename=row["filename"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
        )

    @staticmethod
    def _row_to_mapping(row: sqlite3.Row) -> MappingVersion:
        return MappingVersion(
            version_id=row["id"],
            assay_version=row["assay_version"],
            filename=row["filename"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
        )

    def _latest_mapping_row(
        self, conn: sqlite3.Connection, assay_version: int
    ):
        return conn.execute(
            "SELECT id, assay_version, filename, payload, created_at "
            "FROM mapping_versions WHERE assay_version = ? "
            "ORDER BY id DESC LIMIT 1",
            (assay_version,),
        ).fetchone()

    def latest_assay(self) -> AssayState | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT id, filename, payload, created_at "
                "FROM assay_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            assay = self._row_to_assay(row)
            mapping_row = self._latest_mapping_row(conn, assay.version_id)
            return AssayState(
                assay=assay,
                mapping=(
                    self._row_to_mapping(mapping_row)
                    if mapping_row is not None
                    else None
                ),
            )

    def conversion_snapshot(self) -> ConversionSnapshot:
        """Consistent read of the current baseline and latest original."""

        with self._read() as conn:
            baseline_row = self._current_row(conn, "baseline")
            assay_row = conn.execute(
                "SELECT id, filename, payload, created_at "
                "FROM assay_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assay = None
            if assay_row is not None:
                assay_version = self._row_to_assay(assay_row)
                mapping_row = self._latest_mapping_row(
                    conn, assay_version.version_id
                )
                assay = AssayState(
                    assay=assay_version,
                    mapping=(
                        self._row_to_mapping(mapping_row)
                        if mapping_row is not None
                        else None
                    ),
                )
            return ConversionSnapshot(
                baseline=(
                    self._row_to_input(conn, baseline_row)
                    if baseline_row is not None
                    else None
                ),
                assay=assay,
            )

    def get_input_payload(self, version_id: int, kind: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT payload FROM input_versions WHERE id = ? AND kind = ?",
                (version_id, kind),
            ).fetchone()
            return json.loads(row["payload"]) if row else None

    def get_assay(self, assay_version: int) -> AssayVersion | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT id, filename, payload, created_at "
                "FROM assay_versions WHERE id = ?",
                (assay_version,),
            ).fetchone()
            return self._row_to_assay(row) if row else None

    def get_mapping(self, mapping_version: int) -> MappingVersion | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT id, assay_version, filename, payload, created_at "
                "FROM mapping_versions WHERE id = ?",
                (mapping_version,),
            ).fetchone()
            return self._row_to_mapping(row) if row else None

    def publish_conversion(
        self,
        *,
        baseline_version: int,
        assay_version: int,
        mapping_version: int,
        mapping_payload: dict[str, Any],
        samples_payload: dict[str, Any],
        filename: str | None,
    ) -> ConversionResult | None:
        """Atomically record a new mapping and replace the current samples.

        The conversion is applied to the snapshot read before the call.
        Inside one ``BEGIN IMMEDIATE`` transaction the current baseline must
        still be ``baseline_version``, the latest original must still be
        ``assay_version`` and the latest mapping for it must still be
        ``mapping_version``.  If anything moved, ``None`` is returned and
        nothing is written: no mapping version, no samples version.
        """

        encoded_mapping = json.dumps(
            mapping_payload, ensure_ascii=False, separators=(",", ":")
        )
        encoded_samples = json.dumps(
            samples_payload, ensure_ascii=False, separators=(",", ":")
        )
        now = _utcnow()
        try:
            with self._write() as conn:
                if not self._versions_are_current(
                    conn, baseline_version, -1, check_samples=False
                ):
                    return None
                latest_assay = conn.execute(
                    "SELECT MAX(id) FROM assay_versions"
                ).fetchone()[0]
                if latest_assay != assay_version:
                    return None
                current_mapping = self._latest_mapping_row(conn, assay_version)
                current_mapping_id = (
                    current_mapping["id"] if current_mapping is not None else None
                )
                if current_mapping_id != mapping_version:
                    return None
                if (
                    current_mapping is not None
                    and current_mapping["payload"] == encoded_mapping
                ):
                    # The form merely reused the current mapping: keep the
                    # existing version instead of growing the history.
                    new_mapping_id = current_mapping_id
                else:
                    mapping_cursor = conn.execute(
                        "INSERT INTO mapping_versions "
                        "(assay_version, filename, payload, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (assay_version, filename, encoded_mapping, now),
                    )
                    new_mapping_id = int(mapping_cursor.lastrowid)
                samples_id = self._replace_current(
                    conn,
                    "samples",
                    encoded_samples,
                    filename,
                    now,
                    assay_version=assay_version,
                    mapping_version=new_mapping_id,
                )
                return ConversionResult(
                    samples_version=samples_id,
                    mapping_version=new_mapping_id,
                )
        except sqlite3.Error as exc:
            raise StoreError(f"发布换算结果失败：{exc}") from exc

    # -- reports ---------------------------------------------------------

    @staticmethod
    def _versions_are_current(
        conn: sqlite3.Connection,
        baseline_version: int,
        samples_version: int,
        *,
        check_samples: bool = True,
    ) -> bool:
        baseline_row = conn.execute(
            "SELECT id FROM input_versions "
            "WHERE kind = 'baseline' AND is_current = 1"
        ).fetchone()
        if baseline_row is None or baseline_row["id"] != baseline_version:
            return False
        if not check_samples:
            return True
        samples_row = conn.execute(
            "SELECT id FROM input_versions "
            "WHERE kind = 'samples' AND is_current = 1"
        ).fetchone()
        return samples_row is not None and samples_row["id"] == samples_version

    def publish_report(
        self,
        snapshot: Snapshot,
        payload: dict[str, Any],
    ) -> int | None:
        """Insert a complete report if, and only if, its inputs are current.

        Returns the new report id, or ``None`` when an upload won the race.
        The check and the insert share one ``BEGIN IMMEDIATE`` transaction,
        so an upload cannot slip in between them.  The provenance of the
        current samples version (original sheet and mapping) is copied onto
        the report, so the exact files behind a historical report stay
        retrievable even after they are superseded.
        """

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                if not self._versions_are_current(
                    conn,
                    snapshot.baseline.version_id,
                    snapshot.samples.version_id,
                ):
                    return None
                cursor = conn.execute(
                    "INSERT INTO reports "
                    "(baseline_version, samples_version, assay_version, "
                    " mapping_version, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.baseline.version_id,
                        snapshot.samples.version_id,
                        snapshot.samples.assay_version,
                        snapshot.samples.mapping_version,
                        encoded,
                        _utcnow(),
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存报告失败：{exc}") from exc

    def recent_reports(self, limit: int = 20) -> list[ReportSummary]:
        """Report history, newest first.

        Historical reports stay listed after their inputs are replaced, and
        each row keeps its original/mapping version so the files behind it
        can still be inspected.
        """

        with self._read() as conn:
            rows = conn.execute(
                "SELECT r.id, r.baseline_version, r.samples_version, "
                "       r.assay_version, r.mapping_version, "
                "       r.payload, r.created_at, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples "
                "FROM reports r ORDER BY r.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        summaries: list[ReportSummary] = []
        for row in rows:
            payload = json.loads(row["payload"])
            summaries.append(
                ReportSummary(
                    report_id=row["id"],
                    created_at=row["created_at"],
                    baseline_version=row["baseline_version"],
                    samples_version=row["samples_version"],
                    assay_version=row["assay_version"],
                    mapping_version=row["mapping_version"],
                    sample_count=int(payload.get("sample_count", 0)),
                    non_unique_count=int(payload.get("non_unique_count", 0)),
                    baseline_is_current=row["current_baseline"]
                    == row["baseline_version"],
                    samples_is_current=row["current_samples"]
                    == row["samples_version"],
                )
            )
        return summaries

    def latest_report(self) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT r.id, r.baseline_version, r.samples_version, "
                "       r.assay_version, r.mapping_version, "
                "       r.payload, r.created_at, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples "
                "FROM reports r ORDER BY r.id DESC LIMIT 1"
            ).fetchone()
            return self._row_to_report(row) if row else None

    def get_report(self, report_id: int) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                "SELECT r.id, r.baseline_version, r.samples_version, "
                "       r.assay_version, r.mapping_version, "
                "       r.payload, r.created_at, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'baseline' AND is_current = 1) "
                "         AS current_baseline, "
                "       (SELECT id FROM input_versions "
                "         WHERE kind = 'samples' AND is_current = 1) "
                "         AS current_samples "
                "FROM reports r WHERE r.id = ?",
                (report_id,),
            ).fetchone()
            return self._row_to_report(row) if row else None

    def report_provenance(self, report_id: int) -> ReportProvenance | None:
        """The original sheet and mapping a report was computed from."""        """The original sheet and mapping a report was computed from."""

        with self._read() as conn:
            report = conn.execute(
                "SELECT assay_version, mapping_version FROM reports WHERE id = ?",
                (report_id,),
            ).fetchone()
            if report is None:
                return None
            assay_row = (
                conn.execute(
                    "SELECT id, filename, payload, created_at "
                    "FROM assay_versions WHERE id = ?",
                    (report["assay_version"],),
                ).fetchone()
                if report["assay_version"] is not None
                else None
            )
            mapping_row = (
                conn.execute(
                    "SELECT id, assay_version, filename, payload, created_at "
                    "FROM mapping_versions WHERE id = ?",
                    (report["mapping_version"],),
                ).fetchone()
                if report["mapping_version"] is not None
                else None
            )
            return ReportProvenance(
                assay=self._row_to_assay(assay_row) if assay_row else None,
                mapping=(
                    self._row_to_mapping(mapping_row) if mapping_row else None
                ),
            )

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> ReportRecord:
        return ReportRecord(
            report_id=row["id"],
            baseline_version=row["baseline_version"],
            samples_version=row["samples_version"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            baseline_is_current=row["current_baseline"]
            == row["baseline_version"],
            samples_is_current=row["current_samples"] == row["samples_version"],
            assay_version=row["assay_version"],
            mapping_version=row["mapping_version"],
        )
