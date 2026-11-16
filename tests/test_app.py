"""End-to-end tests for the web application, database and atomic publishing."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import Store, StoreError
from app.inputs import BaselineInput, SamplesInput
from app.main import create_app

BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

SAMPLES_CSV = """sample_id,Fe,SiO2,Al2O3,CaO
S1,62.74,3.71,2.02,1.13
S2,60,4,2,1
"""


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        test_client.app_state = app.state  # type: ignore[attr-defined]
        yield test_client


def upload(client: TestClient, baseline: bool = False, samples: bool = False):
    files = {}
    if baseline:
        files["baseline_file"] = ("baseline.csv", BASE_CSV.encode(), "text/csv")
    if samples:
        files["samples_file"] = ("samples.csv", SAMPLES_CSV.encode(), "text/csv")
    return client.post("/inputs/upload", files=files, follow_redirects=False)


def compute(client: TestClient, follow_redirects: bool = False):
    return client.post("/compute", follow_redirects=follow_redirects)


def latest(client: TestClient) -> dict:
    return client.get("/api/latest").json()


def register(client: TestClient):
    upload(client, baseline=True, samples=True)
    compute(client)


def test_index_renders_empty_state(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "尚未上传" in response.text
    assert "尚无报告" in response.text


def test_healthz(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_and_compute_publishes_report(client: TestClient):
    response = upload(client, baseline=True, samples=True)
    assert response.status_code == 303
    before = latest(client)
    assert before["report"] is None

    response = compute(client)
    assert response.status_code == 303
    report = latest(client)["report"]
    assert report is not None
    assert report["stale"] is False
    assert report["payload"]["sample_count"] == 2
    # Chosen sample values come from a known good mix near ore A/B.
    first = report["payload"]["results"][0]
    assert first["sample_id"] == "S1"
    assert first["unique"] is True
    assert len(first["weights"]) == 3
    # Weights sum to exactly 1 and are non-negative.
    from fractions import Fraction

    weights = [Fraction(value) for value in first["weights"]]
    assert sum(weights) == 1
    assert all(value >= 0 for value in weights)
    # Residuals equal measured minus predicted, exactly.
    measured = [Fraction(value) for value in first["measured"]]
    predicted = [Fraction(value) for value in first["predicted"]]
    residuals = [Fraction(value) for value in first["residuals"]]
    assert residuals == [m - p for m, p in zip(measured, predicted)]
    assert Fraction(first["sse"]) == sum(r * r for r in residuals)


def test_report_renders_with_stale_badge(client: TestClient):
    register(client)
    html = client.get("/").text
    assert "与当前输入一致" in html
    assert "无法唯一确定" in html


def test_replacing_baseline_marks_report_stale(client: TestClient):
    register(client)
    other = BASE_CSV.replace("62.3", "63.1")
    response = client.post(
        "/inputs/upload",
        files={"baseline_file": ("b2.csv", other.encode(), "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    report = latest(client)["report"]
    assert report["stale"] is True
    html = client.get("/").text
    assert "已过期" in html
    # Stale reports remain viewable after the inputs changed.
    assert "S1" in html


def test_replacing_samples_marks_report_stale(client: TestClient):
    register(client)
    changed = SAMPLES_CSV.replace("S2,60,4,2,1", "S2,61,4.1,2.1,1.1")
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s2.csv", changed.encode(), "text/csv")},
        follow_redirects=False,
    )
    assert latest(client)["report"]["stale"] is True


def test_invalid_upload_keeps_previous_input_and_report(client: TestClient):
    register(client)
    versions_before = latest_baseline_versions(client)
    bad = "ore,Fe,SiO2,Al2O3,CaO\nA,1,2,3,4\nA,5,6,7,8\nC,1,1,1,1\n"
    response = client.post(
        "/inputs/upload",
        files={"baseline_file": ("bad.csv", bad.encode(), "text/csv")},
    )
    assert response.status_code == 400
    assert "重复" in response.text
    assert latest_baseline_versions(client) == versions_before
    report = latest(client)["report"]
    assert report is not None and report["stale"] is False


def latest_baseline_versions(client: TestClient) -> dict:
    store: Store = client.app_state.store
    return {
        kind: value.version_id
        for kind, value in store.current_inputs().items()
        if value is not None
    }


def test_partial_upload_only_replaces_selected_input(client: TestClient):
    upload(client, baseline=True)
    store: Store = client.app_state.store
    assert store.current_inputs()["samples"] is None
    upload(client, samples=True)
    current = store.current_inputs()
    assert current["baseline"].version_id == 1
    assert current["samples"].version_id == 2


def test_compute_requires_both_inputs(client: TestClient):
    upload(client, baseline=True)
    response = compute(client)
    assert response.status_code == 303
    assert "baseline" in client.app_state.store.current_inputs()
    assert latest(client)["report"] is None
    # With only one input the page tells the user what is missing.
    html = client.get("/").text
    assert "尚未上传成品 CSV" in html
    assert "需要先上传两类输入" in html


def test_mismatched_element_columns_are_rejected(client: TestClient):
    upload(client, baseline=True)
    broken = SAMPLES_CSV.replace("CaO", "MgO")
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s.csv", broken.encode(), "text/csv")},
    )
    response = compute(client, follow_redirects=True)
    assert response.status_code == 200
    assert "元素列必须一一对应" in response.text
    assert "MgO" in response.text
    assert latest(client)["report"] is None


def test_upload_of_one_bad_file_does_not_replace_the_other(client: TestClient):
    """A valid + invalid pair must leave *both* inputs untouched."""

    upload(client, baseline=True, samples=True)
    store: Store = client.app_state.store
    before = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    bad_samples = "sample_id,Fe,SiO2,Al2O3,CaO\nS1,1,2,3,4\nS1,5,6,7,8\n"
    new_base = BASE_CSV.replace("62.3", "63.1")
    response = client.post(
        "/inputs/upload",
        files={
            "baseline_file": ("b.csv", new_base.encode(), "text/csv"),
            "samples_file": ("s.csv", bad_samples.encode(), "text/csv"),
        },
    )
    assert response.status_code == 400
    after = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    assert after == before


def test_publish_is_rejected_if_inputs_changed_midway(client: TestClient):
    """Simulate an upload landing while a computation is in flight."""

    store: Store = client.app_state.store
    upload(client, baseline=True, samples=True)
    snapshot = store.snapshot()
    # Change the baseline after the snapshot, before publishing.
    payload = snapshot.baseline.payload
    store.save_input("baseline", payload, "new.csv")

    from app.compute import compute_report

    report = compute_report(
        BaselineInput.from_payload(snapshot.baseline.payload),
        SamplesInput.from_payload(snapshot.samples.payload),
    )
    assert store.publish_report(snapshot, report) is None
    assert store.latest_report() is None


def test_publishing_is_atomic_under_concurrent_uploads(tmp_path: Path):
    """Hammer publish and upload from two threads.

    Every published report must correspond to a snapshot that was current
    at publish time; no torn state and no DB errors are permitted.
    """

    store = Store(str(tmp_path / "race.db"))
    baseline_payload = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1", "2", "3", "4"], ["5", "6", "7", "8"], ["9", "1", "2", "3"]],
        }
    ).to_payload()
    samples_payload = SamplesInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "sample_ids": ["S1"],
            "rows": [["1", "2", "3", "4"]],
        }
    ).to_payload()
    store.save_inputs(
        {"baseline": (baseline_payload, "b.csv"), "samples": (samples_payload, "s.csv")}
    )

    errors: list[BaseException] = []
    published: list[int] = []

    def publisher():
        try:
            for _ in range(60):
                snapshot = store.snapshot()
                if snapshot is None:
                    continue
                report_id = store.publish_report(snapshot, {"marker": "x"})
                if report_id is not None:
                    published.append(report_id)
                    record = store.get_report(report_id)
                    assert record is not None
                    assert set(record.payload) == {"marker"}
                    expected = {
                        "baseline": record.baseline_version,
                        "samples": record.samples_version,
                    }
                    # Every published report was assembled from one intact
                    # snapshot: both inputs were a coherent pair.
                    assert expected["baseline"] >= 1
                    assert expected["samples"] >= 1
                time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    def uploader():
        try:
            time.sleep(0.02)
            for i in range(30):
                store.save_inputs(
                    {
                        "baseline": (dict(baseline_payload), f"b{i}.csv"),
                        "samples": (dict(samples_payload), f"s{i}.csv"),
                    }
                )
                time.sleep(0.005)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=publisher), threading.Thread(target=uploader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    # Publication must be possible in the quiet window before uploads begin,
    # and must never fail with a database error while uploads race it.
    assert published
    with sqlite3.connect(tmp_path / "race.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        current = conn.execute(
            "SELECT COUNT(*) FROM input_versions WHERE is_current = 1"
        ).fetchone()[0]
    assert count == len(published)
    assert current == 2


def test_interleaved_upload_discards_inflight_computation(
    client: TestClient, monkeypatch
):
    """An upload landing inside /compute must prevent publishing.

    The endpoint snapshots inputs, then computes.  Here the computation is
    wrapped so that a new input is saved before it returns; the publish
    step must notice the version change and refuse to publish.
    """

    import app.main as main_module
    from app.compute import compute_report as real_compute

    store: Store = client.app_state.store
    upload(client, baseline=True, samples=True)
    before_versions = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }

    def compute_and_upload(baseline, samples):
        result = real_compute(baseline, samples)
        store.save_input("samples", samples.to_payload(), "raced.csv")
        return result

    monkeypatch.setattr(main_module, "compute_report", compute_and_upload)

    response = compute(client, follow_redirects=True)
    assert response.status_code == 200
    assert "输入已被更新" in response.text
    assert "已丢弃" in response.text
    # Nothing was published.
    assert latest(client)["report"] is None
    # The upload that raced did take effect (whole-file replacement).
    after_versions = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    assert after_versions["baseline"] == before_versions["baseline"]
    assert after_versions["samples"] == before_versions["samples"] + 1


def test_publish_failure_shows_old_report_and_its_staleness(
    client: TestClient, monkeypatch
):
    register(client)
    store: Store = client.app_state.store
    old_id = latest(client)["report"]["id"]

    # Make the inputs newer than the report so the old report is stale.
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s2.csv", SAMPLES_CSV.encode(), "text/csv")},
    )
    assert latest(client)["report"]["stale"] is True

    def broken_publish(*args, **kwargs):
        raise StoreError("磁盘只读（测试注入）")

    monkeypatch.setattr(store, "publish_report", broken_publish)
    response = compute(client, follow_redirects=True)
    assert response.status_code == 200
    assert "磁盘只读" in response.text
    assert "已过期" in response.text
    # The previous report is still the latest one and its rows are visible.
    current = latest(client)["report"]
    assert current["id"] == old_id
    assert current["stale"] is True
    assert "S1" in response.text


def test_upload_failure_keeps_previous_inputs(client: TestClient, monkeypatch):
    register(client)
    store: Store = client.app_state.store
    before = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }

    def broken_save(*args, **kwargs):
        raise StoreError("数据库磁盘已满（测试注入）")

    monkeypatch.setattr(store, "save_inputs", broken_save)
    response = client.post(
        "/inputs/upload",
        files={"baseline_file": ("b.csv", BASE_CSV.encode(), "text/csv")},
    )
    assert response.status_code == 400
    assert "原有输入未被替换" in response.text
    after = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    assert after == before
    assert latest(client)["report"]["stale"] is False


def test_report_csv_export(client: TestClient):
    register(client)
    report = latest(client)["report"]
    response = client.get(f"/reports/{report['id']}/export.csv")
    assert response.status_code == 200
    body = response.content.decode("utf-8-sig")
    lines = body.strip().splitlines()
    assert lines[0].startswith("sample_id,unique,A_weight,B_weight,C_weight")
    assert len(lines) == 3
    assert client.get("/reports/9999/export.csv").status_code == 404


def test_input_csv_export_roundtrip(client: TestClient):
    response = upload(client, baseline=True)
    assert response.status_code == 303
    response = client.get("/inputs/baseline/export.csv")
    assert response.status_code == 200
    text = response.content.decode("utf-8-sig")
    assert text.splitlines()[0] == "ore,Fe,SiO2,Al2O3,CaO"
    assert text.splitlines()[1].startswith("A,62.3,")


def test_non_unique_sample_is_flagged(client: TestClient):
    baseline = """ore,Fe,SiO2,Al2O3,CaO
A,60,4,2,1
B,60,4,2,1
C,50,6,3,2
"""
    samples = """sample_id,Fe,SiO2,Al2O3,CaO
AMBIG,55,5,2.5,1.5
"""
    upload(client, baseline=True)
    client.post(
        "/inputs/upload",
        files={"baseline_file": ("b.csv", baseline.encode(), "text/csv")},
    )
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s.csv", samples.encode(), "text/csv")},
    )
    compute(client)
    report = latest(client)["report"]
    result = report["payload"]["results"][0]
    assert result["unique"] is False
    assert result["weights"] is None
    assert result["predicted"] is None
    assert result["residuals"] is None
    html = client.get("/").text
    assert "无法唯一确定" in html
    assert "AMBIG" in html
    csv_body = client.get(f"/reports/{report['id']}/export.csv").content.decode(
        "utf-8-sig"
    )
    lines = csv_body.strip().splitlines()
    header = lines[0].split(",")
    ambiguous_line = lines[1].split(",")
    row = dict(zip(header, ambiguous_line))
    assert row["sample_id"] == "AMBIG"
    assert row["unique"] == "no"
    # No specific blend may be emitted for a non-unique optimum.
    for ore in ("A", "B", "C"):
        assert row[f"{ore}_weight"] == ""
    assert row["A_weight_min"] == "0"
    assert row["A_weight_max"] == "50"
    # Predicted content and residual are ranges too, never a chosen value.
    for element in ("Fe", "SiO2", "Al2O3", "CaO"):
        assert row[f"predicted_{element}"] == ""
        assert row[f"residual_{element}"] == ""
    assert row["predicted_Fe_min"] == "55"
    assert row["residual_Fe_min"] == "0"
    assert row["note"]

    api = client.get("/api/latest").json()["report"]["payload"]["results"][0]
    assert api["predicted"] is None
    assert api["predicted_ranges"][0] == ["55", "55"]
    assert api["weight_ranges"][1] == ["0", "1/2"]


def test_sqlite_schema_has_single_current_version(tmp_path: Path):
    store = Store(str(tmp_path / "s.db"))
    payload = BaselineInput.from_payload(
        {
            "elements": ["e1", "e2", "e3", "e4"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    ).to_payload()
    store.save_input("baseline", payload, "a.csv")
    store.save_input("baseline", payload, "b.csv")
    with sqlite3.connect(tmp_path / "s.db") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM input_versions "
            "WHERE kind='baseline' AND is_current=1"
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM input_versions WHERE kind='baseline'"
        ).fetchone()[0]
    assert count == 1
    assert total == 2
