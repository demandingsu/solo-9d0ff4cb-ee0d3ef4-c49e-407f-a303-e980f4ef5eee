"""Tests for original assay sheets, column mapping and exact conversion.

Covers: one-to-one mapping validation, exact ppm/% conversion, row-numbered
rejection of detection-limit markers / duplicate sample ids / illegal
values, whole-batch rejection, atomic replacement of the current samples,
stale reports, and retrieval of the exact original + mapping behind a
historical report.
"""

from __future__ import annotations

import sqlite3
from fractions import Fraction as F
from pathlib import Path
from urllib.parse import unquote_plus

import pytest
from fastapi.testclient import TestClient

from app.db import Store, StoreError
from app.inputs import (
    AssayConversionError,
    AssayInput,
    BaselineInput,
    InputValidationError,
    SamplesInput,
    build_mapping,
    convert_assay,
    parse_assay,
)
from app.main import create_app

BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

ASSAY_CSV = """sample_no,Fe_pct,Si,Al,Ca
S-1001,62.74,37100,20200,11300
S-1002,61.34,42300,23100,12900
"""

MAPPING = {
    "element_0": "Fe",
    "unit_0": "%",
    "element_1": "SiO2",
    "unit_1": "ppm",
    "element_2": "Al2O3",
    "unit_2": "ppm",
    "element_3": "CaO",
    "unit_3": "ppm",
}


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(str(tmp_path / "assay.db"))
    with TestClient(app) as test_client:
        test_client.app_state = app.state  # type: ignore[attr-defined]
        yield test_client


def up_baseline(client: TestClient, text: str = BASE_CSV) -> None:
    response = client.post(
        "/inputs/upload",
        files={"baseline_file": ("b.csv", text.encode(), "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def up_assay(client: TestClient, text: str = ASSAY_CSV):
    return client.post(
        "/assay/upload",
        files={"assay_file": ("assay.csv", text.encode(), "text/csv")},
        follow_redirects=False,
    )


def convert(client: TestClient, mapping: dict | None = None):
    return client.post(
        "/assay/convert", data=MAPPING if mapping is None else mapping,
        follow_redirects=False,
    )


def samples_of(client: TestClient) -> SamplesInput | None:
    current = client.app_state.store.current_inputs()["samples"]
    return SamplesInput.from_payload(current.payload) if current else None


# -- parsing / mapping ---------------------------------------------------


def test_parse_assay_keeps_raw_text_and_labels():
    assay = parse_assay(ASSAY_CSV.encode())
    assert assay.columns == ("Fe_pct", "Si", "Al", "Ca")
    assert [row.sample_id for row in assay.rows] == ["S-1001", "S-1002"]
    assert assay.rows[0].values == ("62.74", "37100", "20200", "11300")
    assert assay.rows[0].line_no == 2
    # Stored payload round-trips the exact original bytes.
    restored = AssayInput.from_payload(assay.to_payload())
    assert restored.raw_bytes == ASSAY_CSV.encode()
    assert restored.columns == assay.columns


def test_parse_assay_requires_five_columns():
    with pytest.raises(InputValidationError, match="表头必须有 5 列"):
        parse_assay(b"a,b,c\nS1,1,2\n")


def test_parse_assay_rejects_duplicate_headers():
    with pytest.raises(InputValidationError, match="不能重复"):
        parse_assay(b"id,Fe,Fe,Si,Al\nS1,1,2,3,4\n")


def test_build_mapping_requires_one_to_one():
    base = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    assay = parse_assay(ASSAY_CSV.encode())
    with pytest.raises(InputValidationError, match="一一对应"):
        build_mapping(
            assay, base, ["Fe", "Fe", "Al2O3", "CaO"], ["%", "%", "%", "%"]
        )
    with pytest.raises(InputValidationError, match="一一对应"):
        build_mapping(
            assay, base, ["Fe", "SiO2", "Al2O3", "Al2O3"], ["%", "%", "%", "%"]
        )
    with pytest.raises(InputValidationError, match="必须指定"):
        build_mapping(assay, base, ["Fe", "SiO2", "Al2O3", ""], ["%", "%", "%", "%"])
    with pytest.raises(InputValidationError, match="单位必须是"):
        build_mapping(
            assay,
            base,
            ["Fe", "SiO2", "Al2O3", "CaO"],
            ["%", "ppb", "%", "%"],
        )
    with pytest.raises(InputValidationError, match="不在当前基准"):
        build_mapping(
            assay, base, ["Fe", "SiO2", "Al2O3", "MgO"], ["%", "%", "%", "%"]
        )
    mapping = build_mapping(
        assay, base, ["Fe", "SiO2", "Al2O3", "CaO"], ["%", "ppm", "%", "ppm"]
    )
    assert mapping.entry_for("SiO2").unit == "ppm"
    assert mapping.entry_for("Fe").column == 0


def test_conversion_is_exact_decimal():
    baseline = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    assay = parse_assay(
        b"id,Fe_pct,Si,Al,Ca\nS1,62.74,37100,0.1,1e-3\n"
    )
    mapping = build_mapping(
        assay, baseline, ["Fe", "SiO2", "Al2O3", "CaO"], ["%", "ppm", "ppm", "%"]
    )
    samples = convert_assay(assay, mapping, baseline)
    assert samples.elements == ("Fe", "SiO2", "Al2O3", "CaO")
    assert samples.rows[0] == (
        F("62.74"),
        F("37100") / 10000,
        F("0.1") / 10000,
        F("0.001"),
    )
    assert sum(samples.rows[0]) == F("62.74") + F("3.71") + F("0.00001") + F("0.001")


def test_conversion_column_order_follows_mapping():
    baseline = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    # Original columns deliberately in a different order than the baseline.
    assay = parse_assay(b"id,Ca,Fe,Si,Al\nS1,1.2,62.3,3.9,2.1\n")
    mapping = build_mapping(
        assay,
        baseline,
        ["CaO", "Fe", "SiO2", "Al2O3"],
        ["%", "%", "%", "%"],
    )
    samples = convert_assay(assay, mapping, baseline)
    assert samples.rows[0] == (F("62.3"), F("3.9"), F("2.1"), F("1.2"))


@pytest.mark.parametrize(
    "marker", ["<0.01", "≤0.01", "<LOD", "ND", "n.d.", "未检出", "低于检出限"]
)
def test_detection_limit_markers_are_row_numbered(marker):
    baseline = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    assay = parse_assay(
        f"id,Fe_pct,Si,Al,Ca\nS1,62.74,37100,20200,11300\nS2,{marker},40000,20000,10000\n".encode()
    )
    mapping = build_mapping(
        assay, baseline, ["Fe", "SiO2", "Al2O3", "CaO"], ["%", "ppm", "ppm", "ppm"]
    )
    with pytest.raises(AssayConversionError) as info:
        convert_assay(assay, mapping, baseline)
    joined = " ".join(info.value.errors)
    assert "第 3 行" in joined
    assert "检出限" in joined
    assert "Fe_pct" in joined


def test_all_errors_reported_with_line_numbers_and_batch_rejected():
    baseline = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    assay = parse_assay(
        b"id,Fe_pct,Si,Al,Ca\n"
        b"S1,62.74,37100,20200,11300\n"
        b"S1,61.34,42300,23100,12900\n"
        b"S3,abc,40000,20000,10000\n"
        b",60,4,2,1\n"
    )
    mapping = build_mapping(
        assay, baseline, ["Fe", "SiO2", "Al2O3", "CaO"], ["%", "ppm", "ppm", "ppm"]
    )
    with pytest.raises(AssayConversionError) as info:
        convert_assay(assay, mapping, baseline)
    errors = info.value.errors
    assert info.value.total == 3
    assert any("第 3 行" in error and "重复" in error for error in errors)
    assert any("第 4 行" in error and "非法数值" in error for error in errors)
    assert any("第 5 行" in error and "样本编号不能为空" in error for error in errors)


def test_line_numbers_are_physical_lines_with_blank_and_quoted_rows():
    baseline = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    raw = (
        "id,Fe,Si,Al,Ca\n"
        "\n"
        "S1,62,4,2,1\n"
        "\n"
        "\n"
        "S2,<0.01,4,2,1\n"
    )
    assay = parse_assay(raw.encode())
    mapping = build_mapping(
        assay, baseline, ["Fe", "SiO2", "Al2O3", "CaO"], ["%", "%", "%", "%"]
    )
    with pytest.raises(AssayConversionError) as info:
        convert_assay(assay, mapping, baseline)
    assert "第 6 行" in " ".join(info.value.errors)


# -- web flow ------------------------------------------------------------


def test_upload_and_convert_replaces_current_samples(client: TestClient):
    up_baseline(client)
    response = up_assay(client)
    assert response.status_code == 303
    assert "原件 v1" in unquote_plus(response.headers["location"])

    response = convert(client)
    assert response.status_code == 303, response.text
    current = client.app_state.store.current_inputs()["samples"]
    assert current.assay_version == 1
    assert current.mapping_version == 1

    samples = samples_of(client)
    assert samples.sample_ids == ("S-1001", "S-1002")
    assert samples.rows[0][1] == F("3.71")
    assert samples.rows[0][3] == F("1.13")


def test_failed_conversion_keeps_current_samples_and_report(client: TestClient):
    up_baseline(client)
    up_assay(client)
    convert(client)
    client.post("/compute", follow_redirects=False)
    report_before = client.get("/api/latest").json()["report"]
    store: Store = client.app_state.store
    versions_before = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }

    bad = ASSAY_CSV + "S-1002,<0.01,4,2,1\n"
    bad = bad.replace("S-1001,62.74,37100,20200,11300", "S-1001,<0.01,37100,20200,11300")
    up_assay(client, bad)
    response = convert(client)
    assert response.status_code == 400
    assert "第 2 行" in response.text
    assert "整批拒收" in response.text

    versions_after = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    assert versions_after == versions_before
    report = client.get("/api/latest").json()["report"]
    assert report["id"] == report_before["id"]
    # The failed conversion never touched the current product, so the report
    # stays consistent with it; the newly uploaded original is only pending.
    assert report["stale"] is False
    # ...and the report is still viewable.
    assert "S-1001" in client.get("/").text


def test_failed_conversion_after_upload_does_not_replace_samples(client: TestClient):
    up_baseline(client)
    up_assay(client)
    first = convert(client)
    assert first.status_code == 303
    store: Store = client.app_state.store
    before = store.current_inputs()["samples"].version_id

    # A new original containing a bad row: upload succeeds structurally, the
    # conversion must fail and leave the old samples untouched.
    up_assay(client, ASSAY_CSV.replace("61.34", "abc"))
    response = convert(client)
    assert response.status_code == 400
    assert "非法数值" in response.text
    after = store.current_inputs()["samples"].version_id
    assert after == before


def test_mapping_one_to_one_is_enforced_over_http(client: TestClient):
    up_baseline(client)
    up_assay(client)
    bad = dict(MAPPING, element_1="Fe")
    response = convert(client, bad)
    assert response.status_code == 400
    assert "一一对应" in response.text
    assert client.app_state.store.current_inputs()["samples"] is None


def test_convert_retries_after_mapping_edit_uses_same_original(client: TestClient):
    up_baseline(client)
    up_assay(client)
    # First, wrong unit: ppm read as %, giving absurd values but valid numbers.
    wrong = dict(MAPPING, unit_1="%")
    assert convert(client, wrong).status_code == 303
    first = samples_of(client).rows[0][1]
    assert first == F("37100")
    # Fix the mapping; conversion re-runs from the same original.
    response = convert(client, MAPPING)
    assert response.status_code == 303
    assert samples_of(client).rows[0][1] == F("3.71")
    store: Store = client.app_state.store
    assert store.current_inputs()["samples"].assay_version == 1
    assert store.current_inputs()["samples"].mapping_version >= 2


def test_save_mapping_only_does_not_touch_samples(client: TestClient):
    up_baseline(client)
    up_assay(client)
    response = client.post(
        "/assay/mapping", data=MAPPING, follow_redirects=False
    )
    assert response.status_code == 303
    assert client.app_state.store.current_inputs()["samples"] is None
    state = client.app_state.store.latest_assay()
    assert state is not None and state.mapping is not None
    # The page pre-selects the saved mapping.
    html = client.get("/").text
    assert 'value="Fe" selected' in html
    assert 'value="ppm" selected' in html


def test_save_mapping_for_superseded_original_is_refused(client: TestClient):
    up_baseline(client)
    up_assay(client)
    first_version = client.app_state.store.latest_assay().assay.version_id
    up_assay(client, ASSAY_CSV.replace("62.74", "62.75"))
    # The browser still points at the superseded original: the mapping must
    # not be attached to it.
    result = client.app_state.store.save_mapping(
        first_version, {"entries": []}
    )
    assert result is None
    state = client.app_state.store.latest_assay()
    assert state.assay.version_id == first_version + 1
    assert state.mapping is None


def test_saving_identical_mapping_does_not_grow_history(client: TestClient):
    up_baseline(client)
    up_assay(client)
    first = client.app_state.store.save_mapping(1, {"entries": [1]})
    second = client.app_state.store.save_mapping(1, {"entries": [1]})
    assert first == second
    assert count_rows(client, "mapping_versions") == 1


def test_conversion_marks_previous_report_stale(client: TestClient):
    up_baseline(client)
    up_assay(client)
    convert(client)
    client.post("/compute", follow_redirects=False)
    report = client.get("/api/latest").json()["report"]
    assert report["stale"] is False

    # A new original = same numbers but a different version; reconversion
    # creates a new samples version, so the old report must be stale.
    up_assay(client, ASSAY_CSV.replace("62.74", "62.75"))
    assert convert(client).status_code == 303
    report = client.get("/api/latest").json()["report"]
    assert report["stale"] is True
    assert "已过期" in client.get("/").text


def test_stale_form_versions_are_rejected(client: TestClient):
    """A form rendered for an older original must not map a newer sheet."""

    up_baseline(client)
    up_assay(client)
    # A new original with a *different column order* arrives after the form
    # was rendered.  Applying the old positional choices would silently map
    # the wrong columns, so the bound version must be checked.
    new_order = "sample_no,Ca_ppm,Fe_pct,Si_ppm,Al_ppm\n" "S-1001,11300,62.74,37100,20200\n"
    up_assay(client, new_order)
    form = dict(MAPPING, assay_version="1")
    response = convert(client, form)
    assert response.status_code == 400
    assert "当前原件已改变" in response.text
    assert client.app_state.store.current_inputs()["samples"] is None

    # The current version token is accepted.
    form = dict(MAPPING, assay_version="2")
    reordered = {
        "element_0": "CaO", "unit_0": "ppm",
        "element_1": "Fe", "unit_1": "%",
        "element_2": "SiO2", "unit_2": "ppm",
        "element_3": "Al2O3", "unit_3": "ppm",
        "assay_version": "2",
    }
    assert convert(client, reordered).status_code == 303
    assert samples_of(client).rows[0] == (F("62.74"), F("3.71"), F("2.02"), F("1.13"))


def test_form_is_bound_to_baseline_version(client: TestClient):
    up_baseline(client)
    up_assay(client)
    form = dict(MAPPING, baseline_version="99")
    response = convert(client, form)
    assert response.status_code == 400
    assert "当前基准已改变" in response.text
    assert client.app_state.store.current_inputs()["samples"] is None


def test_structurally_invalid_original_keeps_previous_original(client: TestClient):
    up_baseline(client)
    up_assay(client)
    before = client.app_state.store.latest_assay().assay.version_id
    response = up_assay(client, "a,b,c,d\nS1,1,2,3,4\n")
    assert response.status_code == 400
    assert "表头必须有 5 列" in response.text
    assert client.app_state.store.latest_assay().assay.version_id == before


def test_conversion_write_failure_keeps_current_inputs(client: TestClient, monkeypatch):
    up_baseline(client)
    up_assay(client)
    up_assay(client)  # second original, still unconverted
    store: Store = client.app_state.store

    def broken_publish(*args, **kwargs):
        raise StoreError("磁盘只读（测试注入）")

    monkeypatch.setattr(store, "publish_conversion", broken_publish)
    response = convert(client)
    assert response.status_code == 400
    assert "现役成品未替换" in response.text
    assert store.current_inputs()["samples"] is None
    assert count_rows(client, "mapping_versions") == 0
    assert store.latest_assay().mapping is None


def test_conversion_conflict_publishes_nothing(client: TestClient, monkeypatch):
    up_baseline(client)
    up_assay(client)
    convert(client)
    store: Store = client.app_state.store
    before = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    mapping_count = count_rows(client, "mapping_versions")

    import app.main as main_module

    original = main_module.convert_assay

    def convert_then_change_baseline(*args, **kwargs):
        result = original(*args, **kwargs)
        # A baseline upload lands between conversion and publication.
        store.save_input(
            "baseline",
            BaselineInput.from_payload(
                {
                    "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
                    "ores": ["A", "B", "C"],
                    "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
                }
            ).to_payload(),
            "raced.csv",
        )
        return result

    monkeypatch.setattr(main_module, "convert_assay", convert_then_change_baseline)
    response = convert(client)
    assert response.status_code == 400
    assert "未发布" in response.text

    after = {
        kind: value.version_id for kind, value in store.current_inputs().items()
    }
    assert after["samples"] == before["samples"]
    # The racing upload did take effect (whole-file replacement), it is just
    # not the baseline the conversion used, so nothing was published.
    assert after["baseline"] != before["baseline"]
    assert count_rows(client, "mapping_versions") == mapping_count


def test_conversion_conflict_when_mapping_changed_underneath(client: TestClient):
    up_baseline(client)
    up_assay(client)
    store: Store = client.app_state.store
    import app.main as main_module

    original = main_module.convert_assay
    calls = {"n": 0}

    def convert_and_save_mapping(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["n"] += 1
        # Someone edits the mapping (saves it) mid-flight.
        mapping = args[1]
        store.save_mapping(1, mapping.to_payload())
        return result

    main_module.convert_assay = convert_and_save_mapping
    try:
        response = convert(client)
    finally:
        main_module.convert_assay = original
    assert response.status_code == 400
    assert "未发布" in response.text
    assert store.current_inputs()["samples"] is None


# -- provenance ----------------------------------------------------------


def test_historical_report_keeps_original_and_mapping(client: TestClient):
    up_baseline(client)
    up_assay(client)
    convert(client)
    client.post("/compute", follow_redirects=False)
    first_report = client.get("/api/latest").json()["report"]
    assert first_report["assay_version"] == 1
    assert first_report["mapping_version"] == 1

    # Replace the original and mapping, then produce a second report.
    new_assay = ASSAY_CSV.replace("62.74", "12.34")
    up_assay(client, new_assay)
    changed = dict(MAPPING, unit_2="%")
    convert(client, changed)
    client.post("/compute", follow_redirects=False)
    latest = client.get("/api/latest").json()["report"]
    assert latest["assay_version"] == 2
    assert latest["mapping_version"] == 2
    assert latest["stale"] is False

    # The first report's provenance still points at original v1 and mapping v1.
    page = client.get(f"/reports/{first_report['id']}/provenance")
    assert page.status_code == 200
    assert "原件 v1" in page.text
    assert "映射 v1" in page.text
    assert "assay.csv" in page.text  # original file name is shown

    download = client.get("/assay/versions/1/download.csv")
    assert download.status_code == 200
    assert download.content.decode("utf-8") == ASSAY_CSV
    # And the exact converted samples behind that report remain downloadable.
    old_samples = client.get(
        f"/inputs/samples/versions/{first_report['samples_version']}/export.csv"
    )
    assert old_samples.status_code == 200
    assert old_samples.content.decode("utf-8-sig").splitlines()[1].startswith(
        "S-1001,62.74,"
    )


def test_exports_are_exact_and_reimportable(client: TestClient):
    """Converted values (e.g. ppm / 10000) survive the CSV round trip."""

    up_baseline(client)
    up_assay(
        client,
        "sample_no,Fe_pct,Si,Al,Ca\nS-1,12.3456,12345,1,9999\n",
    )
    mapping = {
        "element_0": "Fe",
        "unit_0": "%",
        "element_1": "SiO2",
        "unit_1": "ppm",
        "element_2": "Al2O3",
        "unit_2": "ppm",
        "element_3": "CaO",
        "unit_3": "ppm",
    }
    assert convert(client, mapping).status_code == 303
    exported = client.get("/inputs/samples/export.csv").content.decode("utf-8-sig")
    body = exported.splitlines()[1]
    assert body == "S-1,12.3456,1.2345,0.0001,0.9999"
    # Re-uploading the export reproduces exactly the same values.
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s.csv", exported.encode(), "text/csv")},
        follow_redirects=False,
    )
    samples = samples_of(client)
    assert samples.rows[0] == (
        F("12.3456"),
        F("1.2345"),
        F("0.0001"),
        F("0.9999"),
    )


def test_version_pages_and_downloads(client: TestClient):
    up_baseline(client)
    up_assay(client)
    convert(client)
    assert client.get("/assay/versions/1").status_code == 200
    assert client.get("/mappings/1").status_code == 200
    assert "Fe_pct" in client.get("/assay/versions/1").text
    assert client.get("/assay/versions/99").status_code == 404
    assert client.get("/mappings/99").status_code == 404
    assert client.get("/assay/versions/99/download.csv").status_code == 404


def test_direct_samples_upload_has_no_assay_provenance(client: TestClient):
    up_baseline(client)
    client.post(
        "/inputs/upload",
        files={
            "samples_file": (
                "s.csv",
                b"sample_id,Fe,SiO2,Al2O3,CaO\nS1,62,4,2,1\n",
                "text/csv",
            )
        },
        follow_redirects=False,
    )
    client.post("/compute", follow_redirects=False)
    report = client.get("/api/latest").json()["report"]
    assert report["assay_version"] is None
    page = client.get(f"/reports/{report['id']}/provenance")
    assert page.status_code == 200
    assert "直接上传的成品 CSV" in page.text


def test_conversion_is_atomic_under_concurrent_uploads(tmp_path: Path):
    """Hammer conversions while baselines change: no torn samples versions."""

    import threading

    store = Store(str(tmp_path / "conv.db"))
    baseline_a = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1"] * 4, ["2"] * 4, ["3"] * 4],
        }
    )
    baseline_b = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["4"] * 4, ["5"] * 4, ["6"] * 4],
        }
    )
    store.save_input("baseline", baseline_a.to_payload(), "a.csv")
    assay = parse_assay(ASSAY_CSV.encode())
    assay_version = store.save_assay(assay.to_payload(), "a.csv")
    mapping = build_mapping(
        assay, baseline_a, ["Fe", "SiO2", "Al2O3", "CaO"], ["%", "ppm", "ppm", "ppm"]
    )
    samples = convert_assay(assay, mapping, baseline_a)

    errors: list[BaseException] = []
    successes: list[int] = []

    def converter():
        try:
            for _ in range(40):
                snapshot = store.conversion_snapshot()
                if snapshot.baseline is None:
                    continue
                result = store.publish_conversion(
                    baseline_version=snapshot.baseline.version_id,
                    assay_version=snapshot.assay.assay.version_id,
                    mapping_version=(
                        snapshot.assay.mapping.version_id
                        if snapshot.assay and snapshot.assay.mapping
                        else None
                    ),
                    mapping_payload=mapping.to_payload(),
                    samples_payload=samples.to_payload(),
                    filename="a.csv",
                )
                if result is not None:
                    successes.append(result.samples_version)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def uploader():
        try:
            for i in range(20):
                store.save_input(
                    "baseline",
                    (baseline_a if i % 2 else baseline_b).to_payload(),
                    f"b{i}.csv",
                )
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=converter), threading.Thread(target=uploader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert successes
    with sqlite3.connect(tmp_path / "conv.db") as conn:
        current = conn.execute(
            "SELECT COUNT(*) FROM input_versions "
            "WHERE kind='samples' AND is_current=1"
        ).fetchone()[0]
        all_samples = conn.execute(
            "SELECT COUNT(*) FROM input_versions WHERE kind='samples'"
        ).fetchone()[0]
        # Every conversion that reported success left a version behind.
        assert all_samples == len(set(successes))
    assert current == 1
    assert assay_version >= 1


def count_rows(client: TestClient, table: str) -> int:
    store: Store = client.app_state.store
    with sqlite3.connect(store.path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
