"""FastAPI application for ore-blend proportion reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import formatting
from .compute import ComputeError, compute_report
from .db import (
    ReportRecord,
    Store,
    StoreError,
    VersionedInput,
)
from .inputs import (
    MAX_UPLOAD_BYTES,
    AssayConversionError,
    AssayInput,
    AssayMapping,
    BaselineInput,
    InputValidationError,
    SamplesInput,
    build_mapping,
    convert_assay,
    parse_assay,
    parse_baseline,
    parse_samples,
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_ROW_LIMIT = 500


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(title="矿石配料反推与异常定位")
    store = Store(database_path or os.environ.get("ORE_DB_PATH", "data/app.db"))
    app.state.store = store

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["decimal"] = formatting.decimal
    templates.env.filters["percent"] = formatting.percent
    templates.env.filters["signed"] = formatting.signed
    templates.env.filters["ratio"] = formatting.ratio_text
    templates.env.filters["exact"] = formatting.exact

    app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    # -- helpers ---------------------------------------------------------

    def redirect_notice(
        message: str, kind: str = "ok", anchor: str = ""
    ) -> RedirectResponse:
        query = urlencode({"notice": message, "notice_kind": kind})
        url = f"/?{query}"
        if anchor:
            url += f"#{anchor}"
        return RedirectResponse(url, status_code=303)

    def payload_to_inputs(
        snapshot,
    ) -> tuple[BaselineInput, SamplesInput]:
        return (
            BaselineInput.from_payload(snapshot.baseline.payload),
            SamplesInput.from_payload(snapshot.samples.payload),
        )

    def conversion_context() -> dict[str, Any]:
        """Everything the mapping form needs to render."""

        snapshot = store.conversion_snapshot()
        baseline = (
            BaselineInput.from_payload(snapshot.baseline.payload)
            if snapshot.baseline is not None
            else None
        )
        assay = (
            AssayInput.from_payload(snapshot.assay.assay.payload)
            if snapshot.assay is not None
            else None
        )
        mapping = (
            AssayMapping.from_payload(snapshot.assay.mapping.payload)
            if snapshot.assay is not None and snapshot.assay.mapping is not None
            else None
        )
        mapping_slots: list[dict[str, str] | None] = [None] * (
            len(assay.columns) if assay is not None else 0
        )
        if mapping is not None:
            for entry in mapping.entries:
                if 0 <= entry.column < len(mapping_slots):
                    mapping_slots[entry.column] = {
                        "element": entry.element,
                        "unit": entry.unit,
                        "label": entry.label,
                    }
        return {
            "baseline": baseline,
            "assay": assay,
            "assay_version": (
                snapshot.assay.assay.version_id if snapshot.assay else None
            ),
            "assay_filename": (
                snapshot.assay.assay.filename if snapshot.assay else None
            ),
            "assay_created_at": (
                snapshot.assay.assay.created_at if snapshot.assay else None
            ),
            "mapping": mapping,
            "mapping_slots": mapping_slots,
            "mapping_version": (
                snapshot.assay.mapping.version_id
                if snapshot.assay and snapshot.assay.mapping
                else None
            ),
            "mapping_created_at": (
                snapshot.assay.mapping.created_at
                if snapshot.assay and snapshot.assay.mapping
                else None
            ),
            "raw_preview": (
                _assay_preview(assay, limit=10) if assay is not None else []
            ),
            "rows_total": len(assay.rows) if assay is not None else 0,
        }

    def input_summary(value: VersionedInput | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if value.kind == "baseline":
            parsed = BaselineInput.from_payload(value.payload)
            return {
                "version": value.version_id,
                "filename": value.filename,
                "created_at": value.created_at,
                "elements": list(parsed.elements),
                "names": list(parsed.ores),
                "rows": [
                    [formatting.decimal(v, 8) for v in row] for row in parsed.rows
                ],
                "count": len(parsed.rows),
            }
        parsed_samples = SamplesInput.from_payload(value.payload)
        return {
            "version": value.version_id,
            "filename": value.filename,
            "created_at": value.created_at,
            "elements": list(parsed_samples.elements),
            "names": list(parsed_samples.sample_ids[:5]),
            "rows": [],
            "count": len(parsed_samples.sample_ids),
        }

    def report_view(record: ReportRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        payload = dict(record.payload)
        rows = payload.get("results", [])
        visible = rows[:REPORT_ROW_LIMIT]
        return {
            "id": record.report_id,
            "created_at": record.created_at,
            "baseline_version": record.baseline_version,
            "samples_version": record.samples_version,
            "assay_version": record.assay_version,
            "mapping_version": record.mapping_version,
            "stale": record.is_stale,
            "baseline_current": record.baseline_is_current,
            "samples_current": record.samples_is_current,
            "payload": payload,
            "rows": visible,
            "hidden_rows": max(0, len(rows) - len(visible)),
        }

    def context(request: Request) -> dict[str, Any]:
        current = store.current_inputs()
        record = store.latest_report()
        conversion = conversion_context()
        samples_current = current["samples"]
        conversion["samples_match_assay"] = bool(
            samples_current is not None
            and conversion["assay_version"] is not None
            and samples_current.assay_version == conversion["assay_version"]
            and samples_current.mapping_version
            == conversion["mapping_version"]
        )
        summaries = store.recent_reports()
        return {
            "request": request,
            "current": {
                kind: input_summary(value) for kind, value in current.items()
            },
            "current_samples_source": (
                {
                    "assay_version": samples_current.assay_version,
                    "mapping_version": samples_current.mapping_version,
                }
                if samples_current is not None
                else None
            ),
            "report": report_view(record),
            "report_history": [
                {
                    "id": summary.report_id,
                    "created_at": summary.created_at,
                    "samples_version": summary.samples_version,
                    "assay_version": summary.assay_version,
                    "mapping_version": summary.mapping_version,
                    "sample_count": summary.sample_count,
                    "non_unique_count": summary.non_unique_count,
                    "stale": summary.is_stale,
                }
                for summary in summaries
            ],
            "conversion": conversion,
            "notice": request.query_params.get("notice"),
            "notice_kind": request.query_params.get("notice_kind", "ok"),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", context(request))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inputs/upload")
    async def upload_inputs(
        request: Request,
        baseline_file: UploadFile | None = File(None),
        samples_file: UploadFile | None = File(None),
    ) -> Response:
        provided = {
            "baseline": baseline_file,
            "samples": samples_file,
        }
        if not any(file is not None and file.filename for file in provided.values()):
            return redirect_notice("请至少选择一个 CSV 文件再上传。", "error", "upload")

        parsed: dict[str, Any] = {}
        filenames: dict[str, str] = {}
        labels = {"baseline": "基准", "samples": "成品"}
        try:
            for kind, file in provided.items():
                if file is None or not file.filename:
                    continue
                raw = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise _LabeledError(
                        labels[kind],
                        f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                parser = parse_baseline if kind == "baseline" else parse_samples
                try:
                    parsed[kind] = await run_in_threadpool(parser, raw)
                except InputValidationError as exc:
                    raise _LabeledError(labels[kind], str(exc)) from exc
                filenames[kind] = (
                    file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
                    or "未命名.csv"
                )
        except _LabeledError as exc:
            return _render_with_error(request, f"{exc.label}文件：{exc.message}")

        updates: dict[str, tuple[dict[str, Any], str]] = {}
        for kind, value in parsed.items():
            updates[kind] = (value.to_payload(), filenames[kind])

        had_report = await run_in_threadpool(
            lambda: store.latest_report() is not None
        )
        try:
            versions = await run_in_threadpool(store.save_inputs, updates)
        except StoreError as exc:
            return _render_with_error(
                request, f"输入写入失败，原有输入未被替换：{exc}"
            )

        replaced = []
        if "baseline" in versions:
            replaced.append(f"基准 v{versions['baseline']}")
        if "samples" in versions:
            replaced.append(f"成品 v{versions['samples']}")
        suffix = "。旧报告已标记为过期。" if had_report else "。"
        return redirect_notice(
            "已整版替换：" + "，".join(replaced) + suffix,
            "ok",
            "report",
        )

    def _render_with_error(
        request: Request, message: str, *, status_code: int = 400
    ) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=status_code
        )

    # -- original assay sheet --------------------------------------------

    @app.post("/assay/upload")
    async def upload_assay(
        request: Request, assay_file: UploadFile | None = File(None)
    ) -> Response:
        if assay_file is None or not assay_file.filename:
            return redirect_notice("请选择一个原件 CSV 再上传。", "error", "assay")
        raw = await assay_file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return _render_with_error(
                request,
                f"原件文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
            )
        try:
            assay = await run_in_threadpool(parse_assay, raw)
        except InputValidationError as exc:
            return _render_with_error(request, f"原件文件：{exc}")
        filename = (
            assay_file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
            or "未命名.csv"
        )
        try:
            version = await run_in_threadpool(
                store.save_assay, assay.to_payload(), filename
            )
        except StoreError as exc:
            return _render_with_error(
                request, f"原件保存失败，原有原件仍有效：{exc}"
            )
        return redirect_notice(
            f"原件 v{version} 已保存（{len(assay.rows)} 行）。"
            "请逐列指定基准元素与单位后换算。",
            "ok",
            "assay",
        )

    def _mapping_choices(
        form: list[tuple[str, str]]
    ) -> tuple[list[str], list[str]]:
        """Read the four ``element_*`` / ``unit_*`` pairs in column order."""

        values = dict(form)
        elements = [values.get(f"element_{i}", "") for i in range(4)]
        units = [values.get(f"unit_{i}", "") for i in range(4)]
        return elements, units

    async def _read_form(request: Request) -> list[tuple[str, str]]:
        form = await request.form()
        return [(str(key), str(value)) for key, value in form.multi_items()]

    def _form_versions(
        form: list[tuple[str, str]]
    ) -> tuple[int | None, int | None, int | None]:
        values = dict(form)

        def optional(name: str) -> int | None:
            raw = values.get(name, "")
            return int(raw) if raw.strip().isdigit() else None

        return (
            optional("baseline_version"),
            optional("assay_version"),
            optional("mapping_version"),
        )

    def _render_stale_form(
        request: Request,
        message: str,
    ) -> HTMLResponse:
        return _conversion_failure(
            request,
            message + "请刷新页面，确认当前基准、原件与映射后重新提交。",
        )

    def _form_matches_current(
        request: Request,
        form: list[tuple[str, str]],
        snapshot,
    ) -> HTMLResponse | None:
        """Reject a form rendered against a different set of versions.

        The four element columns are matched positionally, so a new original
        sheet with a different column order would silently receive the old
        choices.  Hidden version fields bind the form to what the user saw.
        """

        baseline_version, assay_version, mapping_version = _form_versions(form)
        current_mapping = snapshot.assay.mapping if snapshot.assay else None
        current_mapping_version = (
            current_mapping.version_id if current_mapping is not None else None
        )
        if (
            baseline_version is not None
            and snapshot.baseline is not None
            and baseline_version != snapshot.baseline.version_id
        ):
            return _render_stale_form(request, "当前基准已改变，本次提交未执行。")
        if (
            assay_version is not None
            and (snapshot.assay is None or assay_version != snapshot.assay.assay.version_id)
        ):
            return _render_stale_form(request, "当前原件已改变，本次提交未执行。")
        if (
            mapping_version is not None
            and mapping_version != current_mapping_version
        ):
            return _render_stale_form(request, "映射已改变，本次提交未执行。")
        return None

    def _conversion_failure(
        request: Request,
        message: str,
        *,
        errors: list[str] | None = None,
        error_heading: str | None = None,
    ) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        ctx["conversion_errors"] = errors or []
        ctx["conversion_error_heading"] = error_heading
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    @app.post("/assay/mapping")
    async def save_mapping(request: Request) -> Response:
        """Store a new mapping version without converting.

        Used when the mapping must be edited first; conversion is a separate
        step so a failed conversion cannot leave a mapping half-applied.
        """

        form = await _read_form(request)
        snapshot = await run_in_threadpool(store.conversion_snapshot)
        if snapshot.baseline is None:
            return _conversion_failure(
                request, "必须先有当前基准，才能建立元素映射。"
            )
        if snapshot.assay is None:
            return _conversion_failure(request, "请先上传原件 CSV。")
        stale = _form_matches_current(request, form, snapshot)
        if stale is not None:
            return stale
        baseline = BaselineInput.from_payload(snapshot.baseline.payload)
        assay = AssayInput.from_payload(snapshot.assay.assay.payload)
        elements, units = _mapping_choices(form)
        try:
            mapping = build_mapping(assay, baseline, elements, units)
        except InputValidationError as exc:
            return _conversion_failure(request, str(exc))
        try:
            mapping_version = await run_in_threadpool(
                store.save_mapping,
                snapshot.assay.assay.version_id,
                mapping.to_payload(),
            )
        except StoreError as exc:
            return _conversion_failure(request, f"映射保存失败：{exc}")
        if mapping_version is None:
            return _conversion_failure(
                request,
                "保存映射期间已有新原件上传，该映射未保存；请刷新后针对当前原件重试。",
            )
        return redirect_notice(
            f"映射 v{mapping_version} 已保存，尚未换算。"
            "点击“按此映射换算”生成成品。",
            "ok",
            "assay",
        )

    @app.post("/assay/convert")
    async def convert(request: Request) -> Response:
        form = await _read_form(request)
        snapshot = await run_in_threadpool(store.conversion_snapshot)
        if snapshot.baseline is None:
            return _conversion_failure(
                request, "当前没有基准 CSV，无法换算原件。"
            )
        if snapshot.assay is None:
            return _conversion_failure(request, "请先上传原件 CSV。")
        stale = _form_matches_current(request, form, snapshot)
        if stale is not None:
            return stale
        baseline = BaselineInput.from_payload(snapshot.baseline.payload)
        assay = AssayInput.from_payload(snapshot.assay.assay.payload)
        elements, units = _mapping_choices(form)
        try:
            mapping = build_mapping(assay, baseline, elements, units)
        except InputValidationError as exc:
            return _conversion_failure(request, str(exc))

        baseline_version = snapshot.baseline.version_id
        assay_version = snapshot.assay.assay.version_id
        current_mapping = snapshot.assay.mapping
        current_mapping_id = (
            current_mapping.version_id if current_mapping is not None else None
        )

        try:
            samples = await run_in_threadpool(
                convert_assay, assay, mapping, baseline
            )
        except AssayConversionError as exc:
            heading = (
                f"原件换算失败：发现 {exc.total} 处问题，整批拒收，"
                "现役成品与旧报告均未改动。"
            )
            if exc.total > len(exc.errors):
                heading += f"以下仅列出前 {len(exc.errors)} 处："
            else:
                heading += "问题如下："
            return _conversion_failure(
                request,
                heading,
                errors=exc.errors,
                error_heading=heading,
            )

        try:
            result = await run_in_threadpool(
                store.publish_conversion,
                baseline_version=baseline_version,
                assay_version=assay_version,
                mapping_version=current_mapping_id,
                mapping_payload=mapping.to_payload(),
                samples_payload=samples.to_payload(),
                filename=snapshot.assay.assay.filename,
            )
        except StoreError as exc:
            return _conversion_failure(
                request, f"换算结果写入失败，现役成品未替换：{exc}"
            )
        if result is None:
            return _conversion_failure(
                request,
                "换算期间基准、原件或映射已改变，本次结果未发布；"
                "现役成品与已有报告均未改动。请刷新后重试。",
            )
        return redirect_notice(
            f"换算成功：成品 v{result.samples_version}（映射 "
            f"v{result.mapping_version}，{len(samples.sample_ids)} 个样本）"
            "已整版替换现役成品，旧报告随之过期。",
            "ok",
            "report",
        )

    @app.post("/compute")
    async def compute(request: Request) -> Response:
        snapshot = await run_in_threadpool(store.snapshot)
        if snapshot is None:
            return redirect_notice(
                "基准与成品输入都必须先上传，才能计算。", "error", "upload"
            )

        try:
            baseline, samples = payload_to_inputs(snapshot)
            payload = await run_in_threadpool(
                compute_report, baseline, samples
            )
        except (ComputeError, InputValidationError) as exc:
            return redirect_notice(str(exc), "error", "report")
        except Exception as exc:  # pragma: no cover - defensive
            return redirect_notice(
                f"计算失败，未产生新报告：{type(exc).__name__}: {exc}",
                "error",
                "report",
            )

        try:
            report_id = await run_in_threadpool(
                store.publish_report, snapshot, payload
            )
        except StoreError as exc:
            return redirect_notice(
                f"报告写入失败，仍显示上一份报告：{exc}", "error", "report"
            )

        if report_id is None:
            return redirect_notice(
                "计算期间输入已被更新，本次结果与当前输入不符，已丢弃，"
                "未发布报告。请基于新输入重新计算。",
                "error",
                "report",
            )

        return redirect_notice(
            f"报告 #{report_id} 已生成并发布（{payload['sample_count']} 个样本，"
            f"{payload['non_unique_count']} 个无法唯一确定）。",
            "ok",
            "report",
        )

    # -- exports ---------------------------------------------------------

    @app.get("/reports/{report_id}/export.csv")
    def export_report(report_id: int) -> Response:
        record = store.get_report(report_id)
        if record is None:
            return PlainTextResponse("报告不存在", status_code=404)
        payload = record.payload
        elements = payload["elements"]
        header = ["sample_id", "unique"]
        for ore in payload["ores"]:
            header.append(f"{ore}_weight")
        for ore in payload["ores"]:
            header.append(f"{ore}_weight_min")
            header.append(f"{ore}_weight_max")
        for element in elements:
            header.append(f"measured_{element}")
        for element in elements:
            header += [
                f"predicted_{element}",
                f"predicted_{element}_min",
                f"predicted_{element}_max",
            ]
        for element in elements:
            header += [
                f"residual_{element}",
                f"residual_{element}_min",
                f"residual_{element}_max",
            ]
        header += ["sse", "note"]

        def triple(point_value, interval):
            """One element's (value, min, max) cells.

            A non-unique optimum has no single value, so the value cell is
            left empty and only the exact attainable range is emitted.
            """

            low, high = interval
            return [
                "" if point_value is None else formatting.decimal(point_value, 10),
                formatting.decimal(low, 10),
                formatting.decimal(high, 10),
            ]

        rows = [",".join(_csv_cell(cell) for cell in header)]
        for result in payload["results"]:
            row = [result["sample_id"], "yes" if result["unique"] else "no"]
            if result["unique"]:
                row += [formatting.percent(v, 8) for v in result["weights"]]
            else:
                row += [""] * len(payload["ores"])
            for low, high in result["weight_ranges"]:
                row += [formatting.percent(low, 8), formatting.percent(high, 8)]
            row += [formatting.decimal(v, 10) for v in result["measured"]]
            for i in range(len(elements)):
                row += triple(
                    None if result["predicted"] is None else result["predicted"][i],
                    result["predicted_ranges"][i],
                )
            for i in range(len(elements)):
                row += triple(
                    None if result["residuals"] is None else result["residuals"][i],
                    result["residual_ranges"][i],
                )
            row.append(formatting.decimal(result["sse"], 12))
            row.append(
                ""
                if result["unique"]
                else "最优占比不唯一：占比、预测值与残差仅以下列 min/max 范围给出，"
                "不存在唯一的配料组合"
            )
            rows.append(",".join(_csv_cell(cell) for cell in row))
        return Response(
            "﻿" + "\r\n".join(rows) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ore-report-{report_id}.csv"'
                )
            },
        )

    @app.get("/inputs/{kind}/export.csv")
    def export_input(kind: str) -> Response:
        if kind not in ("baseline", "samples"):
            return PlainTextResponse("输入类型不存在", status_code=404)
        current = store.current_inputs().get(kind)
        if current is None:
            return PlainTextResponse("输入尚未上传", status_code=404)
        return _input_csv_response(kind, current.payload, "current")

    # -- original / mapping retrieval ------------------------------------

    @app.get("/assay/versions/{version_id}/download.csv")
    def download_assay(version_id: int) -> Response:
        assay = store.get_assay(version_id)
        if assay is None:
            return PlainTextResponse("原件版本不存在", status_code=404)
        raw = bytes(AssayInput.from_payload(assay.payload).raw_bytes)
        return Response(
            raw,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="assay-v{version_id}.csv"'
                )
            },
        )

    @app.get("/assay/versions/{version_id}")
    def assay_version_page(request: Request, version_id: int) -> Response:
        assay = store.get_assay(version_id)
        if assay is None:
            return PlainTextResponse("原件版本不存在", status_code=404)
        parsed = AssayInput.from_payload(assay.payload)
        ctx = context(request)
        ctx["version_detail"] = {
            "kind": "原件",
            "version_id": assay.version_id,
            "filename": assay.filename,
            "created_at": assay.created_at,
            "columns": list(parsed.columns),
            "rows": [
                [str(row.line_no), row.sample_id, *row.values]
                for row in parsed.rows[:200]
            ],
            "total": len(parsed.rows),
            "download_url": f"/assay/versions/{version_id}/download.csv",
        }
        return templates.TemplateResponse(request, "version.html", ctx)

    @app.get("/mappings/{version_id}")
    def mapping_version_page(request: Request, version_id: int) -> Response:
        mapping = store.get_mapping(version_id)
        if mapping is None:
            return PlainTextResponse("映射版本不存在", status_code=404)
        parsed = AssayMapping.from_payload(mapping.payload)
        ctx = context(request)
        ctx["version_detail"] = {
            "kind": "映射",
            "version_id": mapping.version_id,
            "filename": mapping.filename,
            "created_at": mapping.created_at,
            "assay_version": mapping.assay_version,
            "entries": [
                {
                    "label": entry.label,
                    "element": entry.element,
                    "unit": entry.unit,
                }
                for entry in parsed.entries
            ],
            "total": None,
            "download_url": f"/assay/versions/{mapping.assay_version}/download.csv",
        }
        return templates.TemplateResponse(request, "version.html", ctx)

    @app.get("/reports/{report_id}/provenance")
    def report_provenance_page(request: Request, report_id: int) -> Response:
        record = store.get_report(report_id)
        if record is None:
            return PlainTextResponse("报告不存在", status_code=404)
        provenance = store.report_provenance(report_id)
        ctx = context(request)
        ctx["provenance"] = {
            "report_id": report_id,
            "assay_version": record.assay_version,
            "mapping_version": record.mapping_version,
            "samples_version": record.samples_version,
            "assay": (
                {
                    "version_id": provenance.assay.version_id,
                    "filename": provenance.assay.filename,
                    "created_at": provenance.assay.created_at,
                    "download_url": (
                        f"/assay/versions/{provenance.assay.version_id}"
                        "/download.csv"
                    ),
                }
                if provenance and provenance.assay
                else None
            ),
            "mapping": (
                {
                    "version_id": provenance.mapping.version_id,
                    "created_at": provenance.mapping.created_at,
                    "entries": AssayMapping.from_payload(
                        provenance.mapping.payload
                    ).entries,
                }
                if provenance and provenance.mapping
                else None
            ),
        }
        return templates.TemplateResponse(request, "provenance.html", ctx)

    def _input_csv_response(
        kind: str, payload: dict[str, Any], suffix: str
    ) -> Response:
        if kind == "baseline":
            value = BaselineInput.from_payload(payload)
            header = ["ore", *value.elements]
            rows = [
                [ore, *[formatting.exact_decimal(x) for x in row]]
                for ore, row in zip(value.ores, value.rows)
            ]
        else:
            value = SamplesInput.from_payload(payload)
            header = ["sample_id", *value.elements]
            rows = [
                [sample_id, *[formatting.exact_decimal(x) for x in row]]
                for sample_id, row in zip(value.sample_ids, value.rows)
            ]
        lines = [",".join(_csv_cell(cell) for cell in header)]
        lines += [",".join(_csv_cell(cell) for cell in row) for row in rows]
        return Response(
            "﻿" + "\r\n".join(lines) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{kind}-{suffix}.csv"'
                )
            },
        )

    @app.get("/inputs/samples/versions/{version_id}/export.csv")
    def export_samples_version(version_id: int) -> Response:
        """A converted samples version, kept after it stops being current."""

        payload = store.get_input_payload(version_id, "samples")
        if payload is None:
            return PlainTextResponse("成品版本不存在", status_code=404)
        return _input_csv_response("samples", payload, f"v{version_id}")

    @app.get("/api/latest")
    def api_latest() -> JSONResponse:
        record = store.latest_report()
        current = store.current_inputs()
        if record is None:
            return JSONResponse(
                {
                    "report": None,
                    "inputs": {
                        kind: (
                            {
                                "version": value.version_id,
                                "filename": value.filename,
                                "created_at": value.created_at,
                            }
                            if value
                            else None
                        )
                        for kind, value in current.items()
                    },
                }
            )
        return JSONResponse(
            {
                "report": {
                    "id": record.report_id,
                    "created_at": record.created_at,
                    "baseline_version": record.baseline_version,
                    "samples_version": record.samples_version,
                    "assay_version": record.assay_version,
                    "mapping_version": record.mapping_version,
                    "stale": record.is_stale,
                    "payload": record.payload,
                }
            }
        )

    return app


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\r", "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _assay_preview(assay: AssayInput, limit: int) -> list[list[str]]:
    return [
        [str(row.line_no), row.sample_id, *row.values]
        for row in assay.rows[:limit]
    ]


class _LabeledError(Exception):
    def __init__(self, label: str, message: str) -> None:
        super().__init__(message)
        self.label = label
        self.message = message


app = create_app()
