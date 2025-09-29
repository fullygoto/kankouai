"""Admin blueprint for managing pamphlet text files by city."""

from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable

import shutil

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from config import PAMPHLET_CITIES
from services import input_normalizer, pamphlet_rag, pamphlet_store


bp = Blueprint("pamphlets_admin", __name__, url_prefix="/admin")

PREVIEW_MAX_BYTES = 200_000
DEFAULT_EDIT_MAX_BYTES = 2 * 1024 * 1024


def _get_edit_limit() -> int:
    return current_app.config.get("PAMPHLET_EDIT_MAX_BYTES", DEFAULT_EDIT_MAX_BYTES)


def _admin_required(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            abort(403)
        return func(*args, **kwargs)

    return wrapper


@bp.record_once
def _init_dirs(state) -> None:  # type: ignore[override]
    pamphlet_store.ensure_dirs()


@bp.get("/pamphlets")
@_admin_required
def pamphlets_index():
    city = request.args.get("city", "goto")
    if city not in PAMPHLET_CITIES:
        city = next(iter(PAMPHLET_CITIES.keys()))
    try:
        files = pamphlet_store.list_files(city)
    except Exception as exc:
        flash(f"一覧取得に失敗しました: {exc}", "danger")
        files = []

    base_dir = Path(current_app.config.get("DATA_BASE_DIR", "/var/data"))
    pamphlet_root = Path(pamphlet_store.BASE).resolve()
    try:
        usage = shutil.disk_usage(base_dir)
        disk_info = {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }
    except (FileNotFoundError, PermissionError, OSError):
        disk_info = None

    def _format_bytes(num: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(num)
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{num} B"

    storage_info = {
        "base": str(base_dir),
        "pamphlets": str(pamphlet_root),
        "entries": str(Path(current_app.config.get("ENTRIES_DIR", base_dir / "entries"))),
        "uploads": str(Path(current_app.config.get("UPLOADS_DIR", base_dir / "uploads"))),
        "images": str(Path(current_app.config.get("IMAGES_DIR", base_dir / "images"))),
        "disk": disk_info,
        "disk_fmt": {k: _format_bytes(v) for k, v in (disk_info or {}).items()},
    }

    return render_template(
        "admin/pamphlets.html",
        cities=PAMPHLET_CITIES,
        city=city,
        files=files,
        storage_info=storage_info,
    )


@bp.get("/pamphlets/view")
@_admin_required
def pamphlets_view():
    city = request.args.get("city", "goto")
    name = request.args.get("name", "")
    try:
        text, size, mtime = pamphlet_store.read_text(
            city, name, max_bytes=PREVIEW_MAX_BYTES
        )
        truncated = PREVIEW_MAX_BYTES is not None and size > PREVIEW_MAX_BYTES
        return jsonify(
            {
                "name": name,
                "size": size,
                "mtime": mtime,
                "text": text,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        response = jsonify({"error": str(exc)})
        response.status_code = 400
        return response


@bp.get("/pamphlets/edit")
@_admin_required
def pamphlets_edit():
    city = request.args.get("city", "goto")
    name = request.args.get("name", "")
    try:
        info = pamphlet_store.stat_file(city, name)
        size = int(info["size"])
        mtime = float(info["mtime"])
        edit_limit = _get_edit_limit()
        too_large = size > edit_limit
        text = None
        if not too_large:
            text, _size, mtime = pamphlet_store.read_text(city, name)
    except Exception as exc:
        flash(f"読み込みに失敗しました: {exc}", "danger")
        return redirect(url_for("pamphlets_admin.pamphlets_index", city=city))

    return render_template(
        "admin/pamphlets_edit.html",
        cities=PAMPHLET_CITIES,
        city=city,
        name=name,
        size=size,
        mtime=datetime.fromtimestamp(mtime),
        expected_mtime=mtime,
        text=text,
        too_large=too_large,
    )


@bp.post("/pamphlets/save")
@_admin_required
def pamphlets_save():
    city = request.form.get("city", "goto")
    name = request.form.get("name", "")
    expected_mtime_str = request.form.get("expected_mtime")
    content = request.form.get("content", "")

    try:
        expected_mtime = float(expected_mtime_str) if expected_mtime_str else None
    except (TypeError, ValueError):
        expected_mtime = None

    edit_limit = _get_edit_limit()
    byte_len = len(content.encode("utf-8"))
    current_app.logger.debug(
        "Pamphlet edit size check city=%s name=%s bytes=%s limit=%s",
        city,
        name,
        byte_len,
        edit_limit,
    )
    if byte_len > edit_limit:
        flash(
            "編集内容が大きすぎます（{:.1f}KB > 上限 {:.0f}KB）".format(
                byte_len / 1024, edit_limit / 1024
            ),
            "error",
        )
        return redirect(url_for("pamphlets_admin.pamphlets_edit", city=city, name=name))

    try:
        pamphlet_store.write_text(
            city,
            name,
            content,
            expected_mtime=expected_mtime,
        )
        flash("保存しました。", "success")
        return redirect(url_for("pamphlets_admin.pamphlets_edit", city=city, name=name))
    except ValueError:
        flash("他の変更で競合しました。最新の内容を確認してください。", "warning")
        try:
            info = pamphlet_store.stat_file(city, name)
            size = int(info["size"])
            mtime = float(info["mtime"])
            edit_limit = _get_edit_limit()
            too_large = size > edit_limit
            text = None
            if not too_large:
                text, _size, mtime = pamphlet_store.read_text(city, name)
        except Exception as exc:
            flash(f"最新の内容取得に失敗しました: {exc}", "danger")
            return redirect(url_for("pamphlets_admin.pamphlets_index", city=city))
        return render_template(
            "admin/pamphlets_edit.html",
            cities=PAMPHLET_CITIES,
            city=city,
            name=name,
            size=size,
            mtime=datetime.fromtimestamp(mtime),
            expected_mtime=mtime,
            text=text,
            too_large=too_large,
        )
    except Exception as exc:
        flash(f"保存に失敗しました: {exc}", "danger")
        return redirect(url_for("pamphlets_admin.pamphlets_edit", city=city, name=name))


@bp.get("/pamphlets/download")
@_admin_required
def pamphlets_download():
    city = request.args.get("city", "goto")
    name = request.args.get("name", "")
    try:
        path = pamphlet_store.get_file_path(city, name)
        return send_file(path, as_attachment=True, download_name=name)
    except Exception as exc:
        flash(f"ダウンロードに失敗しました: {exc}", "danger")
        return redirect(url_for("pamphlets_admin.pamphlets_index", city=city))


@bp.post("/pamphlets/upload")
@_admin_required
def pamphlets_upload():
    city = request.form.get("city", "goto")
    file_obj = request.files.get("file")
    try:
        pamphlet_store.save_file(city, file_obj)
        flash("アップロードしました。", "success")
    except Exception as exc:
        flash(f"アップロード失敗: {exc}", "danger")
    return redirect(url_for("pamphlets_admin.pamphlets_index", city=city))


@bp.post("/pamphlets/delete")
@_admin_required
def pamphlets_delete():
    city = request.form.get("city", "goto")
    name = request.form.get("name", "")
    try:
        pamphlet_store.delete_file(city, name)
        flash("削除しました。", "success")
    except Exception as exc:
        flash(f"削除に失敗しました: {exc}", "danger")
    return redirect(url_for("pamphlets_admin.pamphlets_index", city=city))


@bp.post("/pamphlets/reindex")
@_admin_required
def pamphlets_reindex():
    city = request.form.get("city", "goto")
    try:
        view = current_app.view_functions.get("admin_pamphlet_reindex")
        if view is None:
            raise RuntimeError("再インデックスAPIが見つかりません。")
        response = view()
        data = getattr(response, "get_json", lambda **_: None)(silent=True) or {}
        state = data.get("pamphlet_index")
        result = data.get("result") or {}
        details = ", ".join(
            f"{PAMPHLET_CITIES.get(key, key)}: {info.get('state', 'unknown')}"
            for key, info in result.items()
        )
        msg = "再インデックスを実行しました。"
        if state:
            msg += f"（全体状態: {state}）"
        if details:
            msg += f" 詳細: {details}"
        flash(msg, "info")
    except Exception as exc:
        flash(f"再インデックス呼出し失敗: {exc}", "danger")
    return redirect(url_for("pamphlets_admin.pamphlets_index", city=city))
@bp.get("/pamphlets/debug")
@_admin_required
def pamphlets_debug():
    city = request.args.get("city", "goto")
    question = input_normalizer.normalize_user_query(request.args.get("q", ""))
    if city not in PAMPHLET_CITIES:
        response = jsonify({"error": "unknown city"})
        response.status_code = 400
        return response
    try:
        result = pamphlet_rag.answer_from_pamphlets(question, city)
    except Exception as exc:  # pragma: no cover - defensive
        response = jsonify({"error": str(exc)})
        response.status_code = 500
        return response

    debug = result.get("debug", {}) or {}
    payload = {
        "question": question,
        "city": city,
        "city_label": PAMPHLET_CITIES.get(city, city),
        "queries": debug.get("queries"),
        "bm25": debug.get("bm25"),
        "embedding": debug.get("embedding"),
        "combined": debug.get("combined"),
        "selection": debug.get("selection"),
        "confidence": result.get("confidence"),
        "prompt": debug.get("prompt"),
        "answer": result.get("answer"),
        "sources": result.get("sources"),
    }
    return jsonify(payload)


