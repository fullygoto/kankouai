"""Admin blueprint for managing pamphlet text files by city."""

from __future__ import annotations

from datetime import datetime
from functools import wraps
from typing import Callable

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
from services import pamphlet_store


bp = Blueprint("pamphlets_admin", __name__, url_prefix="/admin")

PREVIEW_MAX_BYTES = 200_000
MAX_EDIT_BYTES = 2 * 1024 * 1024


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
    return render_template(
        "admin/pamphlets.html",
        cities=PAMPHLET_CITIES,
        city=city,
        files=files,
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
        too_large = size > MAX_EDIT_BYTES
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
        max_edit_bytes=MAX_EDIT_BYTES,
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

    if len(content.encode("utf-8")) > MAX_EDIT_BYTES:
        flash("ファイルが大きすぎます。2MBまで編集できます。", "danger")
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
            too_large = size > MAX_EDIT_BYTES
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
            max_edit_bytes=MAX_EDIT_BYTES,
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
