"""Admin blueprint for managing pamphlet text files by city."""

from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import PAMPHLET_CITIES
from services import pamphlet_store


bp = Blueprint("pamphlets_admin", __name__, url_prefix="/admin")


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
