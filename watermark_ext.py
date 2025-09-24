"""Administrative watermark management views."""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

from flask import (
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

WM_SUFFIX_NONE = "__none"
WM_SUFFIX_GOTO = "__goto"
WM_SUFFIX_FULLY = "__fullygoto"

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
_ALLOWED_ROLES = {"admin", "shop"}


@dataclass
class BatchSummary:
    processed: int = 0
    skipped: int = 0
    errors: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _media_root() -> Path:
    base = (
        current_app.config.get("MEDIA_ROOT")
        or current_app.config.get("IMAGES_DIR")
        or (Path(current_app.root_path) / "media")
    )
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _watermark_suffix() -> str:
    suffix = current_app.config.get("WATERMARK_SUFFIX")
    if isinstance(suffix, str) and suffix.strip():
        return suffix.strip()
    return DEFAULT_WATERMARK_SUFFIX


def _require_login():
    if session.get("user_id"):
        return None
    login_endpoint = current_app.config.get("LOGIN_ENDPOINT", "login")
    try:
        return redirect(url_for(login_endpoint))
    except Exception:
        abort(403)


def _ensure_role() -> None:
    if session.get("role") not in _ALLOWED_ROLES:
        abort(403)


def _list_existing(order: str, query: str | None) -> List[str]:
    names = list_media_files(include_derivatives=False)
    if query:
        q = query.lower()
        names = [n for n in names if q in n.lower()]
    if order == "name":
        names.sort(key=lambda n: n.lower())
    else:
        def _mtime(name: str) -> float:
            try:
                return media_path_for(name).stat().st_mtime
            except Exception:
                return 0.0
        names.sort(key=_mtime, reverse=True)
    return names


def _available_watermarks() -> List[str]:
    files: List[str] = []
    for path in list_watermark_files():
        try:
            files.append(path.name)
        except Exception:
            continue
    return files


def _normalize_selection(name: str | None) -> str:
    if not name:
        return ""
    cleaned = strip_derivative_suffix(name)
    return cleaned.strip().strip("/")


def _generate_single(base_path: Path, options: WatermarkOptions) -> str:
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    data, ext = apply_watermark(base_path, options)
    out_path = derivative_path(base_path, options.suffix, ext=ext)
    atomic_write(out_path, data)
    return out_path.name


def _folder_counts() -> tuple[list[str], dict[str, int]]:
    root = _media_root()
    folders: list[str] = []
    counts: dict[str, int] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("."):
            continue
        folders.append(name)
        try:
            cnt = sum(
                1
                for child in entry.iterdir()
                if child.is_file() and child.suffix.lower() in _ALLOWED_EXTS
            )
        except OSError:
            cnt = 0
        counts[name] = cnt
    return folders, counts


def _list_files_in_folder(folder: str | None) -> list[SimpleNamespace]:
    root = _media_root()
    rel = (folder or "").strip().strip("/")
    base = (root / rel).resolve() if rel else root
    try:
        ensure_within_media(base)
    except ValueError:
        return []
    if not base.exists() or not base.is_dir():
        return []
    items: list[SimpleNamespace] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _ALLOWED_EXTS:
            continue
        rel_name = str(entry.relative_to(root))
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = 0.0
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else ""
        items.append(
            SimpleNamespace(
                name=entry.name,
                fn=entry.name,
                rel=rel_name,
                ts=ts,
                mtime=mtime,
                v=str(int(mtime)) if mtime else "0",
            )
        )
    items.sort(key=lambda ns: getattr(ns, "mtime", 0.0), reverse=True)
    return items


def _build_gallery(include_derivatives: bool = True) -> list[SimpleNamespace]:
    names = list_media_files(include_derivatives=include_derivatives)
    gallery: list[SimpleNamespace] = []
    for name in names:
        try:
            path = media_path_for(name)
        except ValueError:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else ""
        gallery.append(
            SimpleNamespace(name=name, fn=name, rel=name, ts=ts, mtime=mtime, v=str(int(mtime)) if mtime else "0")
        )
    gallery.sort(key=lambda ns: getattr(ns, "mtime", 0.0), reverse=True)
    return gallery


def view_admin_watermark():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    order = (request.values.get("order") or "new").strip().lower()
    if order not in {"new", "name"}:
        order = "new"
    query = (request.values.get("q") or "").strip()
    picker = 1 if (request.values.get("picker") or "").lower() in {"1", "true", "on"} else 0

    existing = _list_existing(order, query)
    watermark_files = _available_watermarks()

    selected_watermark = request.values.get("watermark_file") or (watermark_files[0] if watermark_files else "")
    scale_value = request.values.get("scale") or f"{DEFAULT_SCALE:.2f}"
    opacity_value = request.values.get("opacity") or f"{DEFAULT_OPACITY:.2f}"

    summary = BatchSummary()
    results: list[dict[str, str]] = []

    if request.method == "POST":
        options = WatermarkOptions.from_request(
            request.form.get("watermark_file") or selected_watermark,
            request.form.get("scale") or scale_value,
            request.form.get("opacity") or opacity_value,
            suffix=_watermark_suffix(),
        )
        selected_watermark = options.watermark_path.name
        scale_value = f"{options.scale:.3f}"
        opacity_value = f"{options.opacity:.2f}"

        seen: set[str] = set()
        targets: list[tuple[str, Path]] = []

        for raw_name in request.form.getlist("selected_existing"):
            base_name = _normalize_selection(raw_name)
            if not base_name:
                summary.skipped += 1
                continue
            try:
                base_path = media_path_for(base_name)
            except ValueError:
                summary.errors.append(f"{raw_name}: 無効なパスです")
                summary.skipped += 1
                continue
            if not base_path.exists():
                summary.errors.append(f"{raw_name}: 元画像が見つかりません")
                summary.skipped += 1
                continue
            if base_name in seen:
                continue
            seen.add(base_name)
            targets.append((base_name, base_path))

        uploads = request.files.getlist("files") or []
        for storage in uploads:
            if not storage or not storage.filename:
                continue
            try:
                payload = validate_upload(storage)
            except ValueError as exc:
                summary.errors.append(f"{storage.filename}: {exc}")
                summary.skipped += 1
                continue
            stem, ext = os.path.splitext(payload.filename)
            try:
                unique = choose_unique_filename(stem, ext, existing=seen)
            except ValueError as exc:
                summary.errors.append(f"{storage.filename}: {exc}")
                summary.skipped += 1
                continue
            seen.add(unique)
            try:
                base_path = media_path_for(unique)
                atomic_write(base_path, payload.data)
            except Exception as exc:
                current_app.logger.exception("failed to store upload: %s", storage.filename)
                summary.errors.append(f"{storage.filename}: 保存に失敗しました")
                summary.skipped += 1
                continue
            targets.append((unique, base_path))

        limit = max_batch_size()
        if len(targets) > limit:
            overflow = len(targets) - limit
            summary.errors.append(f"処理上限 {limit} 件を超えたため {overflow} 件をスキップしました")
            summary.skipped += overflow
            targets = targets[:limit]

        for base_name, base_path in targets:
            try:
                generated = _generate_single(base_path, options)
                results.append(
                    {
                        "src": base_name,
                        "wm": generated,
                        "url_src": url_for("admin_media_img", filename=base_name),
                        "url_wm": url_for("admin_media_img", filename=generated),
                    }
                )
                summary.processed += 1
            except Exception as exc:
                current_app.logger.exception("watermark generation failed: %s", base_path)
                summary.errors.append(f"{base_name}: 生成に失敗しました ({exc})")
                summary.skipped += 1

        if summary.processed or summary.skipped:
            flash(f"生成結果: 成功 {summary.processed} 件 / スキップ {summary.skipped} 件")
        for msg in summary.errors:
            flash(msg)

        existing = _list_existing(order, query)

    return render_template(
        "admin_watermark.html",
        existing=existing,
        results=results,
        wm=selected_watermark,
        q=query,
        order=order,
        picker=picker,
        available_watermarks=watermark_files,
        scale_value=scale_value,
        opacity_value=opacity_value,
        summary=summary,
    )


def view_admin_watermark_one():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    src = _normalize_selection(request.args.get("src"))
    if not src:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))
    try:
        base_path = media_path_for(src)
    except ValueError:
        flash("無効なファイル指定です")
        return redirect(url_for("admin_watermark"))
    if not base_path.exists():
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))

    suffix = _watermark_suffix()
    wm_path = derivative_path(base_path, suffix)
    context = {
        "src": src,
        "wm_name": wm_path.name,
        "wm_exists": wm_path.exists(),
        "watermark_files": _available_watermarks(),
        "suffix": suffix,
        "scale_default": f"{DEFAULT_SCALE:.2f}",
        "opacity_default": f"{DEFAULT_OPACITY:.2f}",
    }
    return render_template("admin_watermark_one.html", **context)


def view_admin_watermark_one_regen():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    src = _normalize_selection(request.form.get("src"))
    if not src:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))
    try:
        base_path = media_path_for(src)
    except ValueError:
        flash("無効なファイル指定です")
        return redirect(url_for("admin_watermark"))
    if not base_path.exists():
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))

    options = WatermarkOptions.from_request(
        request.form.get("watermark_file"),
        request.form.get("scale"),
        request.form.get("opacity"),
        suffix=_watermark_suffix(),
    )
    try:
        _generate_single(base_path, options)
        flash("再生成が完了しました")
    except Exception as exc:
        current_app.logger.exception("single regenerate failed: %s", base_path)
        flash(f"再生成に失敗しました: {exc}")
    return redirect(url_for("admin_watermark_one", src=src))


def view_admin_watermark_edit():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    filename = _normalize_selection(request.args.get("filename"))
    if not filename:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))
    try:
        base_path = media_path_for(filename)
    except ValueError:
        flash("無効なファイル指定です")
        return redirect(url_for("admin_watermark"))
    if not base_path.exists():
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))

    suffix = _watermark_suffix()
    wm_path = derivative_path(base_path, suffix)
    preview = SimpleNamespace(
        src=url_for("admin_media_img", filename=filename),
        wm=url_for("admin_media_img", filename=wm_path.name) if wm_path.exists() else None,
    )
    return render_template(
        "admin_watermark_edit.html",
        filename=filename,
        watermark_files=_available_watermarks(),
        selected_watermark=request.args.get("watermark_file"),
        scale=f"{DEFAULT_SCALE:.2f}",
        opacity=f"{DEFAULT_OPACITY:.2f}",
        preview=preview,
    )


def view_admin_watermark_edit_apply():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    filename = _normalize_selection(request.form.get("filename"))
    if not filename:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))
    try:
        base_path = media_path_for(filename)
    except ValueError:
        flash("無効なファイル指定です")
        return redirect(url_for("admin_watermark"))
    if not base_path.exists():
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))

    options = WatermarkOptions.from_request(
        request.form.get("watermark_file"),
        request.form.get("scale"),
        request.form.get("opacity"),
        suffix=_watermark_suffix(),
    )
    try:
        _generate_single(base_path, options)
        flash("透かしを更新しました")
        return redirect(url_for("admin_watermark_one", src=filename))
    except Exception as exc:
        current_app.logger.exception("edit apply failed: %s", base_path)
        flash(f"更新に失敗しました: {exc}")
        return redirect(url_for("admin_watermark_edit", filename=filename))


def view_admin_media_delete():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    name = _normalize_selection(request.form.get("filename"))
    if not name:
        flash("削除対象が指定されていません")
        return redirect(url_for("admin_watermark"))
    try:
        base_path = media_path_for(name)
    except ValueError:
        abort(400)

    suffixes = {"", _watermark_suffix(), WM_SUFFIX_NONE, WM_SUFFIX_GOTO, WM_SUFFIX_FULLY}
    removed = 0
    extensions = {base_path.suffix.lower(), ".jpg", ".jpeg", ".png"}
    for suf in suffixes:
        stem = base_path.stem
        if suf and stem.endswith(suf):
            stem = stem[: -len(suf)]
        for ext in extensions:
            candidate = base_path.with_name(f"{stem}{suf}{ext}")
            try:
                ensure_within_media(candidate)
            except ValueError:
                continue
            if candidate.exists():
                try:
                    candidate.unlink()
                    removed += 1
                except OSError:
                    current_app.logger.exception("failed to remove %s", candidate)
    flash("削除しました" if removed else "削除対象が見つかりませんでした")
    return redirect(url_for("admin_watermark"))


def view_admin_media_folders():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    folder = (request.args.get("folder") or "").strip().strip("/")
    picker = 1 if (request.args.get("picker") or "").lower() in {"1", "true", "on"} else 0
    entry_id = (request.args.get("entry_id") or "").strip()
    next_url = request.args.get("next") or ""

    folders, counts = _folder_counts()
    images = _list_files_in_folder(folder)
    if not folder and not images and folders:
        return render_template(
            "admin_media_folders.html",
            folders=folders,
            folder_counts=counts,
            folder="",
            images=[],
            picker=picker,
            entry_id=entry_id,
            next=next_url,
        )
    return render_template(
        "admin_media_folders.html",
        folder=folder,
        images=images,
        folders=folders,
        folder_counts=counts,
        picker=picker,
        entry_id=entry_id,
        next=next_url,
    )


def view_admin_media_browse():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    items = _build_gallery(include_derivatives=True)
    return render_template("admin_media_browse.html", folder="全体", items=items, picker=0)


def view_admin_media_pick():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    folder = (request.args.get("folder") or "").strip().strip("/")
    entry_id = (request.args.get("entry_id") or "").strip()
    next_url = request.args.get("next") or ""
    images = _list_files_in_folder(folder)
    return render_template(
        "admin_media_pick.html",
        folder=folder or "",
        images=images,
        entry_id=entry_id,
        next=next_url,
    )


def view_admin_media_picker():
    need = _require_login()
    if need:
        return need
    _ensure_role()

    folder = (request.args.get("folder") or "").strip().strip("/")
    return_to = request.args.get("return_to") or url_for("admin_watermark")
    folders, counts = _folder_counts()
    files = _list_files_in_folder(folder)
    folder_objs = [SimpleNamespace(name=name, count=counts.get(name, 0)) for name in folders]
    return render_template(
        "admin_media_picker.html",
        folder=folder or "",
        folders=folder_objs,
        files=files,
        return_to=return_to,
    )


def init_watermark_ext(app) -> None:
    app.view_functions["admin_watermark"] = view_admin_watermark
    app.view_functions["admin_media_delete"] = view_admin_media_delete
    app.add_url_rule(
        "/admin/media/delete",
        endpoint="admin_media_delete",
        view_func=view_admin_media_delete,
        methods=["POST"],
    )
    app.add_url_rule(
        "/admin_media_delete",
        endpoint="admin_media_delete_short",
        view_func=view_admin_media_delete,
        methods=["POST"],
    )
    app.add_url_rule("/admin_watermark", endpoint="admin_watermark_alias", view_func=view_admin_watermark, methods=["GET", "POST"])

    app.add_url_rule("/admin/watermark/one", endpoint="admin_watermark_one", view_func=view_admin_watermark_one, methods=["GET"])
    app.add_url_rule("/admin_watermark_one", endpoint="admin_watermark_one_short", view_func=view_admin_watermark_one, methods=["GET"])

    app.add_url_rule("/admin/watermark/one/regenerate", endpoint="admin_watermark_one_regen", view_func=view_admin_watermark_one_regen, methods=["POST"])
    app.add_url_rule("/admin_watermark_one_regen", endpoint="admin_watermark_one_regen_short", view_func=view_admin_watermark_one_regen, methods=["POST"])

    app.add_url_rule("/admin_watermark_edit", endpoint="admin_watermark_edit", view_func=view_admin_watermark_edit, methods=["GET"])
    app.add_url_rule("/admin_watermark_edit/apply", endpoint="admin_watermark_edit_apply", view_func=view_admin_watermark_edit_apply, methods=["POST"])

    app.add_url_rule("/admin/media/folders", endpoint="admin_media_folders", view_func=view_admin_media_folders, methods=["GET"])
    app.add_url_rule("/admin/media/browse", endpoint="admin_media_browse", view_func=view_admin_media_browse, methods=["GET"])
    app.add_url_rule("/admin/media/pick", endpoint="admin_media_pick", view_func=view_admin_media_pick, methods=["GET"])
    app.add_url_rule("/admin/media/picker", endpoint="admin_media_picker", view_func=view_admin_media_picker, methods=["GET"])
# --- compatibility: provide media_path_for expected by app ---
from pathlib import Path
import os

def media_path_for(filename: str, folder: str | None = None) -> str:
    """
    Resolve an absolute path under the configured media directory.
    - Uses app.config['MEDIA_ROOT'] or ['MEDIA_DIR'] or 'media' as fallback.
    - Guards against path traversal ('..', absolute paths).
    """
    base = Path(
        current_app.config.get("MEDIA_ROOT")
        or current_app.config.get("MEDIA_DIR")
        or "media"
    )
    if folder:
        base = base / folder

    safe = os.path.normpath(str(filename)).replace("\\", "/")
    # prevent traversal / absolute
    if safe.startswith("../") or safe.startswith("/"):
        safe = os.path.basename(safe)

    return str(base / safe)
