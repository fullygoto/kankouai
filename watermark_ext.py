"""Administrative watermark management views."""
from __future__ import annotations

import datetime
import io
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

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

# ============================================================
# Constants / defaults
# ============================================================

WM_SUFFIX_NONE = "__none"
WM_SUFFIX_GOTO = "__goto"
WM_SUFFIX_FULLY = "__fullygoto"

# 既存コードが参照するデフォルト値（無ければここを使う）
DEFAULT_SCALE: float = 0.33
DEFAULT_OPACITY: float = 0.5
DEFAULT_WATERMARK_SUFFIX: str = "__wm"

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_ALLOWED_ROLES = {"admin", "shop"}

# 一括生成の安全上限
DEFAULT_BATCH_LIMIT = 100


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class BatchSummary:
    processed: int = 0
    skipped: int = 0
    errors: List[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass
class WatermarkOptions:
    watermark_path: Path
    scale: float
    opacity: float
    suffix: str

    @classmethod
    def from_request(
        cls,
        wm_name: Optional[str],
        scale: Optional[str],
        opacity: Optional[str],
        suffix: Optional[str] = None,
    ) -> "WatermarkOptions":
        # 透明度・スケール
        try:
            s = float(scale) if scale is not None else DEFAULT_SCALE
        except Exception:
            s = DEFAULT_SCALE
        if s <= 0:
            s = DEFAULT_SCALE

        try:
            op = float(opacity) if opacity is not None else DEFAULT_OPACITY
        except Exception:
            op = DEFAULT_OPACITY
        if op < 0:
            op = 0.0
        if op > 1:
            op = 1.0

        # 透かし素材の解決
        wm_path = _resolve_watermark_path(wm_name)
        if wm_path is None:
            raise ValueError("透かし画像が見つかりません")

        # サフィックス（拡張子前に付くラベル）
        suf = (suffix or current_app.config.get("WATERMARK_SUFFIX") or DEFAULT_WATERMARK_SUFFIX).strip()
        if not suf.startswith("__"):
            suf = "__" + suf

        return cls(watermark_path=wm_path, scale=s, opacity=op, suffix=suf)


# ============================================================
# Internal helpers (paths, scanning, safety)
# ============================================================

def _media_root() -> Path:
    base = (
        current_app.config.get("MEDIA_ROOT")
        or current_app.config.get("IMAGES_DIR")
        or (Path(current_app.root_path) / "media" / "img")
    )
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def ensure_within_media(p: Path) -> None:
    """Mediaルート外の書き込みを防止。外なら ValueError。"""
    root = _media_root()
    rp = p.resolve()
    if root not in rp.parents and rp != root:
        raise ValueError("path is outside MEDIA root")


def strip_derivative_suffix(name: str) -> str:
    """拡張子前の '__xxx' を除去（派生ファイル→元名を推定）"""
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    if "__" in stem:
        stem = stem[: stem.index("__")]
    return f"{stem}{ext}"


def derivative_path(base_path: Path, suffix: str, ext: Optional[str] = None) -> Path:
    """拡張子前に suffix を付けた出力パスを返す。"""
    if not suffix.startswith("__"):
        suffix = "__" + suffix
    use_ext = (ext or base_path.suffix) or ".jpg"
    stem = strip_derivative_suffix(base_path.name)
    stem_no_ext, _ = os.path.splitext(stem)
    return base_path.with_name(f"{stem_no_ext}{suffix}{use_ext}")


def max_batch_size() -> int:
    return int(current_app.config.get("WATERMARK_BATCH_LIMIT") or DEFAULT_BATCH_LIMIT)


def media_path_for(filename: str, folder: str | None = None) -> Path:
    """
    Resolve a path under MEDIA root safely.
    NOTE: Path traversal ('..' / absolute) をガード。
    """
    base = _media_root()
    if folder:
        base = base / folder

    safe = os.path.normpath(str(filename)).replace("\\", "/")
    if safe.startswith("../") or safe.startswith("/"):
        safe = os.path.basename(safe)

    p = (base / safe).resolve()
    ensure_within_media(p)
    return p


def list_media_files(include_derivatives: bool = True) -> List[str]:
    """media配下の画像ファイル一覧（相対パス）。"""
    root = _media_root()
    out: List[str] = []
    if not root.exists():
        return out

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _ALLOWED_EXTS:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            rel = rel.replace("\\", "/")
            if not include_derivatives:
                stem = os.path.splitext(os.path.basename(rel))[0]
                if "__" in stem:
                    # 派生（__suffixあり）は除く
                    continue
            out.append(rel)
    out.sort()
    return out


def list_watermark_files() -> List[Path]:
    """
    透かし素材として使う画像（PNG/WEBP推奨）候補の絶対パス一覧。
    検索パス:
      - config WATERMARK_DIR
      - media/_watermarks, media/watermark
    """
    candidates: List[Path] = []
    cfg = current_app.config.get("WATERMARK_DIR")
    if cfg:
        candidates.append(Path(str(cfg)))
    media_root = _media_root()
    candidates.append(media_root.parent / "_watermarks")
    candidates.append(media_root.parent / "watermark")
    candidates.append(media_root / "_watermarks")  # 念のため

    seen: set[str] = set()
    out: List[Path] = []
    exts = {".png", ".webp", ".jpg", ".jpeg"}

    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        for name in sorted(os.listdir(d)):
            if os.path.splitext(name)[1].lower() in exts:
                p = (d / name).resolve()
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    out.append(p)
    return out


def _resolve_watermark_path(name_or_path: Optional[str]) -> Optional[Path]:
    if not name_or_path:
        files = list_watermark_files()
        return files[0] if files else None

    p = Path(name_or_path)
    if p.exists():
        return p.resolve()

    # 名前一致で候補から探す
    for cand in list_watermark_files():
        if cand.name == name_or_path:
            return cand
    return None


def choose_unique_filename(stem: str, ext: str, existing: Iterable[str] | None = None) -> str:
    """同名があれば '-1', '-2'…でユニーク化。"""
    root = _media_root()
    e = set(existing or ())
    i = 0
    while True:
        suffix = "" if i == 0 else f"-{i}"
        candidate = f"{stem}{suffix}{ext}"
        if candidate not in e and not (root / candidate).exists():
            return candidate
        i += 1


def atomic_write(path: Path, data: bytes) -> None:
    """テンポラリ→rename による原子的書き込み。"""
    ensure_within_media(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_upload(storage) -> SimpleNamespace:
    """
    Werkzeug FileStorage を検証して bytes を返す。
    - 拡張子チェック
    - サイズ（任意。必要なら config で制御）
    """
    filename = (storage.filename or "").strip()
    if not filename:
        raise ValueError("ファイル名が空です")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_EXTS:
        raise ValueError("対応していない拡張子です")
    data = storage.read()
    if not data:
        raise ValueError("空のファイルです")
    return SimpleNamespace(filename=filename, data=data)


# ============================================================
# Watermark processing (Pillow)
# ============================================================

from PIL import Image, ImageOps

def apply_watermark(base_path: Path, opt: WatermarkOptions) -> Tuple[bytes, str]:
    """
    ベース画像の右下に透かしを合成。
    - サイズ: 画像幅 * opt.scale
    - 不透明度: opt.opacity (0..1)
    返り値: (生成バイナリ, 拡張子)
    """
    # open base
    with Image.open(base_path) as im:
        im = im.convert("RGBA")

        # open watermark
        with Image.open(opt.watermark_path) as wm:
            wm = wm.convert("RGBA")

            # scale
            target_w = max(1, int(im.width * opt.scale))
            aspect = wm.height / wm.width if wm.width else 1.0
            target_h = max(1, int(target_w * aspect))
            wm = wm.resize((target_w, target_h), Image.LANCZOS)

            # opacity
            if opt.opacity < 1.0:
                alpha = wm.split()[-1]
                alpha = ImageOps.autocontrast(alpha)
                alpha = alpha.point(lambda a: int(a * float(opt.opacity)))
                wm.putalpha(alpha)

            # paste bottom-right with margin
            margin = max(8, target_w // 16)
            x = im.width - wm.width - margin
            y = im.height - wm.height - margin

            composite = Image.new("RGBA", im.size)
            composite.paste(im, (0, 0))
            composite.paste(wm, (x, y), mask=wm)

            # format selection
            base_ext = base_path.suffix.lower()
            if base_ext in {".jpg", ".jpeg"}:
                out = composite.convert("RGB")
                ext = ".jpg"
                buf = io.BytesIO()
                out.save(buf, format="JPEG", quality=90, optimize=True)
                return buf.getvalue(), ext
            elif base_ext == ".png":
                ext = ".png"
                buf = io.BytesIO()
                composite.save(buf, format="PNG", optimize=True)
                return buf.getvalue(), ext
            else:
                # webpなどはPNGに寄せる
                ext = ".png"
                buf = io.BytesIO()
                composite.save(buf, format="PNG", optimize=True)
                return buf.getvalue(), ext


# ============================================================
# View helpers
# ============================================================

def _watermark_suffix() -> str:
    suffix = current_app.config.get("WATERMARK_SUFFIX")
    if isinstance(suffix, str) and suffix.strip():
        suf = suffix.strip()
    else:
        suf = DEFAULT_WATERMARK_SUFFIX
    if not suf.startswith("__"):
        suf = "__" + suf
    return suf


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
        rel_name = str(entry.relative_to(root)).replace("\\", "/")
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


# ============================================================
# Views
# ============================================================

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
            except Exception:
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
    extensions = {base_path.suffix.lower(), ".jpg", ".jpeg", ".png", ".webp"}
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


# ============================================================
# URL wiring
# ============================================================

def init_watermark_ext(app) -> None:
    # 互換 alias も含めて登録
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
