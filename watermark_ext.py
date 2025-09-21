# -*- coding: utf-8 -*-
"""
watermark_ext.py (rev7)
- /admin/watermark: ピッカー + 並び替え + subdir（フォルダ）対応
- /admin/media/delete: サブフォルダ対応 & 削除後に元の表示条件へ戻る
- /admin/watermark/one: 単体編集
- /admin/watermark/one/regenerate: 単体の再生成
- /admin/media/folders: フォルダ一覧 or 画像一覧（フォルダ無しでもルート直下を表示）
"""

from __future__ import annotations
import os, importlib, datetime
from types import SimpleNamespace
from typing import List, Dict, Iterable, Tuple, Optional
from flask import current_app, render_template, request, session, abort, flash, redirect, url_for

# 透かし派生のサフィックス（app.py 側で上書き可）
WM_SUFFIX_NONE  = "__none"
WM_SUFFIX_GOTO  = "__goto"
WM_SUFFIX_FULLY = "__fullygoto"

_IMG_EXTS = (".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff")

# ---------------------------
# 共通
# ---------------------------
def _get_images_dir() -> str:
    try:
        appmod = importlib.import_module("app")
        v = getattr(appmod, "IMAGES_DIR", None)
        if v:
            return v
    except Exception:
        pass
    app = current_app
    return (
        app.config.get("IMAGES_DIR")
        or app.config.get("MEDIA_DIR")
        or app.config.get("UPLOAD_FOLDER")
        or os.path.join(app.root_path, "media")
    )

def _get_suffixes() -> Tuple[str, str, str]:
    s_none, s_goto, s_fully = WM_SUFFIX_NONE, WM_SUFFIX_GOTO, WM_SUFFIX_FULLY
    try:
        appmod = importlib.import_module("app")
        s_none  = getattr(appmod, "WM_SUFFIX_NONE",  s_none)
        s_goto  = getattr(appmod, "WM_SUFFIX_GOTO",  s_goto)
        s_fully = getattr(appmod, "WM_SUFFIX_FULLY", s_fully)
    except Exception:
        pass
    return s_none, s_goto, s_fully

def _boolish(x) -> bool:
    return str(x).lower() in {"1","true","on","yes"}

def _secure_subdir(subdir: Optional[str]) -> str:
    subdir = (subdir or "").strip().strip("/")
    if not subdir:
        return ""
    if subdir.startswith(".") or ".." in subdir or subdir.startswith("/"):
        return ""
    return subdir

def _require_login():
    if session.get("user_id"):
        return None
    try:
        return redirect(url_for("login"))
    except Exception:
        abort(403)

def _fmt_ts(ts: float) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""

# ---------------------------
# 列挙
# ---------------------------
def _is_src_image(filename: str) -> bool:
    if not filename or filename.startswith("."):
        return False
    root, ext = os.path.splitext(filename)
    if ext.lower() not in _IMG_EXTS:
        return False
    s_none, s_goto, s_fully = _get_suffixes()
    return not any(suf in root for suf in (s_none, s_goto, s_fully))

def _list_folders() -> Tuple[List[str], Dict[str,int]]:
    base = _get_images_dir()
    folders: List[str] = []
    counts: Dict[str,int] = {}
    if not os.path.isdir(base):
        return folders, counts
    for name in os.listdir(base):
        if name.startswith("."): continue
        p = os.path.join(base, name)
        if os.path.isdir(p):
            folders.append(name)
            c = 0
            try:
                for fn in os.listdir(p):
                    if _is_src_image(fn):
                        c += 1
            except Exception:
                c = 0
            counts[name] = c
    folders.sort()
    return folders, counts

def _list_source_images(order: str = "name", subdir: Optional[str] = None) -> List[str]:
    images_dir = _get_images_dir()
    rel = _secure_subdir(subdir)
    base = os.path.join(images_dir, rel) if rel else images_dir
    files: List[str] = []
    if not os.path.isdir(base):
        return files
    for fn in os.listdir(base):
        if _is_src_image(fn):
            files.append(f"{rel}/{fn}" if rel else fn)
    if (order or "").lower() == "new":
        def _mtime(relfn: str) -> float:
            try:    return os.path.getmtime(os.path.join(images_dir, relfn))
            except: return 0.0
        files.sort(key=_mtime, reverse=True)
    else:
        files.sort()
    return files

def _list_all_images_in_folder(subdir: str, order: str = "new") -> List[Dict[str, str]]:
    """フォルダ内の“全画像”（派生も含む）。subdir='' なら IMAGES_DIR 直下。"""
    images_dir = _get_images_dir()
    rel = _secure_subdir(subdir)
    base = os.path.join(images_dir, rel) if rel else images_dir
    out: List[Dict[str,str]] = []
    if not os.path.isdir(base):
        return out

    for fn in os.listdir(base):
        if fn.startswith("."): continue
        root, ext = os.path.splitext(fn)
        if ext.lower() not in _IMG_EXTS: continue
        path = os.path.join(base, fn)
        try:
            mt = os.path.getmtime(path)
        except Exception:
            mt = 0.0
        v = int(mt) if mt else 0
        relfn = f"{rel}/{fn}" if rel else fn
        out.append({
            "name": fn,
            "rel": relfn,                   # ← 相対パス（folder 無し判定用にも使う）
            "ts": _fmt_ts(mt),
            "v": str(v),
        })

    # 新しい順
    if (order or "").lower() == "new":
        out.sort(key=lambda d: int(d.get("v","0")), reverse=True)
    else:
        out.sort(key=lambda d: d["name"])
    return out

# ---------------------------
# 透かし生成（内部委譲）
# ---------------------------
def _generate_variants(abs_path: str, *, force: bool, kinds: Iterable[str]) -> Dict[str, str]:
    appmod = importlib.import_module("app")
    fn_sel = getattr(appmod, "_wm_variants_for_path_selected", None)
    if callable(fn_sel):
        return fn_sel(abs_path, force=force, kinds=list(kinds))
    fn = getattr(appmod, "_wm_variants_for_path", None)
    if not callable(fn):
        raise RuntimeError("_wm_variants_for_path が見つかりません")
    return fn(abs_path, force=force)

# ---------------------------
# /admin/watermark （一覧・ピッカー）
# ---------------------------
def view_admin_watermark():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin","shop"}:
        abort(403)

    wm     = (request.values.get("wm") or "all").strip()
    if wm not in {"all","none","goto","fully"}:
        wm = "all"
    make_none  = wm in {"all","none"}
    make_goto  = wm in {"all","goto"}
    make_fully = wm in {"all","fully"}

    q       = (request.values.get("q") or "").strip().lower()
    order   = (request.values.get("order") or "new").strip().lower()
    picker  = 1 if request.values.get("picker") in {"1","true","on"} else 0
    subdir  = _secure_subdir(request.values.get("subdir"))

    existing = _list_source_images(order=order, subdir=subdir)
    if q:
        existing = [fn for fn in existing if q in fn.lower()]

    results, errors = [], []
    images_dir = _get_images_dir()

    if request.method == "POST":
        force = request.form.get("force") == "1"
        selected = request.form.getlist("selected_existing")
        uploads  = request.files.getlist("files") or []

        targets: List[Tuple[str,str]] = []
        for name in selected:
            name = name.strip("/")
            abs_path = os.path.join(images_dir, name)
            targets.append((name, abs_path))

        appmod = importlib.import_module("app")
        save_fn = getattr(appmod, "_save_jpeg_1080_350kb", None)

        for f in uploads:
            if not f or not f.filename:
                continue
            try:
                if callable(save_fn):
                    saved = save_fn(f, previous=None, delete=False)
                else:
                    fname = os.path.basename(f.filename)
                    out_path = os.path.join(images_dir, fname)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    f.save(out_path)
                    saved = fname
                targets.append((saved, os.path.join(images_dir, saved)))
            except Exception as e:
                errors.append(f"{f.filename}: {e}")

        kinds = []
        if make_none:  kinds.append("none")
        if make_goto:  kinds.append("goto")
        if make_fully: kinds.append("fully")

        for display_rel, abs_path in targets:
            try:
                out = _generate_variants(abs_path, force=force, kinds=kinds)
                r = {"src": display_rel}
                rel_dir = os.path.dirname(display_rel)
                r["url_src"] = url_for("admin_media_img", filename=display_rel)
                def _relfix(x: Optional[str]) -> Optional[str]:
                    if not x: return None
                    if "/" in x or "\\" in x or not rel_dir: return x
                    return f"{rel_dir}/{x}"
                if "none" in kinds and out.get("none"):
                    r["url_none"] = url_for("admin_media_img", filename=_relfix(out["none"]))
                if "goto" in kinds and out.get("goto"):
                    r["url_goto"] = url_for("admin_media_img", filename=_relfix(out["goto"]))
                if "fully" in kinds and out.get("fully"):
                    r["url_fully"] = url_for("admin_media_img", filename=_relfix(out["fully"]))
                results.append(SimpleNamespace(**r))
            except Exception as e:
                current_app.logger.exception("watermark gen failed: %s", abs_path)
                errors.append(f"{display_rel}: 生成に失敗しました（{e}）")

        for m in errors:
            flash(m)
        if results:
            flash(f"{len(results)} 件を処理しました" + ("（上書き）" if force else ""))

    return render_template("admin_watermark.html",
                           wm=wm, q=q, existing=existing, results=results,
                           order=order, picker=picker, subdir=subdir)

# ---------------------------
# /admin/media/delete （削除）
# ---------------------------
def view_admin_media_delete():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin","shop"}:
        abort(403)

    name   = (request.form.get("filename") or "").strip().strip("/")
    order  = (request.form.get("order") or "").strip().lower()
    picker = (request.form.get("picker") or "").strip()
    ret_q  = (request.form.get("q") or "").strip()
    ret_subdir = _secure_subdir(request.form.get("subdir"))

    if not name:
        flash("削除対象が指定されていません")
        return redirect(url_for("admin_watermark",
                                order=(order if order in {"new","name"} else None),
                                picker=("1" if _boolish(picker) else None),
                                subdir=(ret_subdir or None),
                                q=(ret_q or None)))

    images_dir = _get_images_dir()
    s_none, s_goto, s_fully = _get_suffixes()

    rel_dir = os.path.dirname(name)
    stem, ext = os.path.splitext(os.path.basename(name))
    base_dir = os.path.join(images_dir, rel_dir) if rel_dir else images_dir

    targets = [
        os.path.join(base_dir, f"{stem}{ext or '.jpg'}"),
        os.path.join(base_dir, f"{stem}{s_none}.jpg"),
        os.path.join(base_dir, f"{stem}{s_goto}.jpg"),
        os.path.join(base_dir, f"{stem}{s_fully}.jpg"),
    ]

    removed = 0
    base_real = os.path.realpath(images_dir)
    for p in targets:
        try:
            real = os.path.realpath(p)
            if os.path.commonpath([base_real, real]) != base_real:
                continue
            if os.path.exists(real):
                os.remove(real)
                removed += 1
        except Exception:
            current_app.logger.exception("failed to remove: %s", p)

    flash("削除しました" if removed else "削除対象が見つかりませんでした")

    params = {}
    if order in {"new","name"}: params["order"] = order
    if _boolish(picker):        params["picker"] = "1"
    if ret_subdir:              params["subdir"] = ret_subdir
    params["q"] = ret_q if ret_q else stem

    return redirect(url_for("admin_watermark", **params))

# ---------------------------
# /admin/watermark/one （表示）
# ---------------------------
def view_admin_watermark_one():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin","shop"}:
        abort(403)
    src = (request.args.get("src") or "").strip().strip("/")
    if not src:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))
    images_dir = _get_images_dir()
    abs_path = os.path.join(images_dir, src)
    if not os.path.isfile(abs_path):
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))
    return render_template("admin_watermark_one.html", src=src)

# ---------------------------
# /admin/watermark/one/regenerate （単体の再生成）
# ---------------------------
def view_admin_watermark_one_regen():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin","shop"}:
        abort(403)

    src = (request.form.get("src") or "").strip().strip("/")
    if not src:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))

    images_dir = _get_images_dir()
    abs_path = os.path.join(images_dir, src)
    if not os.path.isfile(abs_path):
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))

    wm = (request.form.get("wm") or "all").strip()
    make_none  = wm in {"all","none"}
    make_goto  = wm in {"all","goto"}
    make_fully = wm in {"all","fully"}

    kinds: List[str] = []
    if make_none:  kinds.append("none")
    if make_goto:  kinds.append("goto")
    if make_fully: kinds.append("fully")

    force = request.form.get("force") == "1"

    try:
        _generate_variants(abs_path, force=force, kinds=kinds)
        flash("再生成しました" + ("（上書き）" if force else ""))
    except Exception as e:
        current_app.logger.exception("regen failed for %s", abs_path)
        flash(f"生成に失敗しました: {e}")

    return redirect(url_for("admin_watermark_one", src=src))

# ---------------------------
# /admin/media/folders （フォルダ一覧 or 画像一覧）
# ---------------------------
def view_admin_media_folders():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin","shop"}:
        abort(403)

    entry_id = (request.args.get("entry_id") or "").strip()
    next_url = (request.args.get("next") or "").strip()
    folder   = _secure_subdir(request.args.get("folder"))
    order    = (request.args.get("order") or "new").strip().lower()

    folders, counts = _list_folders()

    # 1) folder=指定 → そのフォルダの画像一覧を表示
    # 2) フォルダが一つも無い → IMAGES_DIR 直下の画像一覧を表示
    if folder or not folders:
        images = _list_all_images_in_folder(folder or "", order=order)
        # サムネ用 URL（キャッシュバスター v 付き）はテンプレ側で組み立て
        return render_template("admin_media_folders.html",
                               folder=(folder or ""), images=images,
                               entry_id=entry_id, next=next_url)

    # それ以外はフォルダ一覧を表示
    return render_template("admin_media_folders.html",
                           folders=folders, folder_counts=counts,
                           entry_id=entry_id, next=next_url)

# ---------------------------
# ルート登録
# ---------------------------
def init_watermark_ext(app):
    app.view_functions["admin_watermark"] = view_admin_watermark
    if "admin_media_delete" not in app.view_functions:
        app.add_url_rule("/admin/media/delete", endpoint="admin_media_delete",
                         view_func=view_admin_media_delete, methods=["POST"])
    if "admin_watermark_one" not in app.view_functions:
        app.add_url_rule("/admin/watermark/one", endpoint="admin_watermark_one",
                         view_func=view_admin_watermark_one, methods=["GET"])
    if "admin_watermark_one_regen" not in app.view_functions:
        app.add_url_rule("/admin/watermark/one/regenerate", endpoint="admin_watermark_one_regen",
                         view_func=view_admin_watermark_one_regen, methods=["POST"])
    if "admin_media_folders" not in app.view_functions:
        app.add_url_rule("/admin/media/folders", endpoint="admin_media_folders",
                         view_func=view_admin_media_folders, methods=["GET"])
