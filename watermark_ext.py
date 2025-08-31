
# -*- coding: utf-8 -*-
"""
watermark_ext.py
- 透かし一括生成（/admin/watermark）の「写真ピッカー」「並び替え（新しい順）」「編集/削除」強化
- 既存アプリの巨大な app.py を触らずに、最小変更で差し込める拡張モジュール
使い方:
  app.py で Flask アプリ生成後のどこかで次を1回呼ぶだけ:
    from watermark_ext import init_watermark_ext
    init_watermark_ext(app)
"""

from __future__ import annotations
import os, io, importlib
from types import SimpleNamespace
from typing import List, Dict, Any, Iterable, Tuple
from flask import Blueprint, current_app, render_template, request, session, abort, flash, redirect, url_for, send_from_directory

# 既定のサフィックス（アプリ本体の定義を優先して利用／なければ既定）
WM_SUFFIX_NONE  = "__none"
WM_SUFFIX_GOTO  = "__goto"
WM_SUFFIX_FULLY = "__fullygoto"

# ------------------------------------------------------------
# 内部ユーティリティ
# ------------------------------------------------------------

def _get_images_dir() -> str:
    """アプリ本体の IMAGES_DIR を取得。なければ MEDIA_DIR/UPLOAD_FOLDER/media """
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
    """アプリ本体側で定義されていればそれを使う"""
    s_none, s_goto, s_fully = WM_SUFFIX_NONE, WM_SUFFIX_GOTO, WM_SUFFIX_FULLY
    try:
        appmod = importlib.import_module("app")
        s_none  = getattr(appmod, "WM_SUFFIX_NONE",  s_none)
        s_goto  = getattr(appmod, "WM_SUFFIX_GOTO",  s_goto)
        s_fully = getattr(appmod, "WM_SUFFIX_FULLY", s_fully)
    except Exception:
        pass
    return s_none, s_goto, s_fully

def _list_source_images(order: str = "name") -> List[str]:
    """IMAGES_DIR から『元画像』（__none/__goto/__fullygoto を含まない）だけ列挙。
    order=\"new\" で更新日時の新しい順／既定は名前順。
    """
    images_dir = _get_images_dir()
    s_none, s_goto, s_fully = _get_suffixes()
    files: List[str] = []
    if not os.path.isdir(images_dir):
        return files
    for fn in os.listdir(images_dir):
        if fn.startswith("."):
            continue
        root, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
            continue
        if any(suf in root for suf in (s_none, s_goto, s_fully)):
            continue
        files.append(fn)

    if (order or "").lower() == "new":
        def _mtime(f: str) -> float:
            try:
                return os.path.getmtime(os.path.join(images_dir, f))
            except Exception:
                return 0.0
        files.sort(key=_mtime, reverse=True)
    else:
        files.sort()
    return files

def _boolish(x) -> bool:
    return str(x).lower() in {"1","true","on","yes"}

def _require_login():
    """簡易ログイン保護（app.py の login_required と同等の挙動に近づける）"""
    if "user_id" not in session:
        flash("ログインしてください")
        return redirect(url_for("login"))
    return None

def _generate_variants(abs_path: str, *, force: bool, kinds: Iterable[str]) -> Dict[str, str]:
    """アプリ本体のジェネレータを呼び出して派生を作る。
    - _wm_variants_for_path_selected(path, force, kinds) があればそれを優先
    - なければ _wm_variants_for_path(path, force)（全種生成）を呼ぶ
    """
    appmod = importlib.import_module("app")
    fn_sel = getattr(appmod, "_wm_variants_for_path_selected", None)
    if callable(fn_sel):
        return fn_sel(abs_path, force=force, kinds=list(kinds))

    fn = getattr(appmod, "_wm_variants_for_path", None)
    if not callable(fn):
        raise RuntimeError("_wm_variants_for_path が見つかりません")
    return fn(abs_path, force=force)

# ------------------------------------------------------------
# ビュー関数（エンドポイント本体）
# ------------------------------------------------------------

def view_admin_watermark():
    # ログイン要求
    need = _require_login()
    if need: return need

    # 権限: admin / shop のどちらでも利用可
    if session.get("role") not in {"admin", "shop"}:
        abort(403)

    wm     = (request.values.get("wm") or "all").strip()
    if wm not in {"all","none","goto","fully"}:
        wm = "all"

    make_none  = (wm in ("all","none"))
    make_goto  = (wm in ("all","goto"))
    make_fully = (wm in ("all","fully"))

    q      = (request.values.get("q") or "").strip().lower()
    order  = (request.values.get("order") or "new").strip().lower()
    picker = 1 if (request.values.get("picker") in {"1","true","on"}) else 0

    existing = _list_source_images(order=order)
    if q:
        existing = [fn for fn in existing if q in fn.lower()]

    results, errors = [], []
    images_dir = _get_images_dir()

    if request.method == "POST":
        force = (request.form.get("force") == "1")
        selected = request.form.getlist("selected_existing")
        uploads  = request.files.getlist("files") or []

        targets: List[Tuple[str, str]] = []
        for name in selected:
            name = os.path.basename(name)
            targets.append((name, os.path.join(images_dir, name)))

        # アップロード保存は app.py の関数を利用（なければスキップ）
        appmod = importlib.import_module("app")
        save_fn = getattr(appmod, "_save_jpeg_1080_350kb", None)

        for f in uploads:
            if not f or not f.filename:
                continue
            try:
                if callable(save_fn):
                    saved = save_fn(f, previous=None, delete=False)
                else:
                    # フォールバック：素のまま保存（拡張子チェックのみ）
                    fname = os.path.basename(f.filename)
                    stem, ext = os.path.splitext(fname)
                    if ext.lower() not in (".jpg",".jpeg",".png",".webp"):
                        raise ValueError("対応拡張子ではありません")
                    out_path = os.path.join(images_dir, fname)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    f.save(out_path)
                    saved = fname

                if not saved:
                    errors.append(f"{f.filename}: 保存に失敗")
                    continue
                targets.append((saved, os.path.join(images_dir, saved)))
            except Exception as e:
                errors.append(f"{f.filename}: {e}")

        kinds: List[str] = []
        if make_none:  kinds.append("none")
        if make_goto:  kinds.append("goto")
        if make_fully: kinds.append("fully")

        for display_name, abs_path in targets:
            try:
                out = _generate_variants(abs_path, force=force, kinds=kinds)
                r = {
                    "src": display_name,
                    "url_src":  url_for("admin_media_img", filename=display_name),
                }
                if "none"  in kinds and out.get("none"):  r["url_none"]  = url_for("admin_media_img", filename=out["none"])
                if "goto"  in kinds and out.get("goto"):  r["url_goto"]  = url_for("admin_media_img", filename=out["goto"])
                if "fully" in kinds and out.get("fully"): r["url_fully"] = url_for("admin_media_img", filename=out["fully"])
                results.append(SimpleNamespace(**r))
            except Exception:
                current_app.logger.exception("watermark gen failed: %s", abs_path)
                errors.append(f"{display_name}: 生成に失敗しました")

        for m in errors:
            flash(m)

    return render_template(
        "admin_watermark.html",
        wm=wm, q=q, existing=existing, results=results,
        order=order, picker=picker,
    )

def view_admin_media_delete():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin", "shop"}:
        abort(403)

    name = (request.form.get("filename") or "").strip()
    if not name:
        flash("削除対象が指定されていません")
        return redirect(url_for("admin_watermark"))

    images_dir = _get_images_dir()
    s_none, s_goto, s_fully = _get_suffixes()

    name = os.path.basename(name)
    stem, ext = os.path.splitext(name)

    targets = [
        name,
        f"{stem}{s_none}.jpg",
        f"{stem}{s_goto}.jpg",
        f"{stem}{s_fully}.jpg",
    ]

    removed = 0
    for fn in targets:
        path = os.path.join(images_dir, fn)
        try:
            base = os.path.realpath(images_dir)
            real = os.path.realpath(path)
            if os.path.commonpath([base, real]) != base:
                continue
            if os.path.exists(real):
                os.remove(real)
                removed += 1
        except Exception:
            current_app.logger.exception("failed to remove: %s", fn)

    flash("削除しました" if removed else "削除対象が見つかりませんでした")
    return redirect(url_for("admin_watermark", q=stem))

def view_admin_watermark_one():
    need = _require_login()
    if need: return need
    if session.get("role") not in {"admin", "shop"}:
        abort(403)

    src = (request.args.get("src") or "").strip()
    if not src:
        flash("対象画像が指定されていません")
        return redirect(url_for("admin_watermark"))
    src = os.path.basename(src)

    images_dir = _get_images_dir()
    path = os.path.join(images_dir, src)
    if not os.path.isfile(path):
        flash("指定の画像が見つかりません")
        return redirect(url_for("admin_watermark"))
    return render_template("admin_watermark_one.html", src=src)

# ------------------------------------------------------------
# 初期化フック
# ------------------------------------------------------------

def init_watermark_ext(app):
    """
    既存の /admin/watermark の関数を差し替え、
    /admin/media/delete と /admin/watermark/one を追加する。
    """
    # /admin/watermark を関数置換（URL ruleは既存のまま使う）
    app.view_functions["admin_watermark"] = view_admin_watermark

    # /admin/media/delete が無ければ追加
    if "admin_media_delete" not in app.view_functions:
        app.add_url_rule("/admin/media/delete", endpoint="admin_media_delete",
                         view_func=view_admin_media_delete, methods=["POST"])

    # /admin/watermark/one が無ければ追加
    if "admin_watermark_one" not in app.view_functions:
        app.add_url_rule("/admin/watermark/one", endpoint="admin_watermark_one",
                         view_func=view_admin_watermark_one, methods=["GET"])
