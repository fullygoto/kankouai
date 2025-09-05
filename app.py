# === Standard Library ===
import os
import json
import re
import math
import datetime
import itertools
import time
import threading
import ipaddress
import logging
import uuid
import hmac, hashlib, base64
import zipfile
import io
from pathlib import Path
from functools import wraps
from collections import Counter
from typing import Any, Dict, List
import urllib.parse as _u  # ← _extract_wm_flags などで使用

# === Third-Party ===
from dotenv import load_dotenv
load_dotenv()

from flask import (
    Flask, render_template, request, redirect, url_for, flash, session,
    jsonify, send_file, abort, send_from_directory, render_template_string, Response,
    current_app as _flask_current_app,  # ← これを追加
)

from werkzeug.exceptions import RequestEntityTooLarge, NotFound
from werkzeug.utils import secure_filename, safe_join
from werkzeug.routing import BuildError
from werkzeug.security import check_password_hash, generate_password_hash

from openai import OpenAI

# --- Pillow / PIL ---
import warnings  # ← 追加！
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL import ImageFile, UnidentifiedImageError

# Pillowのバージョン差を吸収したリサンプリング定数
try:
    # Pillow >= 9.1
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    # Pillow < 9.1 互換
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", Image.BICUBIC))

# 破損気味な画像でも読み込むための安全策（任意）
ImageFile.LOAD_TRUNCATED_IMAGES = True

# DecompressionBomb を警告→例外に（任意）
try:
    warnings.simplefilter("error", Image.DecompressionBombWarning)
except Exception:
    pass

# === LINE Bot ===
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    ImageSendMessage, LocationMessage, FlexSendMessage, LocationAction,
    FollowEvent,
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

# =========================
#  Flask / 設定
# =========================
app = Flask(__name__)


@app.context_processor
def _inject_template_helpers():
    def has_endpoint(name: str) -> bool:
        try:
            return name in app.view_functions
        except Exception:
            return False
    # current_app も使えるように入れておく（既存テンプレ互換）
    return {
        "current_app": _flask_current_app,
        "has_endpoint": has_endpoint,
    }

# 64MB上限（ZIP復元のため拡張）
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB

# 413: サイズ超過時のメッセージ
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("ファイルサイズが大きすぎます（最大64MB）。")
    return redirect(request.referrer or url_for("admin_entries_edit"))

# 既に前の回答で入れていれば流用されます
MYMAP_MID = os.getenv("MYMAP_MID", "")

def build_wm_image_urls(filename: str, mode: str | None = None) -> dict:
    wm_mode = (mode or "fullygoto")  # 既定
    ts = str(int(time.time()))
    base = dict(_external=True, _scheme="https", _sign=True)
    return {
        "original": url_for("serve_image", filename=filename, wm=wm_mode, **base) + f"&_ts={ts}",
        "preview":  url_for("serve_image", filename=filename, wm="thumb",   **base) + f"&_ts={ts}",
    }

def gmaps_url(*, name:str="", address:str="", lat:float|None=None, lng:float|None=None, place_id:str|None=None) -> str:
    base = "https://www.google.com/maps/search/?api=1"
    if place_id:
        q = _u.quote(name or address or "")
        return f"{base}&query={q}&query_place_id={_u.quote(place_id)}"
    if lat is not None and lng is not None:
        return f"{base}&query={lat:.6f},{lng:.6f}"
    q = _u.quote(name or address)
    return f"{base}&query={q}"

def mymap_view_url(*, lat:float|None=None, lng:float|None=None, zoom:int=13) -> str:
    if not MYMAP_MID:
        return ""
    base = f"https://www.google.com/maps/d/viewer?mid={_u.quote(MYMAP_MID)}"
    if lat is not None and lng is not None:
        return f"{base}&ll={lat:.6f},{lng:.6f}&z={int(zoom)}"
    return base

# entries の 1件から lat/lng を安全に取り出す（lat/lngが無ければ map URL から推定）
_float = lambda x: (float(x) if x not in (None, "", "None") else None)

# 追加: “開く用”の最適URLを決める
_MAP_HOST_RE = re.compile(
    r'^https?://(?:www\.)?(?:google\.[^/]+/maps|maps\.app\.goo\.gl|goo\.gl/maps|g\.page|g\.co/kgs)',
    re.I
)


# ==== 展望所マップ（My Maps 専用） =========================
VIEWPOINTS_URL  = os.getenv("VIEWPOINTS_URL", "").strip()
VIEWPOINTS_MID  = os.getenv("VIEWPOINTS_MID", "").strip()
VIEWPOINTS_LL   = os.getenv("VIEWPOINTS_LL", "").strip()   # 例: "32.97,129.52"
VIEWPOINTS_ZOOM = os.getenv("VIEWPOINTS_ZOOM", "10").strip()
VIEWPOINTS_THUMB = os.getenv("VIEWPOINTS_THUMB", "").strip()  # 例: "viewer_thumb.jpg"

def viewpoints_map_url() -> str:
    """
    優先順位:
      1) VIEWPOINTS_URL（フルURL指定）
      2) VIEWPOINTS_MID から組み立て（ll, z があれば付与）
    """
    if VIEWPOINTS_URL:
        return VIEWPOINTS_URL
    if VIEWPOINTS_MID:
        base = f"https://www.google.com/maps/d/viewer?mid={_u.quote(VIEWPOINTS_MID)}&femb=1"
        ll = ""
        if VIEWPOINTS_LL:
            try:
                lat, lng = [s.strip() for s in VIEWPOINTS_LL.split(",", 1)]
                ll = f"&ll={_u.quote(lat)},{_u.quote(lng)}"
            except Exception:
                ll = ""
        z = ""
        if VIEWPOINTS_ZOOM:
            try:
                z = f"&z={int(VIEWPOINTS_ZOOM)}"
            except Exception:
                z = ""
        return base + ll + z
    return ""  # 未設定

def _flex_viewpoints_map():
    """
    “展望所マップ”だけを返す Flex バブル。サムネは任意（VIEWPOINTS_THUMB）。
    """
    url = viewpoints_map_url()
    if not url:
        # URL 未設定時は None（呼び出し側でテキストフォールバック）
        return None

    hero = None
    if VIEWPOINTS_THUMB:
        try:
            thumb_url = build_signed_image_url(VIEWPOINTS_THUMB, wm=True, external=True)
            hero = {"type": "image", "url": thumb_url, "size": "full", "aspectMode": "cover", "aspectRatio": "16:9"}
        except Exception:
            hero = None

    body_contents = [
        {"type": "text", "text": "五島列島 展望所マップ", "weight": "bold", "wrap": True},
        {"type": "text", "text": "Googleマイマップで展望スポットを一覧表示します。", "size": "sm", "color": "#666666", "wrap": True},
    ]

    bubble = {
        "type": "bubble",
        **({"hero": hero} if hero else {}),
        "body": {"type": "box", "layout": "vertical", "spacing": "sm", "contents": body_contents},
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {"type": "button", "style": "primary",
                 "action": {"type": "uri", "label": "マップを開く", "uri": url}}
            ]
        }
    }
    return {"type": "carousel", "contents": [bubble]}

def send_viewpoints_map(event):
    try:
        app.logger.info("[viewpoints] url=%s thumb=%s", viewpoints_map_url(), VIEWPOINTS_THUMB)
        flex = _flex_viewpoints_map()
        if flex:
            line_bot_api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text="展望所マップ", contents=flex)
            )
            return
    except Exception:
        app.logger.exception("send_viewpoints_map: flex failed")

    url = viewpoints_map_url() or "(展望所マップURLが未設定です)"
    _reply_or_push(event, f"展望所マップはこちら：\n{url}")



# ======= LINE 応答 停止/再開 管理（管理者優先・利用者次点・互換対応） =======
PAUSE_DIR = os.getenv("PAUSE_DIR") or os.path.dirname(__file__)
PAUSE_FLAG_ADMIN = os.path.join(PAUSE_DIR, "line_paused.admin.flag")
PAUSE_FLAG_USER  = os.path.join(PAUSE_DIR, "line_paused.user.flag")
# 旧実装で使っていた互換フラグ（残っていると再開できない要因になる）
PAUSE_FLAG_LEGACY = os.path.join(PAUSE_DIR, "line_paused.flag")

def _pause_set_admin(on: bool):
    """管理者の停止フラグをON/OFF（互換フラグも合わせて操作）"""
    try:
        if on:
            # 管理者停止を最優先に立てる（旧フラグも立てておくと古いコード経路でも確実に止まる）
            with open(PAUSE_FLAG_ADMIN, "w", encoding="utf-8") as f:
                f.write("admin\n")
            with open(PAUSE_FLAG_LEGACY, "w", encoding="utf-8") as f:
                f.write("admin\n")
        else:
            # 解除時は必ず両方消す（旧フラグの残骸で“止まりっぱなし”を防止）
            for p in (PAUSE_FLAG_ADMIN, PAUSE_FLAG_LEGACY):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        app.logger.exception("pause_set_admin failed")

def _pause_set_user(on: bool):
    """利用者（メッセージ）による停止フラグをON/OFF"""
    try:
        if on:
            with open(PAUSE_FLAG_USER, "w", encoding="utf-8") as f:
                f.write("user\n")
        else:
            # 利用者停止解除。旧フラグも念のため掃除（古い経路でuser停止を書いたケースの保険）
            for p in (PAUSE_FLAG_USER, ):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        app.logger.exception("pause_set_user failed")

def _pause_state():
    """
    現在の停止状態を判定。
    優先順位: 管理者 > 利用者。旧フラグ(PAUSE_FLAG_LEGACY)も管理者扱いで尊重する。
    """
    s_admin = os.path.exists(PAUSE_FLAG_ADMIN) or os.path.exists(PAUSE_FLAG_LEGACY)
    if s_admin:
        return True, "admin"
    s_user = os.path.exists(PAUSE_FLAG_USER)
    if s_user:
        return True, "user"
    return False, None
# ======= /停止管理 =======


# === 透かしON/OFF（Render環境変数をまとめて判定） ===
def _env_truthy(val: str | None) -> bool:
    if val is None:
        return False
    v = val.strip().lower()
    return v not in ("", "0", "false", "off", "no")

def _watermark_enabled() -> bool:
    # どれか1つでも真なら有効（Renderで WATERMARK_ENABLE=1 ならOK）
    return (
        _env_truthy(os.getenv("WATERMARK_ENABLE")) or
        _env_truthy(os.getenv("WATERMARK_ENABLED")) or
        _env_truthy(os.getenv("WM_ENABLE"))
    )

# === 透かし種別の解釈ヘルパー ===
def _resolve_wm_kind(arg: str | None):
    v = (arg or "").strip().lower()
    if v in ("", "0", "false", "off", "none", "no"):
        return None
    if v in ("1", "true", "on", "yes"):
        return "fullygoto"   # 既定
    if v in ("fullygoto", "fully", "fg"):
        return "fullygoto"
    if v in ("gotocity", "city", "gc"):
        return "gotocity"
    return None
    
# ==== 画像配信（署名 + 透かし対応）=============================
# 依存: Pillow, safe_join, send_file, load_entries(), app など
# 既存の設定が無い場合のデフォルト
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "media/img")
WATERMARK_ENABLE = os.getenv("WATERMARK_ENABLE", "1").lower() in {"1","true","on","yes"}
IMAGE_PROTECT    = os.getenv("IMAGE_PROTECT",    "0").lower() in {"1","true","on","yes"}

# ==== 画像保存/配信のディレクトリ統一（互換アライメント） ====
MEDIA_URL_PREFIX = os.getenv("MEDIA_URL_PREFIX", "/media/img")
IMAGES_DIR = os.getenv("IMAGES_DIR") or MEDIA_ROOT
MEDIA_ROOT = IMAGES_DIR  # ← 配信・保存とも同じ実体を指すように統一
try:
    app.logger.info("[media] MEDIA_ROOT=%s  IMAGES_DIR=%s  URL_PREFIX=%s",
                    MEDIA_ROOT, IMAGES_DIR, MEDIA_URL_PREFIX)
except Exception:
    pass


# === 保存ヘルパー（/media/img 配下に“確実に保存”＋mtimeで新しい順を保証） ===
MEDIA_DIR = Path(MEDIA_ROOT).resolve()
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

def _wm_variant_name(base_filename: str, kind: str, *, out_ext: str | None=None) -> str:
    """kind: 'fullygoto' | 'gotocity' | 'src'"""
    stem, ext = os.path.splitext(base_filename)
    if out_ext and out_ext.lower() != ext.lower():
        ext = out_ext  # webp→jpg など出力拡張子が変わる場合に追従
    suf = "__" + ({"fullygoto":"fullygoto","gotocity":"gotocity","src":"src"}[kind])
    return f"{stem}{suf}{ext}"

def _touch_latest(path: Path, bias_sec: int = 0) -> None:
    now = time.time() + bias_sec
    os.utime(path, (now, now))

def _list_existing_files(order: str = "new") -> list[dict]:
    items = []
    for p in MEDIA_DIR.glob("*"):
        if p.is_file():
            items.append({"name": p.name, "mtime": p.stat().st_mtime})
    items.sort(key=lambda x: x["mtime"], reverse=(order == "new"))
    return items


# ===== 既存ファイル一覧（派生も含める） ======================================
_DERIV_SUFFIX_RE = re.compile(r"__(?:none|goto|fullygoto|gotocity|src)\.[^.]+$", re.I)

def list_media_for_grid(order: str = "new", include_derivatives: bool = True) -> list[str]:
    """
    /media/img 直下の画像ファイルを列挙して、テンプレートの 'existing' に渡すためのリストを返す。
    - include_derivatives=True のとき __none / __goto / __fullygoto（__gotocity も許容）を含める
      False のときはそれらを除外（=元画像だけ）
    - order: 'new'（更新日時降順） or 'name'（名前昇順）
    """
    rows: list[tuple[str, float]] = []
    for p in MEDIA_DIR.glob("*"):
        if not p.is_file():
            continue
        name = p.name
        # 画像だけ
        if (p.suffix or "").lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if not include_derivatives and _DERIV_SUFFIX_RE.search(name):
            continue  # 元画像だけに絞る
        try:
            rows.append((name, p.stat().st_mtime))
        except OSError:
            continue

    if order == "name":
        rows.sort(key=lambda t: t[0].lower())
    else:
        rows.sort(key=lambda t: t[1], reverse=True)

    return [n for n, _ in rows]

def _save_bytes(dst: Path, data: bytes, *, bias_sec: int = 0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    _touch_latest(dst, bias_sec=bias_sec)

def _render_watermark_bytes(src_path: Path, mode: str) -> tuple[bytes, str]:
    """
    src_path を読み込み、'fullygoto' or 'gotocity' の透かしを焼き込んだバイト列を返す。
    戻り値: (bytes, out_ext)  ※webpはJPEG化。
    """
    from PIL import Image, ImageOps
    import io as _io
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA")
        # 既存ヘルパーで右下にロゴ or テキストを重ねる
        out_img = _overlay_png_or_text(im, mode)

        ext = (src_path.suffix or "").lower()
        buf = _io.BytesIO()
        if ext in {".jpg", ".jpeg", ".webp"}:
            out_img.convert("RGB").save(
                buf, format="JPEG", quality=int(os.getenv("WM_JPEG_QUALITY", "88")),
                optimize=True, progressive=True, subsampling=2
            )
            out_ext = ".jpg"
        else:
            out_img.save(buf, format="PNG", optimize=True)
            out_ext = ".png"
        buf.seek(0)
        return buf.read(), out_ext

# --- Watermark assets (static/watermarks 以下) ---
WATERMARK_DIR = os.getenv("WATERMARK_DIR") or os.path.join(app.static_folder, "watermarks")
WATERMARK_FILES = {
    "fullygoto": "wm_fullygoto.png",
    "gotocity":  "wm_gotocity.png",
}

def _wm_overlay_path(mode: str) -> str | None:
    fn = WATERMARK_FILES.get((mode or "").strip().lower())
    if not fn:
        return None
    p = os.path.join(WATERMARK_DIR, fn)
    return p if os.path.isfile(p) else None


def _wm_choice_from_entries(arg, *, fmt: str = "legacy") -> str | None:
    """
    エントリ既定の透かしモードを返す。
    引数 arg: エントリ dict か 画像ファイル名(文字列)。
    fmt:
      - "legacy"     -> "fullygoto" / "gotocity" / None（互換）
      - "canonical"  -> "fully" / "city" / "none" / None

    返り値:
      fmtに応じた文字列。見つからない/未設定時は None。
    """
    def _to_fmt(v: str | None) -> str | None:
        # 既存の正規化関数を流用（'fullygoto'等も 'fully'/'city' に正規化される前提）
        v = _wm_normalize(v)  # -> 'fully'/'city'/'none'/None
        if v is None:
            return None
        if fmt == "legacy":
            return {"fully": "fullygoto", "city": "gotocity", "none": None}.get(v, None)
        # canonical
        return v  # 'fully'/'city'/'none'

    try:
        # 1) dict 直接
        if isinstance(arg, dict):
            # 明示指定があれば最優先
            v = _to_fmt(arg.get("wm_external_choice"))
            if v is not None:
                return v

            # 旧互換のブール群（どれか真なら city を既定、偽なら none）
            legacy_flag = (
                arg.get("wm_external")
                or arg.get("wm_ext_fully")
                or arg.get("wm_ext")
            )
            if isinstance(legacy_flag, bool):
                return _to_fmt("city" if legacy_flag else "none")
            if legacy_flag:  # 真偽値以外の真も city 既定に寄せる
                return _to_fmt("city")

            # さらに旧UIの 'wm_on'（真=city/偽=none）
            if isinstance(arg.get("wm_on"), bool):
                return _to_fmt("city" if arg.get("wm_on") else "none")

            return None  # 未設定

        # 2) 画像ファイル名で探索
        fn = str(arg or "").strip()
        if not fn:
            return None
        for e in load_entries():
            img = (e.get("image_file") or e.get("image") or "").strip()
            if img and img == fn:
                return _wm_choice_from_entries(e, fmt=fmt)

    except Exception:
        app.logger.exception("wm choice lookup failed")

    return None

def _verify_sig_if_available(filename: str, args) -> bool:
    """
    署名検証。既存の検証関数があれば使う。無ければ（または IMAGE_PROTECT=False なら）通す。
    既存候補: verify_signed_media_url(filename, args) / verify_signed_query(filename, args)
    """
    if not IMAGE_PROTECT:
        return True
    try:
        vf = (globals().get("verify_signed_media_url")
              or globals().get("verify_signed_query")
              or None)
        if vf:
            return bool(vf(filename, args))
        # 既存関数が無い場合は通す（既存URLを壊さないため）
        return True
    except Exception:
        app.logger.exception("signature verify failed")
        return False

def _load_image_safe(path: str):
    with Image.open(path) as im:
        return im.convert("RGBA")

def _overlay_png_or_text(base_rgba: Image.Image, mode: str) -> Image.Image:
    """
    右下にロゴPNG（あれば）を重ねる。無ければテキストで代替。
    mode: 'fullygoto' | 'gotocity'
    """
    W, H = base_rgba.size
    # ロゴファイル探す（任意）：static/wm_fullygoto.png / static/wm_gotocity.png
    logo_path = None
    cand = {
        "fullygoto": ["static/wm_fullygoto.png", "static/wm_fully.png"],
        "gotocity":  ["static/wm_gotocity.png", "static/wm_city.png"],
    }.get(mode, [])
    for p in cand:
        if os.path.isfile(p):
            logo_path = p
            break

    out = base_rgba.copy()
    margin = max(6, int(min(W, H) * 0.02))

    if logo_path:
        try:
            with Image.open(logo_path) as wm:
                wm = wm.convert("RGBA")
                # 幅の 18% 目安
                target_w = max(64, int(W * 0.18))
                scale = target_w / wm.width
                wm = wm.resize((target_w, int(wm.height * scale)), Image.LANCZOS)
                x = W - wm.width - margin
                y = H - wm.height - margin
                out.alpha_composite(wm, dest=(x, y))
                return out
        except Exception:
            app.logger.warning("wm logo load failed, fallback to text")

    # テキスト透かし（ロゴが無い時の簡易版）
    try:
        txt = "@fullyGOTO" if mode == "fullygoto" else "@Goto City"
        overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # フォント（無ければデフォルト）
        try:
            # 任意に NotoSansJP 等を置いていれば使う
            font_path = os.getenv("WM_FONT_PATH", "")
            size = max(18, int(min(W, H) * 0.045))
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # テキストサイズ
        tw, th = draw.textbbox((0, 0), txt, font=font)[2:]
        # 背景プレート
        pad = max(4, int(th * 0.25))
        box_w, box_h = tw + pad * 2, th + pad * 2
        bx = W - box_w - margin
        by = H - box_h - margin
        draw.rounded_rectangle((bx, by, bx + box_w, by + box_h), radius=int(pad * 1.2), fill=(0, 0, 0, 76))
        # テキスト（白）
        draw.text((bx + pad, by + pad), txt, font=font, fill=(255, 255, 255, 230))
        out.alpha_composite(overlay)
    except Exception:
        app.logger.exception("text watermark failed")
    return out

def _infer_mimetype(path: str) -> tuple[str, str]:
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext in {".jpg", ".jpeg"}: return "JPEG", "image/jpeg"
    if ext in {".png"}:          return "PNG",  "image/png"
    if ext in {".webp"}:         return "WEBP", "image/webp"
    return "PNG", "image/png"


# === Watermark helpers (ADD) ================================================
def _wm_normalize(v) -> str | None:
    """
    透かしモードの正規化:
      入力: "fullygoto"/"gotocity"/"fully"/"city"/"1"/True/False/"none"... など
      出力: "fullygoto" / "gotocity" / None
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return "gotocity" if v else None
    s = str(v).strip().lower()
    if s in {"fullygoto", "fully"}:
        return "fullygoto"
    if s in {"gotocity", "city"}:
        return "gotocity"
    if s in {"1", "true", "on", "yes"}:
        return "gotocity"
    if s in {"0", "false", "off", "none", ""}:
        return None
    # 未知値は保守的に None
    return None


# ============================================================================



@app.route("/media/img/<path:filename>")
def serve_image(filename):
    """
    署名検証 → 実ファイル解決 → 透かしモード決定（URL優先→エントリ既定） → 返却。
    - wm 未指定 or WATERMARK_ENABLE=False のときは原画像をそのまま返す
    - wm=gotocity / fullygoto のときは動的に合成して返す
    - 一覧サムネ用途（wm=1 / thumb / preview）は常に透かし（既定は gotocity）。※A: gotocity が無ければ fullygoto にフォールバック
    - 返却は元拡張子に合わせて JPEG/PNG を選択（LINE互換のため webp は JPEG）
    - 観測ヘッダ: X-WM-Requested / X-WM-Applied / X-WM-Ratio / X-Img-Signed / X-Img-Exp / X-Img-Fmt / X-Img-Path
    - 追加: ①HEAD は合成スキップ（超軽量化）／②B: サムネ（thumb）だけはキャッシュ有効化
    """
    import os
    from flask import request, abort, send_file
    from flask import Response  # HEAD時の軽量レスポンスで使用

    # --- 環境変数ON/OFF ---
    def _env_truthy(val: str | None) -> bool:
        if val is None: return False
        v = val.strip().lower()
        return v not in ("", "0", "false", "off", "no")
    def _watermark_enabled_local() -> bool:
        return (
            _env_truthy(os.getenv("WATERMARK_ENABLE")) or
            _env_truthy(os.getenv("WATERMARK_ENABLED")) or
            _env_truthy(os.getenv("WM_ENABLE"))
        )

    # サムネキャッシュ秒（B）— 未指定なら 7日
    THUMB_CACHE_SECS = int(os.getenv("WM_THUMB_CACHE_SECS", "604800"))

    # ============================================================
    # Watermark 正規化ヘルパ
    # ============================================================

    _normalize_wm_choice_local = globals().get("_normalize_wm_choice")
    if not callable(_normalize_wm_choice_local):
        def _normalize_wm_choice_local(choice: str | None, wm_on_default: bool = True) -> str:
            c = (str(choice or "")).strip().lower()
            if c in ("none", "fullygoto", "gotocity"):
                return c
            if c in ("1", "true", "on", "yes"):
                return "fullygoto"
            if c in ("0", "false", "off", "no", ""):
                return "none"
            if c == "fully":
                return "fullygoto"
            if c == "city":
                return "gotocity"
            return "fullygoto" if wm_on_default else "none"

    # --- 署名検証（外部があれば利用／無ければfail-open） ---
    def _verify_sig_if_available_local(fn: str, qargs) -> bool:
        if "_verify_sig_if_available" in globals():
            try:
                return bool(globals()["_verify_sig_if_available"](fn, qargs))
            except Exception:
                pass
        return True

    # --- entriesから既定wmを引く（外部があれば利用） ---
    def _wm_choice_from_entries_local(basename: str) -> str | None:
        if "_wm_choice_from_entries" in globals():
            try:
                return globals()["_wm_choice_from_entries"](basename)
            except Exception:
                pass
        try:
            entries = load_entries()
            for e in entries:
                img = (e.get("image_file") or e.get("image") or "").strip()
                if img and os.path.basename(img) == basename:
                    raw = e.get("wm_external_choice")
                    if raw is None:
                        raw = "fullygoto" if e.get("wm_on", True) else "none"
                    return _normalize_wm_choice_local(raw, wm_on_default=True)
        except Exception:
            pass
        return None

    # --- 透かし合成（内蔵。サムネ時は大きめ比率） ---
    def _compose_watermark_local(pil_image, mode: str):
        if "_compose_watermark" in globals():
            try:
                return globals()["_compose_watermark"](pil_image, mode)
            except Exception:
                pass

        from PIL import Image
        kind, _, flag = (mode or "").partition("|")
        is_thumb = (flag == "thumb")

        # 探索パス
        base_candidates = []
        try: base_candidates.append(os.path.join(app.root_path, "static", "watermarks"))
        except Exception: pass
        if "BASE_DIR" in globals():
            try: base_candidates.append(os.path.join(globals()["BASE_DIR"], "static", "watermarks"))
            except Exception: pass
        base_candidates += ["static/watermarks", "./static/watermarks"]

        wm_map = {"fullygoto": "wm_fullygoto.png", "gotocity": "wm_gotocity.png"}
        wm_file = wm_map.get(kind)
        if not wm_file:
            return pil_image
        wm_path = None
        for b in base_candidates:
            p = os.path.join(b, wm_file)
            if os.path.isfile(p):
                wm_path = p
                break
        if not wm_path:
            app.logger.error("Watermark file missing: %s (searched in %s)", wm_file, base_candidates)
            return pil_image

        mark = Image.open(wm_path).convert("RGBA")
        im = pil_image.convert("RGBA")

        ratio = 0.45 if is_thumb else 0.16  # サムネは大きめ
        target_w = max(1, int(im.width * ratio))
        scale = target_w / mark.width
        mark = mark.resize((target_w, int(mark.height * scale)), Image.LANCZOS)

        margin = max(8, int(im.width * 0.01))
        x = im.width - mark.width - margin
        y = im.height - mark.height - margin

        im.alpha_composite(mark, (x, y))
        im._wm_ratio_used = ratio  # デバッグ
        return im

    # --- 画像パス解決 ---
    from werkzeug.utils import safe_join
    def _find_image_path(fn: str) -> str | None:
        candidates = []
        if "MEDIA_ROOT" in globals() and globals().get("MEDIA_ROOT"):
            candidates.append(globals()["MEDIA_ROOT"])
        for d in {globals().get("IMAGES_DIR"), os.getenv("MEDIA_ROOT", "media/img"), "media/img", "media"}:
            if d and d not in candidates:
                candidates.append(d)
        for base in candidates:
            try: p = safe_join(base, fn)
            except Exception: continue
            if p and os.path.isfile(p):
                return p
        return None

    # ===== 1) 署名検証 =====
    signed_req = bool(request.args.get("_sign"))
    if not _verify_sig_if_available_local(filename, request.args):
        abort(403)

    # ===== 2) 実ファイル =====
    abs_path = _find_image_path(filename)
    if not abs_path: abort(404)

    # ===== 3) モード決定 =====
    raw_wm = request.args.get("wm")
    requested_mode = (str(raw_wm) if raw_wm is not None else "")
    mode = _normalize_wm_choice_local(requested_mode, wm_on_default=True)

    # URLに妥当な指定が無ければ、エントリ既定を採用
    if requested_mode in (None, "",) or mode not in ("none", "fullygoto", "gotocity"):
        try:
            from os.path import basename as _bn
            m2 = _wm_choice_from_entries_local(_bn(filename))
            mode = m2 if m2 is not None else _normalize_wm_choice_local(None, wm_on_default=True)
        except Exception:
            mode = _normalize_wm_choice_local(None, wm_on_default=True)

    # サムネ用途は gotocity ＋ “thumb フラグ”
    if str(requested_mode).lower() in {"1", "thumb", "preview"}:
        applied_mode = "gotocity"
        is_thumb = True
        # ★A: gotocity が無ければ fullygoto にフォールバック
        try:
            wm_dir = os.path.join(app.root_path, "static", "watermarks")
            if not os.path.isfile(os.path.join(wm_dir, "wm_gotocity.png")):
                applied_mode = "fullygoto"
                app.logger.info("wm thumb fallback → fullygoto (wm_gotocity.png missing)")
        except Exception:
            pass
    else:
        applied_mode = mode
        is_thumb = False

    # ===== 4) 出力フォーマット =====
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        out_fmt, mime = "JPEG", "image/jpeg"
    elif ext == ".png":
        out_fmt, mime = "PNG", "image/png"
    elif ext == ".webp":
        out_fmt, mime = "JPEG", "image/jpeg"  # LINE互換
    else:
        out_fmt, mime = "JPEG", "image/jpeg"

    # ===== 5) 合成要否 =====
    watermark_enabled = _watermark_enabled_local()
    force = (requested_mode not in (None, "",))  # URLでwm指定があれば強制
    must_compose = (applied_mode in ("fullygoto", "gotocity")) and (watermark_enabled or is_thumb or force)

    # ===== 6) レスポンス =====
    is_head = (request.method == "HEAD")  # ①HEAD最適化
    WM_METRICS = globals().setdefault("WM_METRICS", {
        "requests": 0, "signed": 0, "unsigned": 0, "errors": 0,
        "applied": {"none": 0, "fullygoto": 0, "gotocity": 0, "thumb": 0},
    })
    WM_METRICS["requests"] += 1
    WM_METRICS["signed" if signed_req else "unsigned"] += 1

    try:
        wm_ratio_used = None  # デバッグ

        # ① HEADは合成を完全スキップ（Pillowも開かない）
        if is_head and must_compose:
            resp = Response(status=200, mimetype=mime)
            # キャッシュ方針は GET と揃える
            if is_thumb:
                resp.headers["Cache-Control"] = f"public, max-age={THUMB_CACHE_SECS}, immutable"
            else:
                resp.headers["Cache-Control"] = "private, no-cache, no-store, max-age=0"

        else:
            from PIL import Image, ImageOps
            import io as _io

            if must_compose:
                with Image.open(abs_path) as im:
                    im = ImageOps.exif_transpose(im)
                    comp_mode = applied_mode + ("|thumb" if is_thumb else "")
                    out = _compose_watermark_local(im, comp_mode)
                    wm_ratio_used = getattr(out, "_wm_ratio_used", None)

                    buf = _io.BytesIO()
                    if out_fmt == "JPEG":
                        out = out.convert("RGB")
                        out.save(
                            buf, format="JPEG",
                            quality=int(os.getenv("WM_JPEG_QUALITY", "88")),
                            optimize=True, progressive=True, subsampling=2
                        )
                    elif out_fmt == "PNG":
                        if out.mode != "RGBA":
                            out = out.convert("RGBA")
                        out.save(buf, format="PNG", optimize=True)
                    else:
                        out = out.convert("RGB")
                        out.save(buf, format="JPEG", quality=88, optimize=True, progressive=True, subsampling=2)
                    buf.seek(0)
                    resp = send_file(buf, mimetype=mime, download_name=os.path.basename(abs_path))

                # B: サムネは強キャッシュ（その他の透かしは非キャッシュ）
                if is_thumb:
                    resp.headers["Cache-Control"] = f"public, max-age={THUMB_CACHE_SECS}, immutable"
                else:
                    resp.headers["Cache-Control"] = "private, no-cache, no-store, max-age=0"
                # 透かし版でUA差異を避けるならVaryは付けない（キャッシュ効かせたい）
                # resp.headers["Vary"] = "Accept"

            else:
                # 原画像返却（強キャッシュ）
                resp = send_file(abs_path, mimetype=mime)
                resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"

        # 観測ヘッダ（HEADでも付与）
        resp.headers["X-WM-Requested"] = str(requested_mode or "")
        resp.headers["X-WM-Applied"]   = ("thumb" if is_thumb else applied_mode)
        if wm_ratio_used is not None:
            resp.headers["X-WM-Ratio"] = f"{wm_ratio_used:.2f}"
        if signed_req:
            resp.headers["X-Img-Signed"] = "1"
            if request.args.get("exp"):
                resp.headers["X-Img-Exp"] = str(request.args.get("exp"))
        resp.headers["X-Img-Fmt"]  = out_fmt
        resp.headers["X-Img-Path"] = abs_path

        WM_METRICS["applied"][resp.headers["X-WM-Applied"]] = WM_METRICS["applied"].get(resp.headers["X-WM-Applied"], 0) + 1
        app.logger.info("serve_image ok", extra={
            "wm_req": requested_mode, "wm_applied": resp.headers["X-WM-Applied"],
            "ratio": resp.headers.get("X-WM-Ratio"),
            "signed": signed_req, "fmt": out_fmt, "path": abs_path, "head": is_head
        })
        return resp

    except Exception:
        WM_METRICS["errors"] += 1
        app.logger.exception("serve_image watermark compose failed; fallback to raw")
        resp = send_file(abs_path, mimetype=mime)
        resp.headers["Cache-Control"] = "private, no-cache, no-store, max-age=0"
        resp.headers["X-WM-Requested"] = str(requested_mode or "")
        resp.headers["X-WM-Applied"]   = "error-fallback"
        resp.headers["X-Img-Fmt"]      = out_fmt
        resp.headers["X-Img-Path"]     = abs_path
        return resp

# ==== /画像配信 =================================================

# --- preview URL helper（admin_media_img が無い環境でも壊れないように） ---
def _preferred_media_url(fn: str) -> str:
    try:
        # 既に admin_media_img があればそれを使う
        return url_for("admin_media_img", filename=fn)
    except Exception:
        # 無い場合は /media/img (=serve_image) を使う
        return safe_url_for("serve_image", filename=fn)


# 別名プレフィックス（環境変数で差し替え可）
MEDIA_URL_PREFIX = os.getenv("MEDIA_URL_PREFIX", "/media/img").rstrip("/")

# 上の serve_image を 1つだけ残した状態で、この追記を入れる
if MEDIA_URL_PREFIX != "/media/img":
    try:
        app.add_url_rule(
            f"{MEDIA_URL_PREFIX}/<path:filename>",
            endpoint="serve_image_alias",   # ← エンドポイント名は重複させない
            view_func=serve_image,          # ← 上で定義済みの関数を再利用
            methods=["GET", "HEAD"]
        )
    except AssertionError:
        pass


# ==== My Maps 共通（VIEWER化） ==================================
def _normalize_mymaps_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    u = re.sub(r"/edit(\?|$)", r"/viewer\1", u)  # /edit → /viewer
    if "femb=1" not in u:
        u += ("&" if "?" in u else "?") + "femb=1"
    return u

# ==== 海水浴場マップ（My Maps 専用） ============================
BEACHMAP_URL  = os.getenv("BEACHMAP_URL", "").strip()
BEACHMAP_MID  = os.getenv("BEACHMAP_MID", "").strip()
BEACHMAP_LL   = os.getenv("BEACHMAP_LL", "").strip()
BEACHMAP_ZOOM = os.getenv("BEACHMAP_ZOOM", "11").strip()
BEACHMAP_THUMB = os.getenv("BEACHMAP_THUMB", "").strip()

def beach_map_url() -> str:
    if BEACHMAP_URL:
        return _normalize_mymaps_url(BEACHMAP_URL)
    if BEACHMAP_MID:
        base = f"https://www.google.com/maps/d/viewer?mid={_u.quote(BEACHMAP_MID)}&femb=1"
        ll = ""
        if BEACHMAP_LL:
            try:
                lat, lng = [s.strip() for s in BEACHMAP_LL.split(",", 1)]
                ll = f"&ll={_u.quote(lat)},{_u.quote(lng)}"
            except Exception:
                ll = ""
        z = ""
        if BEACHMAP_ZOOM:
            try:
                z = f"&z={int(BEACHMAP_ZOOM)}"
            except Exception:
                z = ""
        return base + ll + z
    return ""

# ==== 登山・トレッキングマップ（My Maps 専用） ====================
TREKMAP_URL   = os.getenv(
    "TREKMAP_URL",
    # デフォルトに共有された /edit URL をそのまま入れてOK。実行時に /viewer & femb=1 に正規化されます。
    "https://www.google.com/maps/d/u/1/edit?mid=1dXNtYKD1kx273Ux-zLKhMVmeTLpU1YQ&usp=sharing"
).strip()
TREKMAP_THUMB = os.getenv("TREKMAP_THUMB", "").strip()  # 任意サムネ（未設定でも可）

def trek_map_url() -> str:
    """
    My Maps の /edit URL を /viewer に変換し、埋め込み最適化パラメータ femb=1 を付与して返します。
    TREKMAP_URL を .env で差し替えれば運用で変更可能です。
    """
    return _normalize_mymaps_url(TREKMAP_URL)

# ==== さくらマップ（My Maps 専用） ==============================
SAKURAMAP_URL = os.getenv(
    "SAKURAMAP_URL",
    # いただいた共有URL（/edit は実行時に /viewer + femb=1 へ正規化されます）
    "https://www.google.com/maps/d/u/1/edit?mid=1DXUZRnCUkppZ-WSmvInN5kWBgO0zCbw&usp=sharing"
).strip()
SAKURAMAP_THUMB = os.getenv("SAKURAMAP_THUMB", "").strip()  # 任意：サムネ画像（無くてもOK）

def sakura_map_url() -> str:
    """My Maps の /edit URL → /viewer + femb=1 に正規化して返す"""
    return _normalize_mymaps_url(SAKURAMAP_URL)

# ==== 共通フレックス（タイトル・URL・サムネを受け取る） =========
def _flex_mymap(title: str, url: str, thumb: str = "", subtitle: str | None = None):
    """
    単一マップのバブル or カルーセル（1枚）。
    subtitle が空/None のときは2行目を入れない（LINEの空text禁止対策）。
    サムネは署名URL化（ファイル名想定）。http(s) のときのみ素通し。
    """
    title = (title or "マップ").strip()
    if not url:
        return None

    # 署名付きサムネ（ファイル名想定）。http(s) の場合だけ素通し。
    hero = None
    thumb_url = ""
    t = (thumb or "").strip()
    if t:
        try:
            if t.startswith("http://") or t.startswith("https://"):
                thumb_url = t
            else:
                thumb_url = build_signed_image_url(t, wm=True, external=True)
        except Exception:
            thumb_url = ""
    if thumb_url:
        hero = {"type": "image", "url": thumb_url, "size": "full", "aspectMode": "cover", "aspectRatio": "16:9"}

    body_contents = [{"type": "text", "text": title, "weight": "bold", "wrap": True}]
    if subtitle:
        st = str(subtitle).strip()
        if st:
            body_contents.append({"type": "text", "text": st, "size": "sm", "color": "#666666", "wrap": True})


    bubble = {
        "type": "bubble",
        **({"hero": hero} if hero else {}),
        "body": {"type": "box", "layout": "vertical", "spacing": "sm", "contents": body_contents},
        "footer": {
            "type": "box", "layout": "vertical", "spacing": "sm",
            "contents": [{"type": "button", "style": "primary",
                          "action": {"type": "uri", "label": "マップを開く", "uri": url}}]
        }
    }
    return {"type": "carousel", "contents": [bubble]}

def _flex_map_series_carousel(exclude_key: str | None = None):
    """
    MAP_SERIES からカルーセルを作成。
    - exclude_key があればその要素を除外
    - URLが生成できない要素はスキップ
    - subtitle/desc が空なら2行目は入れない（LINEの空text禁止対策）
    - サムネは署名付きURL（失敗時のみ素のHTTP URLを許容）
    """
    bubbles = []
    for m in MAP_SERIES:
        if exclude_key and m.get("key") == exclude_key:
            continue

        # URL生成（callable/url両対応）
        url_fn = m.get("url_fn")
        url = url_fn() if callable(url_fn) else (m.get("url") or "")
        url = (url or "").strip()
        if not url:
            continue

        title = (m.get("title") or "マップ").strip()
        subtitle = (m.get("subtitle") or m.get("desc") or "").strip()
        thumb_in = (m.get("thumb") or "").strip()

        # hero画像（署名URLを優先、失敗時はHTTP直URLのみ許容）
        hero = None
        if thumb_in:
            thumb_url = ""
            try:
                thumb_url = build_signed_image_url(thumb_in, wm=True, external=True)
            except Exception:
                # 署名生成に失敗した場合、http(s) 直URLだけ通す
                if thumb_in.startswith("http://") or thumb_in.startswith("https://"):
                    thumb_url = thumb_in
            if thumb_url:
                hero = {
                    "type": "image",
                    "url": thumb_url,
                    "size": "full",
                    "aspectMode": "cover",
                    "aspectRatio": "16:9",
                }

        # 本文（空行は作らない）
        body_contents = [
            {"type": "text", "text": title, "weight": "bold", "wrap": True},
        ]
        if subtitle:
            body_contents.append(
                {"type": "text", "text": subtitle, "size": "sm", "color": "#666666", "wrap": True}
            )
        
        bubble = {
            "type": "bubble",
            **({"hero": hero} if hero else {}),
            "body": {"type": "box", "layout": "vertical", "spacing": "sm", "contents": body_contents},
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {"type": "button", "style": "primary",
                     "action": {"type": "uri", "label": "マップを開く", "uri": url}}
                ],
            },
        }
        bubbles.append(bubble)

    if not bubbles:
        return None
    return {"type": "carousel", "contents": bubbles}

# ==== マップシリーズ定義（ここに増やしていける） =================
MAP_SERIES = [
    {
        "key": "viewpoints",
        "title": "五島列島 展望所マップ",
        "url_fn": viewpoints_map_url,
        "thumb":  os.getenv("VIEWPOINTS_THUMB","").strip(),
        "keywords": ["展望", "展望所", "展望台", "viewpoint"],
        "examples": ["展望所マップ", "展望台の地図"]
    },
    {
        "key": "beach",
        "title": "五島列島 海水浴場マップ",
        "url_fn": beach_map_url,
        "thumb":  BEACHMAP_THUMB,
        "keywords": ["海水浴", "海水浴場", "ビーチ", "海水浴マップ", "beach"],
        "examples": ["海水浴場マップ", "ビーチの地図"]
    },
    # ← ここから追加
    {
        "key": "trek",
        "title": "五島列島 登山・トレッキングマップ",
        "url_fn": trek_map_url,            # 上で定義した関数
        "thumb": TREKMAP_THUMB,            # 任意（未設定可）
        "keywords": ["登山", "トレッキング", "ハイキング", "山", "hiking", "trekking", "trail"],
        "examples": ["登山マップ", "トレッキングマップ", "ハイキングの地図"]
    },
    {
        "key": "sakura",
        "title": "五島列島 さくらマップ",
        "url_fn": sakura_map_url,
        "thumb": SAKURAMAP_THUMB,
        "keywords": ["桜", "さくら", "花見", "お花見", "sakura"],
        "examples": ["さくらマップ", "桜の地図", "花見マップ"]
    },
]


def _find_map_by_text(text: str):
    t = _n(text)  # NFKC + lower + 空白圧縮
    has_map_word = ("マップ" in t) or ("地図" in t) or (" map" in t)
    for m in MAP_SERIES:
        if any(k in t for k in m["keywords"]) and has_map_word:
            return m
    try:
        if _is_viewpoints_cmd(text):
            return next(ms for ms in MAP_SERIES if ms["key"]=="viewpoints")
    except Exception:
        pass
    return None

# ★ シリーズ一覧をテキスト1通にまとめるヘルパー（マップ本体のFlexとセットで使う）
def _series_text_for_reply(exclude_key: str | None = None) -> str:
    lines = ["他にもこのようなマップがあります。"]
    for m in MAP_SERIES:
        if exclude_key and m.get("key") == exclude_key:
            continue
        title = (m.get("title") or "マップ").strip()
        # URL が作れなければ「後日追加」表記で載せる
        url_fn = m.get("url_fn")
        url_ok = (url_fn() if callable(url_fn) else "") != ""
        if url_ok:
            lines.append(f"・{title}")
        else:
            lines.append(f"・{title}（後日追加）")
    lines.append("マップを見たい場合はそのマップの名前を送ってください。")
    return "\n".join(lines)

def entry_open_map_url(e: dict, *, lat: float|None=None, lng: float|None=None) -> str:
    """
    1) エントリの map / map_url が Google の共有URLならそれを優先（店名が出やすい）
    2) place_id があれば API=1 の query_place_id で “店名つき”検索URLを生成
    3) どちらも無ければ、名前/住所 or 緯度経度で Google Maps の検索URLを組み立てる
    """
    if not isinstance(e, dict):
        return ""

    # map / map_url を優先採用（Googleの共有URLならそのまま返す）
    def _is_google_maps_url(u: str) -> bool:
        try:
            pu = _u.urlparse(u)
            host = (pu.netloc or "").lower()
            path = (pu.path or "")
            # よく使われる Google Maps ホスト群
            if host.endswith(".google.com") or host.endswith(".google.co.jp") or host == "google.com":
                return "/maps" in path or "maps" in host
            # 短縮・モバイル共有
            if host in {"maps.app.goo.gl", "goo.gl", "g.co"}:
                return True
        except Exception:
            pass
        return False

    m = (e.get("map") or e.get("map_url") or "").strip()
    if m and _is_google_maps_url(m):
        return m

    # place_id があれば “店名つき”で開く（API=1）
    pid = (e.get("place_id") or "").strip()
    if pid:
        title = (e.get("title") or "").strip()
        addr  = (e.get("address") or "").strip()
        q = title or addr
        # タイトル・住所が空なら座標で検索名を埋める
        if not q:
            # 引数で来なければエントリから推定
            if lat is None or lng is None:
                try:
                    la, ln = _entry_latlng(e)
                    lat = lat if lat is not None else la
                    lng = lng if lng is not None else ln
                except Exception:
                    pass
            if lat is not None and lng is not None:
                q = f"{lat},{lng}"
        base = "https://www.google.com/maps/search/?api=1"
        q_enc = _u.quote(q) if q else ""
        return f"{base}&query={q_enc}&query_place_id={_u.quote(pid)}"

    # フォールバック：検索URLを生成
    # 1) 座標が無ければエントリから推定
    if lat is None or lng is None:
        try:
            la, ln = _entry_latlng(e)
            lat = lat if lat is not None else la
            lng = lng if lng is not None else ln
        except Exception:
            pass

    base = "https://www.google.com/maps/search/?api=1"
    if lat is not None and lng is not None:
        # 座標があれば座標検索
        return f"{base}&query={_u.quote(f'{lat},{lng}')}"
    else:
        # 名前＋住所で検索（店名が空でも住所だけで可）
        name = (e.get("title") or "").strip()
        addr = (e.get("address") or "").strip()
        q = " ".join(x for x in [name, addr] if x).strip()
        if not q:
            return ""  # 生成できる材料が無い
        return f"{base}&query={_u.quote(q)}"

def _extract_latlng_from_map(map_url: str|None):
    if not map_url:
        return (None, None)
    try:
        # 1) /@lat,lng, の形（Google Maps 共有URLでよくある）
        m = re.search(r'@(-?\d+\.\d+),\s*(-?\d+\.\d+)', map_url)
        if m:
            return (_float(m.group(1)), _float(m.group(2)))
        # 2) クエリパラメータ（ll= / center= / q=）
        parsed = _u.urlparse(map_url)
        qs = _u.parse_qs(parsed.query)
        for key in ("ll", "center", "q"):
            if key in qs:
                s = _u.unquote(qs[key][0])
                m = re.match(r'\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*', s)
                if m:
                    return (_float(m.group(1)), _float(m.group(2)))
    except Exception:
        pass
    return (None, None)

def _entry_latlng(e: dict):
    """
    エントリから (lat, lng) を堅牢に取り出す。
    優先順:
      1) e['lat'], e['lng']（数値化＋範囲チェック）
      2) extras['lat'], extras['lng']
      3) _extract_latlng_from_map(e['map']) があればそれ
      4) map / map_url 文字列からのローカル抽出（/@lat,lng / ?q= / ?ll= / 純粋 lat,lng）
    最後に (lat,lng) を 6桁に丸めて返す。見つからなければ (None, None)。
    """
    if not isinstance(e, dict):
        return (None, None)

    def _num(v):
        try:
            if v is None or v == "":
                return None
            return float(str(v).replace(",", "."))
        except Exception:
            return None

    def _in_range(lat, lng):
        if lat is None or lng is None:
            return False
        try:
            lat = float(lat); lng = float(lng)
            return (-90.0 <= lat <= 90.0) and (-180.0 <= lng <= 180.0)
        except Exception:
            return False

    # 1) 直接フィールド
    lat = _num(e.get("lat"))
    lng = _num(e.get("lng"))
    if _in_range(lat, lng):
        return (round(lat, 6), round(lng, 6))

    # 2) extras 経由（ある場合のみ）
    ex = e.get("extras") or {}
    lat2 = lat if lat is not None else _num(ex.get("lat"))
    lng2 = lng if lng is not None else _num(ex.get("lng"))
    if _in_range(lat2, lng2):
        return (round(lat2, 6), round(lng2, 6))

    # 3) 既存の抽出関数があれば優先
    try:
        _extract = globals().get("_extract_latlng_from_map")
        if callable(_extract):
            a, b = _extract(e.get("map") or e.get("map_url") or "")
            if _in_range(a, b):
                return (round(a, 6), round(b, 6))
    except Exception:
        pass

    # 4) ローカル解析（/@lat,lng / ?q= / ?ll= / 純粋 lat,lng）
    import re
    from urllib.parse import urlparse, parse_qs

    def _from_text(s: str):
        if not s:
            return (None, None)
        s = str(s).strip()

        # /@lat,lng
        m = re.search(r"/@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)))
            except Exception:
                pass

        # ?q=lat,lng / ?query= / ?ll=
        try:
            pu = urlparse(s)
            qs = parse_qs(pu.query or "")
            for key in ("q", "query", "ll"):
                if key in qs and qs[key]:
                    val = qs[key][0]
                    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)", val)
                    if m:
                        return (float(m.group(1)), float(m.group(2)))
        except Exception:
            pass

        # 純粋 lat,lng（カンマ or 空白）
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)))
            except Exception:
                pass

        return (None, None)

    a, b = _from_text(e.get("map") or e.get("map_url") or "")
    # lat/lng の取り違え救済（片側だけ >90 のとき）
    if a is not None and b is not None:
        if abs(a) > 90 and abs(b) <= 90:
            a, b = b, a
        if _in_range(a, b):
            return (round(a, 6), round(b, 6))

    return (None, None)

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    # 地球半径(キロ)
    R = 6371.0
    try:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
        return R * (2*math.asin(math.sqrt(a)))
    except Exception:
        return 1e9  # 計算不可時は超遠距離扱い

# 互換ラッパー：_boolish を呼ぶコードのための別名
def _boolish(x) -> bool:
    return _boolish_strict(x)

def _build_image_urls(img_name: str | None):
    """サムネ用（透かし付き）と原寸用（署名付き）"""
    if not img_name:
        return {"thumb": "", "image": ""}

    try:
        # 署名付き・絶対URL（Flex対応）
        thumb = build_signed_image_url(img_name, wm=True,  external=True)
        orig  = build_signed_image_url(img_name, wm=False, external=True)
    except Exception:
        # フォールバックも絶対URL +（必要なら）署名付きに
        try:
            thumb = safe_url_for("serve_image", filename=img_name, _external=True, _sign=True, wm=1)
            orig  = safe_url_for("serve_image", filename=img_name, _external=True, _sign=True)
        except Exception:
            # それでも失敗したときの最後のフォールバック
            thumb = url_for("serve_image", filename=img_name, _external=True) + "?wm=1"
            orig  = url_for("serve_image", filename=img_name, _external=True)

    return {"thumb": thumb, "image": orig}

def _build_image_urls_for_entry(e: dict) -> dict:
    """
    エントリから画像URL（署名付き）を作る。
    戻り値例: {"image": "...", "thumb": "..."}
    - wm_external_choice が 'fully' → fullygoto, 'city' → gotocity
    - 旧フラグ wm_on=True の場合は gotocity を既定採用
    """
    try:
        img_name = (e.get("image_file") or e.get("image") or "").strip()
        if not img_name:
            return {}
        # 透かしの既定
        choice = (e.get("wm_external_choice") or "").strip().lower()
        if choice in ("fullygoto", "gotocity"):
            mode = choice
        elif choice == "fully":
            mode = "fullygoto"
        elif choice == "city":
            mode = "gotocity"
        elif e.get("wm_on"):
            mode = "gotocity"  # 旧UIのONは市ロゴ既定
        else:
            mode = None

        if mode:
            s = build_signed_image_url(img_name, wm=mode, external=True)
        else:
            s = build_signed_image_url(img_name, wm=False, external=True)

        return {"image": s, "thumb": s}
    except Exception:
        app.logger.exception("_build_image_urls_for_entry failed")
        return {}


def line_image_pair_for_entry(e: dict) -> tuple[str|None, str|None]:
    """
    LINEのImageSendMessage用に (original_url, preview_url) を返す。
    - サムネは使わず、ラジオ選択（wm_external_choice）をそのまま原寸に反映
    - 'none' のときは wm なし（=透かし無し）
    """
    img = (e.get("image_file") or e.get("image") or "").strip()
    if not img:
        return None, None

    choice = (e.get("wm_external_choice") or "").strip().lower()
    if choice in ("fully", "city"):
        choice = "fullygoto" if choice == "fully" else "gotocity"
    elif choice not in ("none", "fullygoto", "gotocity"):
        legacy = e.get("wm_external") or e.get("wm_ext_fully") or e.get("wm_ext")
        choice = "gotocity" if legacy else "none"

    try:
        if choice == "none":
            # 透かしなし
            u = build_signed_image_url(img, wm=False, external=True)
            return u, u
        else:
            # 指定の透かしを入れる
            u = build_signed_image_url(img, wm=choice, external=True)
            return u, u
    except Exception:
        # フォールバック（safe_url_forで署名 + wmを明示）
        try:
            if choice == "none":
                u = safe_url_for("serve_image", filename=img, _external=True, _sign=True)
            else:
                u = safe_url_for("serve_image", filename=img, _external=True, _sign=True, wm=choice)
            return u, u
        except Exception:
            return None, None


def _nearby_core(
    lat: float,
    lng: float,
    *,
    radius_m: int = 1500,
    cat_filter: set[str] | None = None,
    limit: int = 20,
    record_sources: bool = False,
):
    """
    現在地 (lat, lng) から半径 radius_m[m] 以内の entries を距離昇順で返す。
    - cat_filter: {カテゴリー名,...} を指定すると一致するものだけに絞り込み
    - limit: 返す件数（最低1件）
    - record_sources: True のとき、ヒットした各エントリの「出典」テキストを
      g.response_sources に登録（フォームの 'source' が空なら何もしない）
    """
    try:
        items = [_norm_entry(x) for x in load_entries()]
    except Exception:
        items = []

    rows = []
    for i, e in enumerate(items):
        # カテゴリフィルタ
        cat = (e.get("category") or "").strip()
        if cat_filter and cat not in cat_filter:
            continue

        # 位置取得（エントリのlat/lng or map URLから推定）
        elat, elng = _entry_latlng(e)
        if elat is None or elng is None:
            continue

        # 距離フィルタ
        d_km = _haversine_km(lat, lng, elat, elng)
        if d_km * 1000 > radius_m:
            continue

        # 画像URL（サムネは透かし、原寸は外部公開ポリシーに従う）
        imgs = _build_image_urls_for_entry(e)

        row = {
            "idx": i,
            "title": e.get("title", ""),
            "desc": e.get("desc", ""),
            "address": e.get("address", ""),
            "category": cat,
            "areas": e.get("areas") or [],
            "tags": e.get("tags") or [],
            "lat": elat,
            "lng": elng,
            "distance_m": int(round(d_km * 1000)),
            "google_url": entry_open_map_url(e, lat=elat, lng=elng),
            "mymap_url": mymap_view_url(lat=elat, lng=elng) if MYMAP_MID else "",
            "image_thumb": imgs.get("thumb", ""),
            "image_url": imgs.get("image", ""),
        }

        # ★ 出典登録はフラグで制御（フォームの「出典」テキストが空なら何もしない）
        if record_sources:
            record_source_from_entry(e)

        rows.append(row)

    rows.sort(key=lambda x: x["distance_m"])
    # limit は最低 1 件返す
    try:
        lim = int(limit)
    except Exception:
        lim = 20
    return rows[:max(1, lim)]

def _nearby_flex(items: list[dict]):
    bubbles = []
    for it in items:
        body = [
            {"type":"text","text":it["title"] or "無題","weight":"bold","wrap":True},
            {"type":"text","text":f'{it.get("distance_m",0)} m ・ {it.get("category","")}',"size":"sm","color":"#888888"},
        ]
        if it.get("address"):
            body.append({"type":"text","text":it["address"],"size":"sm","wrap":True,"color":"#666666"})
        footer = {"type":"box","layout":"vertical","spacing":"sm","contents":[
            {"type":"button","style":"primary","action":{"type":"uri","label":"地図で開く","uri": it.get("google_url","")}},
        ]}
        if it.get("mymap_url"):
            footer["contents"].append({"type":"button","style":"secondary","action":{"type":"uri","label":"周辺をマイマップ","uri": it["mymap_url"]}})

        hero = build_flex_hero(it)

        bubble = {
            "type": "bubble",
            **({"hero": hero} if hero else {}),
            "body": {"type":"box","layout":"vertical","spacing":"sm","contents": body},
            "footer": footer
        }

        bubbles.append(bubble)
    return {"type":"carousel","contents":bubbles} if bubbles else None

def _ask_location(text="現在地から近い順で探します。位置情報を送ってください。"):
    return TextSendMessage(
        text=text,
        quick_reply=QuickReply(items=[QuickReplyButton(action=LocationAction(label="現在地を送る"))])
    )


def _classify_mode(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["観光", "観る", "名所", "景勝", "スポット"]):
        return "tourism"
    if any(k in t for k in ["店", "お店", "飲食", "食べる", "呑む", "ショップ", "買い物", "カフェ", "レストラン", "宿泊", "ホテル", "旅館"]):
        return "shop"
    return "all"

def _mode_to_cats(mode: str):
    # あなたの CATEGORIES に合わせて調整してOK
    if mode == "tourism":
        return {"観光", "イベント", "癒し"}
    if mode == "shop":
        return {"食べる・呑む", "ショップ", "泊まる", "生活", "きれい"}
    return None  # all


@app.route("/api/nearby", methods=["GET"])
def api_nearby():
    """
    例: /api/nearby?lat=32.7&lng=128.8&r=2000&cat=観光,飲食&limit=20
    - lat,lng: 現在地（必須）
    - r: 半径[m]（デフォルト 1500）
    - cat: カンマ区切りカテゴリ（省略可）
    - limit: 件数（省略可）
    """
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
    except Exception:
        return jsonify({"ok": False, "error": "lat/lng が不正です"}), 400

    try:
        radius_m = int(request.args.get("r", 1500))
    except Exception:
        radius_m = 1500

    cat_param = (request.args.get("cat") or "").strip()
    cats = {c.strip() for c in cat_param.split(",") if c.strip()} if cat_param else None

    try:
        limit = int(request.args.get("limit", 20))
    except Exception:
        limit = 20

    rows = _nearby_core(lat, lng, radius_m=radius_m, cat_filter=cats, limit=limit)
    return jsonify({"ok": True, "count": len(rows), "items": rows})


# =========================
#  正規化ヘルパー
# =========================
def _split_lines_commas(val: str) -> List[str]:
    """改行・カンマ両対応で分割 → 余白除去 → 空要素除去"""
    if not val:
        return []
    parts = re.split(r'[\n,]+', str(val))
    return [p.strip() for p in parts if p and p.strip()]

# 置き換え版（1本に統一・limit省略OK・絵文字安全）
def _split_for_line(text: str,
                    limit: int | None = None,
                    max_len: int | None = None,
                    **_ignored) -> List[str]:
    """
    LINEの1通上限を超える長文を“安全に”分割するユーティリティ。
    - limit 未指定なら env/グローバルの LINE_SAFE_CHARS（既定4800）を採用
    - 段落→文の順に自然に切る。超過時はハードスプリット
    - どんな入力でも最低1要素返す（空配列にしない）
    - len()ではなくUTF-16コードユニット数でカウント（絵文字混在に強い）
    """
    import os, re

    def u16len(s: str) -> int:
        # UTF-16 LE のコードユニット数（LINEの仕様に近い）
        return len(s.encode("utf-16-le")) // 2

    # 実効上限を決定
    if limit is None:
        limit = max_len
    if limit is None:
        # env > グローバル > 既定 の順で採用
        try:
            limit = int(os.getenv("LINE_SAFE_CHARS", str(globals().get("LINE_SAFE_CHARS", 4800))))
        except Exception:
            limit = 4800
    if limit <= 0:
        return ["" if text is None else str(text)]

    s = "" if text is None else str(text)
    if u16len(s) <= limit:
        return [s]

    # 段落単位（空行で分割）
    paragraphs = [p for p in re.split(r"\n\s*\n", s) if p != ""]
    chunks: List[str] = []

    def flush_buf(buf: str) -> None:
        if buf:
            chunks.append(buf)

    def hard_split(token: str) -> None:
        """1トークン自体が長過ぎる場合、UTF-16長を見ながら強制分割"""
        buf = ""
        for ch in token:
            if u16len(buf + ch) > limit:
                flush_buf(buf)
                buf = ch
            else:
                buf += ch
        flush_buf(buf)

    # 1) 段落→2) 文→3) ハードスプリット の順で収める
    buf = ""
    SENT_SPLIT = re.compile(r"(?<=[。．！？!?])")  # 句点等の直後で区切る
    for para in paragraphs:
        para = para.strip("\n")
        # 段落丸ごと入るなら入れる
        if u16len(para) <= limit:
            if u16len(buf + (("\n\n" + para) if buf else para)) <= limit:
                buf = (buf + ("\n\n" if buf else "") + para)
            else:
                flush_buf(buf); buf = para
            continue

        # 文単位で詰める
        sentences = [x for x in SENT_SPLIT.split(para) if x]
        for sent in sentences:
            if u16len(sent) > limit:
                # 文がそもそも長い → いったん今のbufを吐き出してハード分割
                flush_buf(buf); buf = ""
                hard_split(sent)
                continue
            # 既存bufに足せるなら足す
            sep = ("\n" if (buf and not buf.endswith("\n")) else "")
            candidate = buf + (sep + sent if buf else sent)
            if u16len(candidate) <= limit:
                buf = candidate
            else:
                flush_buf(buf); buf = sent

        # 段落の終わりで改行を入れたい場合はここで調整してもOK
    flush_buf(buf)

    # 念のため空にならない保証
    if not chunks:
        chunks = [s[:limit]]

    return chunks

# === 緯度経度：コピペ用のパーサ ===
import re as _re2

def _parse_dms_block(s: str):
    """
    DMS（度分秒）1ブロックを小数に変換（例: 35°41'6.6"N / 北緯35度41分6.6秒 / 139°41'30"E）
    戻り値: (value, axis)  axis は 'lat' / 'lng' / None
    """
    s = s.strip()
    hemi = None
    if _re2.search(r'[N北]', s, _re2.I): hemi = 'N'
    if _re2.search(r'[S南]', s, _re2.I): hemi = 'S'
    if _re2.search(r'[E東]', s, _re2.I): hemi = 'E'
    if _re2.search(r'[W西]', s, _re2.I): hemi = 'W'

    m = _re2.search(
        r'(\d+(?:\.\d+)?)\s*[°度]\s*(\d+(?:\.\d+)?)?\s*[\'’′分]?\s*(\d+(?:\.\d+)?)?\s*["”″秒]?',
        s
    )
    if not m:
        return None, None

    deg = float(m.group(1))
    minutes = float(m.group(2) or 0.0)
    seconds = float(m.group(3) or 0.0)
    val = deg + minutes/60.0 + seconds/3600.0
    if hemi in ('S','W'):
        val = -val

    axis = None
    if hemi in ('N','S'):
        axis = 'lat'
    elif hemi in ('E','W'):
        axis = 'lng'
    return val, axis


def parse_latlng_any(text: str):
    """
    Googleマップのコピペ（URL/小数/DMS/日本語表記）を (lat, lng) へ正規化。
    例:
      35.681236, 139.767125
      https://www.google.com/maps?q=35.681236,139.767125
      https://www.google.com/maps/@35.681236,139.767125,17z
      35°41'6.6"N 139°41'30"E
      北緯35度41分6.6秒 東経139度41分30秒
    """
    if not text:
        return None
    s = text.strip().replace('，', ',').replace('、', ',')
    s = _re2.sub(r'\s+', ' ', s)

    # URL ?q=lat,lng / ?query=lat,lng
    m = _re2.search(r'[?&](?:q|query)=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)', s)
    if m:
        return float(m.group(1)), float(m.group(2))

    # URL /@lat,lng,...
    m = _re2.search(r'/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(?:[,/?]|$)', s)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 純粋な「lat,lng」
    m = _re2.search(r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a) > 90 and abs(b) <= 90:
            a, b = b, a
        return a, b

    # DMS ブロック2つ拾う
    dms_blocks = _re2.findall(
        r'(\d+(?:\.\d+)?\s*[°度]\s*\d*(?:\.\d+)?\s*[\'’′分]?\s*\d*(?:\.\d+)?\s*["”″秒]?\s*[NSEW北南東西]?)',
        s, flags=_re2.I
    )
    if len(dms_blocks) >= 2:
        v1, a1 = _parse_dms_block(dms_blocks[0])
        v2, a2 = _parse_dms_block(dms_blocks[1])
        if v1 is not None and v2 is not None:
            if not a1 and not a2:
                cand = sorted([v1, v2], key=lambda x: abs(x))
                return cand[0], cand[1]
            lat = v1 if (a1 == 'lat' or (a1 is None and abs(v1) <= 90)) else v2
            lng = v2 if lat == v1 else v1
            return lat, lng

    # 「緯度: xx / 経度: yy」
    mlat = _re2.search(r'緯度[:：]\s*(-?\d+(?:\.\d+)?)', s)
    mlng = _re2.search(r'経度[:：]\s*(-?\d+(?:\.\d+)?)', s)
    if mlat and mlng:
        return float(mlat.group(1)), float(mlng.group(1))

    return None


def normalize_latlng(raw_coords: str, lat_str: str, lng_str: str):
    """
    3入力（rawコピペ / lat / lng）から最終 (lat, lng) を決める。小数6桁へ丸め。
    """
    lat = lng = None
    try:
        if lat_str: lat = float(lat_str)
        if lng_str: lng = float(lng_str)
    except ValueError:
        pass

    if (lat is None or lng is None) and raw_coords:
        pair = parse_latlng_any(raw_coords)
        if pair:
            lat, lng = pair

    if lat is not None and not (-90 <= lat <= 90): lat = None
    if lng is not None and not (-180 <= lng <= 180): lng = None

    if lat is not None and lng is not None:
        lat = round(lat, 6); lng = round(lng, 6)
    return lat, lng



def _norm_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """保存前/表示前にデータ構造を正規化"""
    e = dict(e or {})

    # 文字列 -> 配列に寄せる
    for k in ("tags", "areas", "links", "payment"):
        v = e.get(k)
        if v is None:
            e[k] = []
        elif isinstance(v, str):
            e[k] = _split_lines_commas(v)
        elif isinstance(v, (list, tuple)):
            e[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            e[k] = [str(v).strip()]

    # extras を dict に寄せる
    x = e.get("extras")
    if isinstance(x, dict):
        e["extras"] = {str(k): ("" if v is None else str(v)) for k, v in x.items()}
    elif isinstance(x, list):
        tmp: Dict[str, str] = {}
        for it in x:
            if isinstance(it, dict):
                for k, v in it.items():
                    tmp[str(k)] = "" if v is None else str(v)
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                tmp[str(it[0])] = "" if it[1] is None else str(it[1])
        e["extras"] = tmp
    elif isinstance(x, str) and x.strip():
        e["extras"] = {"備考": x.strip()}
    else:
        e["extras"] = {}

    # 型の下駄（無ければ空文字）
    for k in ("category","title","desc","address","map","tel","holiday",
              "open_hours","parking","parking_num","remark",
              "source","source_url"):
        if e.get(k) is None:
            e[k] = ""

    # category の既定
    if not e["category"]:
        e["category"] = "観光"

    # === 透かし選択（後方互換）===
    # 文字列化してから正規化（bool 等が来ても落ちないように）
    wm_choice = str(e.get("wm_external_choice") or "").strip().lower()
    # 新旧トークンを正規化
    if wm_choice in ("fully", "fullygoto"):
        wm_choice = "fullygoto"
    elif wm_choice in ("city", "gotocity"):
        wm_choice = "gotocity"
    elif wm_choice not in ("none", "fullygoto", "gotocity"):
        # 旧ブール系フラグが True なら "gotocity" を既定に
        legacy = any(_boolish(e.get(k)) for k in ("wm_external", "wm_ext_fully", "wm_ext"))
        wm_choice = "gotocity" if legacy else "none"
    e["wm_external_choice"] = wm_choice
    e["wm"] = wm_choice          # ← 追加（新コードが参照するエイリアス）

    return e


# ← ここに追加（この下から各種設定が続く）
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))  # お好みで
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Flask本体と MAX_CONTENT_LENGTH の設定のすぐ後に追加
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))  # 例: 40MP
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS            # これを超えたら Pillow が警告/例外
ImageFile.LOAD_TRUNCATED_IMAGES = False              # 途中で切れた画像は拒否
warnings.simplefilter("error", Image.DecompressionBombWarning)  # 警告を例外化

# ---- カテゴリマスタ（登録フォームで選択させる用）----
CATEGORIES = [
    "求人",
    "五島を贈ろう",
    "観光",
    "泊まる",
    "食べる・呑む",
    "きれい",
    "癒し",
    "ショップ",
    "生活",
    "趣味・習い事",
]
app.jinja_env.globals["CATEGORIES"] = CATEGORIES  # テンプレートから参照できるようにする


# --- LINE 応答制御フラグ（環境変数で上書き可） ---
# 1質問=原則1通（長文で収まらない場合のみ自動分割）
LINE_SINGLE_REPLY = os.getenv("LINE_SINGLE_REPLY", "1").lower() in {"1", "true", "on", "yes"}

# まずエリアを尋ねる（曖昧ワードのとき）
LINE_ASK_AREA_FIRST = os.getenv("LINE_ASK_AREA_FIRST", "1").lower() in {"1", "true", "on", "yes"}

# LINE は使わない環境では callback を閉じる/無効化でもOK

# --- Jinja2 互換用: 'string' / 'mapping' テストが無い環境向け ---
if 'string' not in app.jinja_env.tests:
    app.jinja_env.tests['string'] = lambda v: isinstance(v, str)
if 'mapping' not in app.jinja_env.tests:
    from collections.abc import Mapping
    app.jinja_env.tests['mapping'] = lambda v: isinstance(v, Mapping)

app.config["JSON_AS_ASCII"] = False  # 日本語をJSONでそのまま返す


# === 透かしコマンド／ユーザー選好ユーティリティ =========================
# 永続化先（必要なら .env で差し替え可）
WM_PREFS_PATH = Path(os.getenv("WM_PREFS_PATH", "data/user_prefs.json"))

def _load_user_prefs() -> dict:
    try:
        return json.loads(WM_PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_user_prefs(prefs: dict) -> None:
    WM_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WM_PREFS_PATH.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")

def _get_user_wm(uid: str) -> str:
    """
    返り値: 'none' | 'fullygoto' | 'gotocity'
    既定は 'none'
    """
    prefs = _load_user_prefs()
    return (prefs.get(uid, {}) or {}).get("wm_choice", "none")

def _set_user_wm(uid: str, choice: str) -> None:
    """
    choice は 'none' | 'fullygoto' | 'gotocity'
    """
    if choice not in ("none", "fullygoto", "gotocity"):
        return
    prefs = _load_user_prefs()
    prefs.setdefault(uid, {})
    prefs[uid]["wm_choice"] = choice
    # 旧互換: none以外なら "wm_on": True としておく
    prefs[uid]["wm_on"] = (choice != "none")
    _save_user_prefs(prefs)

def _normalize_at_command(s: str) -> str:
    """
    全角→半角、空白畳み込み、lower化（@FullyGOTO / ＠Goto City / none / なし 等に耐える）
    """
    if not s:
        return ""
    table = str.maketrans({"＠": "@", "　": " "})
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def parse_wm_command(text: str):
    """
    返り値: 'gotocity' | 'fullygoto' | 'none' | None
    完全一致のみ。優先順位: City → fully → none
    """
    t = _normalize_at_command(text)

    # City を先に判定（"goto" を含むため fully へ誤爆しやすい）
    if re.fullmatch(r"(?:@)?goto\s*city", t, flags=re.I):
        return "gotocity"
    if re.fullmatch(r"(?:@)?fully\s*goto", t, flags=re.I):
        return "fullygoto"
    if re.fullmatch(r"(?:@)?(?:なし|none)", t, flags=re.I):
        return "none"
    return None
# ======================================================================

# 本番では必ず環境変数で設定
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")

# セッション設定
SECURE_COOKIE = os.environ.get("SESSION_COOKIE_SECURE", "1").lower() in {"1", "true", "on", "yes"}
app.config.update(
    SESSION_COOKIE_SECURE=SECURE_COOKIE,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=12),
)
# ==== Rate limit setup (put right after Flask settings) ====
from werkzeug.middleware.proxy_fix import ProxyFix

# リバースプロキシ配下で正しい IP/スキームを取る
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "2"))

# 早期に APP_ENV を取得（このブロック内だけで使う）
_APP_ENV_EARLY = os.getenv("APP_ENV", "dev").lower()

# 本番で ProxyHop が 0 以下は危険なのでブロック
if _APP_ENV_EARLY in {"prod", "production"} and TRUSTED_PROXY_HOPS <= 0:
    raise RuntimeError("TRUSTED_PROXY_HOPS must be >= 1 in production")

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXY_HOPS, x_proto=1, x_host=1, x_port=1)

# flask-limiter が無い環境でも落ちないように（util も含めて try）
try:
    from flask_limiter import Limiter
    try:
        from flask_limiter.util import get_remote_address as _limiter_remote
    except Exception:
        _limiter_remote = None
except Exception:
    Limiter = None
    _limiter_remote = None

def _remote_ip():
    # ProxyFix 適用後は remote_addr が実クライアントIP
    return (request.remote_addr or "0.0.0.0").strip()

# 文字列は「;」区切りで複数指定可（例: "20/minute;1000/day"）
ASK_LIMITS = os.environ.get("ASK_LIMITS", "20/minute;1000/day")
# 例: RATE_STORAGE_URI="redis://:pass@redis:6379/0"
RATE_STORAGE_URI = os.environ.get("RATE_STORAGE_URI") or "memory://"

# 本番で memory:// は共有されず危険なので禁止
if _APP_ENV_EARLY in {"prod", "production"} and (not RATE_STORAGE_URI or RATE_STORAGE_URI.startswith("memory://")):
    raise RuntimeError("In production, set RATE_STORAGE_URI to a shared backend (e.g., redis://...)")

if Limiter:
    # ★一度だけ初期化し、必要なら後からデコレータで利用
    limiter = Limiter(key_func=_limiter_remote or _remote_ip, storage_uri=RATE_STORAGE_URI)
    limiter.init_app(app)
    limit_deco = limiter.limit
else:
    limiter = None
    def limit_deco(*a, **k):
        def _wrap(f): return f
        return _wrap

LOGIN_LIMITS = os.getenv("LOGIN_LIMITS", "10/minute;100/day")

@app.errorhandler(429)
def _ratelimit_handler(e):
    return jsonify({"error": "Too Many Requests", "detail": "Rate limit exceeded."}), 429
# ==== /Rate limit setup ====

@app.errorhandler(RequestEntityTooLarge)
@app.errorhandler(413)
def _too_large(e):
    # API っぽいリクエストは JSON、それ以外は画面に戻す
    wants_json = (
        request.is_json
        or "application/json" in (request.headers.get("Accept","") or "")
        or request.path.startswith("/api/")
    )

    # 例外説明に "pixels" が含まれていれば画像のピクセル上限超過とみなす
    desc = (getattr(e, "description", "") or "").lower()
    is_pixels = "pixel" in desc or "pixels" in desc

    if wants_json:
        if is_pixels:
            return jsonify({
                "error": "Image too large",
                "reason": "pixels",
                "max_image_pixels": MAX_IMAGE_PIXELS
            }), 413
        else:
            return jsonify({
                "error": "File too large",
                "reason": "body",
                "limit_mb": MAX_UPLOAD_MB
            }), 413

    # HTML系はフラッシュして元画面へ
    if is_pixels:
        flash(f"画像のピクセル数が大きすぎます（上限 {MAX_IMAGE_PIXELS:,} ピクセル）")
    else:
        flash(f"ファイルが大きすぎます（上限 {MAX_UPLOAD_MB} MB）")
    return redirect(request.referrer or url_for("admin_entry"))

# ==== 追加（Flask設定の近くでOK）====
ADMIN_IP_ENFORCE = os.getenv("ADMIN_IP_ENFORCE", "1").lower() in {"1","true","on","yes"}
# ====================================
# ====================================
# 出典クレジット（フォームのテキストのみ使用）
# ====================================
from flask import g

# 既定は「フォームだけ」。必要なら 0/false で将来ゆるめられます
SOURCES_ONLY_FORM = os.getenv("SOURCES_ONLY_FORM", "1").lower() in {"1","true","on","yes"}

def record_source(note: str, url: str = ""):
    """
    今回の返信に出典を追加登録（同一リクエスト内）。
    """
    try:
        lst = getattr(g, "response_sources", [])
        note = (note or "").strip()
        url  = (url or "").strip()
        # ★ フォームの“出典”テキストが空なら登録しない
        if not note:
            return
        lst.append({"note": note, "url": url})
        g.response_sources = lst
    except RuntimeError:
        pass  # リクエスト外など

def record_source_from_entry(entry: dict | None):
    """
    エントリから“出典”を登録。
    フォームの『出典』テキスト（entry['source']）が空なら何もしない。
    """
    if not entry:
        return
    note = (entry.get("source") or "").strip()
    if not note:
        return  # ★ テキストが空なら脚注を出さない
    url  = (entry.get("source_url") or "").strip()
    record_source(note, url)

def record_source_from_path(path: str):
    """
    パスからの自動出典付与は無効化（フォームのみを採用）。
    必要になったら SOURCES_ONLY_FORM=0 で再度実装してください。
    """
    if SOURCES_ONLY_FORM:
        return
    # ここには何もしない（互換のため関数は残す）

def _unique_sources(srcs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for s in srcs or []:
        key = ((s.get("note","") or "").strip(), (s.get("url","") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def _sources_footer_text() -> str | None:
    """
    送信前に呼ばれ、登録された出典があれば脚注文字列を返す。
    """
    srcs = _unique_sources(getattr(g, "response_sources", []) or [])
    if not srcs:
        return None
    lines = ["— 出典 —"]
    for s in srcs:
        n = (s.get("note","") or "").strip()
        u = (s.get("url","") or "").strip()
        # テキストは必須。URLはあれば併記（不要ならここで外してOK）
        lines.append(f"{n} {u}".strip())
    return "\n".join(lines)

def _append_sources_if_text(s: str) -> str:
    """
    文字列本文の末尾に出典脚注を付ける。
    出典が未登録（=フォームにテキストが無い）なら何も付けない。
    """
    foot = _sources_footer_text()
    if not foot:
        return s
    s = "" if s is None else str(s)
    # 「出典だけ」を送る用途のときは重複防止
    if s.strip().startswith("— 出典 —"):
        return s
    return (s.rstrip() + "\n\n" + foot)


# ===== ここを Flask 設定の直後に追加 =====
from urllib.parse import urlparse

CSRF_EXEMPT_ENDPOINTS = set()
CSRF_EXEMPT_ENDPOINTS.add("callback")

# CSRF 保護対象のパス接頭辞（管理／店舗）
CSRF_PROTECT_PATHS = ("/admin", "/shop", "/api")



CSRF_STRICT = (os.getenv("CSRF_STRICT", "1").lower() in {"1","true","on","yes"})



@app.before_request
def _csrf_referer_origin_guard():
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return
    if request.endpoint in CSRF_EXEMPT_ENDPOINTS:
        return
    path = request.path or ""
    if not any(path.startswith(p) for p in CSRF_PROTECT_PATHS):
        return

    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")
    src = origin or referer
    if not src:
        if CSRF_STRICT:
            abort(403)
        else:
            return  # 運用フラグで緩和

    host_hdr = request.host  # ProxyFix 適用後は正しい値になる
    if urlparse(src).netloc != host_hdr:
        abort(403)

# ===== 追加ここまで =====

# ===== 管理画面IP制限（/admin, /shop） - ProxyFix 適用後に配置 =====
ADMIN_IP_ALLOWLIST = [
    s.strip() for s in os.getenv("ADMIN_IP_ALLOWLIST", "").replace("\n", ",").split(",")
    if s.strip()
]
APP_ENV = os.getenv("APP_ENV", "dev").lower()
if APP_ENV in {"prod", "production"}:
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise RuntimeError("FLASK_SECRET_KEY must be set in production")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in production")

def _load_admin_nets():
    nets = []
    for s in ADMIN_IP_ALLOWLIST:
        try:
            nets.append(ipaddress.ip_network(s, strict=False))
        except ValueError:
            logging.getLogger(__name__).warning("ADMIN_IP_ALLOWLIST: invalid CIDR skipped: %s", s)
    return nets

_ADMIN_NETS = _load_admin_nets()
# ログ
app.logger.info("[admin-ip] allowlist=%s  APP_ENV=%s",
                ", ".join(map(str, _ADMIN_NETS)) or "(empty)", APP_ENV)

def _admin_ip_ok(ip: str) -> bool:
    # 本番は未設定＝拒否、開発系は未設定でも許可
    if not _ADMIN_NETS:
        return APP_ENV not in {"prod", "production"} 
    try:
        ipobj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(ipobj in n for n in _ADMIN_NETS)

@app.before_request
def _restrict_admin_ip():
    if not ADMIN_IP_ENFORCE:
        return  # ← 開発中はここで早期リターン＝IP制限OFF

    p = request.path or ""
    if p.startswith("/admin") or p.startswith("/shop"):
        client_ip = (request.remote_addr or "").strip()
        if not _admin_ip_ok(client_ip):
            abort(403, description="Forbidden: your IP is not allowed for admin/shop.")
# ===== ここまで =====
@app.route("/nearby", methods=["GET"])
def nearby_page():
    html = """
<!DOCTYPE html><html lang="ja"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>近くのスポット検索</title>
<style>
 body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,'Noto Sans JP',sans-serif;background:#f6faf9;margin:0}
 .wrap{max-width:760px;margin:24px auto;background:#fff;padding:18px 16px;border-radius:16px;box-shadow:0 4px 20px #b7d3e04d}
 h1{margin:0 0 12px 0;font-size:1.2rem}
 .ctl{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0 14px}
 .btn{padding:8px 12px;border:none;border-radius:8px;background:#2a8ed8;color:#fff;cursor:pointer}
 .btn.ghost{background:#fff;color:#2a8ed8;border:1px solid #b7d3e0}
 .muted{color:#678;font-size:.92em}
 .card{display:flex;gap:10px;border:1px solid #e5eef5;background:#fbfeff;border-radius:12px;padding:10px;margin:8px 0}
 .thumb{width:96px;height:72px;object-fit:cover;border-radius:8px;border:1px solid #e2e8f0;background:#fff}
 .ttl{font-weight:600}
 .row{font-size:.95em;color:#345}
 .dist{font-weight:600}
</style>
</head><body>
<div class="wrap">
  <h1>近くのスポット検索</h1>
  <div class="ctl">
    半径
    <select id="r">
      <option value="500">500m</option>
      <option value="1000" selected>1km</option>
      <option value="2000">2km</option>
      <option value="5000">5km</option>
    </select>
    カテゴリ
    <select id="cat">
      <option value="">（指定なし）</option>
      <option>観光</option><option>食べる・呑む</option><option>泊まる</option>
      <option>ショップ</option><option>イベント</option>
    </select>
    <button class="btn" id="btn">現在地から検索</button>
    <span id="status" class="muted"></span>
  </div>
  <div id="list"></div>
</div>
<script>
const $ = s=>document.querySelector(s);
const list = $("#list"), status = $("#status"), btn=$("#btn");
async function search(lat,lng){
  const r = $("#r").value, cat = $("#cat").value;
  status.textContent = "検索中…";
  list.innerHTML = "";
  const qs = new URLSearchParams({lat,lng,r});
  if(cat) qs.set("cat", cat);
  const res = await fetch(`/api/nearby?${qs.toString()}`);
  const data = await res.json().catch(()=>({ok:false}));
  if(!data.ok){ status.textContent = "検索に失敗しました"; return; }
  status.textContent = `見つかった件数：${data.count}`;
  for(const it of data.items){
    const a = document.createElement("div");
    a.className = "card";
    a.innerHTML = `
      ${it.image_thumb ? `<a href="${it.image_url}" target="_blank" rel="noopener"><img class="thumb" src="${it.image_thumb}" loading="lazy" referrerpolicy="no-referrer"></a>` : ``}
      <div class="body">
        <div class="ttl">${it.title}</div>
        <div class="row">${it.address ?? ""}</div>
        <div class="row"><span class="dist">${it.distance_m}</span> m / ${it.category ?? ""}</div>
        <div class="row">
          <a href="${it.google_url}" target="_blank" rel="noopener">地図で開く</a>
          ${it.mymap_url ? ` / <a href="${it.mymap_url}" target="_blank" rel="noopener">周辺をマイマップ</a>` : ``}
        </div>
      </div>`;
    list.appendChild(a);
  }
}
btn.addEventListener("click", ()=>{
  if(!("geolocation" in navigator)){
    status.textContent = "このブラウザは位置情報に対応していません"; return;
  }
  status.textContent = "位置情報を取得中…";
  navigator.geolocation.getCurrentPosition(
    p=>{ search(p.coords.latitude, p.coords.longitude); },
    e=>{ status.textContent = "位置情報の取得を許可してください（HTTPSのみ可）"; },
    {enableHighAccuracy:true, timeout:10000, maximumAge:30000}
  );
});
</script>
</body></html>
    """
    return render_template_string(html)

@app.route("/_debug/ip")
def _debug_ip():
    if APP_ENV in {"prod","production"} and request.headers.get("X-Debug-Token") != os.getenv("DEBUG_TOKEN"):
        abort(404)
    return jsonify({
        "remote_addr": request.remote_addr,
        "x_forwarded_for": request.headers.get("X-Forwarded-For"),
        "host": request.host,
        "scheme": request.scheme,
    })

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("ログインしてください")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def _boolish(x) -> bool:
    return str(x).lower() in {"1","true","on","yes"}

def safe_url_for(endpoint, **values):
    """
    url_for の安全版 + 画像エンドポイントだけ“おまかせ署名”対応。
    - endpoint == "serve_image":
        * _sign=True で署名URL、_sign=False で素URL
        * 既定は「外部(未ログイン)アクセス時は署名」
        * wm は True/1/“fully”/“city”/“fullygoto”/“gotocity”/“none” を受け付ける
        * _external=True で絶対URL
    - それ以外は通常の url_for。BuildError は "#" を返す
    """
    try:
        if endpoint == "serve_image":
            filename = values.get("filename") or values.pop("filename", None)
            if not filename:
                return "#"

            # オプション取り出し
            external = bool(values.pop("_external", False))
            wm_val   = values.pop("wm", values.pop("_wm", None))

            # 未定義でも落ちないようフォールバック
            WATERMARK_ON      = bool(globals().get("WATERMARK_ENABLE", True))
            IMAGE_PROTECT_ON  = bool(globals().get("IMAGE_PROTECT", True))

            # wm の正規化（新旧トークンと真偽値を吸収）
            wm_q = None
            if WATERMARK_ON:
                if isinstance(wm_val, str):
                    v = wm_val.strip().lower()
                    if v in {"fully", "fullygoto"}:
                        wm_q = "fullygoto"
                    elif v in {"city", "gotocity"}:
                        wm_q = "gotocity"
                    elif v in {"none", "0", "off", "false", ""}:
                        wm_q = None
                    elif _boolish(v):
                        wm_q = "1"  # 既定の透かし
                elif wm_val is not None and _boolish(wm_val):
                    wm_q = "1"

            # 署名の既定: 外部(=未ログイン)は署名
            # 変更前: must_sign = (IMAGE_PROTECT_ON and not session.get("user_id"))
            must_sign = (IMAGE_PROTECT_ON and not (session.get("user_id") or session.get("role")))
            sign = _boolish(values.pop("_sign", must_sign))

            # 署名URLを生成
            if sign:
                try:
                    if "build_signed_image_url" in globals() and callable(globals()["build_signed_image_url"]):
                        return build_signed_image_url(filename, wm=(wm_q or False), external=external)
                except Exception:
                    # フォールバック（署名できなければ通常URLに wm だけ付与）
                    pass

            # 署名しない（またはフォールバック）: wm をクエリに付与
            if wm_q:
                values["wm"] = wm_q
            values["filename"] = filename
            values["_external"] = external
            return url_for(endpoint, **values)

        # 通常エンドポイント
        return url_for(endpoint, **values)

    except BuildError:
        return "#"
    except Exception:
        # 最後のフォールバック（それでもダメなら #）
        try:
            return url_for(endpoint, **values)
        except Exception:
            return "#"

# Jinja から直接呼べるように（既に設定済ならこの行で上書きされます）
app.jinja_env.globals["safe_url_for"] = safe_url_for
# 明示的に署名URLを作りたいときはこれも使えます（任意）
app.jinja_env.globals["signed_image_url"] = lambda fn, wm=False: build_signed_image_url(fn, wm=wm, external=False)

@app.route("/admin/_sign_image", methods=["GET"])
@login_required
def admin_sign_image():
    """
    filename と wm を受け取り、その条件で署名済みの画像URLを返す。
    wm: 'none' | 'fullygoto' | 'gotocity' （旧 'fully' / 'city' も受ける）
    """
    fn = (request.args.get("filename") or "").strip()
    wm = (request.args.get("wm") or "").strip().lower()
    if not fn:
        return jsonify({"ok": False, "error": "filename required"}), 400

    # 旧トークン互換
    if wm in {"fully", "fullygoto"}:
        wm = "fullygoto"
    elif wm in {"city", "gotocity"}:
        wm = "gotocity"
    elif wm not in {"none", "fullygoto", "gotocity"}:
        wm = "none"

    try:
        # 署名URL作成（safe_url_for -> build_signed_image_url を経由）
        url = safe_url_for("serve_image", filename=fn, _external=True, _sign=True,
                           **({ "wm": wm } if wm and wm != "none" else {}))
        return jsonify({"ok": True, "url": url})
    except Exception as e:
        app.logger.exception("admin_sign_image failed")
        return jsonify({"ok": False, "error": str(e)}), 500

# =========================
#  環境変数 / モデル
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") or ""  # 未設定でも起動OK
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET") or ""              # 未設定でも起動OK

# ← ここを gpt-5-mini をデフォルトに
OPENAI_MODEL_PRIMARY = os.environ.get("OPENAI_MODEL_PRIMARY", "gpt-5-mini")
OPENAI_MODEL_HIGH    = os.environ.get("OPENAI_MODEL_HIGH", "gpt-5-mini")

REASK_UNHIT   = os.environ.get("REASK_UNHIT", "1").lower() in {"1", "true", "on", "yes"}
SUGGEST_UNHIT = os.environ.get("SUGGEST_UNHIT", "1").lower() in {"1", "true", "on", "yes"}
ENABLE_FOREIGN_LANG = os.environ.get("ENABLE_FOREIGN_LANG", "1").lower() in {"1", "true", "on", "yes"}

# OpenAI v1 クライアント
client = OpenAI(timeout=15)


# --- OpenAIラッパ（モデル切替を一元管理：正規版） ---
if 'openai_chat' not in globals():
    def openai_chat(model, messages, **kwargs):
        """
        Responses API で常に gpt-5-mini を使う軽量ラッパ。
        messages: str か [{"role","content"}] のどちらもOK。
        max_tokens 系は max_output_tokens に正規化。
        """
        if not os.environ.get("OPENAI_API_KEY"):
            return ""
        safe_model = "gpt-5-mini"
        mot = (kwargs.pop("max_output_tokens", None)
               or kwargs.pop("max_completion_tokens", None)
               or kwargs.pop("max_tokens", None))
        max_tokens = int(mot) if mot else 600

        if isinstance(messages, str):
            msgs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        else:
            msgs = [{"role": "user", "content": str(messages)}]

        try:
            res = client.responses.create(
                model=safe_model,
                input=[{"role": m["role"], "content": m["content"]} for m in msgs],
                max_output_tokens=max_tokens,
            )
            return (getattr(res, "output_text", "") or "").strip()
        except Exception as e:
            try:
                app.logger.exception("openai_chat failed: %s", e)
            except Exception:
                pass
            return ""


# LINE Bot
# =========================
#  LINE Webhook
# =========================
from linebot.models import TextSendMessage, LocationMessage, FlexSendMessage, LocationAction, PostbackEvent
from linebot.exceptions import LineBotApiError


# === LINE SDK init (safe) ===
def _line_enabled() -> bool:
    return bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)

try:
    if _line_enabled():
        line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        handler = WebhookHandler(LINE_CHANNEL_SECRET)
        # 近く検索の直近状態（超軽量メモ）
        _LAST = {
            "mode": {},       # user_id -> 'all' | 'tourism' | 'shop'
            "location": {},   # user_id -> (lat, lng, ts)
        }
        # --- 送信の重複抑止 & リクエスト世代管理 -----------------------------------
        from collections import defaultdict, deque
        import re, time  # 念のため

        # 「ユーザーごとの現在世代」。新しいユーザー発話が来たら+1する
        REQUEST_GENERATION: dict[str, int] = defaultdict(int)

        # 「直近に送った本文」を保持して同一本文の連投を防ぐ（10分・最大12件）
        _SENT_HISTORY: dict[str, deque] = defaultdict(lambda: deque(maxlen=12))
        _SENT_TTL_SEC = 10 * 60  # 10分

        def _text_key(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "")).strip().lower()

        def _was_sent_recent(target_id: str, text: str, *, mark: bool) -> bool:
            """直近10分に同一本文を送っていれば True。mark=True なら今回送った扱いに記録。"""
            now = time.time()
            dq = _SENT_HISTORY[target_id]
            # 期限切れを掃除
            while dq and now - dq[0][1] > _SENT_TTL_SEC:
                dq.popleft()
            key = _text_key(text)
            hit = any(k == key for k, _ in dq)
            if mark and not hit:
                dq.append((key, now))
            return hit

        # 待ちメッセージ（未定義だと NameError になるので用意）
        WAIT_MESSAGES = (
            "少しお待ちください…",
            "調べています…",
            "候補をまとめています…",
            "もう少々お待ちください…",
        )

        def pick_wait_message(seed: str | None = None) -> str:
            if not WAIT_MESSAGES:
                return "少しお待ちください…"
            idx = (abs(hash(seed)) if seed else 0) % len(WAIT_MESSAGES)
            return WAIT_MESSAGES[idx]

        # ---------------------------------------------------------------------------
        app.logger.info("LINE enabled")
    else:
        line_bot_api = None
        handler = None
        app.logger.warning("LINE disabled: set LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET")
except Exception as e:
    line_bot_api = None
    handler = None
    app.logger.exception("LINE init failed: %s", e)

# === 出典フッター強制付与ラッパ（すべての reply/push を横取り） ===

def _messages_with_sources_footer(messages):
    """
    どんな送信形でも、最後のテキストの末尾に — 出典 — を付ける。
    テキストが無い場合は出典だけの1通を末尾に追加。
    出典が未登録なら何もしない。
    """
    # list へ正規化
    if messages is None:
        return messages
    msgs = list(messages) if isinstance(messages, (list, tuple)) else [messages]

    # 既に登録されている出典を取得
    foot = _sources_footer_text()

    # 出典が未登録のままなら、最後のテキスト1行目をタイトルとして
    # entries.json から推定→record_source_from_entry で登録を試みる
    if not foot:
        try:
            last_txt = None
            for m in reversed(msgs):
                # TextSendMessage だけ拾う
                if isinstance(m, TextSendMessage):
                    last_txt = (m.text or "")
                    break
            if last_txt:
                first = next((ln.strip() for ln in last_txt.splitlines() if ln.strip()), "")
                if first:
                    key = _title_key(first)
                    if key:
                        for e in load_entries():
                            if _title_key(e.get("title","")) == key:
                                # フォームの「出典」テキストだけ採用（空なら登録されない）
                                record_source_from_entry(e)
                                break
            foot = _sources_footer_text()
        except Exception:
            pass

    # まだ出典なしならそのまま返す
    if not foot:
        return messages

    # 最後の TextSendMessage を探して末尾に追記（重複回避）
    for i in range(len(msgs)-1, -1, -1):
        m = msgs[i]
        if isinstance(m, TextSendMessage):
            t = m.text or ""
            if "— 出典 —" not in t:
                m.text = (t.rstrip() + "\n\n" + foot)
            return msgs

    # テキストが1通も無ければ、出典だけを末尾に追加
    msgs.append(TextSendMessage(text=foot))
    return msgs

class _LineApiWithSources:
    """line_bot_api の reply/push を横取りして出典フッターを必ず付けるプロキシ"""
    def __init__(self, impl):
        self._impl = impl
    def reply_message(self, reply_token, messages, *args, **kwargs):
        return self._impl.reply_message(reply_token, _messages_with_sources_footer(messages), *args, **kwargs)
    def push_message(self, to, messages, *args, **kwargs):
        return self._impl.push_message(to, _messages_with_sources_footer(messages), *args, **kwargs)
    def __getattr__(self, name):
        return getattr(self._impl, name)

# ここで元の line_bot_api を差し替え
if line_bot_api is not None:
    line_bot_api = _LineApiWithSources(line_bot_api)


# =========================
#  LINE: Webhook とハンドラ
# =========================

# 2) イベントハンドラ
# handler が無い環境では定義しない（起動エラー回避）
if _line_enabled() and handler:
    # 既存の厳密判定の「少し後ろ」〜 mute判定の「前」に追加
    def _hit_viewpoints_loose(s: str) -> bool:
        try:
            norm = globals().get("_n", lambda x: re.sub(r"\s+", " ", (x or "").lower()))
            t = norm(s)
        except Exception:
            t = (s or "").strip().lower()
        return (("展望" in t) and ("マップ" in t or "地図" in t)) or ("viewpoint" in t and "map" in t)

    # 近く検索の早期処理だけ残す（※デコレータなし）
    def _handle_nearby_text(event, text):
        """
        「近く」系テキストの早期処理。処理したら True を返す（未処理なら False）。
        - "near 32.6977,128.8445" の開発ショートカット
        - 自然文（近く/近場/周辺/付近/現在地/近い/近所/近くの観光地）
        """
        # 停止/再開・全体停止のガード
        try:
            if globals().get("_line_mute_gate") and _line_mute_gate(event, text):
                return True
        except Exception:
            app.logger.exception("_line_mute_gate failed in _handle_nearby_text")

        t = (text or "").strip()
        # user_id の取得（個チャ/グループ両対応）
        try:
            user_id = _line_target_id(event)
        except Exception:
            user_id = getattr(getattr(event, "source", None), "user_id", None) or "anon"

        # --- 開発用ショートカット: "near 32.6977,128.8445"
        if t.lower().startswith("near "):
            try:
                a, b = t[5:].replace("，", ",").split(",", 1)
                lat, lng = float(a), float(b)
            except Exception:
                _reply_or_push(event, "書式: near 32.6977,128.8445")
                return True

            last_mode = globals().get("_LAST", {}).get("mode", {}).get(user_id, "all")
            cats = _mode_to_cats(last_mode)
            items = _nearby_core(lat, lng, radius_m=2000, cat_filter=cats, limit=8)
            if not items:
                _reply_or_push(event, "近くの候補が見つかりませんでした（緯度・経度未登録の可能性）")
                return True

            flex = _nearby_flex(items)
            if flex:
                line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="近くの候補", contents=flex))
            else:
                _reply_or_push(event, "\n".join(
                    [f'{i+1}. {d["title"]}（{d["distance_m"]}m）' for i, d in enumerate(items)]
                ))
            try:
                if "save_qa_log" in globals():
                    save_qa_log(t, "nearby-flex", source="line", hit_db=True,
                                extra={"kind": "nearby", "mode": last_mode})
            except Exception:
                pass
            return True

        # --- 自然文（近く/近場/周辺/付近/現在地/近い/近所/近くの観光地）
        lower = t.lower()
        if any(k in lower for k in ["近く", "近場", "周辺", "付近", "現在地", "近い", "近所"]) or ("近くの観光地" in lower):
            mode = _classify_mode(t)
            try:
                if "_LAST" in globals():
                    _LAST.setdefault("mode", {})[user_id] = mode
            except Exception:
                pass

            last = globals().get("_LAST", {}).get("location", {}).get(user_id)
            if not last:
                # まずは位置情報をもらう
                try:
                    line_bot_api.reply_message(event.reply_token, _ask_location())
                except Exception:
                    _reply_or_push(event, "現在地を取得できませんでした。位置情報を送ってください。")
                return True

            lat, lng, _ts = last
            cats = _mode_to_cats(mode)
            items = _nearby_core(lat, lng, radius_m=2000, cat_filter=cats, limit=8)
            if not items:
                _reply_or_push(event, "近くの候補が見つかりませんでした（緯度・経度未登録の可能性）")
                return True

            flex = _nearby_flex(items)
            if flex:
                line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="近くの候補", contents=flex))
            else:
                _reply_or_push(event, "\n".join(
                    [f'{i+1}. {d["title"]}（{d["distance_m"]}m）' for i, d in enumerate(items)]
                ))
            try:
                if "save_qa_log" in globals():
                    save_qa_log(t, "nearby-flex", source="line", hit_db=True,
                                extra={"kind": "nearby", "mode": mode})
            except Exception:
                pass
            return True

        return False

# ==== ここから：統合した TextMessage ハンドラ（1本だけ残す・修正版） ====
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        text = getattr(event.message, "text", "") or ""
    except Exception:
        text = ""

    # --- ① ミュート/一時停止ゲート -------------------------
    try:
        if _line_mute_gate(event, text):
            return
    except Exception:
        app.logger.exception("_line_mute_gate failed")

    # --- ①.5 透かしコマンド（@Goto City / @fullyGOTO / なし） ----
    # ※ ここで検出したらユーザー設定に保存し、即時に切替メッセージを返して終了
    try:
        uid = None
        try:
            uid = _line_target_id(event)  # 個チャ/グループ両対応
        except Exception:
            uid = getattr(getattr(event, "source", None), "user_id", None) or "anon"

        choice = parse_wm_command(text)  # 'gotocity' | 'fullygoto' | 'none' | None
        if choice is not None:
            _set_user_wm(uid, choice)
            label = {"none": "なし", "fullygoto": "@fullyGOTO", "gotocity": "@Goto City"}[choice]
            try:
                _reply_or_push(event, f"透かしモードを「{label}」に切り替えました。")
            except Exception:
                # 念のため直接replyもフォールバック
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=f"透かしモードを「{label}」に切り替えました。")
                )
            try:
                save_qa_log(text, f"wm_choice={choice}", source="line", hit_db=False, extra={"kind": "wm"})
            except Exception:
                pass
            return
    except Exception:
        app.logger.exception("wm command handler failed")

    # --- ② マップシリーズ（展望所・海水浴場 ほか） ----------------
    try:
        mm = _find_map_by_text(text)
        if mm:
            url_fn = mm.get("url_fn")
            url = url_fn() if callable(url_fn) else ""

            msgs = []

            # 1) ヒットしたマップだけを Flex（1枚）で返信
            if url:
                flex_main = _flex_mymap(
                    title = mm.get("title") or "マップ",
                    url   = url,
                    thumb = mm.get("thumb") or "",
                    subtitle = (mm.get("subtitle") or mm.get("desc") or "").strip() or None
                )
                # 念のため空text防止チェック
                if isinstance(flex_main, dict) and flex_main.get("contents"):
                    msgs.append(FlexSendMessage(
                        alt_text = mm.get("alt") or (mm.get("title") or "マップ"),
                        contents = flex_main
                    ))
                else:
                    msgs.append(TextSendMessage(text=f"{mm.get('title','マップ')}はこちら：\n{url}"))
            else:
                # URL が作れない場合はテキストのみ
                msgs.append(TextSendMessage(text=f"{mm.get('title','マップ')}は現在準備中です。"))

            # 2) 残りのシリーズは「テキスト1通」にまとめて案内
            series_text = _series_text_for_reply(exclude_key=mm.get("key"))
            msgs.append(TextSendMessage(text=series_text))

            # 1回の reply でまとめて送信 → 終了
            line_bot_api.reply_message(event.reply_token, msgs[:5])
            return
    except Exception:
        app.logger.exception("[map series] handler failed")

    # --- ③ 近く検索（どちらの実装も生かす） ------------------
    # 3-1) 既存の早期ハンドラがあれば最優先で使う
    try:
        _nearby_early = globals().get("_handle_nearby_text")
        if callable(_nearby_early) and _nearby_early(event, text):
            return
    except Exception:
        app.logger.exception("nearby early handler failed")

    # 3-2) キーワード版（旧 on_text の簡易実装も維持）
    try:
        t = (text or "").lower()
        if any(k in t for k in ["近く", "近場", "周辺", "現在地", "近い", "近所", "近くの観光地"]):
            mode = _classify_mode(text)
            uid2 = _line_target_id(event)
            if "_LAST" in globals():
                _LAST["mode"][uid2] = mode
            # 位置情報のお願いを reply、失敗時は push でフォールバック（※ force_push は使わない）
            try:
                line_bot_api.reply_message(
                    event.reply_token,
                    _ask_location("現在地から近い順で探します。位置情報を送ってください。")
                )
            except Exception:
                try:
                    line_bot_api.push_message(
                        uid2,
                        TextSendMessage(text="現在地から近い順で探します。位置情報を送ってください。")
                    )
                except Exception:
                    pass
            return
    except Exception:
        app.logger.exception("nearby keyword handler failed")

    # --- ④ 即答リンク（天気・交通） -------------------------
    try:
        w, ok = get_weather_reply(text)
        app.logger.debug(f"[quick] weather_match={ok} text={text!r}")
        if ok and w:
            _reply_quick_no_dedupe(event, w)
            save_qa_log(text, w, source="line", hit_db=False, extra={"kind":"weather"})
            return

        tmsg, ok = get_transport_reply(text)
        app.logger.debug(f"[quick] transport_match={ok} text={text!r}")
        if ok and tmsg:
            _reply_quick_no_dedupe(event, tmsg)
            save_qa_log(text, tmsg, source="line", hit_db=False, extra={"kind":"transport"})
            return
    except Exception as e:
        app.logger.exception(f"quick link reply failed: {e}")

    # --- ⑤ スモールトーク/ヘルプ -----------------------------
    try:
        st = smalltalk_or_help_reply(text)
        if st:
            _reply_or_push(event, st)
            save_qa_log(text, st, source="line", hit_db=False, extra={"kind":"smalltalk"})
            return
    except Exception:
        app.logger.exception("smalltalk failed")

    # --- ⑥ DB回答（画像→テキストを1回の reply にまとめる） ---
    try:
        # ★ ユーザーの透かし設定を取得して回答生成に渡す（新旧両対応）
        uid3 = None
        try:
            uid3 = _line_target_id(event)
        except Exception:
            uid3 = getattr(getattr(event, "source", None), "user_id", None) or "anon"
        wm_mode = _get_user_wm(uid3)  # 'none' | 'fullygoto' | 'gotocity'

        try:
            # 新実装：wm_mode / user_id を受け取れる場合はこちらが使われる
            ans, hit, img_url = _answer_from_entries_min(text, wm_mode=wm_mode, user_id=uid3)
        except TypeError:
            # 旧実装：引数を受け取れない場合は従来通り
            ans, hit, img_url = _answer_from_entries_min(text)

        # --- 画像URL（署名＆透かし適用） ---
        msgs = []

        def _line_img_pair_from_url(img_url_in: str, wm_mode_in: str | None):
            """
            LINE用: original / preview の2本を同じモードで作る。
            優先順位: URLの ?wm=xxx ＞ 引数 wm_mode_in ＞ エントリ側設定(_wm_choice_from_entries)
            - 自サイト配信（/media/img/<name>）以外は元URLをそのまま返す
            戻り値: (original_url, preview_url)
            """
            try:
                if not img_url_in:
                    return None, None

                from urllib.parse import urlparse, unquote, parse_qs
                import re as _re

                pu = urlparse(img_url_in)
                m = _re.search(r"/media/img/([^/?#]+)$", pu.path)
                if not m:
                    # 外部URLは触らない
                    return img_url_in, img_url_in

                filename = unquote(m.group(1))

                # 1) URLパラメータ優先
                qs = parse_qs(pu.query or "")
                wm_in_url = (qs.get("wm", [None])[0] or "").strip().lower()

                mode = None
                if wm_in_url in ("fullygoto", "gotocity"):
                    mode = wm_in_url
                elif wm_in_url in ("fully",):
                    mode = "fullygoto"
                elif wm_in_url in ("city",):
                    mode = "gotocity"
                elif wm_in_url in ("1", "true"):
                    mode = "gotocity"  # 旧真偽値互換の既定
                elif wm_in_url in ("none", "0", "off", "false"):
                    mode = None

                # 2) 引数モード
                if mode is None and wm_mode_in:
                    v = (wm_mode_in or "").strip().lower()
                    if v in ("fullygoto", "gotocity"):
                        mode = v
                    elif v in ("fully",):
                        mode = "fullygoto"
                    elif v in ("city",):
                        mode = "gotocity"
                    elif v in ("none", "0", "off", "false"):
                        mode = None

                # 3) まだ決まらなければエントリ側の選択
                if mode is None:
                    try:
                        choice = _wm_choice_from_entries(filename)
                        if choice in ("fullygoto", "gotocity"):
                            mode = choice
                    except Exception:
                        pass

                # 4) 署名URL生成（original/preview同一）
                if mode in ("fullygoto", "gotocity"):
                    s = build_signed_image_url(filename, wm=mode, external=True)
                    return s, s
                else:
                    s = build_signed_image_url(filename, wm=False, external=True)
                    return s, s

            except Exception:
                app.logger.exception("_line_img_pair_from_url failed; fallback to original")
                return img_url_in, img_url_in

        # 署名URL＋透かしモードを強制（URLに wm が無ければ wm_mode を採用）
        if img_url:
            orig_url, prev_url = _line_img_pair_from_url(img_url, wm_mode)
            if not orig_url or not prev_url:
                # 自サイト画像ならファイル名から再生成（wm_mode を確実に反映）
                try:
                    from urllib.parse import urlparse, unquote
                    import re
                    m = re.search(r"/media/img/([^/?#]+)$", urlparse(img_url).path)
                    if m:
                        fn = unquote(m.group(1))
                        if wm_mode in ("fullygoto", "gotocity"):
                            s = safe_url_for("serve_image", filename=fn, _external=True, _sign=True, wm=wm_mode)
                        else:
                            s = safe_url_for("serve_image", filename=fn, _external=True, _sign=True)
                        orig_url = prev_url = s
                except Exception:
                    pass

            # 先に URL が作れたならそれを使う
            if orig_url and prev_url:
                try:
                    msgs.append(ImageSendMessage(original_content_url=orig_url, preview_image_url=prev_url))
                except Exception as e:
                    app.logger.warning("ImageSendMessage via URL failed: %s", e)

            # URL で送れなかった場合だけ、entry ベースでフォールバック
            if not any(isinstance(m, ImageSendMessage) for m in msgs):
                entry_for_img = None
                try:
                    entry_for_img = (
                        locals().get("entry")
                        or locals().get("entry_edit")
                        or locals().get("best_entry")
                    )
                    if not entry_for_img:
                        top_hit = locals().get("top_hit")
                        if isinstance(top_hit, dict) and top_hit.get("id"):
                            try:
                                entry_for_img = load_entry_by_id(top_hit["id"])  # 無ければこの3行は削除OK
                            except Exception:
                                entry_for_img = None
                except Exception:
                    entry_for_img = None

                img_msg = make_line_image_message(entry_for_img) if entry_for_img else None
                if img_msg:
                    msgs.append(img_msg)

        # 本文は安全分割して複数通に
        for p in _split_for_line(ans):
            msgs.append(TextSendMessage(text=p))

        line_bot_api.reply_message(event.reply_token, msgs)

        joined = ("\n---\n".join(getattr(m, "text", "(image)") for m in msgs)) or "(no-text)"
        save_qa_log(text, joined, source="line", hit_db=hit)

    except LineBotApiError as e:
        global LAST_SEND_ERROR, SEND_ERROR_COUNT
        SEND_ERROR_COUNT = globals().get("SEND_ERROR_COUNT", 0) + 1
        LAST_SEND_ERROR = f"{type(e).__name__}: {e}"
        app.logger.exception("LINE send failed")
    except Exception as e:
        app.logger.exception("answer flow failed: %s", e)
        fallback, _ = build_refine_suggestions(text)
        _reply_or_push(event, "うまく探せませんでした。\n" + fallback)
        save_qa_log(text, "fallback", source="line", hit_db=False, extra={"error": str(e)})
# ==== ここまで：統合ハンドラ ====


@handler.add(FollowEvent)
def on_follow(event):
    """
    友だち追加（またはブロック解除）時の最初の挨拶メッセージを送る。
    ※公式の「あいさつメッセージ」をONにしている場合は二重送信になるので、
      基本は公式側OFF＋このWebhookで送る運用を推奨。
    """
    # 表示名の取得（失敗しても続行）
    display_name = ""
    try:
        if hasattr(event, "source") and event.source and hasattr(event.source, "user_id"):
            prof = line_bot_api.get_profile(event.source.user_id)
            display_name = (prof.display_name or "").strip()
    except Exception as e:
        app.logger.info(f"[follow] get_profile failed: {e}")

    # あいさつ文（必要なら調整してください）
    nick = f"{display_name}さん" if display_name else "はじめまして"
    greet = (
        f"{nick}、fullyGOTO観光AIです。友だち追加ありがとうございます。\n\n"
        "このLINEでは、店名・観光地名を送るだけで基本情報をまとめて返信し、"
        "「今日の天気」「運行状況（船・飛行機）」へのリンク、"
        "「展望所マップ」など◯◯マップの表示にも対応しています。"
        "※試運転（ベータ）中です。\n\n"
        "ご利用前に、下記の説明サイトを必ずお読みください（使い方／できること／トラブル時の停止・再開の方法）\n"
        "▶ https://www.fullygoto.com/fullygoto-ai/\n\n"
        "使い方の例：\n"
        "・「今日の天気」「運行状況」\n"
        "・「高浜海水浴場」「コワーキング＠mitake」\n"
        "・「展望所マップ」「海水浴場マップ」\n"
        "トラブル時のメモ：\n"
        "・「停止」と送る → 応答を一時停止\n"
        "・「解除」と送る → 応答を再開\n"
        "※管理側が全体停止中は復帰できない場合があります（復旧後に再開）。"
    )

    # 既存の安全送信ユーティリティがあればそれを使用
    try:
        _reply_or_push(event, greet)
    except Exception:
        # フォールバック（直接SDKで返信）
        try:
            line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=greet)])
        except Exception as e:
            app.logger.error(f"[follow] reply failed: {e}")

# === LocationMessage は 1 本だけに統合（前半+後半の機能を統合）===
@handler.add(MessageEvent, message=LocationMessage)
def on_location(event):
    # 0) ミュート／一時停止ゲート（先頭で早期return）
    try:
        if _line_mute_gate(event, "location"):
            return
    except Exception:
        app.logger.exception("_line_mute_gate failed")

    try:
        uid = _line_target_id(event)
        lat = float(event.message.latitude)
        lng = float(event.message.longitude)

        # 1) 直近位置を覚える（後半の機能）
        try:
            if "_LAST" in globals() and isinstance(_LAST, dict):
                _LAST.setdefault("location", {})[uid] = (lat, lng, time.time())
        except Exception:
            pass

        # 2) ユーザーモード→カテゴリ
        mode = (globals().get("_LAST", {}).get("mode", {}).get(uid)) or "all"
        cats = _mode_to_cats(mode)

        # 3) “少しお待ちください…” は push（reply_token を結果用に温存）
        try:
            _reply_or_push(event, pick_wait_message(uid), force_push=True)
        except Exception:
            pass

        # 4) 検索（半径/件数は好みで調整）
        radius_m = 2000
        limit    = 10
        rows = _nearby_core(lat, lng, radius_m=radius_m, cat_filter=cats, limit=limit)

        if not rows:
            # 5) 結果なし：reply → 失敗時は push
            try:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="近くの候補が見つかりませんでした（半径を広げる/別のキーワードをお試しください）。")
                )
            except Exception:
                _reply_or_push(
                    event,
                    "近くの候補が見つかりませんでした（緯度・経度未登録の可能性）。",
                    force_push=True
                )
            return

        # 6) Flex返信（失敗時はテキストにフォールバック）
        flex = _nearby_flex(rows)
        try:
            line_bot_api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text="近くのスポット", contents=flex)
            )
        except Exception:
            lines = [f"近くのスポット（上位{min(limit, len(rows))}件）:"]
            for it in rows[:limit]:
                url = it.get("google_url", "")
                lines.append(f"- {it['title']}（{it['distance_m']}m） {url}")
            _reply_or_push(event, "\n".join(lines), force_push=True)

        # 7) ログ（後半の機能）
        try:
            save_qa_log(
                "LOCATION", "nearby-flex", source="line", hit_db=True,
                extra={"kind": "nearby", "mode": mode, "lat": lat, "lng": lng, "radius_m": radius_m, "limit": limit}
            )
        except Exception:
            pass

    except LineBotApiError as e:
        globals()["SEND_ERROR_COUNT"] = globals().get("SEND_ERROR_COUNT", 0) + 1
        globals()["LAST_SEND_ERROR"]  = f"{type(e).__name__}: {e}"
        app.logger.exception("LINE send failed in on_location")
    except Exception as e:
        app.logger.exception("on_location failed: %s", e)
        _reply_or_push(event, "位置情報の処理でエラーが起きました。もう一度お試しください。", force_push=True)


from urllib.parse import parse_qs

@handler.add(PostbackEvent)
def on_postback(event):
    # 0) ミュート／一時停止ゲート
    try:
        if _line_mute_gate(event, "postback"):
            return
    except Exception:
        app.logger.exception("_line_mute_gate failed in postback")

    try:
        data = (getattr(getattr(event, "postback", None), "data", "") or "").strip()
        low  = data.lower()

        # 1) 代表的キーを正規化（ボタン側の実装差を吸収）
        canon = low
        # 例：action=weather&day=today, kind=weather_today などを丸める
        if "=" in low:
            try:
                qs = {k: (v[0] if v else "") for k, v in parse_qs(low).items()}
                if (qs.get("action") == "weather") or (qs.get("kind") in ("weather", "weather_today")):
                    canon = "weather_today"
                elif (qs.get("action") in ("transport", "traffic")) or (qs.get("kind") in ("transport", "traffic_status")):
                    canon = "transport_status"
            except Exception:
                pass

        # 2) 天気・運行の代表表現を拾う（前方一致・別名も吸収）
        WEATHER_KEYS   = ("weather", "weather_today", "天気", "今日の天気", "てんき", "きょうのてんき")
        TRANSPORT_KEYS = ("transport", "transport_status", "traffic", "運行", "運行状況", "交通", "バス運行", "フェリー運行")

        # 完全一致で拾えなくても部分一致で救済
        if any(k in canon for k in WEATHER_KEYS):
            q = "今日の天気"
            msg, ok = get_weather_reply(q)
            if ok and msg:
                _reply_quick_no_dedupe(event, msg)
                save_qa_log(q, msg, source="line", hit_db=False, extra={"kind":"weather", "via":"postback"})
                return

        if any(k in canon for k in TRANSPORT_KEYS):
            q = "運行状況"
            tmsg, ok = get_transport_reply(q)
            if ok and tmsg:
                _reply_quick_no_dedupe(event, tmsg)
                save_qa_log(q, tmsg, source="line", hit_db=False, extra={"kind":"transport", "via":"postback"})
                return

        # 3) 最後の砦：data そのものを投げて判定
        msg, ok = get_weather_reply(canon)
        if ok and msg:
            _reply_quick_no_dedupe(event, msg)
            save_qa_log(canon, msg, source="line", hit_db=False, extra={"kind":"weather", "via":"postback-fallback"})
            return

        tmsg, ok = get_transport_reply(canon)
        if ok and tmsg:
            _reply_quick_no_dedupe(event, tmsg)
            save_qa_log(canon, tmsg, source="line", hit_db=False, extra={"kind":"transport", "via":"postback-fallback"})
            return

        # 4) 何も該当しなければ軽い案内
        _reply_quick_no_dedupe(event, "うまく処理できませんでした。もう一度お試しください。")

    except Exception as e:
        app.logger.exception(f"on_postback failed: {e}")
        _reply_quick_no_dedupe(event, "うまく処理できませんでした。もう一度お試しください。")


def _reply_quick_no_dedupe(event, text: str):
    """重複抑止に引っかけず、とにかく1通返す（reply優先・失敗時push）。"""
    if not text:
        return
    try:
        if getattr(event, "reply_token", None):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
        else:
            tid = _line_target_id(event)
            if tid:
                line_bot_api.push_message(tid, TextSendMessage(text=text))
    except Exception:
        # 片方で失敗した時の保険
        try:
            tid = _line_target_id(event)
            if tid:
                line_bot_api.push_message(tid, TextSendMessage(text=text))
        except Exception:
            pass

# === LINE 返信ユーティリティ（未定義だったので追加） ===
# === LINE 返信ユーティリティ（安全送信 + エラー計測 + force_push互換） ===
def _reply_or_push(event, text: str, *, force_push: bool = False):
    if isinstance(text, str):
        text = _append_sources_if_text(text)    
    """
    長文は自動分割して返信。5通/呼び出し上限を厳守。
    force_push=True のときは reply_token があっても push のみで送る。
    送信失敗は SEND_ERROR_COUNT / SEND_FAIL_COUNT と LAST_SEND_ERROR に記録。
    """
    LINE_SAFE = globals().get("LINE_SAFE_CHARS", 3800)

    if not _line_enabled() or not line_bot_api:
        app.logger.info("[LINE disabled] would send: %r", text)
        return

    def _compress_to(parts, lim):
        """分割済みpartsを、1通あたりlim文字以内でできるだけ結合して個数を減らす"""
        out, buf = [], ""
        for p in parts:
            if not buf:
                buf = p
                continue
            if len(buf) + 1 + len(p) <= lim:
                buf += "\n" + p
            else:
                out.append(buf)
                buf = p
        if buf:
            out.append(buf)
        return out

    lim = LINE_SAFE
    parts = _split_for_line(text, lim) or [""]
    parts = _compress_to(parts, lim)

    MAX_PER_CALL = 5  # reply/push とも5通/呼び出し

    def _do_reply(msgs):
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=m) for m in msgs])

    def _do_push(tid, msgs):
        line_bot_api.push_message(tid, [TextSendMessage(text=m) for m in msgs])

    try:
        reply_token = getattr(event, "reply_token", None)

        # force_push 指定 or reply_token が無い場合は push 送信のみ
        if force_push or not reply_token:
            tid = _line_target_id(event)
            if tid:
                for i in range(0, len(parts), MAX_PER_CALL):
                    _do_push(tid, parts[i:i + MAX_PER_CALL])
            return

        # まず reply で5通まで
        head = parts[:MAX_PER_CALL]
        _do_reply(head)
        rest = parts[MAX_PER_CALL:]

        # 余りは push で後送（ベストエフォート）
        if rest:
            tid = _line_target_id(event)
            if tid:
                for i in range(0, len(rest), MAX_PER_CALL):
                    try:
                        _do_push(tid, rest[i:i + MAX_PER_CALL])
                    except LineBotApiError as e:
                        globals()["SEND_ERROR_COUNT"] = globals().get("SEND_ERROR_COUNT", 0) + 1
                        globals()["LAST_SEND_ERROR"]  = f"{type(e).__name__}: {e}"
                        app.logger.warning("LINE push failed on chunk %s: %s", i // MAX_PER_CALL, e)
                        break

    except LineBotApiError as e:
        globals()["SEND_ERROR_COUNT"] = globals().get("SEND_ERROR_COUNT", 0) + 1
        globals()["LAST_SEND_ERROR"]  = f"{type(e).__name__}: {e}"
        app.logger.exception("LINE send failed")

    except Exception as e:
        globals()["SEND_FAIL_COUNT"]  = globals().get("SEND_FAIL_COUNT", 0) + 1
        globals()["LAST_SEND_ERROR"]  = f"{type(e).__name__}: {e}"
        app.logger.exception("LINE send failed (unexpected)")


# データ格納先
BASE_DIR       = os.environ.get("DATA_BASE_DIR", ".")  # 例: /var/data
ENTRIES_FILE   = os.path.join(BASE_DIR, "entries.json")
DATA_DIR       = os.path.join(BASE_DIR, "data")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
LOG_FILE       = os.path.join(LOG_DIR, "questions_log.jsonl")
SYNONYM_FILE   = os.path.join(BASE_DIR, "synonyms.json")
USERS_FILE     = os.path.join(BASE_DIR, "users.json")
NOTICES_FILE   = os.path.join(BASE_DIR, "notices.json")
SHOP_INFO_FILE = os.path.join(BASE_DIR, "shop_infos.json")

# ★ 一度だけ案内フラグの保存先 & TTL（秒）
PAUSED_NOTICE_FILE    = os.path.join(BASE_DIR, "paused_notice.json")  # もしくは DATA_DIR に置いても可
PAUSED_NOTICE_TTL_SEC = int(os.getenv("PAUSED_NOTICE_TTL_SEC", "86400"))  # 既定: 24時間

# 送信確定ログ（JSON Lines）
SEND_LOG_FILE   = os.path.join(LOG_DIR, "send_log.jsonl")
# ログのサンプリング（1.0=全件、0.0=無効）
SEND_LOG_SAMPLE = float(os.environ.get("SEND_LOG_SAMPLE", "1.0"))


# === Images: config + route + normalized save (唯一の正) ===
MEDIA_URL_PREFIX = "/media/img"  # 画像URLの先頭
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 入力として受け付ける拡張子（出力は常に .jpg）
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ---- 画像アクセス保護（署名付きURL + 透かし）----
IMAGE_PROTECT = os.getenv("IMAGE_PROTECT","1").lower() in {"1","true","on","yes"}
IMAGES_SIGNING_KEY = (os.getenv("IMAGES_SIGNING_KEY") or app.secret_key or "change-me").encode("utf-8")
SIGNED_IMAGE_TTL_SEC = int(os.getenv("SIGNED_IMAGE_TTL_SEC","604800"))  # 既定=7日
# 署名URLのキャッシュ安全マージン（exp までの残時間から引く秒数）
SIGNED_IMAGE_CACHE_SAFETY_SEC = int(os.getenv("SIGNED_IMAGE_CACHE_SAFETY_SEC", "300"))  # 既定: 5分

# 未署名アクセス（管理画面など）時のキャッシュ秒数（数日推奨）
UNSIGNED_IMAGE_MAX_AGE_SEC = int(os.getenv("UNSIGNED_IMAGE_MAX_AGE_SEC", "259200"))  # 既定: 3日


# === Watermark (英語表記に統一) =========================================
WATERMARK_ENABLE  = os.getenv("WATERMARK_ENABLE", "1").lower() in {"1","true","on","yes"}
WATERMARK_TEXT    = os.getenv("WATERMARK_TEXT", "@Goto City")      # 既定は英語
WATERMARK_OPACITY = int(os.getenv("WATERMARK_OPACITY", "160"))     # 0-255
WATERMARK_SCALE   = float(os.getenv("WATERMARK_SCALE", "0.035"))   # 画像幅に対する割合

# 日本語フォントは不要。使いたいフォントがある場合のみパスを環境変数で指定。
# 例) WATERMARK_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
WATERMARK_FONT_PATH = os.getenv("WATERMARK_FONT_PATH", "").strip()

# 透かしプリセット（内部キー → 実際に描く文言）
WM_TEXTS = {
    "none":      None,
    "fullygoto": "@fullyGOTO",
    "gotocity":  "@Goto City",
}


# ============================================================
# Watermark 正規化ヘルパ（どこからでも呼べる場所に配置）
# ============================================================


def _wm_choice_for_entry(entry: dict | None) -> str:
    """
    エントリ（旧/新どちらの保存形式でも）から最終的な wm モードを決定する。
    - 新: entry['wm_external_choice'] が 'none'/'fullygoto'/'gotocity'
    - 旧: entry['wm_on'] が True/False のみ
    """
    if not entry:
        return "fullygoto"
    raw = entry.get("wm_external_choice")
    if raw is None:
        # 旧データの救済（wm_on が True なら fullygoto、False なら none）
        raw = "fullygoto" if entry.get("wm_on", True) else "none"
    return _normalize_wm_choice(raw, wm_on_default=True)

def _abs_media_url(img: str, query: dict | None = None) -> str:
    """/media/img/<img>?... の絶対URLを https で作る"""
    from urllib.parse import urlencode
    from flask import request
    base = (request.url_root or "").rstrip("/")
    url = f"{base}/media/img/{img}"
    if query:
        url += "?" + urlencode(query)
    # LINE 用に https を強制（Renderでhttpが混ざる事故防止）
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url

def _wm_image_urls(entry: dict, img: str) -> tuple[str, str]:
    """
    original: 選択透かし（wm=none のときは素の画像）
    preview : 常に透かし（wm=none のとき PREVIEW_WM_DEFAULT を使用）
    """
    wm = (entry.get("wm") or "none").strip()
    if wm not in WM_VALUES:
        wm = "none"

    # original は wm が none なら素の画像、それ以外は署名つき
    if wm == "none":
        original = _abs_media_url(img)
    else:
        original = _abs_media_url(img, {"_sign": "True", "wm": wm})

    # preview は常に透かし。wm=none のときは既定の透かしを使う
    p_wm = wm if wm != "none" else PREVIEW_WM_DEFAULT
    if p_wm != "none":
        preview = _abs_media_url(img, {"_sign": "True", "wm": p_wm})
    else:
        preview = original  # 既定が none の設定なら original を流用

    return original, preview

def build_flex_hero(entry: dict, ratio: str = "16:9"):
    """
    Flexのhero画像用ユーティリティ。
    - 表示: 透かし付きプレビュー（wm=1）
    - タップ: 原寸（選択モードの透かし＋署名付き）
    """
    if not entry:
        return None
    img_name = (entry.get("image_file") or entry.get("image") or "").strip()
    if not img_name:
        return None
    original, preview = _wm_image_urls(entry, img_name)
    return {
        "type": "image",
        "url": preview,              # 一覧に表示するのは常時透かしサムネ
        "size": "full",
        "aspectMode": "cover",
        "aspectRatio": ratio,
        "action": {                  # タップで原寸（選択モード透かし・署名付き）
            "type": "uri",
            "uri": original
        }
    }

# ==== LINE 送信用：画像URLを一元生成するヘルパ ===========================
PREVIEW_WM_DEFAULT = os.getenv("PREVIEW_WM_DEFAULT", "fullygoto")

def _entry_image_name(entry: dict) -> str | None:
    """entry から主画像ファイル名を1つ取り出す（images優先, imageフォールバック）"""
    if not entry:
        return None
    imgs = entry.get("images")
    if isinstance(imgs, list) and imgs and isinstance(imgs[0], str) and imgs[0]:
        return imgs[0]
    img = entry.get("image")
    if isinstance(img, str) and img:
        return img
    return None


def make_line_image_message(entry: dict) -> ImageSendMessage:
    """
    entry から主画像を1枚取り、選択中の透かしで署名URLを作って返す。
    必要キー:
      - images: List[str]  先頭を使用
      - wm: str            'none' / 'fullygoto' / 'gotocity' のいずれか
    """
    from urllib.parse import urlencode
    from flask import request

    if not entry or not isinstance(entry, dict):
        raise ValueError("make_line_image_message: entry is missing or not a dict")

    images = entry.get("images") or []
    if not images:
        raise ValueError("make_line_image_message: entry['images'] is empty")

    fname = images[0]
    wm = (entry.get("wm") or "none").strip()
    qs = ""
    if wm != "none":
        qs = "?" + urlencode({"_sign": True, "wm": wm})

    # 絶対URL（Render で https を強制）
    base = (request.url_root or "").rstrip("/")
    signed = f"{base}/media/img/{fname}{qs}"

    return ImageSendMessage(
        original_content_url=signed,
        preview_image_url=signed,
    )

def make_flex_hero_image(entry: dict) -> dict | None:
    """
    Flex の hero 用 image コンポーネントを“正規化済みURL”で返す。
    - 表示: preview（常に透かし）
    - タップ: original（署名+選択透かし）に遷移
    """
    img = _entry_image_name(entry)
    if not img:
        return None
    original, preview = _wm_image_urls(entry, img)
    return {
        "type": "image",
        "url": preview,              # 画面表示
        "size": "full",
        "aspectMode": "cover",
        "action": {                  # 画像タップで原寸（署名+選択透かし）を開く
            "type": "uri",
            "label": "open",
            "uri": original
        }
    }


def apply_thumbnail_for_template_column(entry: dict, column: dict) -> dict:
    """
    テンプレート（Buttons/Carousel など）の column にサムネURLを適用。
    既存の column を破壊せず、thumbnailImageUrl だけ入れる/上書きする。
    """
    try:
        img = _entry_image_name(entry)
        if not img:
            return column
        _, preview = _wm_image_urls(entry, img)
        column = dict(column)  # defensively copy
        column["thumbnailImageUrl"] = preview
        return column
    except Exception:
        app.logger.exception("apply_thumbnail_for_template_column failed")
        return column


def _load_wm_font(base_size: int):
    """
    透かし用フォントを読み込む（英語フォールバック）。
    1) 環境変数 WATERMARK_FONT_PATH があれば優先
    2) OSにある一般的な英字フォントを順に探す
    3) 最後は PIL のデフォルトフォント
    """
    from PIL import ImageFont

    if WATERMARK_FONT_PATH:
        try:
            return ImageFont.truetype(WATERMARK_FONT_PATH, base_size)
        except Exception:
            pass

    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS例
    ):
        try:
            return ImageFont.truetype(p, base_size)
        except Exception:
            continue

    return ImageFont.load_default()

def _img_sig(filename: str, exp: int) -> str:
    msg = f"{filename}|{exp}".encode("utf-8")
    return base64.urlsafe_b64encode(
        hmac.new(IMAGES_SIGNING_KEY, msg, hashlib.sha256).digest()
    ).rstrip(b"=").decode("ascii")

# ---- HTTPSの絶対URLを保証するヘルパ／url_forの安全版 ----
def _force_https_abs(u: str) -> str:
    """
    与えられたURLを必ず HTTPS の絶対URLに補正する。
    - '/media/...' のような相対 → リクエストのホスト or EXTERNAL_BASE_URL を付与
    - 'http://' → 'https://' に置換
    - それ以外はそのまま
    """
    if not u:
        return ""
    try:
        import os
        from flask import request

        # //example.com 形式は https を付ける
        if u.startswith("//"):
            return "https:" + u

        # /path から始まる相対URL → ホストを付ける
        if u.startswith("/"):
            try:
                base = (request.url_root or "").rstrip("/")
            except Exception:
                base = ""
            if not base:
                base = (os.getenv("EXTERNAL_BASE_URL") or "").rstrip("/")
            if base.startswith("http://"):
                base = "https://" + base[len("http://"):]
            return (base + u) if base else u

        # http → https
        if u.startswith("http://"):
            return "https://" + u[len("http://"):]
        return u
    except Exception:
        return u


def force_https_url_for(endpoint: str, **values) -> str:
    """
    url_for のシンプル版（絶対URL＋HTTPS補正のみ）
    ※ safe_url_for とは別物の関数名にする！
    """
    from flask import url_for, request
    import os
    from urllib.parse import urlencode

    try:
        values["_external"] = True
        u = url_for(endpoint, **values)
        return _force_https_abs(u)
    except Exception:
        filename = values.get("filename")
        qs = values.copy()
        qs.pop("filename", None)
        qs.pop("_external", None)
        query = ("?" + urlencode(qs)) if qs else ""
        base = (os.getenv("EXTERNAL_BASE_URL") or "").rstrip("/")
        if not base:
            try:
                base = (request.url_root or "").rstrip("/")
            except Exception:
                base = ""
        u = f"{base}{MEDIA_URL_PREFIX}/{filename}{query}" if base else f"{MEDIA_URL_PREFIX}/{filename}{query}"
        return _force_https_abs(u)


# ========== helper: 相対/HTTP を HTTPS の絶対URLに補正 ==========
def _force_https_abs(u: str) -> str:
    """
    - 絶対URL(http/https)なら https に置換して返す
    - 相対URLなら PUBLIC_BASE_URL/EXTERNAL_BASE_URL/BASE_URL または request.url_root を基に絶対化
    - request コンテキストが無い場合でも安全に動作（LINEのpush生成など）
    """
    try:
        if not u:
            return u
        from urllib.parse import urlparse, urlunparse
        import os

        p = urlparse(u)

        # 既にスキームがある場合
        if p.scheme in ("http", "https"):
            scheme = "https"
            # netloc が空のケースは稀だが念のため温存
            return urlunparse((scheme, p.netloc, p.path, p.params, p.query, p.fragment))

        # 相対URL → ベースURLを探す
        base = (os.environ.get("PUBLIC_BASE_URL")
                or os.environ.get("EXTERNAL_BASE_URL")
                or os.environ.get("BASE_URL")
                or "")
        if not base:
            # リクエスト中なら url_root を使う（失敗したら相対のまま返す）
            try:
                from flask import request
                base = request.url_root  # 例: http://xxx/
            except Exception:
                return u

        bp = urlparse(base)
        netloc = bp.netloc
        if not netloc:
            return u  # ベースが不正

        scheme = "https"
        base_prefix = urlunparse((scheme, netloc, "", "", "", ""))  # https://host

        if u.startswith("/"):
            return base_prefix.rstrip("/") + u
        else:
            return base_prefix.rstrip("/") + "/" + u
    except Exception:
        return u


# ========== 置換版 build_signed_image_url ==========
def build_signed_image_url(
    filename: str,
    *,
    wm: str | bool | None = None,
    external: bool = True,
    ttl_sec: int | None = None,
) -> str:
    if not filename:
        return ""

    # --- wm 正規化 ---
    wm_q: str | None = None
    try:
        if isinstance(wm, str):
            v = wm.strip().lower()
            if   v in ("fullygoto", "gotocity"): wm_q = v
            elif v == "fully":                   wm_q = "fullygoto"
            elif v == "city":                    wm_q = "gotocity"
            elif v in ("none","0","off","false",""): wm_q = None
            elif v in ("1","true","on","yes"):   wm_q = "1"
        elif wm is True:
            wm_q = "1"
    except Exception:
        wm_q = None

    from flask import url_for
    from urllib.parse import urlencode
    import time as _t

    # 画像保護の有無
    IMAGE_PROTECT_ON = bool(globals().get("IMAGE_PROTECT", True))

    if not IMAGE_PROTECT_ON:
        # 署名なし
        q = {}
        if wm_q: q["wm"] = wm_q
        try:
            u = url_for("serve_image", filename=filename, _external=external, **q)
        except Exception:
            base = f"{MEDIA_URL_PREFIX}/{filename}"
            if q:
                base = f"{base}?{urlencode(q)}"
            u = base
        return _force_https_abs(u)

    # 署名あり
    ttl = int(ttl_sec or SIGNED_IMAGE_TTL_SEC)
    exp = int(_t.time()) + max(60, ttl)
    sig = _img_sig(filename, exp)
    q = {"sig": sig, "exp": exp}
    if wm_q: q["wm"] = wm_q

    try:
        u = url_for("serve_image", filename=filename, _external=external, **q)
    except Exception:
        base = f"{MEDIA_URL_PREFIX}/{filename}"
        sep  = "&" if "?" in base else "?"
        u    = f"{base}{sep}{urlencode(q)}"

    return _force_https_abs(u)


def _apply_text_watermark(im, text: str):
    """画像に半透明テキスト透かしを描画して返す（英語表記想定）"""
    from PIL import Image, ImageDraw

    if not text:
        return im

    base = im.convert("RGBA")
    W, H = base.size

    size = max(12, int(W * WATERMARK_SCALE))
    sw   = max(1, size // 12)  # 縁取り
    font = _load_wm_font(size)
    draw = ImageDraw.Draw(base)

    # テキストサイズ（stroke を考慮）
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=sw)
        tw, th = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        tw, th = draw.textsize(text, font=font)

    pad = max(6, size // 3)
    x = max(0, W - tw - pad * 2)
    y = max(0, H - th - pad * 2)

    opa = max(0, min(255, int(WATERMARK_OPACITY)))

    # 半透明の下地
    bg = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, int(opa * 0.45)))
    base.alpha_composite(bg, (x, y))

    # 白文字（黒縁取り）
    try:
        draw.text(
            (x + pad, y + pad), text,
            fill=(255, 255, 255, opa),
            font=font,
            stroke_width=sw,
            stroke_fill=(0, 0, 0, min(255, opa + 50)),
        )
    except TypeError:
        draw.text((x + pad, y + pad), text, fill=(255, 255, 255, opa), font=font)

    return base.convert("RGB")

# ===== 透かしバリアント生成（none / gotocity / fullygoto） =====
WM_VARIANTS = {
    "none":      {"suffix": "",           "text": None},
    "gotocity":  {"suffix": "__wm-city",  "text": "＠Goto City"},
    "fullygoto": {"suffix": "__wm-fully", "text": "@fullyGOTO"},
}

def _ensure_wm_variants(basename: str) -> dict:
    """
    IMAGES_DIR/<basename> を元に、3種の静的ファイルを作る。
    戻り値: {"none": "xxx.jpg", "gotocity": "...", "fullygoto": "..."}
    """
    import os
    from PIL import Image

    root, ext = os.path.splitext(basename)
    src = os.path.join(IMAGES_DIR, basename)
    out = {}

    if not os.path.isfile(src):
        return out

    im = Image.open(src).convert("RGB")
    for key, spec in WM_VARIANTS.items():
        fn  = root + spec["suffix"] + ext
        dst = os.path.join(IMAGES_DIR, fn)
        try:
            if spec["text"]:
                im2 = _apply_text_watermark(im, spec["text"])
            else:
                im2 = im
            # JPEG前提で保存（既存のアップロード関数がJPEG化している想定）
            im2.save(dst, "JPEG", quality=85, optimize=True, progressive=True)
            out[key] = fn
        except Exception:
            app.logger.exception("[wm] failed to save %s", fn)
    return out

def _save_jpeg_1080_350kb(file_storage, *, previous: str|None=None, delete: bool=False) -> str|None:
    """
    画像アップロードを『横1080px, JPEG, 350KB以下』に正規化して保存。
    - 新規/置換時: .jpg で保存してファイル名（例: "abcd1234.jpg"）を返す
    - 削除時   : "" を返す
    - 変更なし : None を返す
    - 極端に巨大なピクセル数は 413 (RequestEntityTooLarge) を送出
    """
    import os, io, uuid
    from PIL import Image, ImageOps, UnidentifiedImageError
    from werkzeug.utils import secure_filename
    from werkzeug.exceptions import RequestEntityTooLarge

    # ===== デフォルト値（外部定数が無くても動く） =====
    try:
        TARGET_W = int(globals().get("TARGET_JPEG_MAX_W", 1080))
    except Exception:
        TARGET_W = 1080
    try:
        TARGET_BYTES = int(globals().get("TARGET_JPEG_MAX_BYTES", 350 * 1024))
    except Exception:
        TARGET_BYTES = 350 * 1024
    try:
        IMAGES_DIR = globals()["IMAGES_DIR"]
    except KeyError:
        # 通常はグローバルにありますが、保険
        IMAGES_DIR = os.path.join(os.getcwd(), "data", "images")
    # Pillow のランチョス補間（無ければ BICUBIC）
    try:
        RESAMPLE = globals().get("RESAMPLE_LANCZOS", Image.LANCZOS)
    except Exception:
        RESAMPLE = getattr(Image, "LANCZOS", Image.BICUBIC)

    # 拡張子チェック（HEIC 等も許容。Pillow が開けなければ除外）
    try:
        ALLOWED_EXTS = set(globals().get("ALLOWED_IMAGE_EXTS", {
            ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"
        }))
    except Exception:
        ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}

    # ===== 削除指定 =====
    if delete:
        if previous:
            try:
                os.remove(os.path.join(IMAGES_DIR, previous))
            except Exception:
                # 失敗しても致命ではない
                pass
        return ""

    # ===== アップロード無し =====
    if not file_storage or not getattr(file_storage, "filename", ""):
        return None

    # ファイル名から拡張子確認（厳密には使わず、Pillow で最終判定）
    fname = secure_filename(file_storage.filename or "")
    _, ext = os.path.splitext(fname)
    ext = ext.lower().strip()

    try:
        # 画像読み込み（爆弾対策もここで拾う）
        file_storage.stream.seek(0)
        im = Image.open(file_storage.stream)
        im = ImageOps.exif_transpose(im)  # 向き補正

        # ピクセル数ガード
        w, h = im.size
        limit = getattr(Image, "MAX_IMAGE_PIXELS", None)
        if not limit:
            try:
                limit = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))  # 40MP 目安
            except Exception:
                limit = 40000000
        if (w * h) > int(limit):
            raise RequestEntityTooLarge(f"pixels={w*h} > MAX_IMAGE_PIXELS={limit}")

    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        app.logger.warning("Pillow decompression bomb blocked: %s", e)
        raise RequestEntityTooLarge("Image pixel count too large")
    except UnidentifiedImageError:
        # 画像ではない
        app.logger.warning("Uploaded file is not a valid image: %r", fname)
        return None
    except Exception:
        app.logger.exception("failed to open uploaded image")
        return None

    # 拡張子が未知でも Pillow で開ければ続行（従来の厳しすぎる拡張子判定が登録不可の原因になりやすいため）
    if ext and ext not in ALLOWED_EXTS:
        app.logger.info("unknown ext but image opened by Pillow: %s", ext)

    # JPEG保存用に RGB 化
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    elif im.mode == "L":
        im = im.convert("RGB")

    # 横幅 1080px まで縮小（小さいものは拡大しない）
    w, h = im.size
    if w > TARGET_W:
        new_h = int(h * TARGET_W / w)
        im = im.resize((TARGET_W, new_h), RESAMPLE)

    # エンコード関数
    def encode(quality: int, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(
            buf, format="JPEG",
            quality=int(quality),
            optimize=True,
            progressive=True,
            subsampling=2  # 4:2:0
        )
        return buf.getvalue()

    # まず品質85で試す → バイナリサーチで TARGET_BYTES 以下に
    try:
        best = None
        data = encode(85, im)
        if len(data) <= TARGET_BYTES:
            best = data
        else:
            lo, hi = 40, 88
            while lo <= hi:
                mid = (lo + hi) // 2
                data = encode(mid, im)
                if len(data) <= TARGET_BYTES:
                    best = data
                    lo = mid + 1
                else:
                    hi = mid - 1

            # それでも大きい場合は段階的に 90% 縮小しつつ再探索（最大2回）
            shrink_try = 0
            cur = im
            while (best is None or len(best) > TARGET_BYTES) and shrink_try < 2:
                shrink_try += 1
                cw, ch = cur.size
                cur = cur.resize((max(1, int(cw * 0.9)), max(1, int(ch * 0.9))), RESAMPLE)
                lo, hi = 40, 85
                candidate = None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    data = encode(mid, cur)
                    if len(data) <= TARGET_BYTES:
                        candidate = data
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if candidate:
                    best = candidate

            if best is None:
                best = encode(75, cur)

        # 保存（.jpg 固定）
        os.makedirs(IMAGES_DIR, exist_ok=True)
        new_name = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(IMAGES_DIR, new_name)
        with open(save_path, "wb") as wf:
            wf.write(best)

        # 旧ファイル削除（置換）
        if previous and previous != new_name:
            try:
                os.remove(os.path.join(IMAGES_DIR, previous))
            except Exception:
                pass

        return new_name

    except RequestEntityTooLarge:
        # 413 は上位ハンドラに
        raise
    except Exception:
        app.logger.exception("image normalize & save failed")
        return None


# ========= WM helpers: 透かしモード正規化・ユーザー／エントリ参照 =========
def _norm_wm_mode(v) -> str | None:
    """
    入力を 'fullygoto' / 'gotocity' / None に正規化。
    受け付ける同義語: 'fully'→fullygoto, 'city'→gotocity, '1'→デフォルトは gotocity
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"fullygoto", "gotocity"}:
        return s
    if s in {"fully"}:
        return "fullygoto"
    if s in {"city"}:
        return "gotocity"
    if s in {"1", "true", "on"}:
        # 旧実装の「ON」を既定として city ロゴに合わせる（必要なら fullygoto に変更可）
        return "gotocity"
    if s in {"0", "off", "false", "none", ""}:
        return None
    return s  # 未知値はそのまま（将来拡張用）

# --- per-user の透かしモードを簡易保存（JSON） ---
_WM_PREFS_FILE = os.path.join(DATA_DIR, "user_prefs.json")

def _load_user_prefs() -> dict:
    try:
        if os.path.exists(_WM_PREFS_FILE):
            with open(_WM_PREFS_FILE, "r", encoding="utf-8") as rf:
                return json.load(rf) or {}
    except Exception:
        app.logger.exception("load user_prefs failed")
    return {}

def _save_user_prefs(d: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = _WM_PREFS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as wf:
            json.dump(d, wf, ensure_ascii=False, indent=2)
        os.replace(tmp, _WM_PREFS_FILE)
    except Exception:
        app.logger.exception("save user_prefs failed")

def _get_user_wm(user_id: str | None) -> str | None:
    if not user_id:
        return None
    prefs = _load_user_prefs()
    return _norm_wm_mode((prefs.get("wm_choices") or {}).get(user_id))

def _set_user_wm(user_id: str | None, choice: str | None):
    if not user_id:
        return
    prefs = _load_user_prefs()
    prefs.setdefault("wm_choices", {})
    prefs["wm_choices"][user_id] = _norm_wm_mode(choice)
    _save_user_prefs(prefs)

def parse_wm_command(text: str) -> str | None:
    """
    ユーザーがチャットで透かしを切り替えるコマンドを検出。
    返り値: 'fullygoto'|'gotocity'|'none'|None
    """
    t = (text or "").strip().lower()
    if "@fullygoto" in t or "fullygoto" in t:
        return "fullygoto"
    if "@goto city" in t or "@gotocity" in t or "gotocity" in t:
        return "gotocity"
    if "@wm off" in t or "@wm none" in t or "@wm: none" in t:
        return "none"
    return None


# ========= WM compose: 透かし画像の場所と合成ユーティリティ =========
# デフォルトの透かし画像のパス（必要に応じて配置先を調整）
_WM_DIR = os.path.join(BASE_DIR, "static", "wm")
_WM_FILES = {
    "fullygoto": os.path.join(_WM_DIR, "wm_fullygoto.png"),
    "gotocity":  os.path.join(_WM_DIR, "wm_gotocity.png"),
}
# 合成時の基本パラメータ（お好みで調整可）
_WM_OPACITY = 0.85      # 0.0〜1.0
_WM_SCALE_W = 0.32      # 透かし幅を元画像幅の何倍に縮尺
_WM_PAD_RATIO = 0.03    # 右下余白（元画像幅に対する比）

def _load_wm_image(mode: str):
    """
    透かし画像(PNG, RGBA推奨)を読み込んで RGBA で返す。見つからなければ None。
    """
    try:
        path = _WM_FILES.get(mode or "", "")
        if not path or not os.path.isfile(path):
            return None
        from PIL import Image
        wm = Image.open(path)
        if wm.mode != "RGBA":
            wm = wm.convert("RGBA")
        return wm
    except Exception:
        app.logger.exception("load watermark image failed")
        return None

def _compose_watermark(base_im, mode: str):
    """
    base_im: Pillow Image (RGB/RGBA/L 不問)。返り値は RGB。
    右下に半透明ロゴを重ねる。
    """
    from PIL import Image, ImageOps

    if base_im is None or not mode:
        return base_im

    wm = _load_wm_image(mode)
    if wm is None:
        return base_im

    # RGB 化（最終JPEG想定）
    if base_im.mode not in ("RGB", "L"):
        base = base_im.convert("RGB")
    elif base_im.mode == "L":
        base = base_im.convert("RGB")
    else:
        base = base_im

    bw, bh = base.size
    if bw < 10 or bh < 10:
        return base

    # スケール・位置
    target_w = max(1, int(bw * _WM_SCALE_W))
    scale = target_w / wm.width
    target_h = max(1, int(wm.height * scale))
    wm_resized = wm.resize((target_w, target_h), resample=RESAMPLE_LANCZOS)

    # 透明度
    if _WM_OPACITY < 1.0:
        alpha = wm_resized.split()[3]  # A
        alpha = alpha.point(lambda a: int(a * _WM_OPACITY))
        wm_resized.putalpha(alpha)

    pad = int(bw * _WM_PAD_RATIO)
    x = max(0, bw - wm_resized.width - pad)
    y = max(0, bh - wm_resized.height - pad)

    # 合成
    base_rgba = base.convert("RGBA")
    base_rgba.paste(wm_resized, (x, y), wm_resized)
    return base_rgba.convert("RGB")


def _get_image_meta(filename: str):
    """画像ファイルの URL / バイト数 / 幅高さ(px) / KB を返す（なければ None）"""
    if not filename:
        return None
    try:
        path = os.path.join(IMAGES_DIR, filename)
        size_b = os.path.getsize(path)
    except Exception:
        return None

    w = h = None
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        pass

    try:
        url = build_signed_image_url(filename, wm=True, external=True)
    except Exception:
        url = f"{MEDIA_URL_PREFIX}/{filename}"

    return {
        "name": filename,
        "bytes": size_b,
        "kb": round(size_b / 1024, 1),
        "w": w,
        "h": h,
        "url": url,
    }

# --- LINE安全対策（ミュート＆全体一時停止）---
MUTES_FILE = os.path.join(BASE_DIR, "line_mutes.json")          # 会話単位のミュート管理
GLOBAL_LINE_PAUSE_FILE = os.path.join(BASE_DIR, "line_paused.flag")  # 全体停止フラグ（存在すれば停止）
# 全体停止中にユーザーの「再開」で動かすか（既定=OFF）
ALLOW_RESUME_WHEN_PAUSED = os.getenv("ALLOW_RESUME_WHEN_PAUSED", "0").lower() in {"1","true","on","yes"}
LINE_RETHROW_ON_SEND_ERROR = os.getenv("LINE_RETHROW_ON_SEND_ERROR", "0").lower() in {"1","true","on","yes"}

# （ここから追記）停止中案内の一回通知設定
LINE_PAUSE_NOTICE = os.getenv("LINE_PAUSE_NOTICE", "1").lower() in {"1","true","on","yes"}

def _notice_paused_once(event, target_id: str):
    """全体一時停止中のとき、対象に一度だけ“停止中”メッセージを返す"""
    if not LINE_PAUSE_NOTICE:
        return
    try:
        # 送信履歴がある実装ならスパム防止で使う
        if "_was_sent_recent" in globals():
            key = "__paused_notice__"
            if _was_sent_recent(target_id, key, mark=True):
                return
        _reply_or_push(event,
            "（現在メンテナンスのため一時停止中です。再開までお待ちください）\n"
            "※再開後に応答が必要なら「再開」と送ってください。"
        )
    except Exception:
        app.logger.exception("paused notice failed")
# （追記ここまで）

# 1通原則＋長文だけ分割のための閾値（既に定義済みならそのままでOK）
LINE_SAFE_CHARS = int(os.getenv("LINE_SAFE_CHARS", "3800"))


# ---- synonyms 進捗＆キュー（追記）----
PENDING_SYNONYMS_FILE = os.path.join(BASE_DIR, "synonyms_autogen_queue.json")

def _compute_tag_sets_for_synonyms():
    """
    entries.json から全タグ集合を作り、
    ・have    = 辞書に類義語が「ある」タグ
    ・missing = 辞書に類義語が「ない」タグ
    を返すユーティリティ。
    """
    entries = load_entries()
    all_tags = set()
    for e in entries:
        for t in (e.get("tags") or []):
            t = (t or "").strip()
            if t:
                all_tags.add(t)

    syn = load_synonyms()
    have = sorted([t for t in all_tags if syn.get(t)])
    missing = sorted([t for t in all_tags if not syn.get(t)])
    return sorted(all_tags), have, missing

def _load_syn_queue():
    """
    生成待ち行列（キュー）を JSON から読み込む。
    無ければ {} を返す（新規作成時に使う）。
    """
    return _safe_read_json(PENDING_SYNONYMS_FILE, {})

def _save_syn_queue(data):
    """
    生成待ち行列（キュー）を安全に保存する。
    """
    _atomic_json_dump(PENDING_SYNONYMS_FILE, data)


# 必要フォルダ作成
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 便利関数: JSONをアトミックに保存
def _atomic_json_dump(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, obj: dict):
    """JSONLに1行追記（失敗してもアプリは落とさない）"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        app.logger.exception("sendlog append failed")

def _extract_wm_flags(url: str | None) -> dict | None:
    """URLから wm / _sign / exp などを抽出"""
    if not url:
        return None
    try:
        pr = _u.urlparse(url)
        qs = _u.parse_qs(pr.query or "")
        wm  = (qs.get("wm", [None])[0] or "")
        exp = (qs.get("exp", [None])[0] or None)
        signed = ("_sign" in qs)
        return {"url": url, "wm": wm, "signed": bool(signed), "exp": exp}
    except Exception:
        return {"url": url}

def _iter_image_links_from_messages(messages) -> list[dict]:
    """
    LINEメッセージ配列から、実際に送る画像URL群をフラット収集。
    - ImageSendMessage: original/preview
    - Flex: どこにあっても type=image の url を走査
    """
    out: list[dict] = []

    def _walk(obj):
        if isinstance(obj, dict):
            if (obj.get("type") == "image") and obj.get("url"):
                info = _extract_wm_flags(obj.get("url"))
                if info: out.append(info)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it)

    for m in (messages or []):
        # ImageSendMessage
        if getattr(m, "original_content_url", None) or getattr(m, "preview_image_url", None):
            oi = _extract_wm_flags(getattr(m, "original_content_url", None))
            pi = _extract_wm_flags(getattr(m, "preview_image_url", None))
            if oi: out.append({"kind": "image:original", **oi})
            if pi: out.append({"kind": "image:preview",  **pi})
            continue

        # FlexSendMessage
        if m.__class__.__name__ == "FlexSendMessage":
            try:
                payload = m.as_json_dict()  # line-bot-sdk のAPI
                _walk(payload)
            except Exception:
                pass

    return out

def _log_sent_messages(event, messages, status: str):
    """
    送信確定ログ（確実に“このURLで投げた”という証跡）
    status: 'replied' | 'pushed' | 'dryrun' | 'error' | 'noop'
    """
    import random, time
    try:
        if SEND_LOG_SAMPLE <= 0.0:
            return
        if random.random() > SEND_LOG_SAMPLE:
            return

        rt  = getattr(event, "reply_token", None)
        tid = None
        try:
            tid = _line_target_id(event)
        except Exception:
            pass

        record = {
            "ts": int(time.time()),
            "kind": "line_send",
            "status": status,
            "reply": bool(rt),
            "target": tid,
            "count": len(messages or []),
            "images": _iter_image_links_from_messages(messages),
        }
        _append_jsonl(SEND_LOG_FILE, record)
    except Exception:
        # ログでアプリを止めない
        app.logger.exception("send confirm log failed")


# ★ここから追加：一度だけ案内 用の保存/読込/クリア
def _load_paused_notices() -> dict:
    """{ target_id: last_notified_ts } を返す"""
    return _safe_read_json(PAUSED_NOTICE_FILE, {})

def _save_paused_notices(obj: dict):
    _atomic_json_dump(PAUSED_NOTICE_FILE, obj or {})

def _clear_paused_notices():
    """キャッシュ全消し（全体一時停止 解除時に呼ぶ）"""
    try:
        if os.path.exists(PAUSED_NOTICE_FILE):
            os.remove(PAUSED_NOTICE_FILE)
    except Exception:
        app.logger.exception("clear paused notices failed")
    # ★ 追加：メモリ上の「一度だけ案内」フラグも全消し
    try:
        _clear_pause_notice_cache_all()
    except Exception:
        pass

def _safe_read_json(path: str, default_obj):
    """
    JSONを安全に読み込む。壊れていたら .bad-YYYYmmdd-HHMMSS に退避し、
    default_obj を書き戻してから default_obj を返す。
    """
    try:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_obj, f, ensure_ascii=False, indent=2)
            return default_obj
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        app.logger.exception(f"[safe_read_json] JSON read failed: {path}: {e}")
        # 破損ファイルを退避
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            os.replace(path, f"{path}.bad-{ts}")
        except Exception:
            pass
        # 既定値で初期化
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_obj, f, ensure_ascii=False, indent=2)
        except Exception:
            app.logger.exception(f"[safe_read_json] JSON rewrite failed: {path}")
        return default_obj

import unicodedata
def _n(s: str) -> str:
    """NFKC正規化 + 小文字化 + 連続空白圧縮"""
    import unicodedata, re
    t = unicodedata.normalize("NFKC", (s or ""))
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def _is_stop_cmd(tnorm: str) -> bool:
    # 「停止」「中止」「やめて」や英語、語尾つきも拾う
    if tnorm in STOP_COMMANDS:
        return True
    if tnorm.startswith("停止") or tnorm.startswith("中止") or tnorm.startswith("やめて"):
        return True
    return bool(re.search(r"\b(stop|mute|silence)\b", tnorm))


def _load_mutes() -> dict:
    return _safe_read_json(MUTES_FILE, {})

def _save_mutes(obj: dict):
    _atomic_json_dump(MUTES_FILE, obj or {})

def _line_target_id(event) -> str:
    return getattr(event.source, "user_id", None) \
        or getattr(event.source, "group_id", None) \
        or getattr(event.source, "room_id", None) \
        or ""

def _is_muted_target(target_id: str) -> bool:
    m = _load_mutes()
    rec = m.get(target_id)
    return bool(rec and rec.get("muted"))

def _set_muted_target(target_id: str, muted: bool, who="user"):
    m = _load_mutes()
    m[target_id] = m.get(target_id, {})
    m[target_id]["muted"] = bool(muted)
    m[target_id]["by"] = who
    m[target_id]["ts"] = time.time()
    _save_mutes(m)

def _is_global_paused() -> bool:
    return os.path.exists(GLOBAL_LINE_PAUSE_FILE)

def _set_global_paused(paused: bool):
    try:
        if paused:
            open(GLOBAL_LINE_PAUSE_FILE, "a", encoding="utf-8").close()
        else:
            if os.path.exists(GLOBAL_LINE_PAUSE_FILE):
                os.remove(GLOBAL_LINE_PAUSE_FILE)
            _clear_paused_notices()    
    except Exception:
        app.logger.exception("set_global_paused failed")

# ==== Mute/Resume 判定ヘルパ（強化版） ====
import re

def _norm_cmd(s: str) -> str:
    if s is None:
        return ""
    # 全角スペース→半角、改行/タブ削除、英字小文字化、空白削除
    s = s.replace("　", " ").strip().lower()
    s = re.sub(r"\s+", "", s)
    return s

# 正規化後の代表語（スペース無し・小文字・NFKC前提）
RESUME_COMMANDS = {
    "再開", "解除", "ミュート解除", "応答再開",
    "resume", "unmute", "start", "muteoff", "mute_off", "muteオフ"
}
PAUSE_COMMANDS = {
    "停止", "一時停止", "ミュート", "黙って", "黙る",
    "stop", "pause", "muteon", "mute_on", "muteオン"
}

def _is_pause_cmd(text: str) -> bool:
    """停止コマンド判定：停止/一時停止/ミュート/stop/pause/mute on 等"""
    t = _norm_cmd(text)
    if t in PAUSE_COMMANDS:
        return True
    if any(k in t for k in ("停止", "一時停止", "ミュート", "黙って", "黙る")):
        return True
    return any(k in t for k in ("stop", "pause", "muteon"))

def _is_resume_cmd(text: str) -> bool:
    """
    再開コマンド判定：
    - 日本語：「再開」「解除」「ミュート解除」「再開してください」等（前方一致OK）
    - 英語   ：resume / unmute / start / mute off 等
    """
    t = _norm_cmd(text)
    if t in RESUME_COMMANDS:
        return True
    if t.startswith("再開") or t.startswith("解除") or t.startswith("応答再開"):
        return True
    return any(k in t for k in ("resume", "unmute", "start", "muteoff"))

def _clear_pause_notice_cache(tid: str):
    """
    “全体一時停止中の案内を一度だけ返す”系の既存キャッシュがあればクリア。
    名前は環境差があるので代表的なキーを総当たり。
    """
    for k in ("_PAUSE_NOTICE_SENT", "PAUSE_NOTICE_SENT", "_NOTICE_PAUSED_ONCE"):
        d = globals().get(k)
        if isinstance(d, dict):
            try:
                d.pop(tid, None)
            except Exception:
                pass
            

def _clear_pause_notice_cache_all():
    """
    上記キャッシュをプロセス内で全消し（管理者の全体再開で呼ぶ用）。
    """
    for k in ("_PAUSE_NOTICE_SENT", "PAUSE_NOTICE_SENT", "_NOTICE_PAUSED_ONCE"):
        d = globals().get(k)
        if isinstance(d, dict):
            try:
                d.clear()
            except Exception:
                pass
            
# ==== 統一ゲート（管理者最優先＋再開で旧ミュートも解除） ====
def _line_mute_gate(event, text: str) -> bool:
    """
    True  -> ここで処理完了（以降の通常応答は行わない）
    False -> 通常の応答処理を継続
    優先順位: 管理者 > ユーザー
    """
    t = (text or "").strip()

    # 停止状態（新API: _pause_state、無ければ旧APIにフォールバック）
    try:
        paused, by = _pause_state()  # 期待: (bool, 'admin'|'user'|None)
    except Exception:
        try:
            paused = bool(_is_global_paused())
        except Exception:
            paused = False
        by = "admin" if paused else None

    # ターゲットID
    try:
        tid = _line_target_id(event)
    except Exception:
        tid = getattr(getattr(event, "source", None), "user_id", None) or "anon"

    # 1) 管理者停止中は常にミュート（ユーザーの再開も無効・案内のみ）
    if paused and by == "admin":
        if _is_pause_cmd(t) or _is_resume_cmd(t):
            try:
                _reply_quick_no_dedupe(event, "現在、管理者によって応答が停止されています。管理画面から再開されるまでお待ちください。")
            except Exception:
                pass
        return True

    # 2) ユーザー「再開」：全体（user）＆旧ミュートの両方を解除＋案内
    if _is_resume_cmd(t):
        try:
            _pause_set_user(False)  # 全体（ユーザー）停止解除
        except Exception:
            pass
        try:
            _set_muted_target(tid, False, who="user")  # 旧・会話ミュート解除
        except Exception:
            pass
        _clear_pause_notice_cache(tid)  # “一度だけ案内”キャッシュを消す
        try:
            _reply_quick_no_dedupe(event, "了解です。応答を再開します。")
        except Exception:
            pass
        return True

    # 3) ユーザー「停止」：両方ON（互換のため）＋案内
    if _is_pause_cmd(t):
        try:
            _pause_set_user(True)   # 全体（ユーザー）停止ON
        except Exception:
            pass
        try:
            _set_muted_target(tid, True, who="user")  # 旧・会話ミュートON
        except Exception:
            pass
        try:
            _reply_quick_no_dedupe(event, "了解です。しばらく応答を停止します。「解除」と送ると元に戻します。")
        except Exception:
            pass
        return True

    # 4) ユーザー停止中は案内のみ返してミュート
    if paused and by == "user":
        try:
            _reply_quick_no_dedupe(event, "（現在、応答を一時停止しています。「解除」と送ると再開します）")
        except Exception:
            pass
        return True

    # 5) 念のため：旧・会話ミュートが残っていたら案内してミュート
    try:
        if "_is_muted_target" in globals() and _is_muted_target(tid):
            try:
                _reply_quick_no_dedupe(event, "（この会話はミュート中です。「解除」と送ると再開します）")
            except Exception:
                pass
            return True
    except Exception:
        pass

    # 6) 通常処理へ
    return False

# ==== 展望所マップ: コマンド検出（表記ゆれ対応） ====
_CMD_RE_VIEWPOINTS = re.compile(
    r"(?:展望\s*(?:所|台)?\s*(?:マップ|地図))|(?:viewpoints?\s*map)",
    re.I
)

def _is_viewpoints_cmd(text: str) -> bool:
    """
    表記ゆれ（NFKC・大小文字・全半角・スペース・句読点）を吸収して判定
    """
    t = _norm_cmd(text)  # NFKC + lower + trim
    # 単語間スペース・全角スペース・句読点/記号を除去して緩く判定
    t = re.sub(r"[ \t\u3000／/、。,．。!！?？\-ー〜~]+", "", t)
    return bool(_CMD_RE_VIEWPOINTS.search(t))


# 初回ブートストラップ
ADMIN_INIT_USER = os.environ.get("ADMIN_INIT_USER", "admin")
ADMIN_INIT_PASSWORD = os.environ.get("ADMIN_INIT_PASSWORD")  # 初回のみ使用推奨

def _ensure_json(path, default_obj):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f, ensure_ascii=False, indent=2)

def _bootstrap_files_and_admin():
    app.logger.info(f"[boot] BASE_DIR={BASE_DIR}")
    app.logger.info(f"[boot] USERS_FILE path: {USERS_FILE}")

    _ensure_json(ENTRIES_FILE, [])
    _ensure_json(SYNONYM_FILE, {})
    _ensure_json(NOTICES_FILE, [])
    _ensure_json(SHOP_INFO_FILE, {})

    users = []
    users_exists = os.path.exists(USERS_FILE)
    if users_exists:
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = []

    if not users_exists or not users:
        if ADMIN_INIT_PASSWORD:
            users = [{
                "user_id": ADMIN_INIT_USER,
                "name": "管理者",
                "password_hash": generate_password_hash(ADMIN_INIT_PASSWORD),  # 既定=pbkdf2:sha256
                "role": "admin",
            }]
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            app.logger.warning(
                "users.json を新規作成し、管理者ユーザー '%s' を作成しました。初回ログイン後 ADMIN_INIT_PASSWORD を環境変数から削除してください。",
                ADMIN_INIT_USER,
            )
        else:
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            app.logger.warning(
                "users.json を作成しましたが管理者は未作成です。ADMIN_INIT_PASSWORD を設定して再デプロイするか、手動で users.json を用意してください。"
            )

_bootstrap_files_and_admin()

# 必須キーが未設定なら警告（起動は継続）
if not OPENAI_API_KEY:
    app.logger.warning("OPENAI_API_KEY が未設定です。OpenAI 呼び出しは失敗します。")
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    app.logger.warning("LINE_CHANNEL_* が未設定です。/callback は正常動作しません。")

# === OpenAIの出力から安全にJSONオブジェクトを取り出すユーティリティ ===
def _extract_json_object(text: str):
    """
    OpenAIが時々つける前置き/後置きや ```json フェンス、全角引用符、末尾カンマなどを掃除し、
    最初の JSON オブジェクト（{...}）だけを抜き出して dict として返す。
    失敗したら None を返す。
    """
    if not text:
        return None
    s = str(text)

    # コードフェンスを優先抽出（```json ... ``` or ``` ... ```）
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1)

    # よく混ざる説明行を刈り取る
    # 例: "以下のJSONです:" などの前置き・後置きを雑に除去
    # JSONらしき最初の { と最後の } をスタックで対応付け
    start = s.find("{")
    if start == -1:
        return None

    # 全角引用符 → 半角に正規化
    s = s.replace("“", '"').replace("”", '"').replace("‟", '"').replace("＂", '"').replace("’", "'").replace("‘", "'")

    # JSONコメント/余分な制御文字を除去（緩め）
    s = re.sub(r"^\s*//.*?$", "", s, flags=re.M)         # 行頭コメント //...
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)           # ブロックコメント /* ... */
    s = re.sub(r"[^\S\r\n]+\n", "\n", s)                  # 余分な空白

    # { ... } の対応をとり、最初の完全なオブジェクトを取り出す
    depth = 0
    end = None
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    jtxt = s[start:end+1].strip()

    # 末尾カンマの軽微な修正（キー/配列の末尾にあるケース）
    jtxt = re.sub(r",(\s*[}\]])", r"\1", jtxt)

    try:
        return json.loads(jtxt)
    except Exception:
        # ダブルクォートが欠け気味のケースに軽く対応（キーはダブルクォート必須）
        # ここでは過度に変換せず諦める
        return None


# === 透かしモード正規化（共通 util） ===
WM_CANON_MAP = {
    "none": "none", "no": "none", "off": "none", "0": "none", False: "none",
    "fully": "fully", "full": "fully", "all": "fully", "1": "fully", True: "fully",
    "city": "city",
    # 互換（旧名称の吸収）
    "fullygoto": "fully",
    "gotocity": "city",
}

def _wm_normalize(v) -> str | None:
    """
    透かしモードの入力値を正規化して 'none' / 'fully' / 'city' のいずれかにする。
    不明値は None を返す（呼び出し側で既定値を決める）。
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true","false"}:
        return "fully" if (s == "true") else "none"
    return WM_CANON_MAP.get(s, None)

# =========================
#  重複統合（タイトル基準）
# =========================

DEDUPE_ON_SAVE = os.environ.get("DEDUPE_ON_SAVE", "1").lower() in {"1","true","on","yes"}
DEDUPE_USE_AI  = os.environ.get("DEDUPE_USE_AI",  "1").lower() in {"1","true","on","yes"}

def _title_key(s: str) -> str:
    """タイトルを比較用に正規化（全半角・空白のゆらぎ吸収。過剰には潰さない）"""
    s = unicodedata.normalize("NFKC", (s or "").strip())
    # 連続空白を1つに
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _uniq_keep_order(items):
    def _norm_key(x):
        s = unicodedata.normalize("NFKC", str(x)).strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s
    seen = set()
    out = []
    for x in items or []:
        if not x:
            continue
        k = _norm_key(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip() if isinstance(x, str) else x)
    return out

def _mode_or_longest(values):
    """文字列候補から最頻値→同率なら最長→それでも同率なら先勝ち"""
    vs = [str(v).strip() for v in values if v and str(v).strip()]
    if not vs:
        return ""
    # 正規化して頻度
    norm = [re.sub(r"\s+", " ", unicodedata.normalize("NFKC", v)) for v in vs]
    cnt = Counter(norm)
    best_norm, _ = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0])))
    # 同じ正規化群の中から最長を返す
    candidates = [v for v, n in zip(vs, norm) if n == best_norm]
    return sorted(candidates, key=lambda x: len(x), reverse=True)[0]

def _merge_extras(dicts: List[dict]) -> dict:
    """extras を統合。キーごとに最頻→最長を採用。"""
    all_keys = set()
    for d in dicts or []:
        all_keys.update((d or {}).keys())
    out = {}
    for k in all_keys:
        vals = []
        for d in dicts or []:
            v = (d or {}).get(k, "")
            if v:
                vals.append(v)
        out[k] = _mode_or_longest(vals)
    return {k: v for k, v in out.items() if v}

# ===== 重複統合の採用ロジック強化（地図URLなど） =====
MAP_PREFER_HOSTS = (
    "google.com/maps",
    "goo.gl/maps",
    "maps.app.goo.gl",
    "g.page",
    "g.co/kgs",
)

def _pick_best_map_url(urls):
    """候補URL群から“より良い”地図URLを1つ選ぶ（Google系を優先）"""
    if not urls:
        return ""
    cand = [str(u or "").strip() for u in urls if str(u or "").strip()]
    if not cand:
        return ""

    def score(u: str):
        s = 0
        # 形式がちゃんとしているものを優遇
        if u.startswith(("http://", "https://")):
            s += 2
        # Google系・Maps らしさを優遇
        if any(h in u for h in MAP_PREFER_HOSTS):
            s += 10
        if "maps" in u:
            s += 2
        # 極端に短いダミーURLよりは、ある程度長さがある方を少しだけ優遇
        if len(u) >= 20:
            s += 1
        return (s, len(u))

    cand.sort(key=score, reverse=True)
    return cand[0]

def _pick_best_string(values):
    """文字列系（住所/電話/営業時間など）は 最頻→最長 を採用"""
    return _mode_or_longest(values)

def _union_list(*lists):
    """複数の配列を結合しつつユニーク化＆順序維持"""
    out = []
    seen = set()
    for lst in lists:
        for x in (lst or []):
            s = str(x).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out

def _ai_optimize_description(title: str, descs: List[str], tags: List[str], areas: List[str]) -> str:
    base = _mode_or_longest(descs)
    # 追加：実質1件ならAIスキップ
    norm = {re.sub(r"\s+", " ", unicodedata.normalize("NFKC", d.strip())) for d in descs if d and d.strip()}
    if len(norm) <= 1:
        return base
    if not DEDUPE_USE_AI or not OPENAI_API_KEY:
        return base    
    try:
        prompt = (
            "以下の複数の説明文を、重複を避けて1本の日本語説明に統合してください。\n"
            "・事実関係の矛盾があれば最も一般的な説明に寄せる\n"
            "・観光ガイドとして初見でも分かるように、200〜350字を目安\n"
            "・誇張・未確認表現は避け、固有名/見どころ/注意点を簡潔に\n"
            f"【タイトル】{title}\n"
            f"【タグ】{', '.join(tags)}\n"
            f"【エリア】{', '.join(areas)}\n"
            "【説明候補】\n- " + "\n- ".join([d for d in descs if d.strip()][:10])
        )
        out = openai_chat(
            OPENAI_MODEL_HIGH,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return (out or "").strip() or base
    except Exception:
        return base

def dedupe_entries_by_title(entries: List[dict], use_ai: bool = DEDUPE_USE_AI, dry_run: bool = False):
    """
    同一タイトル（正規化キー一致）を統合。
    - 説明: AI（任意）で統合、失敗時は最長
    - タグ/エリア/リンク/支払い: ユニーク結合
    - 住所/電話/営業時間/定休/駐車/備考/地図等: 最頻→最長
    - 画像/透かし/緯度経度/出典/ユーザーID も保全
    - extras: キー単位で最頻→最長
    戻り値: (新entries, stats, preview)
    """
    import os

    def _first_number(x):
        try:
            if x is None or x == "":
                return None
            return float(x)
        except Exception:
            return None

    def _pick_best_image_name(names):
        # ベース名＋空白除去 → 最頻→最長
        cands = [os.path.basename(str(n).strip()) for n in names if str(n or "").strip()]
        return _mode_or_longest(cands)

    # 1) _pick_best_wm_choice 修正
    def _pick_best_wm_choice(group):
        vals = []
        for g in group:
            w_raw = g.get("wm_external_choice")
            # ここを _wm_normalize -> _normalize_wm_choice に
            w = _normalize_wm_choice(w_raw, wm_on_default=bool(g.get("wm_on", True)))
            if w:
                vals.append(w)
            else:
                if isinstance(g.get("wm_on"), bool):
                    vals.append("fullygoto" if g.get("wm_on") else "none")
        best = _mode_or_longest(vals) if vals else ""
        return best or "none"

    groups = {}
    for e in entries:
        e0 = _norm_entry(e)  # 先に正規化
        k = _title_key(e0.get("title", ""))
        if not k:
            groups[f"__keep_{id(e0)}"] = [e0]
            continue
        groups.setdefault(k, []).append(e0)

    new_list = []
    removed = 0
    merged_groups = 0
    preview = []

    for k, group in groups.items():
        if len(group) == 1:
            new_list.append(group[0])
            continue

        merged_groups += 1

        titles = [g.get("title","") for g in group]
        title = _mode_or_longest(titles)  # タイトル：最頻→最長

        # 主要文字列項目
        address     = _pick_best_string([g.get("address","")     for g in group])
        tel         = _pick_best_string([g.get("tel","")         for g in group])
        holiday     = _pick_best_string([g.get("holiday","")     for g in group])
        open_hours  = _pick_best_string([g.get("open_hours","")  for g in group])
        parking     = _pick_best_string([g.get("parking","")     for g in group])
        parking_num = _pick_best_string([g.get("parking_num","") for g in group])
        remark      = _pick_best_string([g.get("remark","")      for g in group])
        map_url     = _pick_best_map_url([g.get("map","")        for g in group])  # Google系優先
        category    = _mode_or_longest([g.get("category","")     for g in group]) or "観光"
        # 追記：出典の保全
        source      = _pick_best_string([g.get("source","")      for g in group])
        source_url  = _pick_best_string([g.get("source_url","")  for g in group])

        # リスト系は結合ユニーク
        tags   = _uniq_keep_order(itertools.chain.from_iterable([g.get("tags",[])    for g in group]))
        areas  = _uniq_keep_order(itertools.chain.from_iterable([g.get("areas",[])   for g in group])) or ["五島市"]
        links  = _uniq_keep_order(itertools.chain.from_iterable([g.get("links",[])   for g in group]))
        pay    = _uniq_keep_order(itertools.chain.from_iterable([g.get("payment",[]) for g in group]))

        # extras をマージ
        extras = _merge_extras([g.get("extras",{}) for g in group])

        # --- 画像名の保全（image_file / image のどちらでも） ---
        img_names = []
        for g in group:
            v = (g.get("image_file") or g.get("image") or "").strip()
            if v:
                img_names.append(v)
        best_image = _pick_best_image_name(img_names) if img_names else ""

        # --- 透かし設定の保全 ---
        wm_choice = _pick_best_wm_choice(group)
        wm_on = any(bool(g.get("wm_on")) for g in group)  # 互換フラグは「どれかTrueならTrue」

        # --- 緯度・経度の保全（両方揃っている候補を優先） ---
        lat = lng = None
        # 1) lat/lng 両方あるレコードを優先
        for g in group:
            la = _first_number(g.get("lat"))
            ln = _first_number(g.get("lng"))
            if la is not None and ln is not None:
                lat, lng = la, ln
                break
        # 2) 片方しか無い場合の救済
        if lat is None:
            for g in group:
                la = _first_number(g.get("lat"))
                if la is not None:
                    lat = la
                    break
        if lng is None:
            for g in group:
                ln = _first_number(g.get("lng"))
                if ln is not None:
                    lng = ln
                    break

        # --- user_id の保全（最頻→最長）---
        user_id = _mode_or_longest([g.get("user_id","") for g in group])

        # 説明はAI統合（フォールバックあり）
        desc_candidates = [g.get("desc","") for g in group if (g.get("desc","").strip())]
        norm_descs = {
            re.sub(r"\s+", " ", unicodedata.normalize("NFKC", d.strip()))
            for d in desc_candidates
        }
        if len(norm_descs) <= 1:
            desc = desc_candidates[0] if desc_candidates else ""
        elif use_ai:
            desc = _ai_optimize_description(title, desc_candidates, tags, areas)
        else:
            desc = _mode_or_longest(desc_candidates)

        # 統合結果
        merged = {
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas,
            "links": links,
            "payment": pay,
            "tel": tel,
            "holiday": holiday,
            "open_hours": open_hours,
            "parking": parking,
            "parking_num": parking_num,
            "remark": remark,
            "extras": extras,
        }

        # 追記：出典
        if source:     merged["source"] = source
        if source_url: merged["source_url"] = source_url

        # 追記：画像（両フィールドを揃えて保持）
        if best_image:
            merged["image_file"] = best_image
            merged["image"] = best_image

        # 追記：透かし
        if wm_choice:
            merged["wm_external_choice"] = wm_choice
        merged["wm_on"] = bool(wm_on)

        # 追記：緯度経度
        if lat is not None: merged["lat"] = round(float(lat), 6)
        if lng is not None: merged["lng"] = round(float(lng), 6)

        # 追記：user_id
        if user_id:
            merged["user_id"] = user_id

        # 最後に正規化（ここで画像やwmが落ちないよう、_norm_entry の後付けパッチが有効）
        merged = _norm_entry(merged)

        new_list.append(merged)
        removed += (len(group) - 1)
        preview.append({"title": title, "merged_from": len(group)})

    stats = {"merged_groups": merged_groups, "removed": removed, "total_after": len(new_list)}
    if dry_run:
        # 乾式の場合は元データを返して外側で表示だけに使う
        return entries, stats, preview
    return new_list, stats, preview

# ============================================================
# Watermark 正規化（グローバル唯一の定義）
# ============================================================
def _normalize_wm_choice(choice: str | None, wm_on_default: bool = True) -> str:
    """
    入力のゆらぎ（'fully' / 'city' / '1' / True / None など）を
    保存・配信で使う正規形 'none' / 'fullygoto' / 'gotocity' に統一する。

    受理する主な入力:
      - 'none' / 'fullygoto' / 'gotocity' → そのまま返す
      - 'fully' → 'fullygoto'（全面透かし）
      - 'city'  → 'gotocity'（市ロゴのみ）
      - 真偽っぽい値:
          '1', 'true', 'on', 'yes'  → 'fullygoto'
          '0', 'false', 'off', 'no', '' → 'none'
      - それ以外/未指定 → wm_on_default が True なら 'fullygoto'、False なら 'none'
    """
    c = (str(choice or "")).strip().lower()
    if c in ("none", "fullygoto", "gotocity"):
        return c
    if c in ("1", "true", "on", "yes"):
        return "fullygoto"
    if c in ("0", "false", "off", "no", ""):
        return "none"
    if c == "fully":
        return "fullygoto"
    if c == "city":
        return "gotocity"
    return "fullygoto" if wm_on_default else "none"

# =========================
#  基本I/O
# =========================
def load_users():
    return _safe_read_json(USERS_FILE, [])

def save_qa_log(question, answer, source="web", hit_db=False, extra=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": source,
        "question": question,
        "answer": answer,
        "hit_db": hit_db,
        "extra": extra or {},
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

def _messages_to_prompt(messages):
    # role付きメッセージを1本のテキストにまとめる
    if isinstance(messages, str):
        return messages
    lines = []
    for m in messages:
        role = m.get("role","user")
        content = m.get("content","")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


# 言語検出＆翻訳
OPENAI_MODEL_TRANSLATE = os.environ.get("OPENAI_MODEL_TRANSLATE", OPENAI_MODEL_HIGH)

def detect_lang_simple(text: str) -> str:
    if not text:
        return "ja"
    t = text.strip()
    if re.search(r"[ぁ-んァ-ン]", t):
        return "ja"
    zh_markers = [
        "今天","天氣","天气","請問","请问","交通","景點","景点","門票","门票",
        "美食","住宿","船班","航班","預約","预约","營業","营业","營業時間","营业时间"
    ]
    if any(m in t for m in zh_markers):
        return "zh-Hant"
    if re.fullmatch(r"[A-Za-z0-9\s\-\.,!?'\"/()]+", t):
        return "en"
    return "ja"

def translate_text(text: str, target_lang: str) -> str:
    if not text:
        return text
    lang_label = {'ja': '日本語', 'zh-Hant': '繁體中文', 'en': '英語'}.get(target_lang, '日本語')
    prompt = (
        f"次のテキストを{lang_label}に自然に翻訳してください。"
        "意味は変えず、URL・数値・記号・改行はそのまま維持。"
        "箇条書きや見出しの構造も保持してください。\n===\n" + text
    )
    model_t = OPENAI_MODEL_TRANSLATE
    out = openai_chat(
        model_t,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200,
    )
    if not out and model_t != OPENAI_MODEL_PRIMARY:
        out = openai_chat(
            OPENAI_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )
    return out or text

# =========================
#  スモールトーク／使い方（スタブ）
# =========================
if 'get_weather_reply' not in globals():
    def get_weather_reply(text: str):
        return None, False

if 'get_transport_reply' not in globals():
    def get_transport_reply(text: str):
        return None, False

if 'build_refine_suggestions' not in globals():
    def build_refine_suggestions(text: str):
        return "キーワードを増やす（例：地名＋カテゴリ）／別表記でもう一度お試しください。", []

if 'smalltalk_or_help_reply' not in globals():
    def smalltalk_or_help_reply(text: str):
        t = _n(text)
        if t in {"help", "使い方", "ヘルプ"}:
            return "使い方：施設名やカテゴリで聞いてください。現在地から探す場合は「近く」と送ると位置情報ボタンが出ます。"
        if t in {"こんにちは", "こんちは", "hi", "hello"}:
            return "こんにちは！五島の情報をお探しですか？"
        return None

# =========================
#  お知らせ・データI/O
# =========================
def load_notices():
    """お知らせ一覧（必ず list を返す）"""
    data = _safe_read_json(NOTICES_FILE, [])
    return data if isinstance(data, list) else []

def save_notices(notices):
    """お知らせ一覧をアトミック保存（不正値でも list 化）"""
    _atomic_json_dump(NOTICES_FILE, list(notices or []))


def load_entries():
    """スポット一覧（正規化して返す）"""
    raw = _safe_read_json(ENTRIES_FILE, [])
    return [_norm_entry(e) for e in raw if isinstance(e, dict)]

def save_entries(entries):
    """
    保存前に正規化＋（設定オンなら）タイトル重複統合。
    壊れ値が混じっても dict のみ採用して落ちないように。
    """
    items = [_norm_entry(e) for e in (entries or []) if isinstance(e, dict)]
    if DEDUPE_ON_SAVE:
        try:
            items, stats, _ = dedupe_entries_by_title(items, use_ai=DEDUPE_USE_AI, dry_run=False)
            app.logger.info(
                "[dedupe] merged_groups=%s removed=%s total_after=%s",
                stats.get("merged_groups"), stats.get("removed"), stats.get("total_after")
            )
        except Exception:
            app.logger.exception("[dedupe] failed")
    _atomic_json_dump(ENTRIES_FILE, items)


def load_synonyms():
    """タグ類義語辞書（必ず dict を返す）"""
    data = _safe_read_json(SYNONYM_FILE, {})
    return data if isinstance(data, dict) else {}

def save_synonyms(synonyms):
    """類義語辞書をアトミック保存（不正値でも dict 化）"""
    _atomic_json_dump(SYNONYM_FILE, dict(synonyms or {}))


def load_shop_info(user_id):
    """
    店舗用プロフィール情報を取得（必ず dict）
    ※ user_id は JSON キーとして文字列化して扱う
    """
    infos = _safe_read_json(SHOP_INFO_FILE, {})
    if not isinstance(infos, dict):
        return {}
    rec = infos.get(str(user_id)) or {}
    return rec if isinstance(rec, dict) else {}

def save_shop_info(user_id, info):
    """
    店舗用プロフィール情報を保存（アトミック・壊れた JSON 自己修復）
    """
    infos = _safe_read_json(SHOP_INFO_FILE, {})
    if not isinstance(infos, dict):
        infos = {}
    infos[str(user_id)] = dict(info or {})
    _atomic_json_dump(SHOP_INFO_FILE, infos)

# =========================
#  シノニム ヘルパー & 自動生成
# =========================
def merge_synonyms(base: dict, incoming: dict) -> dict:
    """
    synonyms.json のマージ:
      - 既存のキー（タグ）は残しつつ、新規は追加
      - 値（シノニム配列）はユニーク化・空要素除去
      - 大文字小文字や前後空白も正規化
    """
    base = dict(base or {})
    for tag, syns in (incoming or {}).items():
        if not tag:
            continue
        tag_norm = str(tag).strip()
        cur = set([s.strip() for s in base.get(tag_norm, []) if s and str(s).strip()])
        add = set([s.strip() for s in (syns or []) if s and str(s).strip()])
        base[tag_norm] = sorted(cur.union(add))
    return base

def ai_propose_synonyms(tags, context_text="", model=None) -> dict:
    """
    OpenAIを使ってタグごとのシノニム案をJSONで返す。
    出力の前後に説明が混ざっても落ちないよう、_extract_json_object() で回収。
    """
    if not OPENAI_API_KEY or not tags:
        return {}
    model = model or OPENAI_MODEL_HIGH
    prompt = (
        "次の観光・生活関連の『タグ』ごとに、日本語の同義語・言い換え・表記ゆれを1〜6語で返してください。"
        "出力は必ず JSON オブジェクトのみ。キー=元タグ、値=文字列配列。"
        "元タグと完全一致の語、スラングは含めないでください。\n"
        f"【文脈参考】\n{context_text[:1000]}\n"
        f"【タグ一覧】\n{', '.join(sorted(set(tags)))}"
    )
    try:
        content = openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        ) or ""
        data = _extract_json_object(content)
        if not isinstance(data, dict):
            app.logger.warning("[ai_propose_synonyms] JSON抽出に失敗。content(先頭200): %r", content[:200])
            return {}

        # 値は list[str] に正規化
        cleaned = {}
        for k, v in data.items():
            if not k:
                continue
            if isinstance(v, str):
                v = _split_lines_commas(v)
            elif isinstance(v, (list, tuple)):
                v = [str(x).strip() for x in v if str(x).strip()]
            else:
                v = []
            cleaned[str(k).strip()] = v
        return cleaned
    except Exception as e:
        app.logger.warning(f"[ai_propose_synonyms] fail: {e}")
        return {}

def auto_update_synonyms_from_entries(entries_like):
    """
    エントリ群からタグを集めてAIでシノニムを提案→synonyms.json をマージ保存。
    """
    try:
        if not entries_like:
            return
        # まとめて提案（API呼び出し1回）
        all_tags = set()
        contexts = []
        for e in entries_like:
            all_tags.update(e.get("tags", []) or [])
            # タイトル/説明/住所/備考を少し文脈として
            contexts.append(" / ".join([
                e.get("title",""), e.get("desc",""),
                e.get("address",""), " ".join((e.get("extras") or {}).values())
            ]))
        if not all_tags:
            return
        proposal = ai_propose_synonyms(sorted(all_tags), "\n".join(contexts))
        if not proposal:
            return
        syn0 = load_synonyms()
        syn1 = merge_synonyms(syn0, proposal)
        save_synonyms(syn1)
        app.logger.info(f"[auto_update_synonyms] updated for tags: {sorted(all_tags)}")
    except Exception:
        app.logger.exception("[auto_update_synonyms] error")


# =========================
#  タグ・サジェスト
# =========================
def suggest_tags_and_title(question, answer, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    prompt = (
        f"以下は観光案内AIへの質問とそのAI回答例です。\n"
        f"【質問】\n{question}\n"
        f"【AI回答】\n{answer}\n"
        f"---\n"
        f"この内容にふさわしい「登録用タイトル」と「カンマ区切りの日本語タグ案（5～10個）」をそれぞれ1行ずつ、下記の形式で出力してください。\n"
        f"タイトル: ～\n"
        f"タグ: tag1, tag2, tag3, tag4, tag5\n"
        f"---\n"
        f"※タグには質問の意図や主要キーワード、関連ジャンルも含めてください。"
    )
    try:
        content = openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        title, tags = "", ""
        for line in content.splitlines():
            if line.startswith("タイトル:"):
                title = line.replace("タイトル:", "").strip()
            elif line.startswith("タグ:"):
                tags = line.replace("タグ:", "").strip()
        return title, tags
    except Exception as e:
        print("[AIサジェストエラー]", e)
        return "", ""

def ai_suggest_faq(question, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    prompt = (
        f"以下の質問に対し、観光案内AIとして分かりやすいFAQ回答文（最大400文字）を作成してください。\n"
        f"質問: {question}\n"
        f"---\n"
        f"回答:"
    )
    try:
        return openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        ).strip()
    except Exception as e:
        print("[FAQサジェストエラー]", e)
        return ""

def generate_unhit_report(n=7):
    """直近n日分の未ヒット・誤答ログ集計"""
    import datetime as dt

    logs = []
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                logs.append(d)
    except Exception as e:
        print("ログ集計エラー:", e)
        return []

    today = dt.datetime.now()
    recent = []
    for log in logs[::-1]:
        t = log.get("timestamp", "")
        try:
            dtv = dt.datetime.fromisoformat(t[:19])
            if (today - dtv).days <= n and not log.get("hit_db", False):
                recent.append(log)
        except:
            continue

    questions = [l["question"] for l in recent]
    counter = Counter(questions)
    return counter.most_common()


# --- 簡易正規化（NFKC + 連続空白を1つに + lower）---
def _n(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", (s or "").strip())).lower()

def find_tags_by_synonym(question, synonyms):
    """
    質問文とタグ/類義語を NFKC + 空白圧縮 + 小文字化 で正規化して部分一致判定。
    例: if _n(syn) in _n(question): ...
    """
    qn = _n(question)
    tags = set()

    for tag, synlist in (synonyms or {}).items():
        # タグ名自体のマッチ
        if _n(tag) and _n(tag) in qn:
            tags.add(tag)
            continue
        # 類義語のマッチ
        for syn in (synlist or []):
            sn = _n(syn)
            if sn and sn in qn:
                tags.add(tag)
                break

    return list(tags)


# =========================
#  未ヒット時の絞り込み候補
# =========================
def get_top_tags(k=8):
    entries = load_entries()
    counter = Counter()
    for e in entries:
        for t in e.get("tags", []) or []:
            if t:
                counter[t] += 1
    if not counter:
        return ["教会", "五島うどん", "海水浴", "釣り", "温泉", "レンタカー", "カフェ", "体験"]
    return [t for t, _ in counter.most_common(k)]

# === 追質問（意図深掘り）ビルダー ===
def build_probe_questions(question: str):
    """
    あいまいな問い合わせのときに、追加で確認したいポイントを返す。
    返り値は「箇条書きに載せる1行文」リスト。
    """
    q = (question or "").strip()
    items = []

    # ドライブ系（モデルコース含む）
    if any(k in q for k in ["ドライブ", "モデルコース", "コース", "ルート"]):
        items.append("どんなドライブが良いですか？（海岸線／教会巡り／展望台／灯台／夕日）")
        items.append("出発エリアはどちらですか？（五島市／新上五島町／小値賀町／宇久町）")
        items.append("所要時間は？（2〜3時間／半日／1日）")
        items.append("お車はありますか？（あり／なし）")

    # 教会系
    if "教会" in q:
        items.append("内部見学の可否は重視しますか？（はい／いいえ）")
        items.append("世界遺産関連を優先しますか？（はい／いいえ）")

    # ビーチ／海水浴
    if any(k in q for k in ["ビーチ", "海水浴"]):
        items.append("遊泳重視ですか？景観重視ですか？")
        items.append("シャワー・更衣室の有無は必要ですか？（必要／不要）")

    # 子連れ
    if any(k in q for k in ["子連れ", "家族", "ファミリー"]):
        items.append("お子さまの年齢層は？（未就学／小学生／中高生）")
        items.append("移動時間はどれくらいが良いですか？（短め／こだわらない）")

    # 雨
    if "雨" in q:
        items.append("屋内中心で探しますか？（はい／いいえ）")

    return items


def build_refine_suggestions(question):
    areas = ["五島市", "新上五島町", "小値賀町", "宇久町"]
    top_tags = get_top_tags()

    def rank_tags(tags, q):
        q = q or ""
        prefs = {
            "海": ["海水浴", "ビーチ", "釣り"],
            "ビーチ": ["海水浴", "ビーチ"],
            "釣り": ["釣り", "船釣り"],
            "教会": ["教会"],
            "うどん": ["五島うどん"],
            "温泉": ["温泉"],
            "雨": ["温泉", "カフェ", "資料館", "美術館", "屋内", "体験"],
            "子連れ": ["体験", "公園", "動物"],
        }
        score = {t: 0 for t in tags}
        for key, pref_list in prefs.items():
            if key in q:
                for w, t in enumerate(pref_list):
                    if t in score:
                        score[t] += (100 - w)
        ranked = sorted(tags, key=lambda t: (-score.get(t, 0), tags.index(t)))
        return ranked
    

    def has_events_this_week():
        try:
            notices = load_notices()
        except Exception:
            return False
        today = datetime.date.today()
        start_week = today - datetime.timedelta(days=today.weekday())
        end_week = start_week + datetime.timedelta(days=6)
        for n in notices or []:
            if n.get("category") != "イベント":
                continue
            s = n.get("start_date") or n.get("date")
            e = n.get("end_date") or s
            try:
                sd = datetime.date.fromisoformat(s) if s else None
                ed = datetime.date.fromisoformat(e) if e else None
            except Exception:
                continue
            if not sd and not ed:
                continue
            if sd is None:
                sd = ed
            if ed is None:
                ed = sd
            if not (ed < start_week or sd > end_week):
                return True
        return False

    top_tags = rank_tags(top_tags, question)
    filters = []
    if "雨" in (question or ""):
        filters.append("雨の日OK（屋内スポット）")
    if has_events_this_week():
        filters.append("今週のイベント")
    if any(w in (question or "") for w in ["子連れ", "家族", "ファミリー"]):
        filters.append("子連れOK")

    area_line = " / ".join(areas)
    tag_line = ", ".join(top_tags)
    examples = []
    if areas and top_tags:
        examples.append(f"『{areas[0]}の{top_tags[0]}』")
    if len(areas) > 1 and len(top_tags) > 1:
        examples.append(f"『{areas[1]}の{top_tags[1]}』")
    if len(top_tags) > 2:
        examples.append(f"『{top_tags[2]} の人気スポット』")

    lines = [
        "🔍 絞り込み候補",
        f"- エリア: {area_line}",
        f"- タグ例: {tag_line}",
    ]
    if filters:
        lines.append(f"- フィルタ: {', '.join(filters)}")
    if examples:
        lines.append("- 例: " + " / ".join(examples))
    # ★ここから追記：意図深掘りの追質問を合体
    probe_lines = build_probe_questions(question)
    if probe_lines:
        lines.append("")  # 空行で区切り
        lines.append("❓ もう少し教えてください")
        for p in probe_lines:
            lines.append(f"- {p}")

    msg = "\n".join(lines)
    # 返り値の meta に probe も載せる（後方互換）
    return msg, {"areas": areas, "tags": top_tags, "filters": filters, "probe": probe_lines}


# ---- 画像メタ情報（共通化） ----
def _image_meta(img_name: str | None):
    """
    画像ファイルの存在/サイズ/解像度とURLを返す（どこからでも利用可）。
    ※ 後方互換の別名キー(size_kb/size_bytes)と、人間向けラベル(label)を追加。
    """
    if not img_name:
        return None
    try:
        # 画像URL（存在しなくてもURLは生成を試みる）
        try:
            url = url_for("serve_image", filename=img_name, _external=True)
        except Exception:
            url = ""

        path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.isfile(path):
            # 存在しない場合も、UIが扱いやすい形で返す
            return {
                "name": img_name, "exists": False, "url": url,
                "bytes": None, "kb": None, "width": None, "height": None,
                # 互換・表示向け
                "size_bytes": None, "size_kb": None, "kb_str": None, "label": None
            }

        size_b = os.path.getsize(path)
        kb = max(1, (size_b + 1023) // 1024)  # 切り上げKB（最低1KB）
        w = h = None
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            pass

        # 人間向けの表示ラベル（例: "243 KB / 1920×1080px"）
        kb_str = f"{kb:,}"
        label = f"{kb_str} KB"
        if w and h:
            label += f" / {w}×{h}px"

        return {
            "name": img_name, "exists": True, "url": url,
            "bytes": size_b, "kb": kb, "width": w, "height": h,
            # ★後方互換の別名
            "size_bytes": size_b, "size_kb": kb,
            # ★表示用
            "kb_str": kb_str, "label": label,
        }
    except Exception:
        # 何かあっても呼び出し側を壊さない
        return {
            "name": img_name or "", "exists": False, "url": "",
            "bytes": None, "kb": None, "width": None, "height": None,
            "size_bytes": None, "size_kb": None, "kb_str": None, "label": None
        }

# =========================
#  管理画面: 観光データ登録・編集
# =========================
@app.route("/admin/entry", methods=["GET", "POST"])
@login_required
def admin_entry():
    import os  # ローカルimportで安全に
    import re as _re
    from werkzeug.exceptions import RequestEntityTooLarge  # ★ 追加：except用に確実に定義

    if session.get("role") == "shop":
        return redirect(url_for("shop_entry"))

    # ---- 座標ユーティリティ（全角/URL/DMSなど何でも受ける）----
    def _zen2han(s: str) -> str:
        if not s:
            return s
        table = str.maketrans({
            "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
            "，":",","、":",","．":".","。":".","＋":"+","－":"-","−":"-",
        })
        return s.translate(table)

    def _parse_dms_block(s: str):
        """DMS 1ブロックを小数に。例: 35°41'6.6\"N / 北緯35度41分6.6秒"""
        s = (_zen2han(s or "")).strip()
        hemi = None
        if _re.search(r'[N北]', s, _re.I): hemi = 'N'
        if _re.search(r'[S南]', s, _re.I): hemi = 'S'
        if _re.search(r'[E東]', s, _re.I): hemi = 'E'
        if _re.search(r'[W西]', s, _re.I): hemi = 'W'
        # ★ ここを「ダブルクォートの raw 文字列」に修正
        m = _re.search(
            r"(\d+(?:\.\d+)?)\s*[°度]\s*(\d+(?:\.\d+)?)?\s*['’′分]?\s*(\d+(?:\.\d+)?)?\s*[\"”″秒]?",
            s
        )
        if not m:
            return None, None
        deg = float(m.group(1))
        minutes = float(m.group(2) or 0.0)
        seconds = float(m.group(3) or 0.0)
        val = deg + minutes/60.0 + seconds/3600.0
        if hemi in ('S','W'):
            val = -val
        axis = None
        if hemi in ('N','S'):
            axis = 'lat'
        elif hemi in ('E','W'):
            axis = 'lng'
        return val, axis

    def _parse_latlng_any(text: str | None):
        """URL/小数『lat,lng』/DMS/日本語表記 → (lat,lng) or (None,None)"""
        if not text:
            return (None, None)
        s = _zen2han(text).strip()
        s = _re.sub(r'\s+', ' ', s)

        # 1) Google Maps URL: /@lat,lng または ?q= / ?query= / ?ll=
        m = _re.search(r'/@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)', s)
        if not m:
            m = _re.search(r'(?:[?&](?:q|query|ll)=)\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)', s)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except Exception:
                pass

        # 2) 純粋な「lat,lng」（カンマ/空白区切り）
        m = _re.search(r'(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)', s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            # 万一順序が逆っぽい場合の救済（>90 は経度とみなす）
            if abs(a) > 90 and abs(b) <= 90:
                a, b = b, a
            return a, b

        # 3) DMS ブロック×2（★ ここもダブルクォートの raw 文字列）
        dms_blocks = _re.findall(
            r"(\d+(?:\.\d+)?\s*[°度]\s*\d*(?:\.\d+)?\s*['’′分]?\s*\d*(?:\.\d+)?\s*[\"”″秒]?\s*[NSEW北南東西]?)",
            s, flags=_re.I
        )
        if len(dms_blocks) >= 2:
            v1, a1 = _parse_dms_block(dms_blocks[0])
            v2, a2 = _parse_dms_block(dms_blocks[1])
            if v1 is not None and v2 is not None:
                if not a1 and not a2:
                    cand = sorted([v1, v2], key=lambda x: abs(x))
                    return cand[0], cand[1]
                lat = v1 if (a1 == 'lat' or (a1 is None and abs(v1) <= 90)) else v2
                lng = v2 if lat == v1 else v1
                return lat, lng

        # 4) 「緯度: xx / 経度: yy」
        mlat = _re.search(r'緯度[:：]\s*(-?\d+(?:\.\d+)?)', s)
        mlng = _re.search(r'経度[:：]\s*(-?\d+(?:\.\d+)?)', s)
        if mlat and mlng:
            return float(mlat.group(1)), float(mlng.group(1))

        return (None, None)

    def _normalize_latlng(raw_coords: str, lat_str: str, lng_str: str, prev_entry: dict | None):
        """3入力（raw/lat/lng）＋前回値から最終 (lat,lng) を決定。片側だけ空なら前回値を引継ぎ。"""
        lat = lng = None
        # 直接欄（全角/カンマ小数対応）
        try:
            if lat_str:
                lat = float(_zen2han(lat_str).replace(',', '.'))
        except ValueError:
            lat = None
        try:
            if lng_str:
                lng = float(_zen2han(lng_str).replace(',', '.'))
        except ValueError:
            lng = None

        # raw（なんでもコピペ欄）から抽出
        if (lat is None or lng is None) and raw_coords:
            a, b = _parse_latlng_any(raw_coords)
            if lat is None: lat = a
            if lng is None: lng = b

        # 片側未入力は前回値を引継ぎ
        if prev_entry:
            if lat is None and "lat" in prev_entry: lat = prev_entry.get("lat")
            if lng is None and "lng" in prev_entry: lng = prev_entry.get("lng")

        # 範囲チェック & 丸め
        if lat is not None and not (-90 <= float(lat) <= 90): lat = None
        if lng is not None and not (-180 <= float(lng) <= 180): lng = None
        if lat is not None and lng is not None:
            lat = round(float(lat), 6)
            lng = round(float(lng), 6)
        return lat, lng

    # ---- ここから従来どおり ----
    entries = [_norm_entry(x) for x in load_entries()]  # 表示前にも正規化

    edit_id = request.values.get("edit")
    entry_edit = None
    if edit_id not in (None, "", "None"):
        try:
            idx = int(edit_id)
            if 0 <= idx < len(entries):
                entry_edit = entries[idx]
        except Exception:
            entry_edit = None

    if request.method == "POST":
        # 共通
        category = (request.form.get("category") or "").strip() or "観光"
        title    = (request.form.get("title") or "").strip()
        desc     = (request.form.get("desc") or "").strip()
        address  = (request.form.get("address") or "").strip()
        map_url  = (request.form.get("map") or "").strip()

        # 任意のコピペ欄（テンプレに無くてもOK）
        coords_raw = (request.form.get("coords") or "").strip()

        # リスト系（改行/カンマ両対応）
        tags   = _split_lines_commas(request.form.get("tags", ""))
        areas  = request.form.getlist("areas") or _split_lines_commas(request.form.get("areas",""))
        links  = _split_lines_commas(request.form.get("links", ""))
        pay_in = request.form.getlist("payment") or _split_lines_commas(request.form.get("payment",""))

        # 店舗系
        tel         = (request.form.get("tel") or "").strip()
        holiday     = (request.form.get("holiday") or "").strip()
        open_hours  = (request.form.get("open_hours") or "").strip()
        parking     = (request.form.get("parking") or "").strip()
        parking_num = (request.form.get("parking_num") or "").strip()
        remark      = (request.form.get("remark") or "").strip()

        # 出典（任意入力）
        source     = (request.form.get("source") or "").strip()
        source_url = (request.form.get("source_url") or "").strip()

        # 追加情報
        extras_keys = request.form.getlist("extras_key[]")
        extras_vals = request.form.getlist("extras_val[]")
        extras = {}
        for k, v in zip(extras_keys, extras_vals):
            k = (k or "").strip()
            v = (v or "").strip()
            if k:
                extras[k] = v

        # 編集時の既存エントリ/画像名
        edit_hidden = request.form.get("edit_id")
        prev_img = None
        prev_entry = None
        idx_edit = None
        if edit_hidden not in (None, "", "None"):
            try:
                idx_edit = int(edit_hidden)
                if 0 <= idx_edit < len(entries):
                    prev_entry = entries[idx_edit]
                    prev_img = (prev_entry.get("image_file") or prev_entry.get("image") or "") or None
            except Exception:
                prev_entry = None
                prev_img = None
                idx_edit = None

        # ★ 透かし：ラジオ優先、未送信なら旧UIチェックボックスをフォールバック
        wm_raw = (request.form.get("wm_external_choice") or
                  request.form.get("wm_external") or
                  "").strip().lower()
        if not wm_raw and ('wm_on' in request.form):
            wm_raw = "city"   # 旧UIでONのみのときの既定
        wm_choice = _normalize_wm_choice(
            wm_raw,
            wm_on_default=bool(prev_entry.get("wm_on", True)) if prev_entry else True
        )

        if not areas:
            flash("エリアは1つ以上選択してください")
            # ★ 失敗時も編集中行を維持
            if idx_edit is not None:
                return redirect(url_for("admin_entry", edit=idx_edit))
            return redirect(url_for("admin_entry"))

        # 新規エントリの骨格
        new_entry = {
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas,
            "links": links,
            "payment": pay_in,
            "tel": tel,
            "holiday": holiday,
            "open_hours": open_hours,
            "parking": parking,
            "parking_num": parking_num,
            "remark": remark,
            "extras": extras,
            "source": source,           # ← 出典フィールド（任意）
            "source_url": source_url,   # ← 出典URL（任意）
            "wm_external_choice": wm_choice,     # 正規化済み: none / fullygoto / gotocity
            "wm_on": (wm_choice != "none"),      # 互換ブール（テンプレ側が参照）
        }
        new_entry = _norm_entry(new_entry)

        # === 画像アップロード系（完成済みアップロードを最優先） =======================
        # 1) ローカルで“透かし済み”の完成画像：最適化のみで登録（透かしは重ねない）
        result_final = None
        upload_final = request.files.get("image_file_final")
        if upload_final and upload_final.filename:
            try:
                # 既存画像があれば置換できるよう previous を渡す
                result_final = _save_jpeg_1080_350kb(upload_final, previous=prev_img, delete=False)
            except RequestEntityTooLarge:
                flash(f"画像のピクセル数が大きすぎます（上限 {MAX_IMAGE_PIXELS:,} ピクセル）")
                if idx_edit is not None:
                    return redirect(url_for("admin_entry", edit=idx_edit))
                return redirect(url_for("admin_entry"))
            except Exception:
                result_final = None
                app.logger.exception("image handler (final) failed")

        if result_final:
            # 完成済み（透かし込み）をそのまま使う：透かしは「none」に固定
            new_entry["image_file"] = result_final
            new_entry["image"] = result_final
            new_entry["wm_external_choice"] = "none"   # ここで固定
            new_entry["wm_on"] = False                 # 互換ブールも下げる
            # ※ 完成品なので __none/__goto/__fullygoto の派生は作らない（ダブル透かし防止）

        else:
            # 2) 従来のアップロード/削除（フォルダ選択や前画像維持の流れ）
            upload = request.files.get("image_file")
            delete_flag = (request.form.get("image_delete") == "1")
            try:
                result = _save_jpeg_1080_350kb(upload, previous=prev_img, delete=delete_flag)
            except RequestEntityTooLarge:
                flash(f"画像のピクセル数が大きすぎます（上限 {MAX_IMAGE_PIXELS:,} ピクセル）")
                if idx_edit is not None:
                    return redirect(url_for("admin_entry", edit=idx_edit))
                return redirect(url_for("admin_entry"))
            except Exception:
                result = None
                app.logger.exception("image handler failed")

            if result is None:
                # 変更なし → 前画像を維持（削除指示がない限り）
                if prev_img and not delete_flag:
                    new_entry["image_file"] = prev_img
                    new_entry["image"] = prev_img
                    # 既存画像に派生ファイルが無ければ作る（従来どおり）
                    try:
                        _ensure_wm_variants(prev_img)
                    except Exception:
                        app.logger.exception("[wm] backfill variants failed for %s", prev_img)

            elif result == "":
                # 明示削除
                new_entry.pop("image_file", None)
                new_entry.pop("image", None)

            else:
                # 置換/新規保存（従来どおり派生3種を事前生成）
                new_entry["image_file"] = result
                new_entry["image"] = result
                try:
                    _ensure_wm_variants(result)
                except Exception:
                    app.logger.exception("[wm] variants generation failed for %s", result)
        # === /画像アップロード系 =======================================================

        # === 緯度・経度（raw/lat/lng/map を総合して決定、片側空は前回値を継承） ===
        lat, lng = _normalize_latlng(
            coords_raw,
            (request.form.get("lat") or "").strip(),
            (request.form.get("lng") or "").strip(),
            prev_entry
        )
        if (lat is None or lng is None) and map_url:
            a, b = _parse_latlng_any(map_url)
            if lat is None: lat = a
            if lng is None: lng = b

        if lat is not None: new_entry["lat"] = lat
        if lng is not None: new_entry["lng"] = lng

        # （※ 旧UI互換 'wm_on' の**再代入はしない**。上の wm_choice を真にする）
        # new_entry["wm_on"] = ('wm_on' in request.form)  # ← 上書きしない

        # === 保存 ===
        if (idx_edit is not None) and (0 <= idx_edit < len(entries)):
            try:
                entries[idx_edit] = new_entry
                flash("編集しました")
                idx_after = idx_edit
            except Exception:
                entries.append(new_entry)
                flash("編集ID不正のため新規で追加しました")
                idx_after = len(entries) - 1
        else:
            entries.append(new_entry)
            flash("登録しました")
            idx_after = len(entries) - 1

        # 保存（重複統合も内部で実行）
        save_entries(entries)

        # （任意）タグ類義語オート更新の簡易キュー
        try:
            q = _load_syn_queue()
            q.setdefault("tags", {})
            for t in (new_entry.get("tags") or []):
                if t:
                    q["tags"][t] = True
            _save_syn_queue(q)
        except Exception:
            pass

        # ★ 保存後も編集中の行を開いた状態に戻す
        return redirect(url_for("admin_entry", edit=idx_after))

    # ---- 一覧用: “サムネ/容量/サイズ” を各エントリに付加（テンプレ未使用なら無害） ----
    entries_view = []
    for e in entries:
        e2 = dict(e)  # テンプレ用コピー（元データは変更しない）
        img_name = e.get("image_file") or e.get("image")
        e2["__image"] = _image_meta(img_name) if img_name else None
        entries_view.append(e2)

    # 最終表示直前に追加
    paused, by = _pause_state()

    return render_template(
        "admin_entry.html",
        entries=entries_view,
        entry_edit=entry_edit,
        edit_id=edit_id if edit_id not in (None, "", "None") else None,
        role=session.get("role", ""),
        global_paused=paused,   # ← _is_global_paused() の代わりに _pause_state の結果
        paused_by=by,           # ← 追加：誰が停止中か（'admin' / 'user' / None）
    )


# ==== 管理画面用：画像メタAPI（要ログイン） ====
@app.get("/admin/_img_meta")
@login_required
def admin_img_meta():
    fn = (request.args.get("filename") or "").strip()
    if not fn:
        return jsonify(ok=False, error="missing filename"), 400

    meta = _image_meta(fn)
    # ok/exists のフラグも一緒に返す
    return jsonify(ok=True, **meta)
# ==== / 管理画面用API ====

@app.route("/admin/_wm_diag", methods=["GET"])
@login_required
def admin_wm_diag():
    """
    /admin/_wm_diag?idx=<entriesのindex>
    - 対象エントリの画像について:
      * プレビュー(常時透かし)URL
      * 原寸(選択透かし/署名付き)URL
      * 明示モード別( none / fullygoto / gotocity )の原寸URL
    を一覧表示し、同時にアプリ内部リクエストで実レスポンスヘッダも確認する。
    """
    if session.get("role") != "admin":
        abort(403)

    try:
        idx = int(request.args.get("idx", ""))
    except Exception:
        return "idx=? を指定してください（例: /admin/_wm_diag?idx=0）", 400

    # 正規化済みで読む
    entries = [_norm_entry(x) for x in load_entries()]
    if idx < 0 or idx >= len(entries):
        return f"idx={idx} は範囲外です（0〜{len(entries)-1}）", 404

    entry = entries[idx]
    img_name = (entry.get("image_file") or entry.get("image") or "").strip()
    if not img_name:
        return f"idx={idx} のエントリには画像がありません", 404

    # 実際に適用されるモード（旧/新保存形式どちらでもOK）
    effective_mode = _wm_choice_for_entry(entry)

    # URL（外部表示用=絶対URL）と、内部プローブ用の相対パスを両方作る
    def _abs_rel(endpoint, **kwargs):
        # 表示用(絶対)と内部GET用(相対)の両方を返す
        abs_url = safe_url_for(endpoint, _external=True, **kwargs)
        rel_url = safe_url_for(endpoint, _external=False, **kwargs)
        return abs_url, rel_url

    # 代表URLセット
    preview_abs, preview_rel = _abs_rel("serve_image", filename=img_name, wm=1)
    original_abs, original_rel = _abs_rel("serve_image", filename=img_name, _sign=True, wm=effective_mode)

    # 明示モード（テスト用）
    none_abs, none_rel       = _abs_rel("serve_image", filename=img_name, _sign=True, wm="none")
    fully_abs, fully_rel     = _abs_rel("serve_image", filename=img_name, _sign=True, wm="fullygoto")
    city_abs, city_rel       = _abs_rel("serve_image", filename=img_name, _sign=True, wm="gotocity")

    # ---- アプリ内部で実際に叩いてヘッダを採取（外部HTTPに出ない） ----
    def _probe(rel_path):
        # 注意: ここは本番アプリの中から test_client を使って内部GETします
        # 画像本体は重いので HEAD にしたいところですが、署名や動的変換の都合で GET を推奨
        with app.test_client() as c:
            rv = c.get(rel_path)
        return {
            "status": rv.status_code,
            "content_type": rv.headers.get("Content-Type"),
            "content_length": rv.headers.get("Content-Length"),
            # ここは serve_image 側で付けていれば拾える（後述の追加②を参照）
            "X-WM-Requested": rv.headers.get("X-WM-Requested"),
            "X-WM-Applied": rv.headers.get("X-WM-Applied"),
            "X-Img-Signed": rv.headers.get("X-Img-Signed"),
            "X-Img-Exp": rv.headers.get("X-Img-Exp"),
        }

    rows = [
        ("Preview (常時透かし)", preview_abs, _probe(preview_rel)),
        (f"Original (選択透かし: {effective_mode})", original_abs, _probe(original_rel)),
        ("Original 明示 none",      none_abs,  _probe(none_rel)),
        ("Original 明示 fullygoto", fully_abs, _probe(fully_rel)),
        ("Original 明示 gotocity",  city_abs,  _probe(city_rel)),
    ]

    # シンプルに表で出力
    html = """
    <!doctype html><meta charset="utf-8">
    <title>WM Diag</title>
    <style>
      body{font-family:system-ui, -apple-system, Segoe UI, Roboto, "Hiragino Kaku Gothic ProN", Meiryo, sans-serif;padding:16px}
      table{border-collapse:collapse;width:100%}
      th,td{border:1px solid #ddd;padding:6px 8px;font-size:14px;vertical-align:top}
      th{background:#f5f7fb;text-align:left}
      code{background:#f1f4f7;padding:2px 4px;border-radius:4px}
      .mono{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace}
    </style>
    <h1>Watermark Diagnostics</h1>
    <p>idx=<b>{{ idx }}</b> / タイトル: <b>{{ title }}</b> / 適用モード: <code>{{ effective_mode }}</code></p>
    <table>
      <tr><th>種別</th><th>URL</th><th>レスポンス(抜粋)</th></tr>
      {% for label, url, meta in rows %}
      <tr>
        <td>{{ label }}</td>
        <td class="mono"><a href="{{ url }}" target="_blank" rel="noopener">{{ url }}</a></td>
        <td>
          status: {{ meta.status }}<br>
          Content-Type: {{ meta.content_type }} / Content-Length: {{ meta.content_length }}<br>
          X-WM-Requested: {{ meta["X-WM-Requested"] }} / X-WM-Applied: {{ meta["X-WM-Applied"] }}<br>
          X-Img-Signed: {{ meta["X-Img-Signed"] }} / X-Img-Exp: {{ meta["X-Img-Exp"] }}
        </td>
      </tr>
      {% endfor %}
    </table>
    <p style="margin-top:10px;"><a href="{{ url_for('admin_entry', edit=idx) }}">← このエントリを編集に戻る</a></p>
    """
    return render_template_string(
        html,
        idx=idx,
        title=(entry.get("title") or ""),
        effective_mode=effective_mode,
        rows=rows,
    )


@app.route("/admin/entry/delete/", defaults={"idx": None}, methods=["POST"])
@app.route("/admin/entry/delete/<int:idx>", methods=["POST"])
@login_required
def delete_entry(idx):
    if session.get("role") != "admin":
        abort(403)

    if idx is None:
        form_idx = request.form.get("idx")
        if form_idx is not None and str(form_idx).isdigit():
            idx = int(form_idx)

    if idx is None:
        flash("削除対象が指定されていません")
        return redirect(url_for("admin_entry"))

    entries = load_entries()
    if 0 <= idx < len(entries):
        entries.pop(idx)
        save_entries(entries)
        flash("削除しました")
    else:
        flash("指定された項目が見つかりません")
    return redirect(url_for("admin_entry"))



def _wm_safe_basename(name: str) -> str:
    import os, re, unicodedata
    if not name:
        return "image"
    n = os.path.basename(name)
    n = unicodedata.normalize("NFKC", n)
    n = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龠々ー_\-\.]+", "_", n).strip("._")
    if not n:
        n = "image"
    root, _ = os.path.splitext(n)
    return root[:80] or "image"

def _wm_draw_text(im: Image.Image, text: str) -> Image.Image:
    """右下に半透明帯＋白文字（黒縁取り）で簡易透かし"""
    if not text:
        return im.copy()
    base = im.convert("RGBA")
    W, H = base.size
    draw = ImageDraw.Draw(base)

    # 画像幅に応じてフォントサイズを決める（幅の5%目安）
    size = max(18, int(W * 0.05))
    try:
        # 既存のフォントローダがあれば使ってもOK。ここはデフォルトで十分。
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 文字サイズ取得（stroke対応）
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(1, size // 10))
        tw, th = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        tw, th = draw.textsize(text, font=font)

    pad = max(6, size // 3)
    x = max(0, W - tw - pad * 2)
    y = max(0, H - th - pad * 2)

    # 半透明の黒帯
    bg = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 110))
    base.alpha_composite(bg, (x, y))

    # 白文字＋黒縁取り
    try:
        draw.text(
            (x + pad, y + pad),
            text,
            fill=(255, 255, 255, 230),
            font=font,
            stroke_width=max(1, size // 10),
            stroke_fill=(0, 0, 0, 230),
        )
    except TypeError:
        draw.text((x + pad, y + pad), text, fill=(255, 255, 255, 230), font=font)

    return base.convert("RGB")

def _wm_save_jpeg(im: Image.Image, out_path: str, quality: int = 90):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, format="JPEG", quality=quality, optimize=True)

# === 管理プレビュー用 admin_media_img（重複安全・ダブル透かし防止） ===
def _admin_media_img_impl(filename: str):
    # 派生ファイルはダブル透かし防止で wm=none、元画像はプレビュー用に wm=1
    low = (filename or "").lower()
    if any(s in low for s in ("__goto", "__gotocity", "__fullygoto", "__none")):
        wm = "none"
    else:
        wm = "1"
    # 署名付きURLに 302 リダイレクト（実体配信は serve_image が担当）
    return redirect(
        safe_url_for("serve_image", filename=filename, _sign=True, wm=wm)
    )

# 既に登録済みなら追加しない（重複エラー回避）
if "admin_media_img" not in app.view_functions:
    app.add_url_rule(
        "/admin/_media_img/<path:filename>",
        endpoint="admin_media_img",
        view_func=login_required(_admin_media_img_impl),
        methods=["GET"]
    )



# === 新規アップロード→透かし生成→保存（統一版） ============================
@app.post("/admin/_watermark_generate")
@login_required
def admin_watermark_generate():
    """
    フロントの「新規だけ即時生成」ボタン専用。
    1) アップロード画像を回転補正・リサイズ・形式正規化して保存（md5名）
    2) gotocity / fullygoto の派生を作って保存
    3) 一覧へ即時差し込み用の saved を返す
       ※ 返却URLは必ず wm=none を付け、ダブル透かしを防ぐ
    返却: {"ok": true, "saved":[{"name":..., "kind":"src|gotocity|fullygoto","url":"..."}]}
    """
    import io as _io
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(ok=False, error="file required"), 400

    # 入力拡張子チェック（任意：.jpeg を .jpg に寄せる）
    ext_in = (os.path.splitext(file.filename)[1] or "").lower()
    if ext_in == ".jpeg":
        ext_in = ".jpg"
    if ext_in not in {".jpg", ".png", ".webp"}:
        return jsonify(ok=False, error="unsupported file type"), 400

    # 1) 正規化（回転補正＋長辺リサイズ＋形式統一）
    try:
        raw = file.read()
        if not raw:
            return jsonify(ok=False, error="empty file"), 400

        im = Image.open(_io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGBA")

        MAXW = int(os.getenv("UPLOAD_MAX_W", "2560"))
        MAXH = int(os.getenv("UPLOAD_MAX_H", "2560"))
        w, h = im.size
        r = min(MAXW / w, MAXH / h, 1.0)
        if r < 1.0:
            im = im.resize((int(w * r), int(h * r)), RESAMPLE_LANCZOS)

        buf = _io.BytesIO()
        # WEBP→JPEG など、LINE互換も考慮して JPEG/PNG に正規化
        if ext_in in {".jpg", ".webp"}:
            im.convert("RGB").save(
                buf, format="JPEG",
                quality=int(os.getenv("WM_JPEG_QUALITY", "88")),
                optimize=True, progressive=True, subsampling=2
            )
            out_ext = ".jpg"
        else:
            im.save(buf, format="PNG", optimize=True)
            out_ext = ".png"

        buf.seek(0)
        normalized_bytes = buf.getvalue()
    except UnidentifiedImageError:
        return jsonify(ok=False, error="invalid image"), 400
    except Exception as e:
        app.logger.exception("upload normalize failed: %s", e)
        return jsonify(ok=False, error="normalize failed"), 500

    # 2) md5名で保存（先頭に来るよう mtime を少し進める）
    md5 = hashlib.md5(normalized_bytes).hexdigest()
    base_name = md5 + out_ext
    base_path = MEDIA_DIR / base_name
    if not base_path.exists():
        _save_bytes(base_path, normalized_bytes, bias_sec=+1)

    # 返すURLは常に wm=none（ダブル透かし防止）
    def _u(fn: str) -> str:
        try:
            return safe_url_for("serve_image", filename=fn, _external=True, _sign=True, wm="none")
        except Exception:
            return url_for("serve_image", filename=fn, _external=True) + "?wm=none"

    saved = [{"name": base_name, "kind": "src", "url": _u(base_name)}]

    # 3) 派生（@Goto City / @fullyGOTO）も“実体ファイル”として作成・保存
    for kind in ("gotocity", "fullygoto"):
        try:
            wm_bytes, out_ext2 = _render_watermark_bytes(base_path, kind)
            deriv_name = _wm_variant_name(base_name, kind, out_ext=out_ext2)
            _save_bytes(MEDIA_DIR / deriv_name, wm_bytes, bias_sec=+2)
            saved.append({"name": deriv_name, "kind": kind, "url": _u(deriv_name)})
        except Exception as e:
            app.logger.exception("watermark build failed: kind=%s base=%s", kind, base_name)

    return jsonify(ok=True, saved=saved)


@app.route("/shop/entry", methods=["GET", "POST"])
@login_required
def shop_entry():
    if session.get("role") != "shop":
        return redirect(url_for("admin_entry"))
    user_id = session["user_id"]

    if request.method == "POST":
        category = request.form.get("category", "")
        title = request.form.get("title", "")
        desc = request.form.get("desc", "")
        address = request.form.get("address", "")
        tel = request.form.get("tel", "")
        holiday = request.form.get("holiday", "")
        open_hours = request.form.get("open_hours", "")
        parking = request.form.get("parking", "")
        parking_num = request.form.get("parking_num", "")
        payment = request.form.getlist("payment")
        map_url = request.form.get("map", "")

        tags = _split_lines_commas(request.form.get("tags", ""))
        areas = request.form.getlist("areas")
        remark = request.form.get("remark", "")

        links = _split_lines_commas(request.form.get("links", ""))

        extras_keys = request.form.getlist("extras_key[]")
        extras_vals = request.form.getlist("extras_val[]")
        extras = {k.strip(): v.strip() for k, v in zip(extras_keys, extras_vals) if k.strip()}

        if not areas:
            flash("エリアは1つ以上選択してください")
            return redirect(url_for("shop_entry"))

        entry_data = {
            "user_id": user_id,
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "tel": tel,
            "holiday": holiday,
            "open_hours": open_hours,
            "parking": parking,
            "parking_num": parking_num,
            "payment": payment,
            "remark": remark,
            "tags": tags,
            "areas": areas,
            "map": map_url,
            "links": links,
            "extras": extras
        }
        entry_data = _norm_entry(entry_data)
        # 画像アップロード（任意）
        up = request.files.get("image_file")
        if up and up.filename:
            # 既存データがあれば過去の画像名を previous に渡して置換できるようにする
            entries = load_entries()
            prev_idx = next((i for i, e in enumerate(entries) if e.get("user_id") == user_id), None)
            prev_img = entries[prev_idx].get("image_file") if prev_idx is not None else None

        res = _save_jpeg_1080_350kb(up, previous=prev_img, delete=False)
        if res is None:
            flash("画像アップロードに失敗しました")
        else:
            entry_data["image_file"] = res
            # ▼【追記】3種を事前生成
            try:
                _ensure_wm_variants(res)
            except Exception:
                app.logger.exception("[wm] variants generation failed (shop) for %s", res)

        entries = load_entries()
        entry_idx = next((i for i, e in enumerate(entries) if e.get("user_id") == user_id), None)
        if entry_idx is not None:
            entries[entry_idx] = entry_data
        else:
            entries.append(entry_data)
        save_entries(entries)

        flash("店舗情報を保存しました")
        return redirect(url_for("shop_entry"))

    user_entries = [e for e in load_entries() if e.get("user_id") == user_id]
    shop_entry_data = user_entries[-1] if user_entries else None
    shop_entry_data = _norm_entry(shop_entry_data) if shop_entry_data else None
    return render_template("shop_entry.html", role="shop", shop_edit=shop_entry_data)

@app.route("/admin/entries_edit", methods=["GET", "POST"])
@login_required
def admin_entries_edit():
    if session.get("role") != "admin":
        abort(403)

    import json, os
    from datetime import datetime as _dt

    old_entries = load_entries()

    # ① 生JSONでの上書き（従来互換）
    if request.method == "POST" and request.form.get("entries_raw"):
        raw_json = request.form.get("entries_raw")
        try:
            data = json.loads(raw_json)
            if not isinstance(data, list):
                raise ValueError("ルート要素は配列(list)にしてください")
            data = [_norm_entry(e) for e in data]
            save_entries(data)
            flash("entries.jsonを上書きしました")
            return redirect(url_for("admin_entries_edit"))
        except Exception as e:
            flash("JSONエラー: " + str(e))

    # ② UIフォームからの保存（未表示フィールドは既存からマージして保持）
    if request.method == "POST" and not request.form.get("entries_raw"):
        getl = request.form.getlist

        row_ids       = getl("row_id[]")            # ★ hidden（未導入なら空配列）
        cats          = getl("category[]")
        titles        = getl("title[]")
        descs         = getl("desc[]")
        addresses     = getl("address[]")
        maps          = getl("map[]")
        tags_list     = getl("tags[]")
        areas_list    = getl("areas[]")
        tels          = getl("tel[]")
        holidays      = getl("holiday[]")
        opens         = getl("open_hours[]")
        parkings      = getl("parking[]")
        parking_nums  = getl("parking_num[]")
        payments      = getl("payment[]")
        remarks       = getl("remark[]")
        links_list    = getl("links[]")            # テンプレに無ければ空で来るのでOK

        def _safe(lst, i):  # インデックス安全取得
            return lst[i] if i < len(lst) else ""

        # いずれかの最大長で回す（テンプレ列ズレの保険）
        n = max(len(titles), len(descs), len(cats), len(addresses), len(areas_list))

        new_entries = []
        for i in range(n):
            title = (_safe(titles, i) or "").strip()
            desc  = (_safe(descs,  i) or "").strip()

            # 完全空行はスキップ（従来互換）
            if not title and not desc:
                continue

            # 既存行にマッピング：row_id が数字ならそれを優先
            base = {}
            rid = (_safe(row_ids, i) or "").strip()
            if rid.isdigit():
                j = int(rid)
                if 0 <= j < len(old_entries):
                    base = dict(old_entries[j])   # ← 未表示フィールドを保持
            elif i < len(old_entries):
                # row_id 未導入テンプレ互換：同じインデックスをベースに（完璧ではないが後方互換）
                base = dict(old_entries[i])

            # フォーム値で上書き（空文字は“クリア”として反映）
            base["category"]     = (_safe(cats, i) or "観光").strip() or "観光"
            base["title"]        = title
            base["desc"]         = desc
            base["address"]      = _safe(addresses, i)
            base["map"]          = _safe(maps, i)

            # 配列系はユーティリティで正規化
            base["tags"]         = _split_lines_commas(_safe(tags_list,  i))
            base["areas"]        = _split_lines_commas(_safe(areas_list, i))
            base["links"]        = _split_lines_commas(_safe(links_list,  i))
            base["payment"]      = _split_lines_commas(_safe(payments,   i))

            base["tel"]          = _safe(tels, i)
            base["holiday"]      = _safe(holidays, i)
            base["open_hours"]   = _safe(opens, i)
            base["parking"]      = _safe(parkings, i)
            base["parking_num"]  = _safe(parking_nums, i)
            base["remark"]       = _safe(remarks, i)

            # ★ 触らない＝保持：images/image/thumb/wm_external_choice/lat/lng/place_id/gmaps_share_url/extras 等

            e = _norm_entry(base)

            # エリア必須（従来挙動を維持）
            if not e.get("areas"):
                continue

            new_entries.append(e)

        # 保存前に自動バックアップ（復元用）
        try:
            os.makedirs("backups", exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            with open(os.path.join("backups", f"entries-backup-{ts}.json"), "w", encoding="utf-8") as f:
                json.dump(old_entries, f, ensure_ascii=False, indent=2)
        except Exception as ex:
            app.logger.warning(f"[admin_entries_edit] backup failed: {ex}")

        save_entries(new_entries)
        flash(f"{len(new_entries)} 件保存しました（未表示フィールドは保持）")
        return redirect(url_for("admin_entries_edit"))

    # GET（またはPOSTエラー後の再表示）
    entries = [_norm_entry(x) for x in old_entries]
    paused, by = _pause_state()  # ← 既存の一時停止表示に合わせて維持
    return render_template(
        "admin_entries_edit.html",
        entries=entries,
        global_paused=paused,
        paused_by=by,
    )


@app.route("/admin/entries_dedupe", methods=["GET", "POST"])
@login_required
def admin_entries_dedupe():
    if session.get("role") != "admin":
        abort(403)

    entries = load_entries()

    # 実行（保存）: POST
    if request.method == "POST":
        # フォーム/JSON どちらでも受ける
        use_ai_param = request.form.get("use_ai")
        if use_ai_param is None and request.is_json:
            try:
                use_ai_param = (request.get_json() or {}).get("use_ai")
            except Exception:
                use_ai_param = None
        use_ai = DEDUPE_USE_AI if use_ai_param is None else _boolish(use_ai_param)

        new_entries, stats, preview = dedupe_entries_by_title(entries, use_ai=use_ai, dry_run=False)
        save_entries(new_entries)
        return jsonify({"ok": True, "saved": True, "stats": stats, "preview": preview})

    # プレビュー: GET
    detail = (request.args.get("detail") == "1")
    _, stats, preview = dedupe_entries_by_title(entries, use_ai=DEDUPE_USE_AI, dry_run=True)

    if not detail:
        return jsonify({"ok": True, "stats": stats, "preview": preview})

    # detail=1 のとき、重複グループの中身を返す
    groups = {}
    for e in entries:
        k = _title_key(e.get("title", ""))
        if not k:
            continue
        groups.setdefault(k, []).append(e)

    detail_list = []
    for k, gs in groups.items():
        if len(gs) <= 1:
            continue
        item = {
            "key": k,
            "count": len(gs),
            "titles":     [g.get("title", "") for g in gs],
            "descs":      [g.get("desc", "") for g in gs if (g.get("desc", "").strip())],
            "maps":       [g.get("map", "") for g in gs if (g.get("map", "").strip())],
            "addresses":  [g.get("address", "") for g in gs if (g.get("address", "").strip())],
            "tags":       sorted({t for g in gs for t in (g.get("tags") or []) if t}),
            "areas":      sorted({a for g in gs for a in (g.get("areas") or []) if a}),
            "links":      sorted({l for g in gs for l in (g.get("links") or []) if l}),
            "payments":   sorted({p for g in gs for p in (g.get("payment") or []) if p}),
            "tel_list":       [g.get("tel", "") for g in gs if g.get("tel", "").strip()],
            "holiday_list":   [g.get("holiday", "") for g in gs if g.get("holiday", "").strip()],
            "open_hours_list":[g.get("open_hours", "") for g in gs if g.get("open_hours", "").strip()],
        }
        detail_list.append(item)

    detail_list.sort(key=lambda x: x["count"], reverse=True)
    return jsonify({"ok": True, "stats": stats, "groups": detail_list})

# =========================
#  CSV取り込み（既存に追加）
# =========================
@app.route("/admin/entries_import_csv", methods=["POST"])
@login_required
def admin_entries_import_csv():
    if session.get("role") != "admin":
        abort(403)

    file = request.files.get("csv_file")
    if not file:
        flash("CSVファイルが選択されていません")
        return redirect(url_for("admin_entries_edit"))

    import csv, io as _io, json as _json

    # --- 文字コードを自動判定して読み込む（UTF-8/UTF-8-SIG/CP932対応） ---
    raw = file.read()  # bytes
    if not raw:
        flash("アップロードされたCSVが空です")
        return redirect(url_for("admin_entries_edit"))

    enc_tried = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    decoded = None
    used_enc = None
    for enc in enc_tried:
        try:
            decoded = raw.decode(enc)
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        # 最後の手段：置換ありでCP932として読む（文字化け箇所は□になる）
        decoded = raw.decode("cp932", errors="replace")
        used_enc = "cp932(replace)"

    buf = _io.StringIO(decoded, newline="")
    reader = csv.DictReader(buf)

    new_entries = []
    for row in reader:
        category = (row.get("category") or "観光").strip()
        title    = (row.get("title") or "").strip()
        desc     = (row.get("desc") or "").strip()
        address  = (row.get("address") or "").strip()
        map_url  = (row.get("map") or "").strip()
        tags     = _split_lines_commas(row.get("tags"))
        areas    = _split_lines_commas(row.get("areas"))
        links    = _split_lines_commas(row.get("links"))

        # 必須チェック（title/desc/areas）
        if not title or not desc or not areas:
            continue

        # extras: "extras"列(JSON) + extra_*/extra:* 列
        extras = {}
        extras_raw = row.get("extras")
        if extras_raw:
            try:
                obj = _json.loads(extras_raw)
                if isinstance(obj, dict):
                    extras.update({str(k): str(v) for k, v in obj.items()})
            except Exception:
                pass
        for k, v in row.items():
            if k is None:
                continue
            ks = str(k)
            if ks.startswith("extra_") or ks.startswith("extra:"):
                name = ks.split("_", 1)[1] if ks.startswith("extra_") else ks.split(":", 1)[1]
                name = name.strip()
                if name:
                    extras[name] = str(v or "").strip()

        e = {
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas,
        }
        if links:
            e["links"] = links
        if extras:
            e["extras"] = extras

        e = _norm_entry(e)
        new_entries.append(e)

    if not new_entries:
        flash(f"CSVから有効な行が見つかりませんでした（title/desc/areas 必須）［encoding: {used_enc}］")
        return redirect(url_for("admin_entries_edit"))

    entries = load_entries()
    entries.extend(new_entries)
    save_entries(entries)

    try:
        auto_update_synonyms_from_entries(new_entries)
        flash(f"CSVから {len(new_entries)} 件を追加＋シノニム更新完了（encoding: {used_enc}）")
    except Exception:
        flash(f"CSVから {len(new_entries)} 件を追加（encoding: {used_enc}／シノニム自動更新は失敗）")

    return redirect(url_for("admin_entries_edit"))

# =========================
#  管理: JSONインポート（entries / synonyms）
# =========================
@app.route("/admin/import", methods=["GET", "POST"])
@login_required
def admin_import():
    if session.get("role") != "admin":
        abort(403)

    if request.method == "POST":
        import json as _json
        # entries.json の処理
        if "entries_json" in request.files and request.files["entries_json"].filename:
            mode = request.form.get("entries_mode", "merge")  # merge|replace
            file = request.files["entries_json"]
            try:
                data = _json.load(file.stream)
                if not isinstance(data, list):
                    raise ValueError("entries.json は配列(list)である必要があります")
                data = [_norm_entry(e) for e in data]
                if mode == "replace":
                    save_entries(data)
                    flash(f"entries.json を {len(data)} 件で上書きしました")
                    # 上書き時もシノニム自動生成（重いのでタグだけまとめて）
                    try:
                        auto_update_synonyms_from_entries(data[:80])  # 安全のため上位80件だけ文脈に
                        flash("シノニムを自動更新しました")
                    except Exception:
                        flash("シノニム自動更新に失敗しました")
                else:
                    cur = load_entries()
                    cur.extend(data)
                    save_entries(cur)
                    flash(f"entries.json に {len(data)} 件マージしました（追記）")
                    try:
                        auto_update_synonyms_from_entries(data)
                        flash("シノニムを自動更新しました")
                    except Exception:
                        flash("シノニム自動更新に失敗しました")
            except Exception as e:
                flash(f"entries.json の読み込みに失敗: {e}")

        # synonyms.json の処理
        if "synonyms_json" in request.files and request.files["synonyms_json"].filename:
            mode = request.form.get("synonyms_mode", "merge")  # merge|replace
            file = request.files["synonyms_json"]
            try:
                data = _json.load(file.stream)
                if not isinstance(data, dict):
                    raise ValueError("synonyms.json はオブジェクト(dict)である必要があります")
                if mode == "replace":
                    save_synonyms(data)
                    flash("synonyms.json を上書きしました")
                else:
                    cur = load_synonyms()
                    merged = merge_synonyms(cur, data)
                    save_synonyms(merged)
                    flash("synonyms.json をマージしました（重複は自動でユニーク化）")
            except Exception as e:
                flash(f"synonyms.json の読み込みに失敗: {e}")

        return redirect(url_for("admin_import"))

    # GET: 画面表示
    return render_template("admin_import.html")



# =========================
#  ログ・未ヒット確認
# =========================
from datetime import datetime as _dt

def _boolish_strict(v) -> bool:
    """
    hit_db の厳密判定:
      True 扱い: True, 1, "1", "true", "yes", "y", "on"
      False扱い: False, 0, "0", "false", "no", "n", "off", "", None
      想定外の文字列は False（=未ヒット扱い）に倒す
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
        return False
    return False

def _parse_ts(ts_str: str):
    """ISO風タイムスタンプを datetime に。失敗時は最小値（ソート用）。"""
    if not ts_str:
        return _dt.min
    s = (ts_str or "").strip()
    try:
        # "Z" が来る可能性も一応吸収
        return _dt.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return _dt.strptime(s[:19], fmt)
            except Exception:
                pass
    return _dt.min


@app.route("/admin/logs")
@login_required
def admin_logs():
    if session.get("role") != "admin":
        abort(403)

    # 追加: 簡易検索 & 上限（既定300は従来と同じ体感）
    q = (request.args.get("q") or "").strip().lower()
    try:
        limit = int(request.args.get("limit", "300"))
    except Exception:
        limit = 300
    limit = max(1, min(limit, 2000))

    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    log = json.loads((line or "").strip() or "{}")
                    if q:
                        blob = (
                            (log.get("question") or "") + "\n" +
                            (log.get("answer") or "")   + "\n" +
                            (log.get("source") or "")   + "\n" +
                            json.dumps(log.get("extra") or {}, ensure_ascii=False)
                        ).lower()
                        if q not in blob:
                            continue
                    logs.append(log)
                except Exception:
                    pass

    # 新しい順に統一
    logs.sort(key=lambda x: _parse_ts(x.get("timestamp")), reverse=True)
    logs = logs[:limit]

    # ?fmt=json でそのまま確認も可能
    if (request.args.get("fmt") or "").lower() == "json":
        return jsonify({"ok": True, "items": logs, "count": len(logs)})

    return render_template("admin_logs.html", logs=logs, q=q, limit=limit, role=session.get("role",""))


@app.route("/admin/unhit_questions")
@login_required
def admin_unhit_questions():
    if session.get("role") != "admin":
        abort(403)

    # 追加: 簡易検索 & 上限（既定100は従来と同じ）
    q = (request.args.get("q") or "").strip().lower()
    try:
        limit = int(request.args.get("limit", "100"))
    except Exception:
        limit = 100
    limit = max(1, min(limit, 2000))

    unhit_all = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    log = json.loads((line or "").strip() or "{}")
                    # 厳密な未ヒット判定
                    hit = _boolish_strict(log.get("hit_db", False))
                    if hit:
                        continue
                    if q:
                        blob = (
                            (log.get("question") or "") + "\n" +
                            (log.get("answer") or "")   + "\n" +
                            (log.get("source") or "")   + "\n" +
                            json.dumps(log.get("extra") or {}, ensure_ascii=False)
                        ).lower()
                        if q not in blob:
                            continue
                    unhit_all.append(log)
                except Exception:
                    pass

    # 新しい順で上位 limit 件
    unhit_all.sort(key=lambda x: _parse_ts(x.get("timestamp")), reverse=True)
    unhit_logs = unhit_all[:limit]

    stats = {
        "total_unhit": len(unhit_all),
        "showing": len(unhit_logs),
        "limit": limit,
        "q": q,
    }

    # ?fmt=json でAPI返却（運用チェック用）
    if (request.args.get("fmt") or "").lower() == "json":
        return jsonify({"ok": True, "stats": stats, "items": unhit_logs})

    return render_template(
        "admin_unhit.html",
        unhit_logs=unhit_logs,
        stats=stats,
        q=q,
        limit=limit,
        role=session.get("role",""),
    )


@app.route("/api/faq_suggest", methods=["POST"])
@limit_deco(ASK_LIMITS)   # レート制限は維持
@login_required
def api_faq_suggest():
    if session.get("role") != "admin":
        abort(403)

    # フォーム or JSON どちらでも受け付け
    payload = request.get_json(silent=True) or {}
    q = (request.form.get("q") or payload.get("q") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "q is required"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 500

    text = ai_suggest_faq(q, model=OPENAI_MODEL_PRIMARY) or ""
    return jsonify({"ok": True, "text": text})


@app.route("/admin/add_entry", methods=["POST"])
@login_required
def admin_add_entry():
    if session.get("role") != "admin":
        abort(403)

    entries = load_entries()
    title = request.form.get("title", "")
    desc  = request.form.get("desc", "")
    tags  = _split_lines_commas(request.form.get("tags", ""))
    areas = _split_lines_commas(request.form.get("areas", ""))

    if not title or not desc:
        flash("タイトルと説明は必須です")
        return redirect(url_for("admin_unhit_questions"))

    entry = _norm_entry({
        "title": title,
        "desc": desc,
        "address": "",
        "map": "",
        "tags": tags,
        "areas": areas
    })
    entries.append(entry)
    save_entries(entries)

    try:
        # その場で触れたタグからシノニムを軽く更新（失敗しても致命ではない）
        auto_update_synonyms_from_entries([entry])
        flash("DBに追加しました（シノニムも自動更新）")
    except Exception:
        flash("DBに追加しました（シノニム自動更新でエラーが出ました。ログを確認してください）")

    return redirect(url_for("admin_entry"))
# =========================



# =========================
#  バックアップ/復元
# =========================
def _safe_extractall(zf: zipfile.ZipFile, dst: str):
    base = os.path.abspath(dst)
    for member in zf.namelist():
        target = os.path.abspath(os.path.join(dst, member))
        if not target.startswith(base + os.sep) and target != base:
            raise Exception("Unsafe path found in ZIP (zip slip)")
    os.makedirs(dst, exist_ok=True)
    zf.extractall(dst)

def write_full_backup_zip(out_dir: str) -> str:
    """アプリ内バックアップを out_dir に保存し、ファイルパスを返す"""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"gotokanko_{ts}.zip"
    out_path = os.path.join(out_dir, filename)

    # 追加: アプリコードの場所を特定
    app_root = os.path.dirname(os.path.abspath(__file__))  # app.py があるディレクトリ
    app_py   = os.path.abspath(__file__)
    templates_dir = os.path.join(app_root, "templates")

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # ---- 既存: データ類 ----
        if os.path.exists(ENTRIES_FILE):   zf.write(ENTRIES_FILE,   arcname="entries.json")
        if os.path.exists(SYNONYM_FILE):   zf.write(SYNONYM_FILE,   arcname="synonyms.json")
        if os.path.exists(NOTICES_FILE):   zf.write(NOTICES_FILE,   arcname="notices.json")
        if os.path.exists(SHOP_INFO_FILE): zf.write(SHOP_INFO_FILE, arcname="shop_infos.json")

        if os.path.exists(DATA_DIR):
            for root, dirs, files in os.walk(DATA_DIR):
                for fname in files:
                    fpath   = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, BASE_DIR)
                    zf.write(fpath, arcname)

        if os.path.exists(LOG_DIR):
            for root, dirs, files in os.walk(LOG_DIR):
                for fname in files:
                    fpath   = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, BASE_DIR)
                    zf.write(fpath, arcname)

        # ---- 追加: コードスナップショット ----
        # app.py 本体
        try:
            if os.path.isfile(app_py):
                zf.write(app_py, arcname="code/app.py")
        except Exception as e:
            app.logger.exception("[backup] add app.py failed: %s", e)

        # templates/ ディレクトリ
        try:
            if os.path.isdir(templates_dir):
                for root, dirs, files in os.walk(templates_dir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        # code/templates/... というパスで格納
                        rel   = os.path.relpath(fpath, app_root)
                        zf.write(fpath, arcname=os.path.join("code", rel))
        except Exception as e:
            app.logger.exception("[backup] add templates/ failed: %s", e)

    return out_path

@app.route("/admin/backup")
@login_required
def admin_backup():
    if session.get("role") != "admin":
        abort(403)

    if request.args.get("download"):
        out_dir = os.path.join(BASE_DIR, "manual_backups")
        path = write_full_backup_zip(out_dir)
        app.logger.info(f"[backup] manual saved: {path}")
        return send_file(path, as_attachment=True,
                         download_name=os.path.basename(path),
                         mimetype="application/zip")

    logs_count = 0
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, encoding="utf-8") as f:
                logs_count = sum(1 for _ in f)
        except Exception:
            app.logger.exception("count log lines failed")

    stats = {
        "entries_count": len(load_entries()),
        "synonyms_count": len(load_synonyms()),
        "notices_count": len(load_notices()),
        "logs_count": logs_count,
    }
    return render_template("admin_backup.html", stats=stats)

@app.route("/admin/storage_stats")
@login_required
def admin_storage_stats():
    if session.get("role") != "admin":
        abort(403)
    import os
    base = BASE_DIR
    out = {"base_dir": base, "dirs": {}, "files": {}}

    def dir_stats(path):
        total = 0
        count = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                    count += 1
                except Exception:
                    pass
        return {"bytes": total, "files": count}

    # 主要ディレクトリの統計
    for d in ["data", "data/images", "logs", "manual_backups", "auto_backups"]:
        p = os.path.join(base, d)
        if os.path.isdir(p):
            out["dirs"][d] = dir_stats(p)

    # 主要ファイルのサイズだけ
    def add_file(relpath):
        p = os.path.join(base, relpath)
        if os.path.isfile(p):
            out["files"][relpath] = {"bytes": os.path.getsize(p)}

    for f in ["entries.json", "synonyms.json", "notices.json", "shop_infos.json", "logs/questions_log.jsonl"]:
        add_file(f)

    return jsonify(out)

@app.route("/admin/restore", methods=["POST"])
@login_required
def admin_restore():
    if session.get("role") != "admin":
        abort(403)
    file = request.files.get("backup_zip")
    if not file:
        flash("アップロードファイルがありません")
        return redirect(url_for("admin_entry"))
    with zipfile.ZipFile(file, "r") as zf:
        _safe_extractall(zf, BASE_DIR)
    flash("復元が完了しました。データを確認してください。")
    return redirect(url_for("admin_entry"))


@app.route("/internal/backup", methods=["POST"])
def internal_backup():
    token = request.headers.get("X-Backup-Token", "")
    if token != os.environ.get("BACKUP_JOB_TOKEN"):
        abort(403)

    out_dir = os.path.join(BASE_DIR, "auto_backups")
    path = write_full_backup_zip(out_dir)

    # ローテーション（最新10個だけ残す）
    try:
        files = sorted(
            [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".zip")],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for old in files[10:]:
            os.remove(old)
    except Exception:
        app.logger.exception("backup rotation failed")

    app.logger.info(f"[backup] saved: {path}")
    return jsonify({"ok": True, "saved": path})


# === 管理者ボタン：最優先で停止/再開 ==========================================
@app.post("/admin/line/pause")
def admin_line_pause():  # 既存の endpoint 名が admin_line_pause ならそちらに合わせて
    _pause_set_admin(True)   # 管理者停止ON
    # 利用者停止はそのままでもOK（残しておく）。必要なら同時に消すなら _pause_set_user(False)
    flash("LINE応答を一時停止しました（管理者）")
    return redirect(url_for("admin_entry"))

@app.post("/admin/line/resume")
def admin_line_resume():  # 既存の endpoint 名が admin_line_resume ならそちらに合わせて
    _pause_set_admin(False)  # 管理者停止OFF
    _pause_set_user(False)   # ついでに利用者停止も全解除（“全ての返事を再開”）
    flash("LINE応答を再開しました（管理者）")
    return redirect(url_for("admin_entry"))
# ============================================================================

@app.route("/admin/line/mutes", methods=["GET","POST"])
@login_required
def admin_line_mutes():
    if session.get("role") != "admin":
        abort(403)
    if request.method == "POST":
        tid = (request.form.get("target_id") or "").strip()
        if tid:
            _set_muted_target(tid, False, who="admin")
            flash(f"ミュート解除: {tid}")
        return redirect(url_for("admin_line_mutes"))
    # 画面を追加したくない場合は JSON で確認できるようにしておく
    m = _load_mutes()
    rows = [{"target_id": k, **(v or {})} for k, v in m.items()]
    return jsonify({"paused": _is_global_paused(), "mutes": rows})

# =========================
#  認証
# =========================
@app.route("/login", methods=["GET", "POST"])
@limit_deco(LOGIN_LIMITS)   # ← これを付けるだけ
def login():
    if request.method == "POST":
        user_id = request.form.get("username")
        pw = request.form.get("password")
        users = load_users()
        user = next((u for u in users if u["user_id"] == user_id), None)
        if user and check_password_hash(user["password_hash"], pw):
            session.permanent = True  # ← 追加: 有効期限を効かせる
            session["user_id"] = user_id
            session["role"] = user["role"]
            flash("ログインしました")
            if user["role"] == "admin":
                return redirect(url_for("admin_entry"))
            else:
                return redirect(url_for("shop_entry"))
        else:
            flash("ユーザーIDまたはパスワードが違います")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("ログアウトしました")
    return redirect(url_for("login"))


# =========================
#  マスター管理（復活＆強化）
# =========================
@app.route("/admin/synonyms", methods=["GET", "POST"])
@login_required
def admin_synonyms():
    if session.get("role") != "admin":
        abort(403)

    if request.method == "POST":
        tags_in_dom_order = [t.strip() for t in request.form.getlist("tag")]

        syn_keys = [k for k in request.form.keys() if k.startswith("synonyms_")]
        syn_indices = sorted(
            [int(k.split("_", 1)[1]) for k in syn_keys if k.split("_", 1)[1].isdigit()]
        )

        new_synonyms = {}
        for tag, idx in zip(tags_in_dom_order, syn_indices):
            if not tag:
                continue
            syn_str = request.form.get(f"synonyms_{idx}", "") or ""
            syn_list = [s.strip() for s in syn_str.split(",") if s.strip()]
            new_synonyms[tag] = syn_list

        add_tag = (request.form.get("add_tag") or "").strip()
        add_syns_str = request.form.get("add_synonyms", "") or ""
        if add_tag:
            add_list = [s.strip() for s in add_syns_str.split(",") if s.strip()]
            new_synonyms[add_tag] = add_list

        save_synonyms(new_synonyms)
        flash("類義語辞書を保存しました")
        return redirect(url_for("admin_synonyms"))

    synonyms = load_synonyms()
    # 追加: 進捗とキュー残
    all_tags, have, missing = _compute_tag_sets_for_synonyms()
    q = _load_syn_queue()
    pending_count = len(q.get("pending", []))
    return render_template(
        "admin_synonyms.html",
        synonyms=synonyms,
        stats={"total": len(all_tags), "with": len(have), "missing": len(missing)},
        queue_pending=pending_count
    )

@app.route("/admin/synonyms/export_missing")
@login_required
def admin_synonyms_export_missing():
    if session.get("role") != "admin":
        abort(403)
    all_tags, have, missing = _compute_tag_sets_for_synonyms()
    payload = {t: [] for t in missing}  # 空配列のスケルトン
    buf = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="synonyms_missing.json",
                     mimetype="application/json")


@app.route("/admin/synonyms/queue/reset")
@login_required
def admin_synonyms_queue_reset():
    if session.get("role") != "admin":
        abort(403)
    target = request.args.get("target", "missing")
    all_tags, have, missing = _compute_tag_sets_for_synonyms()
    pending = sorted(missing if target != "all" else all_tags)
    data = {
        "target": target,
        "created": datetime.datetime.now().isoformat(),
        "pending": pending
    }
    _save_syn_queue(data)
    flash(f"AI生成キューを作成: {len(pending)} 件")
    return redirect(url_for("admin_synonyms"))


@app.route("/admin/synonyms/queue/status")
@login_required
def admin_synonyms_queue_status():
    if session.get("role") != "admin":
        abort(403)
    q = _load_syn_queue()
    return jsonify({"pending": len(q.get("pending", []))})



# ========== 類義語：インポート（上書き） ==========
@app.route("/admin/synonyms/import", methods=["POST"])
@login_required
def admin_synonyms_import():
    if session.get("role") != "admin":
        abort(403)

    # ファイル or テキストのどちらか
    up = request.files.get("json_file")
    text = (request.form.get("json_text") or "").strip()

    try:
        if up and up.filename:
            raw = up.read().decode("utf-8-sig")
            new_dict = json.loads(raw)
        elif text:
            new_dict = json.loads(text)
        else:
            flash("JSONファイルまたはテキストを指定してください")
            return redirect(url_for("admin_synonyms"))

        if not isinstance(new_dict, dict):
            raise ValueError('JSONは {"タグ": ["別名", ...]} の形にしてください')

        # 正規化：各値を list[str] 化
        norm = {}
        for k, v in new_dict.items():
            key = str(k).strip()
            if isinstance(v, str):
                vals = _split_lines_commas(v)
            elif isinstance(v, (list, tuple)):
                vals = [str(x).strip() for x in v if str(x).strip()]
            elif v:
                vals = [str(v).strip()]
            else:
                vals = []
            norm[key] = vals

        # 上書き保存
        save_synonyms(norm)
        flash(f"類義語辞書を {len(norm)} タグ分で上書きしました")
    except Exception as e:
        app.logger.exception("synonyms import failed")
        flash("インポートに失敗: " + str(e))

    return redirect(url_for("admin_synonyms"))


# ========== 類義語：エクスポート（ダウンロード） ==========
@app.route("/admin/synonyms/export", methods=["GET", "POST"])
@login_required
def admin_synonyms_export():
    if session.get("role") != "admin":
        abort(403)
    syn = load_synonyms()
    buf = io.BytesIO(json.dumps(syn, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="synonyms.json", mimetype="application/json")


# ========== 類義語：AI自動生成（missing/all × append/overwrite） ==========
@app.route("/admin/synonyms/autogen", methods=["POST", "GET"])
@login_required
def admin_synonyms_autogen():
    """
    ...（既存のdocstringはそのまま）...
    """
    try:
        if session.get("role") != "admin":
            abort(403)
        if not OPENAI_API_KEY:
            flash("OPENAI_API_KEY が未設定です")
            return redirect(url_for("admin_synonyms"))

        target = request.values.get("target", "missing")      # "missing" or "all"
        mode   = request.values.get("mode", "append")         # "append" or "overwrite"

        try:
            MAX_TAGS_PER_REQUEST = max(1, min(60, int(request.values.get("limit", 20))))
        except Exception:
            MAX_TAGS_PER_REQUEST = 20
        try:
            CHUNK = max(1, min(10, int(request.values.get("chunk", 4))))
        except Exception:
            CHUNK = 4

        HARD_DEADLINE_SEC = int(os.getenv("SYN_AUTOGEN_DEADLINE", "20"))
        start_time = time.time()
        use_queue = (request.values.get("use_queue", "1").lower() in {"1","true","on","yes"})

        def _norm(s: str) -> str:
            # NFKC + 空白正規化 + 小文字
            return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "").strip())).lower()

        all_tags, have, missing = _compute_tag_sets_for_synonyms()

        if use_queue:
            q = _load_syn_queue()
            # キューが空、またはターゲットが変わったら再構築
            if not q.get("pending") or q.get("target") != target:
                pending = sorted(missing if target != "all" else all_tags)
                q = {"target": target, "created": datetime.datetime.now().isoformat(), "pending": pending}
                _save_syn_queue(q)
            target_tags = list(q.get("pending", [])[:MAX_TAGS_PER_REQUEST])
        else:
            target_tags = sorted(missing if target != "all" else all_tags)[:MAX_TAGS_PER_REQUEST]

        if not target_tags:
            flash("処理対象のタグがありません（未生成なし or キュー空）")
            return redirect(url_for("admin_synonyms"))

        # 成功したタグのみ収集（→ この分だけキューから消す）
        success_results: Dict[str, List[str]] = {}
        processed_count = 0  # 試行した件数（タイムアウトで部分実行の可能性あり）

        i = 0
        while i < len(target_tags):
            if time.time() - start_time > HARD_DEADLINE_SEC:
                flash("タイムアウト回避のため途中まで保存しました。もう一度実行してください。")
                break

            chunk = target_tags[i:i+CHUNK]
            prompt = (
                "次の観光関連タグごとに、日本語の類義語・言い換え・表記揺れを最大5個ずつ返してください。"
                "出力は必ず JSON で、キー=タグ、値=文字列配列の形にしてください。\n"
                "タグ: " + ", ".join(chunk)
            )

            content = ""
            try:
                content = openai_chat(
                    OPENAI_MODEL_HIGH,
                    [{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=900,
                ) or ""
            except Exception as e:
                app.logger.exception("autogen openai (HIGH) failed: %s", e)

            if not content:
                try:
                    content = openai_chat(
                        OPENAI_MODEL_PRIMARY,
                        [{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_output_tokens=900,
                    ) or ""
                except Exception as e2:
                    app.logger.exception("autogen openai (PRIMARY) failed: %s", e2)
                    content = ""

            data = _extract_json_object(content)
            if isinstance(data, dict) and data:
                # 正規化マップ（AI出力→要求chunkのどれかに対応付け）
                chunk_norm_map = {_norm(t): t for t in chunk}
                for k, v in data.items():
                    kn = _norm(k)
                    if kn in chunk_norm_map:  # 要求したタグに限って採用
                        if isinstance(v, str):
                            syns = _split_lines_commas(v)
                        elif isinstance(v, (list, tuple)):
                            syns = [str(x).strip() for x in v if str(x).strip()]
                        else:
                            syns = []
                        success_results[chunk_norm_map[kn]] = syns

            processed_count += len(chunk)
            i += CHUNK

        # 反映（成功分のみ）
        cur = load_synonyms()
        updated = 0
        for tag, syns in success_results.items():
            if mode == "overwrite":
                cur[tag] = syns
            else:
                cur.setdefault(tag, [])
                for s in syns:
                    if s and s not in cur[tag]:
                        cur[tag].append(s)
            updated += 1
        save_synonyms(cur)

        if use_queue:
            q = _load_syn_queue()
            pending = q.get("pending", [])
            # 成功したものだけ取り除く（順序は維持しつつフィルタ）
            success_norms = {_norm(t) for t in success_results.keys()}
            new_pending = [t for t in pending if _norm(t) not in success_norms]
            dropped = len(pending) - len(new_pending)
            q["pending"] = new_pending
            q["target"] = target  # 念のため同期
            _save_syn_queue(q)
            flash(f"AI生成: {updated}/{processed_count} タグ反映（キューから {dropped} 件削除 / 残り {len(new_pending)} 件）")
        else:
            # 非キュー時は試行件数ベースで進捗表示
            if updated == 0:
                flash("AI生成に失敗しました。もう一度実行してください。")
            elif updated < processed_count:
                flash(f"AI生成（部分実行）: {updated}/{processed_count} タグを反映しました。")
            else:
                flash(f"AI生成: {updated} タグ分を{'上書き' if mode=='overwrite' else '追記'}しました")

        return redirect(url_for("admin_synonyms"))

    except Exception as e:
        app.logger.exception("autogen endpoint fatal error: %s", e)
        flash("内部エラーが発生しました（ログを確認してください）。")
        return redirect(url_for("admin_synonyms"))


# （互換用：以前のエンドポイント名を使っていた場合のエイリアス）
@app.route("/admin/synonyms/auto", methods=["POST"])
@login_required
def admin_synonyms_auto():
    return admin_synonyms_autogen()

# ===== ここから貼り付け（既存の /callback 定義を丸ごと置換OK）=====

# 直近のコールバック状況を可視化するメモリ変数
LAST_CALLBACK_AT = None
CALLBACK_HIT_COUNT = 0
LAST_SIGNATURE_BAD = 0
LAST_LINE_ERROR = ""
# ← 追加：送信側の失敗も可視化
LAST_SEND_ERROR = ""
SEND_ERROR_COUNT = 0
SEND_FAIL_COUNT = 0 

@app.route("/_debug/line_status")
def _debug_line_status():
    """
    LINEの稼働状況を確認するデバッグ用エンドポイント。
    本番では X-Debug-Token が一致しないと 404 にする。
    """
    if APP_ENV in {"prod","production"} and request.headers.get("X-Debug-Token") != os.getenv("DEBUG_TOKEN"):
        abort(404)

    try:
        muted_count = len(_load_mutes())
    except Exception:
        muted_count = -1

    return jsonify({
        "enabled": bool(_line_enabled() and handler),
        "have_access_token": bool(LINE_CHANNEL_ACCESS_TOKEN),
        "have_channel_secret": bool(LINE_CHANNEL_SECRET),
        "handler_inited": bool(handler is not None),
        "paused": _is_global_paused(),
        "muted_count": muted_count,
        "last_callback_at": LAST_CALLBACK_AT,
        "callback_hit_count": CALLBACK_HIT_COUNT,
        "invalid_signature_count": LAST_SIGNATURE_BAD,
        "send_fail_count": SEND_FAIL_COUNT,
        "last_error": LAST_LINE_ERROR,
        "send_error_count": SEND_ERROR_COUNT,
        "last_send_error": LAST_SEND_ERROR,
        "webhook_url_hint": "/callback (POST)",  # LINEコンソールにこのパスで登録されているか確認
    })

# LINE webhook（可視化＆Rate Limit付き・1本化）
@app.route("/callback", methods=["POST"])
@limit_deco(ASK_LIMITS)  # ← ここで Rate Limit を適用（limiter無い環境ではノーオペ関数）
def callback():
    """
    LINE Webhook 受け口（統合版）
    - キー未設定や初期化失敗時: 200 'LINE disabled'（LINE側の再試行を抑止）
    - 署名不正: 400 'NG'
    - 予期せぬ例外: 500 'NG'（LINE側に再試行させる）
    - Rate Limit: ASK_LIMITS を適用
    """
    # 軽量メトリクス（未定義でも壊れないように安全更新）
    try:
        globals()["CALLBACK_HIT_COUNT"] = int(globals().get("CALLBACK_HIT_COUNT", 0)) + 1
        globals()["LAST_CALLBACK_AT"] = datetime.datetime.utcnow().isoformat() + "Z"
    except Exception:
        pass

    # キー未設定や初期化失敗
    if not _line_enabled() or not handler:
        app.logger.warning("[LINE] callback hit but LINE disabled (check env vars / init)")
        return "LINE disabled", 200

    signature = request.headers.get("X-Line-Signature", "") or ""
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        try:
            globals()["LAST_SIGNATURE_BAD"] = int(globals().get("LAST_SIGNATURE_BAD", 0)) + 1
        except Exception:
            pass
        app.logger.warning("[LINE] Invalid signature. Check LINE_CHANNEL_SECRET / webhook origin.")
        return "NG", 400
    except Exception as e:
        try:
            globals()["LAST_LINE_ERROR"] = f"{type(e).__name__}: {e}"
        except Exception:
            pass
        app.logger.exception("[LINE] handler error")
        return "NG", 500

    return "OK", 200

# 管理者用：任意ユーザーに push して疎通確認（userId はログやLINEの開発者ツールで取得）
@app.route("/admin/line/test_push", methods=["POST","GET"])
@login_required
def admin_line_test_push():
    if session.get("role") != "admin":
        abort(403)
    to = (request.values.get("to") or "").strip()
    text = (request.values.get("text") or "テスト配信です。").strip()

    if not _line_enabled() or not line_bot_api:
        flash("LINEが無効です（環境変数を確認）")
        return redirect(url_for("admin_entry"))

    if not to:
        flash("to（userId）が指定されていません")
        return redirect(url_for("admin_entry"))

    try:
        parts = _split_for_line(text, LINE_SAFE_CHARS)
        msgs = [TextSendMessage(text=p) for p in parts]
        line_bot_api.push_message(to, msgs)
        flash("push 送信を実行しました")
    except LineBotApiError as e:
        global LAST_SEND_ERROR, SEND_ERROR_COUNT
        SEND_ERROR_COUNT += 1
        LAST_SEND_ERROR = f"{type(e).__name__}: {e}"
        app.logger.exception("LINE send failed: %s", e)
        if LINE_RETHROW_ON_SEND_ERROR:
            # /callback 側まで例外を伝播させ、500 を返して気付けるようにする
            raise
    return redirect(url_for("admin_entry"))
# ===== ここまで貼り付け =====

# --- 最低限のDB回答（ヒット/複数/未ヒット） ---
# 1) まず _answer_from_entries_min を修正
def _answer_from_entries_min(question: str, *, wm_mode: str | None = None, user_id: str | None = None):
    import unicodedata, re
    from datetime import datetime, timedelta

    # ========== 内部ユーティリティ ==========
    def _n_local(s: str) -> str:
        """NFKC正規化 + 連続空白圧縮 + lower"""
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", (s or "")).strip()).lower()

    def _first_int_in_text(text: str) -> int | None:
        m = re.search(r"([0-9０-９]+)", text or "")
        if not m:
            return None
        try:
            return int(unicodedata.normalize("NFKC", m.group(1)))
        except Exception:
            return None

    def _uniq_preserve(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    def _format_choose_lines(title: str, options: list[str]) -> str:
        # 番号選択と“リセット”案内つき
        lines = [title]
        for i, v in enumerate(options, 1):
            lines.append(f"{i}. {v}")
        lines.append("\n※番号 or 名称 で選択できます。やり直す場合は「リセット」")
        return "\n".join(lines)

    def _areas_from(entries):
        arr = []
        for e in entries:
            for a in (e.get("areas") or []):
                if a:
                    arr.append(a)
        return _uniq_preserve(arr)

    def _categories_from(entries):
        arr = []
        for e in entries:
            c = e.get("category")
            if c:
                arr.append(c)
        return _uniq_preserve(arr)

    def _tag_norm(s: str) -> str:
        """タグ比較用に normalize。前後クォート/空白を除去し、NFKC+lower で比較できる形に"""
        t = (s or "").strip()
        # 前後の ' や " を剥がす（複数重なっていてもOK）
        while len(t) >= 2 and ((t[0] == t[-1]) and t[0] in {"'", '"'}):
            t = t[1:-1].strip()
        # 片側だけ付いているケースも軽くケア
        if t[:1] in {"'", '"'}: t = t[1:].strip()
        if t[-1:] in {"'", '"'}: t = t[:-1].strip()
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).lower()
        return t

    def _discriminative_tags(entries, limit: int = 12):
        """候補間で“かぶっていない”タグ（弁別力のあるタグ）を、表示用にクォートを剥がして返す"""
        raw_sets = [set(e.get("tags") or []) for e in entries if e.get("tags")]
        if not raw_sets:
            return []
        union = set().union(*raw_sets)
        inter = set.intersection(*raw_sets) if len(raw_sets) > 1 else set()
        # 弁別候補（共通タグを除く）
        cand_raw = list(union - inter) if inter else list(union)
        # 表示用にクォートを剥がし、正規化キーでユニーク化
        seen, out = set(), []
        for t in sorted(cand_raw):
            disp = t.strip().strip("'").strip('"').strip()
            key  = _tag_norm(disp)
            if key and key not in seen:
                seen.add(key)
                out.append(disp)
            if len(out) >= limit:
                break
        return out

    # ====== セッション状態（関数内だけで完結）======
    #   user_id が無い場合はセッションを使わず“従来表示”にフォールバック
    gf = globals().setdefault("_GUIDED_FLOW", {})   # {user_id: {stage, cand_titles, ...}}
    now = datetime.utcnow()
    # 15分でGC
    try:
        for k, st_gc in list(gf.items()):
            if st_gc.get("exp_at") and st_gc["exp_at"] < now:
                gf.pop(k, None)
    except Exception:
        pass

    def _flow_clear(uid):
        if uid in gf: gf.pop(uid, None)

    def _flow_set(uid, st):
        st["exp_at"] = now + timedelta(minutes=15)
        gf[uid] = st

    def _flow_get(uid):
        return gf.get(uid)

    def _is_reset_cmd(text: str) -> bool:
        t = (text or "").strip()
        return t in {"リセット", "やり直し", "キャンセル", "reset", "clear"}

    # ========= 共通：質問テキスト =========
    q = (question or "").strip()
    if not q:
        return "（内容が読み取れませんでした）", False, ""

    # （任意）★ 交通は即答優先：進行中フローが無ければだけ割り込み
    try:
        st0 = _flow_get(user_id) if user_id else None
    except Exception:
        st0 = None
    if not st0:
        m0, ok0 = get_transport_reply(q)
        if ok0:
            return m0, False, ""  # 画像なし/即答扱い

    # ========= 1) 継続フローの処理（user_id がある場合のみ）=========
    if user_id:
        # 明示リセット
        if _is_reset_cmd(q):
            _flow_clear(user_id)
            return "フローをリセットしました。知りたいスポット名やキーワードを送ってください。", True, ""

        st = _flow_get(user_id)
        if st:
            # いまの候補を再構築（常に最新エントリから）
            es_all = [ _norm_entry(e) for e in load_entries() ]
            cand = [e for e in es_all if (e.get("title") or "") in (st.get("cand_titles") or [])]
            if not cand:
                _flow_clear(user_id)
                # 続行できないので通常検索にフォールバック
            else:
                s = q.strip()
                idx = _first_int_in_text(s)

                # ---- AREA ----
                if st["stage"] == "area":
                    areas = st.get("areas") or _areas_from(cand) or ["五島市","新上五島町","小値賀町","宇久町"]
                    picked = None
                    if idx and 1 <= idx <= len(areas):
                        picked = areas[idx-1]
                    elif s in areas:
                        picked = s
                    if not picked:
                        return _format_choose_lines("まずエリアを選んでください：", areas), True, ""

                    cand2 = [e for e in cand if picked in (e.get("areas") or [])] or cand
                    if len(cand2) == 1:
                        _flow_clear(user_id)
                        e = cand2[0]
                        # ---- 最終1件の組み立て（既存の仕様を踏襲）----
                        record_source_from_entry(e)
                        urls = _build_image_urls_for_entry(e)
                        img_url = urls.get("image") or urls.get("thumb") or ""
                        lat, lng = _entry_latlng(e)
                        murl = entry_open_map_url(e, lat=lat, lng=lng)
                        lines = []
                        title = (e.get("title","") or "").strip()
                        desc  = (e.get("desc","")  or "").strip()
                        if title: lines.append(title)
                        if desc:  lines.append(desc)
                        def add(label, val):
                            if isinstance(val, list):
                                v = " / ".join([str(x).strip() for x in val if str(x).strip()])
                            else:
                                v = (val or "").strip()
                            if v:
                                lines.append(f"{label}：{v}")
                        add("住所", e.get("address"))
                        add("電話", e.get("tel"))
                        add("地図", murl or e.get("map"))
                        add("エリア", e.get("areas") or [])
                        add("休み", e.get("holiday"))
                        add("営業時間", e.get("open_hours"))
                        add("駐車場", e.get("parking"))
                        add("支払方法", e.get("payment") or [])
                        add("備考", e.get("remark"))
                        add("リンク", e.get("links") or [])
                        add("カテゴリー", e.get("category"))
                        return "\n".join(lines), True, img_url

                    # 次=カテゴリー
                    cats = _categories_from(cand2) or ["観光","飲食","宿泊"]
                    _flow_set(user_id, {"stage":"category", "cand_titles":[(e.get("title") or "") for e in cand2], "cats": cats})
                    return _format_choose_lines("次にカテゴリーを選んでください：", cats), True, ""

                # ---- CATEGORY ----
                if st["stage"] == "category":
                    cats = st.get("cats") or _categories_from(cand)
                    picked = None
                    if idx and 1 <= idx <= len(cats):
                        picked = cats[idx-1]
                    elif q.strip() in cats:
                        picked = q.strip()
                    else:
                        # エイリアス（表記ゆれ）吸収
                        ALIASES = {
                            "観光": ["観光","観光地"],
                            "飲食": ["飲食","食事","グルメ","食べる・呑む"],
                            "宿泊": ["宿泊","ホテル","旅館","民宿","泊まる"],
                        }
                        for k, vs in ALIASES.items():
                            if q.strip() == k or q.strip() in vs:
                                picked = k; break
                    if not picked:
                        return _format_choose_lines("次にカテゴリーを選んでください：", cats), True, ""

                    cand2 = [e for e in cand if e.get("category") == picked] or cand
                    if len(cand2) == 1:
                        _flow_clear(user_id)
                        e = cand2[0]
                        record_source_from_entry(e)
                        urls = _build_image_urls_for_entry(e)
                        img_url = urls.get("image") or urls.get("thumb") or ""
                        lat, lng = _entry_latlng(e)
                        murl = entry_open_map_url(e, lat=lat, lng=lng)
                        lines = []
                        title = (e.get("title","") or "").strip()
                        desc  = (e.get("desc","")  or "").strip()
                        if title: lines.append(title)
                        if desc:  lines.append(desc)
                        def add(label, val):
                            if isinstance(val, list):
                                v = " / ".join([str(x).strip() for x in val if str(x).strip()])
                            else:
                                v = (val or "").strip()
                            if v:
                                lines.append(f"{label}：{v}")
                        add("住所", e.get("address"))
                        add("電話", e.get("tel"))
                        add("地図", murl or e.get("map"))
                        add("エリア", e.get("areas") or [])
                        add("休み", e.get("holiday"))
                        add("営業時間", e.get("open_hours"))
                        add("駐車場", e.get("parking"))
                        add("支払方法", e.get("payment") or [])
                        add("備考", e.get("remark"))
                        add("リンク", e.get("links") or [])
                        add("カテゴリー", e.get("category"))
                        return "\n".join(lines), True, img_url

                    # 次=タグ
                    tags = _discriminative_tags(cand2)
                    if not tags:
                        # そのまま番号選択へ
                        titles = [(e.get("title") or "") for e in cand2][:8]
                        _flow_set(user_id, {"stage":"pick", "cand_titles":[(e.get("title") or "") for e in cand2], "titles": titles})
                        return _format_choose_lines("候補が複数あります。番号で選んでください：", titles), True, ""
                    _flow_set(user_id, {"stage":"tag", "cand_titles":[(e.get("title") or "") for e in cand2], "tags": tags})
                    return _format_choose_lines("最後にタグで絞り込みましょう：", tags), True, ""

                # ---- TAG ----
                if st["stage"] == "tag":
                    # 表示用タグ（クォート除去済み）
                    tags = st.get("tags") or _discriminative_tags(cand)
                    # ユーザー入力 → 正規化
                    s_norm = _tag_norm(q)
                    picked = None

                    # 1) 番号選択
                    if idx and 1 <= idx <= len(tags):
                        picked = tags[idx-1]
                    # 2) 名称選択（正規化一致）
                    elif s_norm:
                        for opt in tags:
                            if _tag_norm(opt) == s_norm:
                                picked = opt
                                break

                    if not picked:
                        return _format_choose_lines("最後にタグで絞り込みましょう：", tags or []), True, ""

                    # タグでフィルタ（エントリ側のタグも正規化して比較）
                    def _has_tag(e) -> bool:
                        for t in (e.get("tags") or []):
                            if _tag_norm(t) == _tag_norm(picked):
                                return True
                        return False

                    cand2 = [e for e in cand if _has_tag(e)] or cand
                    if len(cand2) == 1:
                        _flow_clear(user_id)
                        e = cand2[0]
                        record_source_from_entry(e)
                        urls = _build_image_urls_for_entry(e)
                        img_url = urls.get("image") or urls.get("thumb") or ""
                        lat, lng = _entry_latlng(e)
                        murl = entry_open_map_url(e, lat=lat, lng=lng)
                        lines = []
                        title = (e.get("title","") or "").strip()
                        desc  = (e.get("desc","")  or "").strip()
                        if title: lines.append(title)
                        if desc:  lines.append(desc)
                        def add(label, val):
                            if isinstance(val, list):
                                v = " / ".join([str(x).strip() for x in val if str(x).strip()])
                            else:
                                v = (val or "").strip()
                            if v:
                                lines.append(f"{label}：{v}")
                        add("住所", e.get("address"))
                        add("電話", e.get("tel"))
                        add("地図", murl or e.get("map"))
                        add("エリア", e.get("areas") or [])
                        add("休み", e.get("holiday"))
                        add("営業時間", e.get("open_hours"))
                        add("駐車場", e.get("parking"))
                        add("支払方法", e.get("payment") or [])
                        add("備考", e.get("remark"))
                        add("リンク", e.get("links") or [])
                        add("カテゴリー", e.get("category"))
                        return "\n".join(lines), True, img_url

                    # まだ複数 → 番号選択へ
                    titles = [(e.get("title") or "") for e in cand2][:8]
                    _flow_set(user_id, {"stage":"pick", "cand_titles":[(e.get("title") or "") for e in cand2], "titles": titles})
                    return _format_choose_lines("まだ複数あります。番号で選んでください：", titles), True, ""

                # ---- PICK ----
                if st["stage"] == "pick":
                    titles = st.get("titles") or [(e.get("title") or "") for e in cand][:8]
                    idx = _first_int_in_text(q)
                    picked_title = None
                    if idx and 1 <= idx <= len(titles):
                        picked_title = titles[idx-1]
                    elif q.strip() in titles:
                        picked_title = q.strip()
                    if not picked_title:
                        return _format_choose_lines("番号で選んでください：", titles), True, ""
                    e = next((x for x in cand if (x.get("title") or "") == picked_title), None)
                    _flow_clear(user_id)
                    if not e:
                        return "すみません、選択肢が見つかりませんでした。キーワードからやり直してください。", True, ""
                    record_source_from_entry(e)
                    urls = _build_image_urls_for_entry(e)
                    img_url = urls.get("image") or urls.get("thumb") or ""
                    lat, lng = _entry_latlng(e)
                    murl = entry_open_map_url(e, lat=lat, lng=lng)
                    lines = []
                    title = (e.get("title","") or "").strip()
                    desc  = (e.get("desc","")  or "").strip()
                    if title: lines.append(title)
                    if desc:  lines.append(desc)
                    def add(label, val):
                        if isinstance(val, list):
                            v = " / ".join([str(x).strip() for x in val if str(x).strip()])
                        else:
                            v = (val or "").strip()
                        if v:
                            lines.append(f"{label}：{v}")
                    add("住所", e.get("address"))
                    add("電話", e.get("tel"))
                    add("地図", murl or e.get("map"))
                    add("エリア", e.get("areas") or [])
                    add("休み", e.get("holiday"))
                    add("営業時間", e.get("open_hours"))
                    add("駐車場", e.get("parking"))
                    add("支払方法", e.get("payment") or [])
                    add("備考", e.get("remark"))
                    add("リンク", e.get("links") or [])
                    add("カテゴリー", e.get("category"))
                    return "\n".join(lines), True, img_url

                # 不明状態 → クリアして通常に戻す
                _flow_clear(user_id)

    # ========= 2) 通常検索（新規開始 or user_id無し or フローが無い）=========
    qn = _n_local(q)
    es = [ _norm_entry(e) for e in load_entries() ]  # 念のため正規化

    # --- タイトル最優先のスコアリング ---
    ranked = []  # (score, tie_breaker, entry)
    for e in es:
        title = e.get("title", "")
        desc  = e.get("desc", "")
        addr  = e.get("address", "")
        tags  = e.get("tags", []) or []
        areas = e.get("areas", []) or []

        tn = _n_local(title)
        dn = _n_local(desc)
        an = _n_local(addr)

        score = None
        # 優先度：タイトル完全一致＞タイトル部分一致＞説明＞住所＞タグ・エリア
        if qn and qn == tn:
            score = 100
        elif qn and qn in tn:
            score = 80
        elif qn and qn in dn:
            score = 60
        elif qn and qn in an:
            score = 50
        elif any(qn in _n_local(t) for t in tags):
            score = 40
        elif any(qn in _n_local(a) for a in areas):
            score = 30

        if score is not None:
            # 近いタイトル長を優先（同点時の並び安定化）
            tie = abs(len(tn) - len(qn))
            ranked.append((score, tie, e))

    if not ranked:
        # 即返し（天気 / 交通）
        m, ok = get_weather_reply(q)
        if ok:
            return m, False, ""
        m, ok = get_transport_reply(q)
        if ok:
            return m, False, ""
        refine, _meta = build_refine_suggestions(q)
        return "該当が見つかりませんでした。\n" + refine, False, ""

    # スコア降順 → タイトル長の近さ昇順
    ranked.sort(key=lambda t: (-t[0], t[1]))
    hits = [e for _, __, e in ranked]

    # タイトル一致（完全 or 部分）の最上位が単独なら、それを即採用
    top_score = ranked[0][0]
    same_top_count = sum(1 for s, _, __ in ranked if s == top_score)
    if same_top_count == 1:
        hits = [hits[0]]

    # ---- 1件 → 既存の最終出力 ----
    if len(hits) == 1:
        e = hits[0]

        # ★ 出典（フォームの「出典」テキストがあればのみ付与）
        record_source_from_entry(e)

        # ★ 画像URL（wm_external_choice を尊重）
        urls = _build_image_urls_for_entry(e)
        img_url = urls.get("image") or urls.get("thumb") or ""

        # ★ 地図URL（place_id/共有URL優先 → 無ければ緯度経度/名称から生成）
        lat, lng = _entry_latlng(e)
        murl = entry_open_map_url(e, lat=lat, lng=lng)

        # 本文は「タイトル1行＋説明1行」→以降に項目
        lines = []
        title = (e.get("title","") or "").strip()
        desc  = (e.get("desc","")  or "").strip()
        if title: lines.append(title)
        if desc:  lines.append(desc)

        def add(label, val):
            if isinstance(val, list):
                v = " / ".join([str(x).strip() for x in val if str(x).strip()])
            else:
                v = (val or "").strip()
            if v:
                lines.append(f"{label}：{v}")

        add("住所", e.get("address"))
        add("電話", e.get("tel"))
        if murl:
            add("地図", murl)  # ← map文字列ではなく最適URL
        else:
            add("地図", e.get("map"))
        add("エリア", e.get("areas") or [])
        add("休み", e.get("holiday"))
        add("営業時間", e.get("open_hours"))
        add("駐車場", e.get("parking"))
        add("支払方法", e.get("payment") or [])
        add("備考", e.get("remark"))
        add("リンク", e.get("links") or [])
        add("カテゴリー", e.get("category"))

        # タグは返信に入れない（現状維持）
        return "\n".join(lines), True, img_url

    # ---- 2件以上 → 段階的フロー開始（user_id がある場合）----
    if user_id:
        # 自動前進（エリア/カテゴリーが1種類しかないなら先に進める）
        cand = hits[:]
        # エリア自動決定
        areas = _areas_from(cand)
        stage = "area"
        if len(areas) == 1:
            picked = areas[0]
            cand = [e for e in cand if picked in (e.get("areas") or [])] or cand
            if len(cand) == 1:
                # 1件に確定
                e = cand[0]
                record_source_from_entry(e)
                urls = _build_image_urls_for_entry(e)
                img_url = urls.get("image") or urls.get("thumb") or ""
                lat, lng = _entry_latlng(e)
                murl = entry_open_map_url(e, lat=lat, lng=lng)
                lines = []
                title = (e.get("title","") or "").strip()
                desc  = (e.get("desc","")  or "").strip()
                if title: lines.append(title)
                if desc:  lines.append(desc)
                def add(label, val):
                    if isinstance(val, list):
                        v = " / ".join([str(x).strip() for x in val if str(x).strip()])
                    else:
                        v = (val or "").strip()
                    if v:
                        lines.append(f"{label}：{v}")
                add("住所", e.get("address"))
                add("電話", e.get("tel"))
                add("地図", murl or e.get("map"))
                add("エリア", e.get("areas") or [])
                add("休み", e.get("holiday"))
                add("営業時間", e.get("open_hours"))
                add("駐車場", e.get("parking"))
                add("支払方法", e.get("payment") or [])
                add("備考", e.get("remark"))
                add("リンク", e.get("links") or [])
                add("カテゴリー", e.get("category"))
                return "\n".join(lines), True, img_url
            stage = "category"
            # カテゴリー自動決定
            cats = _categories_from(cand)
            if len(cats) == 1:
                picked = cats[0]
                cand = [e for e in cand if e.get("category") == picked] or cand
                if len(cand) == 1:
                    e = cand[0]
                    record_source_from_entry(e)
                    urls = _build_image_urls_for_entry(e)
                    img_url = urls.get("image") or urls.get("thumb") or ""
                    lat, lng = _entry_latlng(e)
                    murl = entry_open_map_url(e, lat=lat, lng=lng)
                    lines = []
                    title = (e.get("title","") or "").strip()
                    desc  = (e.get("desc","")  or "").strip()
                    if title: lines.append(title)
                    if desc:  lines.append(desc)
                    def add(label, val):
                        if isinstance(val, list):
                            v = " / ".join([str(x).strip() for x in val if str(x).strip()])
                        else:
                            v = (val or "").strip()
                        if v:
                            lines.append(f"{label}：{v}")
                    add("住所", e.get("address"))
                    add("電話", e.get("tel"))
                    add("地図", murl or e.get("map"))
                    add("エリア", e.get("areas") or [])
                    add("休み", e.get("holiday"))
                    add("営業時間", e.get("open_hours"))
                    add("駐車場", e.get("parking"))
                    add("支払方法", e.get("payment") or [])
                    add("備考", e.get("remark"))
                    add("リンク", e.get("links") or [])
                    add("カテゴリー", e.get("category"))
                    return "\n".join(lines), True, img_url
                stage = "tag"

        # 状態を保存して最初の質問を返す
        _flow_set(user_id, {
            "stage": "area" if stage == "area" else ("category" if stage == "category" else "tag"),
            "cand_titles": [(e.get("title") or "") for e in cand],
            "areas": _areas_from(cand) if stage == "area" else None,
            "cats": _categories_from(cand) if stage == "category" else None,
            "tags": _discriminative_tags(cand) if stage == "tag" else None,
        })
        if stage == "area":
            areas = _areas_from(cand) or ["五島市","新上五島町","小値賀町","宇久町"]
            return _format_choose_lines("まずエリアを選んでください：", areas), True, ""
        if stage == "category":
            cats = _categories_from(cand) or ["観光","飲食","宿泊"]
            return _format_choose_lines("次にカテゴリーを選んでください：", cats), True, ""
        # stage == "tag"
        tags = _discriminative_tags(cand)
        if tags:
            return _format_choose_lines("最後にタグで絞り込みましょう：", tags), True, ""
        # タグが無ければ番号選択へ
        titles = [(e.get("title") or "") for e in cand][:8]
        _flow_set(user_id, {"stage":"pick", "cand_titles":[(e.get("title") or "") for e in cand], "titles": titles})
        return _format_choose_lines("候補が複数あります。番号で選んでください：", titles), True, ""

    # ---- user_id が無い場合は従来の一覧表示（互換モード）----
    lines = ["候補が複数見つかりました。気になるものはありますか？"]
    for i, e in enumerate(hits[:8], 1):
        area_list = e.get("areas", []) or []
        area = " / ".join(area_list) if area_list else ""
        suffix = f"（{area}）" if area else ""
        lines.append(f"{i}. {e.get('title','')}{suffix}")
    if len(hits) > 8:
        lines.append(f"…ほか {len(hits)-8} 件")

    refine, _meta = build_refine_suggestions(q)
    return "\n".join(lines) + "\n\n" + refine, True, ""

#  即返し（天気 / 運行状況） - 保証版
# =========================
import unicodedata as _unic

def _norm_text_jp(s: str) -> str:
    return _unic.normalize("NFKC", (s or "")).strip().lower()

def get_weather_reply(text: str):
    """
    「天気/天候/予報/weather」を含めば即リンクを返す
    """
    t = _norm_text_jp(text)
    if not any(k in t for k in ["天気", "天候", "予報", "weather"]):
        return "", False

    msg = (
        "【五島列島の主な天気情報リンク】\n"
        "五島市: https://weathernews.jp/onebox/tenki/nagasaki/42211/\n"
        "新上五島町: https://weathernews.jp/onebox/tenki/nagasaki/42411/\n"
        "小値賀町: https://tenki.jp/forecast/9/45/8440/42383/\n"
        "宇久町: https://weathernews.jp/onebox/33.262381/129.131027/q=%E9%95%B7%E5%B4%8E%E7%9C%8C%E4%BD%90%E4%B8%96%E4%BF%9D%E5%B8%82%E5%AE%87%E4%B9%85%E7%94%BA&v=da56215a2617fc2203c6cae4306d5fd8c92e3e26c724245d91160a4b3597570a&lang=ja&type=week"
    )
    return msg, True

def get_transport_reply(text: str):
    """
    交通に関する即答ロジックを統合：
    - 「運行/運航/運休/欠航/状況/情報/status」＋（船/飛行機 等）があれば、従来どおり運行状況リンクを即答
    - それ以外の「交通/移動/レンタカー/タクシー/バス/レンタサイクル など」は、
      エリア別の『交通機関一覧』ページを即答（エリアが特定できなければ4エリア一覧）
    戻り値: (message:str, ok:bool)
    """
    t = _norm_text_jp(text)

    # -------------------------
    # 1) 運行状況（従来ロジックを維持）
    # -------------------------
    state_hit = any(k in t for k in ["運行", "運航", "運休", "欠航", "状況", "情報", "status"])
    vehicle_hit = any(k in t for k in [
        "船","フェリー","ジェットフォイル","高速船","太古","九州商船","産業汽船",
        "飛行機","空港","福江空港","五島つばき空港","ana","jal","フライト"
    ])

    if state_hit and vehicle_hit:
        wants_ship = any(k in t for k in ["船","フェリー","ジェットフォイル","高速船","太古","九州商船","産業汽船"])
        wants_fly  = any(k in t for k in ["飛行機","空港","フライト","福江空港","五島つばき空港","ana","jal"])

        ship_section = (
            "【長崎ー五島航路 運行状況】\n"
            "・野母商船「フェリー太古」運航情報  \n"
            "  http://www.norimono-info.com/frame_set.php?usri=&disp=group&type=ship\n"
            "・九州商船「フェリー・ジェットフォイル」運航情報  \n"
            "  https://kyusho.co.jp/status\n"
            "・五島産業汽船「フェリー」運航情報  \n"
            "  https://www.goto-sangyo.co.jp/\n"
            "その他の航路や詳細は各リンクをご覧ください。"
        )
        fly_section = (
            "五島つばき空港の最新の運行状況は、公式Webサイトでご確認いただけます。\n"
            "▶ https://www.fukuekuko.jp/"
        )

        if wants_ship and not wants_fly:
            return ship_section, True
        if wants_fly and not wants_ship:
            return fly_section, True
        return ship_section + "\n\n" + fly_section, True

    # -------------------------
    # 2) 交通機関一覧（新規ロジック）
    # -------------------------
    # 交通インテント（必要に応じて語を追加）
    transport_intent = any(k in t for k in [
        "交通","移動","レンタカー","レンタサイクル","タクシ","タクシー",
        "配車","送迎","バス","路線バス","交通機関","transport","taxi","bus","car"
    ])
    if not (transport_intent or vehicle_hit):  # vehicle_hit 単独（=船/飛行機ワードのみ）の場合も一覧へ誘導したいので含める
        return "", False

    AREA_URL = {
        "五島市":     "https://www.fullygoto.com/kotsuu/",
        "新上五島町": "https://www.fullygoto.com/kamigotokotuu/",
        "小値賀町":   "https://www.fullygoto.com/odikakotuu/",
        "宇久町":     "https://www.fullygoto.com/ukukotsuu/",
    }
    AREA_ALIASES = {
        "五島市":     {"五島市","福江","福江島","富江","玉之浦","岐宿","三井楽","奈留","奈留島"},
        "新上五島町": {"新上五島町","上五島","中通島","若松","有川","奈良尾","青方"},
        "小値賀町":   {"小値賀町","小値賀","小値賀島","野崎島"},
        "宇久町":     {"宇久町","宇久","宇久島"},
    }

    def detect_area(text_norm: str) -> str | None:
        for area, names in AREA_ALIASES.items():
            for name in names:
                if _norm_text_jp(name) in text_norm:
                    return area
        return None

    area = detect_area(t)
    if area:
        url = AREA_URL[area]
        msg = (
            f"{area} の交通機関一覧はこちらです。\n{url}\n\n"
            "※レンタカー／タクシー／バス等の連絡先と最新の案内は、このページに集約しています。"
        )
        return msg, True

    # エリア未特定 → 4エリアの一覧を提示（即答）
    lines = ["交通機関のエリアをお選びください（リンクをタップ）："]
    for a, u in AREA_URL.items():
        lines.append(f"- {a} 交通機関一覧：{u}")
    lines.append("\n例：『五島市の交通』『上五島のレンタカー』『小値賀 タクシー』などでもOK。")
    return "\n".join(lines), True

@app.route("/admin/upload_image", methods=["POST"])
@login_required
def admin_upload_image():
    if session.get("role") not in {"admin", "shop"}:
        abort(403)

    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "ファイルがありません"}), 400

    # 前半にある唯一の正の保存関数を使用
    res = _save_jpeg_1080_350kb(f, previous=None, delete=False)
    if res is None:
        return jsonify({"ok": False, "error": "画像の保存に失敗しました"}), 400
    if res == "":
        return jsonify({"ok": False, "error": "削除指定は許可されていません"}), 400

    # ▼【追記】3種を事前生成
    try:
        _ensure_wm_variants(res)
    except Exception:
        app.logger.exception("[wm] variants generation failed in admin_upload_image for %s", res)

    url = url_for("serve_image", filename=res, _external=True)
    return jsonify({"ok": True, "file": res, "url": url})


# === 透かし一括生成ツール（WEB UI用） =========================
from PIL import Image, ImageDraw, ImageFont, ImageOps

WM_TEXT_GOTO  = "@Goto City"
WM_TEXT_FULLY = "@fullyGOTO"
WM_SUFFIX_NONE  = "__none"
WM_SUFFIX_GOTO  = "__goto"
WM_SUFFIX_FULLY = "__fullygoto"

def _wm_load_font(size: int):
    """環境依存なく安全にフォントを確保"""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        # よくあるパスを順に
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        ]:
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()

def _wm_draw(im: Image.Image, text: str, scale=0.05, opacity=180, margin_ratio=0.02):
    """右下に半透明テキスト透かし"""
    if not text:
        return im.copy()
    base = ImageOps.exif_transpose(im).convert("RGBA")
    W, H = base.size
    size = max(12, int(W * float(scale)))
    font = _wm_load_font(size)
    draw = ImageDraw.Draw(base)
    stroke_w = max(1, size // 12)

    try:
        bbox = draw.textbbox((0,0), text, font=font, stroke_width=stroke_w)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        tw, th = draw.textlength(text, font=font), size

    pad = max(6, int(size*0.3))
    margin = int(min(W,H) * float(margin_ratio))
    x = max(0, W - tw - pad*2 - margin)
    y = max(0, H - th - pad*2 - margin)
    opa = max(0, min(255, int(opacity)))

    # 半透明の黒下地
    bg = Image.new("RGBA", (int(tw+pad*2), int(th+pad*2)), (0,0,0, int(opa*0.45)))
    base.alpha_composite(bg, (x, y))
    # 白文字＋黒縁
    draw.text((x+pad, y+pad), text, fill=(255,255,255,opa), font=font,
              stroke_width=stroke_w, stroke_fill=(0,0,0, min(255, opa+50)))
    return base.convert("RGB")

def _wm_prep_image(path: str, max_size: int | None = 1080) -> Image.Image:
    """EXIF回転を正し、必要なら長辺max_sizeに縮小"""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if max_size:
        w,h = im.size
        sc = max(w/max_size, h/max_size)
        if sc > 1:
            im = im.resize((int(w/sc), int(h/sc)), Image.LANCZOS)
    return im


# --- 透かしテキストを右下に描画 ---
def _wm_draw(im, text, scale=0.05, opacity=180, margin_ratio=0.02):
    from PIL import Image, ImageDraw, ImageFont
    if not text:
        return im

    base = im.convert("RGBA")
    W, H = base.size

    size = max(18, int(W * float(scale)))
    margin = max(6, int(W * float(margin_ratio)))
    stroke_w = max(1, size // 12)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(base)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    x = max(0, W - tw - margin * 2)
    y = max(0, H - th - margin * 2)

    # 半透明の下地
    bg = Image.new("RGBA", (tw + margin * 2, th + margin * 2), (0, 0, 0, int(opacity * 0.45)))
    base.alpha_composite(bg, (x, y))

    # 本体文字（白＋黒縁）
    try:
        draw.text(
            (x + margin, y + margin),
            text,
            fill=(255, 255, 255, int(opacity)),
            font=font,
            stroke_width=stroke_w,
            stroke_fill=(0, 0, 0, min(255, int(opacity) + 50)),
        )
    except TypeError:
        draw.text((x + margin, y + margin), text, fill=(255, 255, 255, int(opacity)), font=font)

    return base.convert("RGB")


# --- JPEG 保存ヘルパー（品質指定・最適化） ---
def _wm_save_jpeg(im, out_path, quality=85):
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im = im.convert("RGB")
    im.save(out_path, format="JPEG", quality=int(quality), optimize=True, subsampling=0)



def _wm_save(im: Image.Image, dest: str, quality=85):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ext = os.path.splitext(dest)[1].lower()
    out = im
    params = {}
    if ext in (".jpg", ".jpeg"):
        if im.mode not in ("RGB","L"):
            out = im.convert("RGB")
        params = dict(quality=int(quality), optimize=True, progressive=True)
    elif ext == ".webp":
        if im.mode not in ("RGB","L"):
            out = im.convert("RGB")
        params = dict(quality=int(quality), method=6)
    elif ext == ".png":
        params = dict(optimize=True)
    out.save(dest, **params)

def _wm_variants_for_path(src_path: str, force=False, scale=0.05, opacity=180, margin=0.02, quality=85):
    """
    1枚のオリジナルから __none / __goto / __fullygoto を IMAGES_DIR に生成。
    既存があればスキップ（force=Trueで上書き）。
    戻り値: {"none":fn, "goto":fn, "fully":fn}
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)

    # 事前にEXIF回転補正＋リサイズ
    im = _wm_prep_image(src_path, max_size=1080)

    stem, _ext = os.path.splitext(os.path.basename(src_path))
    out_ext = ".jpg"  # ここを固定しないと PNG 等で quality 指定が例外になる
    out_none = os.path.join(IMAGES_DIR, f"{stem}{WM_SUFFIX_NONE}{out_ext}")
    out_goto = os.path.join(IMAGES_DIR, f"{stem}{WM_SUFFIX_GOTO}{out_ext}")
    out_full = os.path.join(IMAGES_DIR, f"{stem}{WM_SUFFIX_FULLY}{out_ext}")

    if force or not os.path.exists(out_none):
        _wm_save_jpeg(im, out_none, quality=quality)
    if force or not os.path.exists(out_goto):
        _wm_save_jpeg(_wm_draw(im, WM_TEXT_GOTO,  scale=scale, opacity=opacity, margin_ratio=margin), out_goto, quality=quality)
    if force or not os.path.exists(out_full):
        _wm_save_jpeg(_wm_draw(im, WM_TEXT_FULLY, scale=scale, opacity=opacity, margin_ratio=margin), out_full, quality=quality)

    return {
        "none":  os.path.basename(out_none),
        "goto":  os.path.basename(out_goto),
        "fully": os.path.basename(out_full),
    }

def _list_source_images():
    """IMAGES_DIR から『元画像っぽいもの（__none/__goto/__fullygotoが付いてない）』だけ列挙"""
    files = []
    if not os.path.isdir(IMAGES_DIR):
        return files
    for fn in os.listdir(IMAGES_DIR):
        if fn.startswith("."): 
            continue
        root, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"):
            continue
        if any(suf in root for suf in (WM_SUFFIX_NONE, WM_SUFFIX_GOTO, WM_SUFFIX_FULLY)):
            continue
        files.append(fn)
    files.sort()
    return files

@app.route("/admin/watermark", methods=["GET", "POST"])
@login_required
def admin_watermark():
    # 権限: admin / shop のどちらでも利用可
    if session.get("role") not in {"admin", "shop"}:
        abort(403)

    # --- 1) ラジオ値・検索語を関数冒頭で取得（GET/POST共通） ---
    wm = (request.values.get("wm") or "all").strip()
    if wm not in {"all", "none", "goto", "fully"}:
        wm = "all"
    make_none  = (wm in ("all", "none"))
    make_goto  = (wm in ("all", "goto"))
    make_fully = (wm in ("all", "fully"))

    q = (request.values.get("q") or "").strip().lower()

    # --- 2) 既存ソース画像の一覧を取得してから検索で絞り込み ---
    existing = _list_source_images()  # 例: 透かし未適用の元画像ファイル名リスト
    if q:
        existing = [fn for fn in existing if q in fn.lower()]

    results = []
    errors  = []

    if request.method == "POST":
        force = (request.form.get("force") == "1")

        # (a) 既存ファイルの選択分
        selected = request.form.getlist("selected_existing")

        # (b) 新規アップロード分を保存
        uploads = request.files.getlist("files") or []

        # 生成対象の (表示名, 絶対パス) を集約
        targets = []
        for name in selected:
            name = os.path.basename(name)
            targets.append((name, os.path.join(IMAGES_DIR, name)))

        for f in uploads:
            if not f or not f.filename:
                continue
            try:
                saved = _save_jpeg_1080_350kb(f, previous=None, delete=False)
                if not saved:
                    errors.append(f"{f.filename}: 保存に失敗")
                    continue
                targets.append((saved, os.path.join(IMAGES_DIR, saved)))
            except Exception as e:
                errors.append(f"{f.filename}: {e}")

        # 選択種別のリスト（none/goto/fully）
        kinds = []
        if make_none:  kinds.append("none")
        if make_goto:  kinds.append("goto")
        if make_fully: kinds.append("fully")

        # 互換ラッパー: 選択生成対応関数があればそちらを使う
        def _generate_variants(path: str, *, force: bool, kinds: list[str]):
            fn = globals().get("_wm_variants_for_path_selected")
            if callable(fn):
                try:
                    return fn(path, force=force, kinds=kinds)  # 期待: {"none":fn, "goto":fn, "fully":fn} のうち選択分のみ
                except TypeError:
                    # 引数不一致などは従来関数にフォールバック
                    pass
            # 従来: 常に3種返す想定（{"none":..., "goto":..., "fully":...}）
            return _wm_variants_for_path(path, force=force)

        # 生成＆URL作成（選択された種類だけ r に載せる）
        for display_name, abs_path in targets:
            try:
                out = _generate_variants(abs_path, force=force, kinds=kinds)
                r = {"src": display_name, "url_src": url_for("admin_media_img", filename=display_name)}

                if make_none and out.get("none"):
                    r["url_none"] = url_for("admin_media_img", filename=out["none"])
                if make_goto and out.get("goto"):
                    r["url_goto"] = url_for("admin_media_img", filename=out["goto"])
                if make_fully and out.get("fully"):
                    r["url_fully"] = url_for("admin_media_img", filename=out["fully"])

                results.append(r)
            except Exception as e:
                errors.append(f"{display_name}: {e}")

        flash(f"{len(results)} 件を処理しました" + ("（上書き）" if force else ""))
        if errors:
            flash("一部エラー: " + " / ".join(errors))

    # --- 3) テンプレへ現在の選択状態も渡す（ラジオ初期化・検索語保持用） ---
    return render_template(
        "admin_watermark.html",
        existing=existing,
        results=results,
        role=session.get("role", ""),
        wm=wm,   # ← ラジオ初期値
        q=q      # ← 検索語初期値（テンプレの <input value="{{ q|e }}"> に反映可能）
    )


@app.route("/_debug/where")
def _debug_where():
    import os
    return jsonify({"app_file": __file__, "cwd": os.getcwd()})


@app.route("/_debug/test_weather")
def _debug_test_weather():
    q = request.args.get("q", "") or ""
    m, ok = get_weather_reply(q)
    return jsonify({"ok": ok, "answer": m})


@app.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    # 必要なら軽い自己診断をここに追加可
    return ("ok", 200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-store",
    })


@app.route("/_debug/test_transport")
def _debug_test_transport():
    q = request.args.get("q", "") or ""
    m, ok = get_transport_reply(q)
    return jsonify({"ok": ok, "answer": m})

def _format_entry_text(e: dict) -> str:
    """指定仕様どおりの並びで1本のテキストを組む（タグは入れない）"""
    lines = []
    title = (e.get("title") or "").strip()
    desc  = (e.get("desc")  or "").strip()
    if title: lines.append(title)            # 1行目：タイトル
    if desc:  lines.append(desc)             # 2行目：説明

    if e.get("address"):    lines.append(f"住所：{e['address']}")
    if e.get("tel"):        lines.append(f"電話：{e['tel']}")
    if e.get("map"):        lines.append(f"地図：{e['map']}")
    areas = " / ".join(e.get("areas") or [])
    if areas:               lines.append(f"エリア：{areas}")
    if e.get("holiday"):    lines.append(f"休み：{e['holiday']}")
    if e.get("open_hours"): lines.append(f"営業時間：{e['open_hours']}")
    # 駐車場（台数があれば併記）
    if e.get("parking"):
        if e.get("parking_num"):
            lines.append(f"駐車場：{e['parking']}（{e['parking_num']}台）")
        else:
            lines.append(f"駐車場：{e['parking']}")
    # 支払方法（配列をカンマ結合）
    pay = ", ".join(e.get("payment") or [])
    if pay: lines.append(f"支払方法：{pay}")
    if e.get("remark"): lines.append(f"備考：{e['remark']}")
    # リンク（複数想定）
    links = e.get("links") or []
    if links:
        lines.append("リンク：")
        lines.extend(links[:5])  # 多すぎ防止で上位5件
    if e.get("category"):
        lines.append(f"カテゴリー：{e['category']}")
    return "\n".join(lines)


def _format_entry_messages(e: dict):
    """
    画像があれば先に画像、続けて仕様どおりのテキストを送るための
    LINEメッセージ配列を返すユーティリティ。

    戻り値: list[TextSendMessage | ImageSendMessage]
    """
    msgs = []

    # 画像（image_file 優先 / 後方互換で image も見る）
    img_name = (e.get("image_file") or e.get("image") or "").strip()
    if img_name:
        try:
            # 透かし/署名/プレビューはヘルパーで統一処理
            img_msg = make_line_image_message(e)  # ← 修正: entry ではなく e を渡す
            if img_msg:
                msgs.append(img_msg)
        except Exception:
            # LINE SDKの型エラー等は握りつぶしてテキストのみ送る
            app.logger.exception("failed to build ImageSendMessage")

    # 本文テキスト（長文は安全分割）
    text = _format_entry_text(e)
    parts = _split_for_line(text, LINE_SAFE_CHARS)
    if not parts:
        parts = ["（表示できる内容がありませんでした）"]

    for p in parts:
        try:
            msgs.append(TextSendMessage(text=p))
        except Exception:
            app.logger.exception("failed to build TextSendMessage; falling back to plain string")
            # 最悪、プレーンな辞書にしても LINE SDK は受け取らないのでログのみ
            # 呼び出し側で TextSendMessage を期待しているためここでは append しない

    return msgs


def _search_entries_prioritized(q: str):
    """タイトル一致＞タイトル部分一致＞その他（説明など）の順で優先度付きヒットを返す"""
    qn = _norm_text_jp(q)
    exact, part, others = [], [], []
    for e in load_entries():
        title_n = _norm_text_jp(e.get("title", ""))
        desc_n  = _norm_text_jp(e.get("desc", ""))
        address_n = _norm_text_jp(e.get("address", ""))
        tags_n  = _norm_text_jp(" ".join(e.get("tags") or []))
        areas_n = _norm_text_jp(" ".join(e.get("areas") or []))

        if not qn:
            continue
        if title_n == qn:
            exact.append(e)
        elif qn in title_n:
            part.append(e)
        elif (qn in desc_n) or (qn in address_n) or (qn in tags_n) or (qn in areas_n):
            others.append(e)
    return exact, part, others


# ===== ガイド付き絞り込みフロー（エリア→カテゴリー→タグ→番号） =====
ALLOWED_AREAS = ["五島市", "新上五島町", "小値賀町", "宇久町"]
CATEGORY_ALIASES = {
    "観光": ["観光", "観光地"],
    "飲食": ["飲食", "食事", "グルメ", "食べる・呑む"],
    "宿泊": ["宿泊", "ホテル", "旅館", "民宿", "泊まる"],
    # 必要に応じて追記（ショップ/イベント など）
}

def _flow_box():
    return globals().setdefault("_LAST", {}).setdefault("guided_flow", {})

def _flow_get(uid): 
    return _flow_box().get(uid)

def _flow_set(uid, state): 
    _flow_box()[uid] = state

def _flow_clear(uid): 
    _flow_box().pop(uid, None)

def _is_flow_reset_cmd(text: str) -> bool:
    t = (text or "").strip()
    return t in {"リセット", "やり直し", "キャンセル", "reset", "clear"}

def _first_int_in_text(text: str) -> int | None:
    import re, unicodedata
    m = re.search(r'([0-9０-９]+)', text or "")
    return int(unicodedata.normalize("NFKC", m.group(1))) if m else None

def _uniq_preserve(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _format_choose_lines(title: str, options: list[str]) -> str:
    lines = [title]
    for i, v in enumerate(options, 1):
        lines.append(f"{i}. {v}")
    lines.append("\n※番号 or 名称 で選択できます。やり直す場合は「リセット」")
    return "\n".join(lines)

def _entry_min(e: dict) -> dict:
    # 状態に保存する最小限（必要ならフィールドを足してください）
    return {
        "title": e.get("title",""),
        "areas": e.get("areas") or [],
        "category": e.get("category",""),
        "tags": e.get("tags") or [],
        "address": e.get("address",""),
        "image_url": e.get("image_url",""),
        "google_url": e.get("google_url",""),
        "mymap_url": e.get("mymap_url",""),
    }

def _flow_start(uid: str, entries: list[dict], question: str) -> str:
    cand = [_entry_min(e) for e in entries]
    # まずエリア
    areas = _uniq_preserve([a for e in cand for a in (e.get("areas") or []) if a in ALLOWED_AREAS]) or ALLOWED_AREAS
    st = {"step": "area", "q": question, "cand": cand, "areas": areas}
    _flow_set(uid, st)
    return _format_choose_lines("まずエリアを選んでください：", areas)

def _flow_step(uid: str, text: str) -> tuple[list[str] | None, list[dict] | None]:
    """
    フローを1ステップ進める：
    戻り値:
      (reply_lines, final_entries)
        reply_lines: 次の質問テキスト（複数行）を Text で返す場合
        final_entries: 1件に確定したときにそのエントリ配列（len==1想定）
    """
    st = _flow_get(uid)
    if not st:
        return (["（フローが見つかりません。最初からキーワードを送ってください）"], None)

    cand = st["cand"]
    step = st["step"]

    # 入力の正規化
    s = (text or "").strip()

    # 番号指定の共通処理
    idx = _first_int_in_text(s)

    if step == "area":
        areas = st.get("areas", [])
        picked = None
        if idx and 1 <= idx <= len(areas):
            picked = areas[idx-1]
        elif s in areas:
            picked = s
        if not picked:
            return (["うまく読み取れませんでした。番号か名称でエリアを選んでください。"], None)

        cand2 = [e for e in cand if picked in (e.get("areas") or [])] or cand
        if len(cand2) == 1:
            _flow_clear(uid); return (None, cand2)

        # 次=カテゴリー
        cats_all = _uniq_preserve([e.get("category","") for e in cand2 if e.get("category")])
        # 表記ゆれ吸収
        if not cats_all:
            cats_all = list(CATEGORY_ALIASES.keys())
        _flow_set(uid, {"step":"category", "q": st["q"], "cand": cand2, "cats": cats_all})
        return ([_format_choose_lines("次にカテゴリーを選んでください：", cats_all)], None)

    if step == "category":
        cats = st.get("cats", [])
        # エイリアス対応
        picked = None
        if idx and 1 <= idx <= len(cats):
            picked = cats[idx-1]
        else:
            for k, vs in CATEGORY_ALIASES.items():
                if s == k or s in vs:
                    picked = k; break
            if not picked and s in cats:
                picked = s
        if not picked:
            return (["うまく読み取れませんでした。番号か名称でカテゴリーを選んでください。"], None)

        cand2 = [e for e in cand if e.get("category") == picked] or cand
        if len(cand2) == 1:
            _flow_clear(uid); return (None, cand2)

        # 次=タグ（共通タグは除外）
        tag_lists = [set(e.get("tags") or []) for e in cand2 if e.get("tags")]
        tag_union = set().union(*tag_lists) if tag_lists else set()
        tag_inter = set.intersection(*tag_lists) if tag_lists else set()
        tag_opts = _uniq_preserve([t for t in (sorted(tag_union - tag_inter)) if t])[:12] or list(tag_union)[:12]
        if not tag_opts:
            # タグが無ければ最終番号選択へ
            titles = [e["title"] for e in cand2][:8]
            _flow_set(uid, {"step":"pick", "q": st["q"], "cand": cand2, "titles": titles})
            return ([_format_choose_lines("候補が複数あります。番号で選んでください：", titles)], None)

        _flow_set(uid, {"step":"tag", "q": st["q"], "cand": cand2, "tags": tag_opts})
        return ([_format_choose_lines("最後にタグで絞り込みましょう：", tag_opts)], None)

    if step == "tag":
        tags = st.get("tags", [])
        picked = None
        if idx and 1 <= idx <= len(tags):
            picked = tags[idx-1]
        elif s in tags:
            picked = s
        if not picked:
            return (["うまく読み取れませんでした。番号か名称でタグを選んでください。"], None)

        cand2 = [e for e in cand if picked in (e.get("tags") or [])] or cand
        if len(cand2) == 1:
            _flow_clear(uid); return (None, cand2)

        titles = [e["title"] for e in cand2][:8]
        _flow_set(uid, {"step":"pick", "q": st["q"], "cand": cand2, "titles": titles})
        return ([_format_choose_lines("まだ複数あります。番号で選んでください：", titles)], None)

    if step == "pick":
        titles = st.get("titles", [])
        picked = None
        if idx and 1 <= idx <= len(titles):
            picked = titles[idx-1]
        elif s in titles:
            picked = s
        if not picked:
            return (["番号で選んでください。例：『2』"], None)
        cand2 = [e for e in cand if e["title"] == picked]
        _flow_clear(uid); 
        return (None, cand2[:1])

    return (["（不明なステップです。リセットしてやり直してください）"], None)


def _answer_from_entries_rich(question: str):
    """
    タイトル最優先の検索で見つけ、画像＋所定フォーマットで返す。
    戻り値: ([Message], hit:bool)
    """
    exact, part, others = _search_entries_prioritized(question)
    hits = exact or part or others

    if not hits:
        refine, _ = build_refine_suggestions(question)
        txt = "該当が見つかりませんでした。\n" + refine
        return [TextSendMessage(text=p) for p in _split_for_line(txt, LINE_SAFE_CHARS)], False

    if len(hits) == 1:
        e = hits[0]
        return _format_entry_messages(e), True

    # 複数ヒット：候補リスト（タイトル優先順）
    lines = ["候補が複数見つかりました。店名で指定してください。"]
    for i, e in enumerate(hits[:8], 1):
        area = " / ".join(e.get("areas") or [])
        suffix = f"（{area}）" if area else ""
        lines.append(f"{i}. {e.get('title','')}{suffix}")

    refine, _ = build_refine_suggestions(question)
    if refine:
        lines.append("")
        lines.append(refine)

    txt = "\n".join(lines)
    return [TextSendMessage(text=p) for p in _split_for_line(txt, LINE_SAFE_CHARS)], True

@app.route("/admin/manual")
@login_required
def admin_manual():
    if session.get("role") != "admin":
        abort(403)
    return render_template("admin_manual.html")

# ======== ▼▼▼ ここから追記：テキストファイル管理（/admin/data_files） ▼▼▼ ========
from flask import send_from_directory

# ① 許可する文字クラスを拡張（全角括弧・句読点・記号などを許可）
ALLOWED_TXT_EXTS = {".txt", ".md"}

def _safe_txt_name(name: str) -> str:
    if not name:
        return ""
    name = os.path.basename(name)
    name = unicodedata.normalize("NFKC", name)
    # スラッシュを禁止（UIはDATA_DIR直下のみを一覧）
    name = re.sub(
        r'[^0-9A-Za-zぁ-んァ-ヶ一-龠々ー_\-\.\(\)\[\]（）【】「」『』・!！?？&＆:：;；,，.。 　]+',
        '',
        name
    )
    name = name.strip().lstrip(".")
    if not name:
        return ""
    root, ext = os.path.splitext(name)
    if not ext:
        name += ".txt"; ext = ".txt"
    if ext.lower() not in {".txt", ".md"}:
        return ""
    return (name[:120]).strip()

def _ensure_in_data_dir(path: str) -> bool:
    """DATA_DIR 配下に収まっているかの安全確認"""
    base = os.path.abspath(DATA_DIR)
    target = os.path.abspath(path)
    return target.startswith(base + os.sep) or target == base

def _read_text_any(path: str):
    encs = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "utf-16"]  # ← utf-16 を追加
    with open(path, "rb") as rf:
        raw = rf.read()
    for enc in encs:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # 最後の手段：置換ありでCP932
    return raw.decode("cp932", errors="replace"), "cp932(replace)"

def _write_text(path: str, text: str, encoding: str = "utf-8"):
    """
    テキストを書き出す。CRLF/CR を LF に正規化。
    encoding='cp932' を選ぶと日本語Windows互換保存（文字により失敗時はutf-8にフォールバック）
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path + ".tmp", "w", encoding=encoding, newline="\n") as wf:
            wf.write(text)
        os.replace(path + ".tmp", path)
        return encoding, None
    except UnicodeEncodeError as e:
        if encoding.lower() == "cp932":
            # CP932に載らない文字があった → UTF-8で保存して通知
            with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as wf:
                wf.write(text)
            os.replace(path + ".tmp", path)
            return "utf-8", f"一部の文字がCP932に無いため UTF-8 で保存しました（{e}）"
        else:
            raise

def _list_txt_files():
    files = []
    if not os.path.isdir(DATA_DIR):
        return files
    for fn in os.listdir(DATA_DIR):
        if fn.startswith("."):
            continue
        root, ext = os.path.splitext(fn)
        if ext.lower() not in ALLOWED_TXT_EXTS:
            continue
        fp = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(fp):
            continue
        st = os.stat(fp)
        files.append({
            "name": fn,
            "ext": (ext[1:].upper() if ext else ""),  # ← これをテンプレで使う
            "size": st.st_size,
            "mtime_ts": st.st_mtime,
        })
    files.sort(key=lambda x: x["mtime_ts"], reverse=True)
    for f in files:
        f["mtime"] = datetime.datetime.fromtimestamp(f["mtime_ts"]).strftime("%Y-%m-%d %H:%M:%S")
        del f["mtime_ts"]
    return files

def _answer_from_data_txt(question: str) -> str:
    """
    DATA_DIR 配下の .txt / .md をざっくりスキャンし、
    質問語にヒットしたファイルの冒頭～近傍を OpenAI で 300～400字に要約。
    ヒットなしなら空文字を返す。
    """
    try:
        files = _list_txt_files()
        if not files:
            return ""
        q = _norm_text_jp(question)
        # 2文字以上のキーワードだけ使う（日本語想定の超簡易スコア）
        kws = [w for w in re.split(r"[ \u3000、。・\n\r\t]", q) if len(w) >= 2]
        best = None
        best_score = 0
        for f in files:
            path = os.path.join(DATA_DIR, f["name"])
            txt, _enc = _read_text_any(path)
            tnorm = _norm_text_jp(txt)
            score = sum(tnorm.count(k) for k in kws) if kws else 0
            if score > best_score:
                best_score, best = score, txt[:5000]  # 長すぎると要約が重いので頭5kに制限
        if best_score <= 0 or not best:
            return ""
        # 要約（OpenAIなしでもそのまま返す）
        try:
            prompt = (
                "以下の資料から、質問者に役立つ部分だけを抽出して日本語で300～400字に要約してください。"
                "箇条書き可。URLや地名は残してください。\n"
                f"【質問】{question}\n---\n{best[:8000]}\n---"
            )
            out = openai_chat(
                OPENAI_MODEL_PRIMARY,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            ).strip()
            return out or best[:400]
        except Exception:
            return best[:400]
    except Exception:
        app.logger.exception("_answer_from_data_txt failed")
        return ""

# ② 例外を握ってログ＋フラッシュにして 500 を防ぐ
@app.route("/admin/data_files", methods=["GET"])
@login_required
def admin_data_files():
    if session.get("role") != "admin":
        abort(403)
    files = _list_txt_files()
    edit = (request.args.get("edit") or "").strip()
    content = ""
    used_enc = ""
    if edit:
        try:
            safe = _safe_txt_name(edit)
            if not safe:
                flash("不正なファイル名です")
                return redirect(url_for("admin_data_files"))
            path = os.path.join(DATA_DIR, safe)
            if not _ensure_in_data_dir(path) or not os.path.exists(path):
                flash("指定ファイルが見つかりません")
                return redirect(url_for("admin_data_files"))
            content, used_enc = _read_text_any(path)
        except Exception as e:
            app.logger.exception("[admin_data_files] load failed: %r -> %s", edit, e)
            flash("ファイル読込でエラーが発生しました。ダウンロードで内容確認か、ファイル名を短くして再試行してください。")
            return redirect(url_for("admin_data_files"))
    return render_template(
        "admin_data_files.html",
        files=files, edit=edit, content=content, used_enc=used_enc
    )

@app.route("/admin/data_files/upload", methods=["POST"])
@login_required
def admin_data_files_upload():
    if session.get("role") != "admin":
        abort(403)
    ups = request.files.getlist("txt_files")
    count = 0
    for f in ups:
        if not f or not f.filename:
            continue
        safe = _safe_txt_name(f.filename)
        if not safe:
            flash(f"スキップ: 不正なファイル名 {f.filename}")
            continue

        # 同名回避の連番付与
        base, ext = os.path.splitext(safe)
        candidate = safe
        i = 1
        while os.path.exists(os.path.join(DATA_DIR, candidate)):
            candidate = f"{base} ({i}){ext}"
            i += 1

        dst = os.path.join(DATA_DIR, candidate)
        if not _ensure_in_data_dir(dst):
            flash(f"スキップ: 保存先エラー {candidate}")
            continue

        data = f.read()
        if not data:
            flash(f"スキップ: 空ファイル {candidate}")
            continue

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(dst, "wb") as wf:
            wf.write(data)
        count += 1

    flash(f"{count} ファイルをアップロードしました")
    return redirect(url_for("admin_data_files"))

@app.route("/admin/data_files/new", methods=["POST"])
@login_required
def admin_data_files_new():
    if session.get("role") != "admin":
        abort(403)
    name = (request.form.get("new_name") or "").strip()
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です（拡張子は .txt / .md のみ）")
        return redirect(url_for("admin_data_files"))

    path = os.path.join(DATA_DIR, safe)
    if not _ensure_in_data_dir(path):
        flash("保存先が不正です")
        return redirect(url_for("admin_data_files"))

    if os.path.exists(path):
        flash("同名ファイルが既にあります")
        return redirect(url_for("admin_data_files"))

    _write_text(path, "", encoding="utf-8")
    flash("新規ファイルを作成しました")
    return redirect(url_for("admin_data_files", edit=safe))

@app.route("/admin/data_files/save", methods=["POST"])
@login_required
def admin_data_files_save():
    if session.get("role") != "admin":
        abort(403)

    name = (request.form.get("name") or "").strip()
    content = request.form.get("content") or ""
    encoding = (request.form.get("encoding") or "utf-8").strip()

    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files"))

    path = os.path.join(DATA_DIR, safe)
    if not _ensure_in_data_dir(path):
        flash("保存先エラー")
        return redirect(url_for("admin_data_files"))

    try:
        used_enc, note = _write_text(path, content, encoding=encoding)
        msg = f"保存しました（{used_enc}）"
        if note:
            msg += f" / {note}"
        flash(msg)
    except Exception as e:
        app.logger.exception("[admin_data_files_save] failed: %s", e)
        flash("保存に失敗しました")
    return redirect(url_for("admin_data_files", edit=safe))

@app.route("/admin/data_files/delete", methods=["POST"])
@login_required
def admin_data_files_delete():
    if session.get("role") != "admin":
        abort(403)
    name = (request.form.get("del_name") or "").strip()
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files"))
    path = os.path.join(DATA_DIR, safe)
    if not _ensure_in_data_dir(path) or not os.path.exists(path):
        flash("ファイルが見つかりません")
        return redirect(url_for("admin_data_files"))
    try:
        os.remove(path)
        flash("削除しました")
    except Exception as e:
        app.logger.exception("delete failed")
        flash("削除に失敗しました: " + str(e))
    return redirect(url_for("admin_data_files"))

@app.route("/admin/data_files/download")
@login_required
def admin_data_files_download():
    if session.get("role") != "admin":
        abort(403)
    name = (request.args.get("name") or "").strip()
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files"))
    # send_from_directory はパス検証込みで安全
    return send_from_directory(DATA_DIR, safe, as_attachment=True)

@app.route("/admin/data_files/rename", methods=["POST"])
@login_required
def admin_data_files_rename():
    if session.get("role") != "admin":
        abort(403)
    old = (request.form.get("old_name") or "").strip()
    new = (request.form.get("new_name") or "").strip()
    old_s = _safe_txt_name(old)
    new_s = _safe_txt_name(new)
    if not old_s or not new_s:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files", edit=old_s))
    src = os.path.join(DATA_DIR, old_s)
    dst = os.path.join(DATA_DIR, new_s)
    if not (_ensure_in_data_dir(src) and _ensure_in_data_dir(dst)) or not os.path.exists(src):
        flash("リネーム対象が見つかりません")
        return redirect(url_for("admin_data_files"))
    if os.path.exists(dst):
        flash("指定の新ファイル名は既に存在します")
        return redirect(url_for("admin_data_files", edit=old_s))
    os.rename(src, dst)
    flash("名前を変更しました")
    return redirect(url_for("admin_data_files", edit=new_s))
# ======== ▲▲▲ 追記ここまで ▲▲▲

@app.route("/_debug/quick")
def _debug_quick():
    # 本番はトークン必須にしたい場合は下を有効化
    # if APP_ENV in {"prod","production"} and request.headers.get("X-Debug-Token") != os.getenv("DEBUG_TOKEN"):
    #     abort(404)

    q = request.args.get("q", "") or ""
    t = _norm_text_jp(q)

    w_msg, w_ok   = get_weather_reply(q)
    tr_msg, tr_ok = get_transport_reply(q)

    def _hit(s, words):
        return [w for w in words if w in s]

    weather_words  = ["天気","天候","予報","weather"]
    primary_words  = ["運行","運航","運休","欠航","状況","情報","status"]
    vehicle_words  = ["船","フェリー","ジェットフォイル","高速船","太古","飛行機","空港","福江空港","五島つばき空港","ana","jal"]

    return jsonify({
        "q": q,
        "normalized": t,
        "weather": {
            "matched": bool(w_ok and w_msg),
            "kw_hit": _hit(t, weather_words),
            "message": w_msg,
        },
        "transport": {
            "matched": bool(tr_ok and tr_msg),
            "kw1_hit": _hit(t, primary_words),
            "kw2_hit": _hit(t, vehicle_words),
            "message": tr_msg,
        }
    })

# =========================
#  トップ/ヘルスチェック
# =========================
@app.route("/")
def home():
    return "<a href='/admin/entry'>[観光データ管理]</a>"


# =========================
#  天気・フェリー情報
# =========================
WEATHER_LINKS = [
    {"area": "五島市", "url": "https://weathernews.jp/onebox/tenki/nagasaki/42211/"},
    {"area": "新上五島町", "url": "https://weathernews.jp/onebox/tenki/nagasaki/42411/"},
    {"area": "小値賀町", "url": "https://tenki.jp/forecast/9/45/8440/42383/"},
    {"area": "宇久町", "url": "https://weathernews.jp/onebox/33.262381/129.131027/q=%E9%95%B7%E5%B4%8E%E7%9C%8C%E4%BD%90%E4%B8%96%E4%BF%9D%E5%B8%82%E5%AE%87%E4%B9%85%E7%94%BA&v=da56215a2617fc2203c6cae4306d5fd8c92e3e26c724245d91160a4b3597570a&lang=ja&type=week"}
]
FERRY_INFO = """【長崎ー五島航路 運行状況】
・野母商船「フェリー太古」運航情報
http://www.norimono-info.com/frame_set.php?usri=&disp=group&type=ship
・九州商船「フェリー・ジェットフォイル」運航情報
https://kyusho.co.jp/status
・五島産業汽船「フェリー」運航情報
https://www.goto-sangyo.co.jp/
その他の航路や詳細は各リンクをご覧ください。
"""


SMALLTALK_PATTERNS = {
    "greet": ["おはよう", "こんにちは", "こんばんは", "やあ", "はじめまして"],
    "howareyou": ["元気", "調子どう", "調子は？"],
    "thanks": ["ありがとう", "助かった", "サンキュー"],
    "capability": ["何ができる", "使い方", "ヘルプ", "help"]
}

def smalltalk_or_help_reply(q_ja: str):
    t = q_ja.strip()
    if any(k in t for k in SMALLTALK_PATTERNS["greet"]):
        return "こんにちは！ご旅行の予定や気分をざっくり教えてくれれば、ぴったりのスポットを探します。"
    if any(k in t for k in SMALLTALK_PATTERNS["howareyou"]):
        return "元気ですよ！今日は海・教会・展望・温泉・グルメなど、気分はどれですか？"
    if any(k in t for k in SMALLTALK_PATTERNS["thanks"]):
        return "どういたしまして！他にも聞きたいことがあればどうぞ。"
    if any(k in t for k in SMALLTALK_PATTERNS["capability"]):
        return (
            "できること：\n"
            "・スポット検索（例：『五島市のビーチ』『教会の見学』）\n"
            "・基本情報の回答（住所／地図／営業時間／駐車場など）\n"
            "・天気・航路リンク案内、未ヒット時の絞り込み提案\n"
            "気軽に話しかけてOK。会話から意図を読み取って探します。"
        )
    return None

# =========================
#  観光データ横断検索
# =========================
def clean_query_for_search(question):
    ignore_patterns = [
        r"(について)?教えて.*$", r"の情報.*$", r"の場所.*$", r"どこ.*$", r"案内.*$", r"を教えて.*$", r"を知りたい.*$", r"アクセス.*$", r"詳細.*$"
    ]
    q = question
    for pat in ignore_patterns:
        q = re.sub(pat, "", q)
    return q.strip()

def find_entry_info(question):
    entries = load_entries()
    synonyms = load_synonyms()
    areas_master = ["五島市", "新上五島町", "宇久町", "小値賀町"]
    target_areas = [area for area in areas_master if area in question]
    tags_from_syn = find_tags_by_synonym(question, synonyms)

    cleaned_query = clean_query_for_search(question)
    key_q = _title_key(cleaned_query)

    # 1. タイトル“正規化後”の完全一致
    hits = [e for e in entries
            if cleaned_query and _title_key(e.get("title","")) == key_q
            and (not target_areas or any(a in e.get("areas",[]) for a in target_areas))]
    if hits:
        return hits
    
    # 2. 各カラム部分一致（links/extras を含む）
    hits = []
    for e in entries:
        haystacks = [
            e.get("title", ""),
            " ".join(e.get("tags", [])),
            e.get("desc", ""),
            e.get("address", ""),
            " ".join(e.get("links", []) or []),
            " ".join((e.get("extras") or {}).values())
        ]
        target_str = " ".join(haystacks)
        if cleaned_query and cleaned_query in target_str:
            if not target_areas or any(area in e.get("areas", []) for area in target_areas):
                hits.append(e)
    if hits:
        return hits

    # 3. 元のtitle, desc部分一致
    hits = [e for e in entries if question in e.get("title", "") or question in e.get("desc", "")]
    if hits:
        return hits

    # 4. タグ一致・類義語一致
    hits = []
    for e in entries:
        tag_hit = (
            any(tag in question for tag in e.get("tags", [])) or
            any(tag in tags_from_syn for tag in e.get("tags", []))
        )
        if tag_hit and (not target_areas or any(area in e.get("areas", []) for area in target_areas)):
            hits.append(e)
    if hits:
        return hits
    return []

def ai_summarize(snippets, question, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    text = "\n\n".join([s.strip() for s in snippets if s and str(s).strip()])
    if not text:
        return ""
    prompt = (
        "以下の資料断片のみを根拠に、質問に答える日本語要約を300字以内で作成してください。"
        "推測や未確認情報は避け、根拠が無ければ『不明』と明記。"
        f"\n[質問]\n{question}\n[資料]\n{text[:4000]}"
    )
    try:
        return (openai_chat(model, [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=500) or "").strip()
    except Exception:
        app.logger.exception("ai_summarize failed")
        return text[:500]

def search_text_files(question, data_dir=DATA_DIR, max_snippets=5, window=80):
    """
    DATA_DIR 配下の .txt/.md をざっくり全文検索して、ヒット周辺の抜粋を返す。
    戻り値: list[str] or None
    """
    q = (question or "").strip()
    words = [w for w in re.split(r"[\s　,、。]+", q) if w]
    # 1文字語のノイズは除く
    words = [w for w in words if len(w) > 1]
    if not words:
        return None

    pat = re.compile("|".join(map(re.escape, words)), re.I)
    snippets = []

    for root, _, files in os.walk(data_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in {".txt", ".md"}:
                continue
            path = os.path.join(root, fn)
            try:
                text, _enc = _read_text_any(path)
            except Exception:
                continue

            for m in pat.finditer(text):
                start = max(0, m.start() - window)
                end   = min(len(text), m.end() + window)
                frag  = text[start:end].replace("\r\n", "\n").replace("\r", "\n").strip()
                if frag and frag not in snippets:
                    snippets.append(frag)
                if len(snippets) >= max_snippets:
                    return snippets
    return snippets

def fetch_and_search_web(question):
    return None

def format_entry_detail(e: dict) -> str:
    lines = []
    if e.get("title"):
        lines.append(f"◼︎ {e['title']}")
    if e.get("areas"):
        lines.append(f"エリア: {', '.join(e['areas'])}")
    if e.get("desc"):
        lines.append(f"説明: {e['desc']}")
    if e.get("address"):
        lines.append(f"住所: {e['address']}")
    if e.get("map"):
        lines.append(f"地図: {e['map']}")
    if e.get("tel"):
        lines.append(f"TEL: {e['tel']}")
    if e.get("open_hours"):
        lines.append(f"営業時間: {e['open_hours']}")
    if e.get("holiday"):
        lines.append(f"定休日: {e['holiday']}")
    if e.get("parking"):
        pn = f"（{e['parking_num']}台）" if e.get("parking_num") else ""
        lines.append(f"駐車場: {e['parking']}{pn}")
    if e.get("payment"):
        lines.append(f"支払い: {', '.join(e['payment'])}")
    if e.get("tags"):
        lines.append(f"タグ: {', '.join(e['tags'])}")
    if e.get("links"):
        lines.append("リンク:\n" + "\n".join(f"- {u}" for u in e["links"]))
    extras = e.get("extras") or {}
    for k, v in extras.items():
        if v:
            lines.append(f"{k}: {v}")
    if e.get("remark"):
        lines.append(f"備考: {e['remark']}")
    return "\n".join(lines)


def smart_search_answer_with_hitflag(question):
    meta = {"model_primary": OPENAI_MODEL_PRIMARY, "fallback": None}

    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        return weather_reply, True, meta
    
    # ★ここから追加：雑談/使い方
    st = smalltalk_or_help_reply(question)
    if st:
        meta["kind"] = "smalltalk"
        return st, True, meta


    if any(word in question for word in ["飛行機", "空港", "航空便", "欠航", "到着", "出発"]):
        return (
            "五島つばき空港の最新の運行状況は、公式Webサイトでご確認ください。\n"
            "▶ https://www.fukuekuko.jp/",
            True,
            meta,
        )

    if any(word in question for word in ["フェリー", "船", "運航", "ジェットフォイル", "太古"]):
        return FERRY_INFO, True, meta

    entries = find_entry_info(question)
    if entries:
        if len(entries) == 1:
            e = entries[0]
            lines = []
            if e.get("title"):      lines.append(f"◼︎ {e['title']}")
            if e.get("areas"):      lines.append(f"エリア: {', '.join(e['areas'])}")
            if e.get("desc"):       lines.append(f"説明: {e['desc']}")
            if e.get("address"):    lines.append(f"住所: {e['address']}")
            if e.get("tel"):        lines.append(f"TEL: {e['tel']}")
            if e.get("open_hours"): lines.append(f"営業時間: {e['open_hours']}")
            if e.get("holiday"):    lines.append(f"定休日: {e['holiday']}")
            if e.get("parking"):    lines.append(f"駐車場: {e['parking']}（台数: {e.get('parking_num','')}）")
            if e.get("payment"):    lines.append("支払い: " + " / ".join(e['payment']))
            if e.get("extras"):
                for k, v in (e.get("extras") or {}).items():
                    if v: lines.append(f"{k}: {v}")
            if e.get("map"):        lines.append(f"地図: {e['map']}")
            if e.get("links"):      lines.append("リンク: " + " / ".join(e['links']))
            return "\n".join(lines), True, meta
        
        else:
            # 複数ヒット時：まず要約を試す → 失敗/空なら「短い候補一覧＋深掘り質問」にフォールバック
            try:
                snippets = [
                    f"タイトル: {e.get('title','')}\n説明: {e.get('desc','')}\n住所: {e.get('address','')}\n"
                    for e in entries
                ]
                ai_ans = ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY)
                # 生成が空/極端に短い場合は失敗扱いにする
                if not ai_ans or len(ai_ans.strip()) < 30:
                    raise RuntimeError("summarize_failed_or_too_short")
                return ai_ans, True, meta
            except Exception:
                # --- フォールバック：短い候補一覧（最大8件）＋意図深掘り ---
                max_list = 8
                short = entries[:max_list]

                lines = ["候補が複数見つかりました。気になるものはありますか？"]
                for i, e in enumerate(short, 1):
                    name = (e.get("title") or "（無題）").strip()
                    areas = ", ".join((e.get("areas") or []))
                    suffix = f"（{areas}）" if areas else ""
                    lines.append(f"{i}. {name}{suffix}")

                remaining = len(entries) - len(short)
                if remaining > 0:
                    lines.append(f"…ほか {remaining} 件")

                # 絞り込み候補＋深掘り質問（build_refine_suggestions は近くで定義済み）
                try:
                    suggest_text, suggest_meta = build_refine_suggestions(question)
                    meta["suggestions"] = suggest_meta
                    lines.append("")
                    lines.append(suggest_text)  # ※probe 質問も含めて返る実装にしていればここで出ます
                except Exception:
                    # build_refine_suggestions が未改修でも最低限の深掘りだけ添える
                    lines.append("")
                    lines.append("❓ もう少し教えてください")
                    lines.append("- 出発エリアは？（五島市／新上五島町／小値賀町／宇久町）")
                    lines.append("- どんな雰囲気？（海岸線／教会巡り／展望台／夕日）")
                    lines.append("- 所要時間は？（2〜3時間／半日／1日）")
                    lines.append("- お車はありますか？（あり／なし）")

                msg = "\n".join(lines)
                return msg, False, meta

    snippets = search_text_files(question, data_dir=DATA_DIR)
    if snippets:
        try:
            return ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY), True, meta
        except Exception as e:
            return "AI要約でエラー: " + str(e), False, meta

    web_texts = fetch_and_search_web(question)
    if web_texts:
        try:
            return ai_summarize(web_texts, question, model=OPENAI_MODEL_PRIMARY), True, meta
        except Exception as e:
            return "Web要約でエラー: " + str(e), False, meta

    suggest_text, suggest_meta = ("", {})
    if SUGGEST_UNHIT:
        suggest_text, suggest_meta = build_refine_suggestions(question)
        meta["suggestions"] = suggest_meta

    if REASK_UNHIT:
        fallback_text = ai_suggest_faq(question, model=OPENAI_MODEL_HIGH)
        if fallback_text:
            meta["fallback"] = {"mode": "high_faq", "model": OPENAI_MODEL_HIGH}
            answer = "【参考回答（データ未登録のため要確認）】\n" + fallback_text
            if suggest_text:
                answer += "\n---\n" + suggest_text
            return answer, False, meta

        if suggest_text:
            return ("申し訳ありません。該当データが見つかりませんでした。\n" + suggest_text, False, meta)

        return ("申し訳ありません。現在この質問には確実な情報を持っていません。", False, meta)

    if suggest_text:
        return ("申し訳ありません。該当データが見つかりませんでした。\n" + suggest_text, False, meta)

    return ("申し訳ありません。現在この質問には確実な情報を持っていません。", False, meta)


# =========================
#  API: /ask
# =========================
@app.route("/ask", methods=["POST"])
@limit_deco(ASK_LIMITS) 
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        save_qa_log(question, weather_reply, source="web", hit_db=True, extra={"kind": "weather"})
        return jsonify({"answer": weather_reply, "hit_db": True, "meta": {"kind": "weather"}})

    trans_reply, trans_hit = get_transport_reply(question)
    if trans_hit:
        save_qa_log(question, trans_reply, source="web", hit_db=True, extra={"kind": "transport"})
        return jsonify({"answer": trans_reply, "hit_db": True, "meta": {"kind": "transport"}})

    if not question:
        return jsonify({"error": "質問が空です"}), 400

    orig_lang = detect_lang_simple(question)
    lang = orig_lang
    if not ENABLE_FOREIGN_LANG:
        lang = "ja"

    q_for_logic = question if orig_lang == "ja" else translate_text(question, "ja")
    answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)

    if lang == "ja":
        answer = answer_ja
    else:
        target = "zh-Hant" if lang == "zh-Hant" else "en"
        answer = translate_text(answer_ja, target)

    save_qa_log(question, answer, source="web", hit_db=hit_db, extra=meta)
    return jsonify({"answer": answer, "hit_db": hit_db, "meta": meta})


@app.route("/admin/unhit_report")
@login_required
def admin_unhit_report():
    if session.get("role") != "admin":
        abort(403)
    report = generate_unhit_report(7)
    return render_template("admin_unhit_report.html", unhit_report=report)


_LINE_THROTTLE = {}  # ループ暴走の最終安全弁（会話ごとの最短インターバル）

def _send_messages(event, messages):
    """テキスト/画像など複合メッセージ送信（出典フッターは line_bot_api のプロキシで付与されます）"""
    if not _line_enabled() or not line_bot_api:
        for m in messages:
            app.logger.info("[LINE disabled] would send: %s", getattr(m, "text", "(non-text)"))
        # ← 確定ログ（ドライラン扱い）
        try: _log_sent_messages(event, messages, status="dryrun")
        except Exception: pass
        return

    status = "noop"
    try:
        rt = getattr(event, "reply_token", None)
        if rt:
            line_bot_api.reply_message(rt, messages)
            status = "replied"
        else:
            tid = _line_target_id(event)
            if tid:
                line_bot_api.push_message(tid, messages)
                status = "pushed"
            else:
                status = "noop"
    except LineBotApiError as e:
        global LAST_SEND_ERROR, SEND_ERROR_COUNT
        SEND_ERROR_COUNT += 1
        LAST_SEND_ERROR = f"{type(e).__name__}: {e}"
        app.logger.exception("LINE send failed: %s", e)
        # ← 失敗時もログ
        try: _log_sent_messages(event, messages, status="error")
        except Exception: pass
        return
    except Exception:
        app.logger.exception("LINE send unexpected error")
        try: _log_sent_messages(event, messages, status="error")
        except Exception: pass
        return

    # ← 成功時の確定ログ
    try: _log_sent_messages(event, messages, status=status)
    except Exception: pass

def _notice_paused_once(event, target_id: str) -> bool:
    """
    全体一時停止中に『一度だけ案内』を送る。
    同じ target_id には TTL 以内は再送しない。
    戻り値: 送ったら True、スキップなら False
    """
    try:
        cache = _load_paused_notices()
    except Exception:
        cache = {}

    now = time.time()
    last = float(cache.get(target_id) or 0)
    if (now - last) < PAUSED_NOTICE_TTL_SEC:
        # まだTTL内なので今回は案内しない
        return False

    # メッセージ文面（ALLOW_RESUME_WHEN_PAUSED によって注記を変える）
    if ALLOW_RESUME_WHEN_PAUSED:
        text = (
            "現在は『全体一時停止』中です。管理者が解除するまで返信は原則止まります。\n"
            "※ この会話のミュート解除は「再開」で可能ですが、全体停止中は返信が届かないことがあります。"
        )
    else:
        text = (
            "現在は『全体一時停止』中です。管理者が再開するまで返信は停止します。\n"
            "※ この会話のミュート解除は「再開」で可能ですが、全体停止中は返信は戻りません。"
        )

    # 送信（_reply_or_push があれば優先。なければ _send_messages にフォールバック）
    try:
        if "_reply_or_push" in globals() and callable(_reply_or_push):
            _reply_or_push(event, text)
        else:
            parts = _split_for_line(text, LINE_SAFE_CHARS)
            msgs = [TextSendMessage(text=p) for p in parts]
            _send_messages(event, msgs)
    except Exception:
        app.logger.exception("paused notice send failed")

    # 送った印として時刻を保存
    try:
        cache[target_id] = now
        _save_paused_notices(cache)
    except Exception:
        app.logger.exception("paused notice save failed")

    return True


# --- LINE 送信先IDの取得ヘルパー（_line_target_id もありますが、こちらは on-demand で使う場合用） ---
def _target_id_from_event(event):
    return (
        getattr(event.source, "user_id", None)
        or getattr(event.source, "group_id", None)
        or getattr(event.source, "room_id", None)
    )


# ===== 送信ユーティリティ（長文分割・複数送信） =====
from linebot.models import TextSendMessage

LINE_MAX_PER_REQUEST = 5  # 1リクエスト最大5メッセージ（※ LINE_SAFE_CHARS は上位の正規版を使用）

def _split_for_messaging(text: str, chunk_size: int = None) -> List[str]:
    """長文を自然な区切り（段落→句点）で2〜複数通に分割。最低1通は返す。"""
    # 上位で定義済みの安全値を使う
    lim = int(chunk_size) if chunk_size else int(globals().get("LINE_SAFE_CHARS", 3800))
    if not text:
        return [""]
    text = text.strip()
    if len(text) <= lim:
        return [text]

    parts: List[str] = []
    rest = text
    while len(rest) > lim:
        cut = rest.rfind("\n\n", 0, lim)
        if cut < int(lim * 0.5):
            cut = rest.rfind("。", 0, lim)
        if cut < int(lim * 0.5):
            cut = lim
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


# === LINE 返信ユーティリティ（安全送信 + エラー計測）
# ※ 正規版 _reply_or_push（force_push 対応／出典フッター対応）は既に上で定義済みなのでここでは定義しません。


def _push_multi_by_id(target_id: str, texts, *, reqgen: int | None = None):
    """複数テキストを順に push。重複抑止＋世代ガード付き。"""
    if not texts:
        return
    for t in texts:
        if not t:
            continue
        for ch in _split_for_line(t, max_len=4900):
            # 世代ガード：新しいユーザー発話が来て世代が進んでいれば以降は送らない
            if (reqgen is not None) and (REQUEST_GENERATION.get(target_id, 0) != reqgen):
                app.logger.info("abort stale push uid=%s", target_id)
                return
            # 直近重複の抑止
            if _was_sent_recent(target_id, ch, mark=False):
                app.logger.info("skip dup push uid=%s", target_id)
                continue
            try:
                line_bot_api.push_message(target_id, TextSendMessage(text=ch))
                _was_sent_recent(target_id, ch, mark=True)
            except LineBotApiError as e:
                app.logger.warning("push error: %s", e)
                return
            time.sleep(0.1)


# --- 最終回答を計算して push する非同期処理（環境に応じて使用。未使用なら残っていても無害） ---
def _compute_and_push_async(event, user_message: str, reqgen=None):
    from linebot.models import TextSendMessage
    target_id = _target_id_from_event(event)
    if not target_id:
        return
    try:
        # 言語判定 → ロジックは既存関数をそのまま利用
        orig_lang = detect_lang_simple(user_message)
        lang = orig_lang if ENABLE_FOREIGN_LANG else "ja"
        q_for_logic = user_message if orig_lang == "ja" else translate_text(user_message, "ja")

        answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)
        answer = answer_ja if lang == "ja" else translate_text(answer_ja, "zh-Hant" if lang == "zh-Hant" else "en")

        save_qa_log(user_message, answer, source="line", hit_db=hit_db, extra=meta)
        texts = _split_for_messaging(answer)
        _push_multi_by_id(target_id, texts, reqgen=reqgen)
    except Exception as e:
        app.logger.exception("compute/push failed: %s", e)
        try:
            line_bot_api.push_message(target_id, TextSendMessage(text="検索中にエラーが発生しました。もう一度お試しください。"))
        except Exception:
            pass


# =========================
#  お知らせ管理
# =========================
@app.route("/admin/notices", methods=["GET", "POST"])
@login_required
def admin_notices():
    # 必要なら _require_admin() を使ってもOK
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))
    notices = load_notices()
    edit_id = request.values.get("edit")
    edit_notice = None

    if edit_id:
        for n in notices:
            if str(n.get("id")) == str(edit_id):
                edit_notice = n
                break

    if request.method == "POST":
        category = request.form.get("category")
        title = request.form.get("title")
        content = request.form.get("content")
        date = datetime.date.today().isoformat()
        image_url = request.form.get("image_url", "")

        if category in ["イベント", "特売"]:
            start_date = request.form.get("start_date", "")
            end_date = request.form.get("end_date", "")
            expire_date = ""
        else:
            start_date = ""
            end_date = ""
            expire_date = request.form.get("expire_date", "")

        if request.form.get("edit_id"):
            for i, n in enumerate(notices):
                if str(n.get("id")) == str(request.form.get("edit_id")):
                    notices[i].update({
                        "title": title,
                        "content": content,
                        "category": category,
                        "start_date": start_date,
                        "end_date": end_date,
                        "expire_date": expire_date,
                        "image_url": image_url
                    })
                    flash("お知らせを更新しました")
                    break
        else:
            notice = {
                "id": notices[-1]["id"] + 1 if notices else 1,
                "title": title,
                "content": content,
                "category": category,
                "date": date,
                "start_date": start_date,
                "end_date": end_date,
                "expire_date": expire_date,
                "image_url": image_url,
                "created_by": session["user_id"]
            }
            notices.append(notice)
            flash("お知らせを追加しました")
        save_notices(notices)
        return redirect(url_for("admin_notices"))
    return render_template("admin_notices.html", notices=notices, edit_notice=edit_notice)


@app.route("/admin/notices/delete/<int:idx>", methods=["POST"])
@login_required
def delete_notice(idx):
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))
    notices = load_notices()
    notices = [n for n in notices if n.get("id") != idx]
    save_notices(notices)
    flash("お知らせを削除しました")
    return redirect(url_for("admin_notices"))


@app.route("/notices")
def notices():
    notices = load_notices()
    return render_template("notices.html", notices=notices)


# ===== 正規化の後付けパッチ（画像系フィールドが消えるのを防ぐ）=====
# 既存の _norm_entry を壊さず、結果を補正するラッパーです。
# 貼り付け位置：ファイル末尾の「メイン起動」直前が安全
try:
    if '_norm_entry' in globals() and callable(_norm_entry):
        __orig_norm_entry = _norm_entry  # 退避

        def _norm_entry(entry):
            # まず元の正規化を実行
            try:
                e = __orig_norm_entry(entry)
            except Exception:
                # 万一でも壊さない
                e = {}
            if not isinstance(e, dict):
                try:
                    e = dict(e or {})
                except Exception:
                    e = {}

            # ---- 画像フィールドの保全（image_file / image）----
            src_img = None
            try:
                for k in ('image_file', 'image'):
                    v = (entry or {}).get(k)
                    if isinstance(v, str) and v.strip():
                        src_img = v.strip()
                        break
            except Exception:
                pass

            # 正規化後に消えていたら復元。どちらか一方しか無ければ揃える
            if src_img:
                if not (e.get('image_file') or e.get('image')):
                    e['image_file'] = src_img
                    e['image'] = src_img
                else:
                    cur = (e.get('image_file') or e.get('image') or '').strip()
                    if cur:
                        e['image_file'] = cur
                        e['image'] = cur

            # ---- 透かし指定の保全（wm_external_choice / wm_on）----
            wm_choice_in = (entry or {}).get('wm_external_choice')
            if isinstance(wm_choice_in, str) and wm_choice_in.strip():
                e['wm_external_choice'] = wm_choice_in.strip().lower()
            elif 'wm_external_choice' not in e:
                # 旧UIの wm_on から導出（後方互換）
                wm_on_in = (entry or {}).get('wm_on')
                if isinstance(wm_on_in, bool):
                    e['wm_external_choice'] = 'fully' if wm_on_in else 'none'

            if isinstance((entry or {}).get('wm_on'), bool) and 'wm_on' not in e:
                e['wm_on'] = bool((entry or {}).get('wm_on'))

            # ---- lat/lng の取りこぼし救済（任意）----
            for k in ('lat', 'lng'):
                v_in = (entry or {}).get(k)
                if (k not in e or e.get(k) in (None, "")) and v_in not in (None, ""):
                    try:
                        e[k] = float(v_in)
                    except Exception:
                        e[k] = v_in

            return e
except Exception:
    # ここで失敗してもアプリ動作は継続
    app.logger.exception("norm-entry post patch failed")
# ===== ここまでパッチ =====
# ... 全ての @app.route(...) 定義が終わった一番最後に置く
from watermark_ext import init_watermark_ext
init_watermark_ext(app)


# メイン起動（重複禁止：これ1つだけ残す）
if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port, debug=(APP_ENV not in {"prod","production"}))
