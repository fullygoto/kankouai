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
# 追加が必要な標準ライブラリ
import shutil      # _has_free_space で disk_usage を使用
import tarfile     # _stream_backup_targz で使用
import queue       # ストリーミング用の内部キュー
from pathlib import Path
from functools import wraps
from collections import Counter
from typing import Any, Dict, List
import urllib.parse as _u  # ← _extract_wm_flags などで使用
from difflib import get_close_matches  # ★追加
import secrets          # ← これを追加


# === Third-Party ===
from dotenv import load_dotenv
load_dotenv()

from flask import (
    Flask, render_template, request, redirect, url_for, flash, session,
    jsonify, send_file, abort, send_from_directory, render_template_string, Response,
    current_app as _flask_current_app,  # ← これを追加


)

# --- Flask app instance (top-level; required for `from app import app`) ---
app = Flask(__name__)
try:
    app.secret_key  # already set?
except Exception:
    app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "change-me"


# --- login_required 互換デコレータ（flask_login が無い場合のフォールバック） ---
    # Pillow のランチョス補間（無ければ BICUBIC）
# ==== TEMP: health endpoints & app-factory shim ====
try:
    # 既存 app が未定義なら作る
    app
except NameError:
    from flask import Flask
    app = Flask(__name__)

from flask import jsonify

@app.get("/healthz")
def _healthz():
    return "ok", 200

@app.get("/readyz")
def _readyz():
    import os, pathlib
    errors = []
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("missing_env:OPENAI_API_KEY")
    base = pathlib.Path(os.getenv("DATA_BASE_DIR", "."))
    users = base / "users.json"
    if not users.exists():
        errors.append(f"missing_file:{users}")
    elif not users.is_file():
        errors.append(f"not_a_file:{users}")
    status = 200 if not errors else 503
    payload = {
        "ok": not errors,
        "errors": errors,
        "env": os.getenv("APP_ENV", "dev"),
        "version": os.getenv("APP_VERSION", "dev"),
    }
    return jsonify(payload), status

def create_app():
    # 後で本格対応に差し替える。いまは既存 app を返すだけ
    return app
# ==== /TEMP ====
