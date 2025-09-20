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
