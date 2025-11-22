import base64
import datetime
import json
import os
import secrets
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, current_app, request, session, url_for
from werkzeug.routing import BuildError
from werkzeug.security import generate_password_hash

from admin_pamphlets import bp as admin_pamphlets_bp
from admin_rollback import bp as admin_rollback_bp
from config import get_config
from coreapp.storage import (
    BASE_DIR as STORAGE_BASE_DIR,
    ensure_dirs,
    migrate_from_legacy_paths,
    seed_from_repo_if_empty,
)
from services import pamphlet_search
from services.paths import ensure_data_directories, get_data_base_dir

load_dotenv()

MEDIA_ROOT: str | None = None
IMAGES_DIR: str | None = None
MEDIA_DIR: Path | None = None
MAX_UPLOAD_MB: int = 64
BASE_DIR: str = ""
ENTRIES_FILE: str = ""
DATA_DIR: str = ""
LOG_DIR: str = ""
LOG_FILE: str = ""
SYNONYM_FILE: str = ""
USERS_FILE: str = ""
NOTICES_FILE: str = ""
SHOP_INFO_FILE: str = ""
PAUSED_NOTICE_FILE: str = ""
SEND_LOG_FILE: str = ""
_DATA_PATHS_INITIALIZED = False
_APP_BOOTSTRAPPED = False


def _configure_data_paths(flask_app: Flask) -> None:
    global _DATA_PATHS_INITIALIZED
    if _DATA_PATHS_INITIALIZED:
        return
    global BASE_DIR, ENTRIES_FILE, DATA_DIR, LOG_DIR, LOG_FILE
    global SYNONYM_FILE, USERS_FILE, NOTICES_FILE, SHOP_INFO_FILE
    global PAUSED_NOTICE_FILE, SEND_LOG_FILE

    base_path = get_data_base_dir(flask_app.config)
    BASE_DIR = str(base_path)
    flask_app.config["DATA_BASE_DIR"] = BASE_DIR
    ENTRIES_FILE = os.path.join(BASE_DIR, "entries.json")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    LOG_FILE = os.path.join(LOG_DIR, "questions_log.jsonl")
    SYNONYM_FILE = os.path.join(BASE_DIR, "synonyms.json")
    USERS_FILE = os.path.join(BASE_DIR, "users.json")
    NOTICES_FILE = os.path.join(BASE_DIR, "notices.json")
    SHOP_INFO_FILE = os.path.join(BASE_DIR, "shop_infos.json")
    PAUSED_NOTICE_FILE = os.path.join(BASE_DIR, "paused_notice.json")
    SEND_LOG_FILE = os.path.join(LOG_DIR, "send_log.jsonl")
    global IMAGES_DIR
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    flask_app.config["IMAGES_DIR"] = IMAGES_DIR

    ensure_data_directories(
        Path(BASE_DIR),
        pamphlet_dir=flask_app.config.get("PAMPHLET_BASE_DIR"),
    )

    _DATA_PATHS_INITIALIZED = True


def _ensure_json(path: str, default_obj: Any) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f, ensure_ascii=False, indent=2)


ADMIN_INIT_USER = os.environ.get("ADMIN_INIT_USER", "admin")
ADMIN_INIT_PASSWORD = os.environ.get("ADMIN_INIT_PASSWORD")


def _bootstrap_files_and_admin(app: Flask) -> None:
    app.logger.info(f"[boot] BASE_DIR={BASE_DIR}")
    app.logger.info(f"[boot] USERS_FILE path: {USERS_FILE}")

    _ensure_json(ENTRIES_FILE, [])
    _ensure_json(SYNONYM_FILE, {})
    _ensure_json(NOTICES_FILE, [])
    _ensure_json(SHOP_INFO_FILE, {})

    users: list[dict[str, str]] = []
    users_exists = os.path.exists(USERS_FILE)
    if users_exists:
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = []

    if not users_exists or not users:
        if ADMIN_INIT_PASSWORD:
            users = [
                {
                    "user_id": ADMIN_INIT_USER,
                    "name": "管理者",
                    "password_hash": generate_password_hash(ADMIN_INIT_PASSWORD),
                    "role": "admin",
                }
            ]
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            app.logger.warning(
                "users.json を新規作成し、管理者ユーザー '%s' を作成しました。初回ログイン後 ADMIN_INIT_PASSWORD を環境変数から削除してください。",
                ADMIN_INIT_USER,
            )
        else:
            factory_password = secrets.token_urlsafe(16)
            factory_user = ADMIN_INIT_USER or "admin"
            users = [
                {
                    "user_id": factory_user,
                    "name": "工場出荷管理者",
                    "password_hash": generate_password_hash(factory_password),
                    "role": "admin",
                }
            ]
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            init_path = Path(USERS_FILE).with_suffix(Path(USERS_FILE).suffix + ".init")
            init_payload = {
                "user_id": factory_user,
                "password": factory_password,
                "note": "初回ログイン後にこのファイルを削除し、パスワードを変更してください。",
            }
            try:
                init_path.write_text(json.dumps(init_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                app.logger.warning("failed to write factory credential hint to %s", init_path, exc_info=True)
            app.logger.warning(
                "users.json が見つかりませんでした。工場出荷管理者を作成し、初期認証情報を %s に出力しました。",
                init_path,
            )


def create_app() -> Flask:
    global MEDIA_ROOT, IMAGES_DIR, MEDIA_DIR, MAX_UPLOAD_MB, _APP_BOOTSTRAPPED

    app = Flask(__name__)

    ensure_dirs()
    _bootstrap_sentinel = Path(STORAGE_BASE_DIR / ".bootstrap.done")
    if not _bootstrap_sentinel.exists():
        migrate_from_legacy_paths()
        seed_from_repo_if_empty()
        try:
            _bootstrap_sentinel.write_text("ok", encoding="utf-8")
        except OSError:
            app.logger.warning("Failed to write bootstrap sentinel", exc_info=True)

    @app.template_global()
    def safe_url_for(endpoint: str, **values: Any) -> str:
        try:
            return url_for(endpoint, **values)
        except BuildError:
            return ""

    app.config.from_object(get_config())

    MEDIA_ROOT = os.getenv("MEDIA_ROOT") or app.config.get("MEDIA_ROOT") or MEDIA_ROOT
    IMAGES_DIR = os.getenv("IMAGES_DIR") or app.config.get("IMAGES_DIR") or MEDIA_ROOT
    MEDIA_ROOT = IMAGES_DIR
    app.config["MEDIA_ROOT"] = MEDIA_ROOT
    app.config["IMAGES_DIR"] = IMAGES_DIR
    app.config["MEDIA_DIR"] = os.getenv("MEDIA_DIR") or app.config.get("MEDIA_DIR") or MEDIA_ROOT
    MEDIA_DIR = Path(app.config["MEDIA_DIR"]).expanduser()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    @app.template_filter("b64encode")
    def jinja_b64encode(s: object) -> str:
        if s is None:
            return ""
        if not isinstance(s, (bytes, bytearray)):
            s = str(s).encode("utf-8")
        return base64.b64encode(s).decode("ascii")

    def _ensure_csrf_token() -> str:
        tok = session.get("_csrf_token")
        if not tok:
            tok = secrets.token_urlsafe(32)
            session["_csrf_token"] = tok
        return tok

    @app.context_processor
    def _inject_csrf_token() -> dict[str, object]:
        return dict(csrf_token=_ensure_csrf_token)

    BUILD_ID = os.getenv("BUILD_ID") or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    @app.context_processor
    def _inject_build_id() -> dict[str, object]:
        return dict(BUILD_ID=BUILD_ID)

    @app.after_request
    def _no_cache_admin(resp):
        p = (request.path or "")
        if p.startswith(("/admin", "/notices")):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp

    @app.context_processor
    def _inject_template_helpers() -> dict[str, object]:
        def has_endpoint(name: str) -> bool:
            try:
                return name in app.view_functions
            except Exception:
                return False

        return {
            "current_app": current_app,
            "has_endpoint": has_endpoint,
        }

    MAX_UPLOAD_MB = app.config.get("MAX_UPLOAD_MB", 64)

    if not _APP_BOOTSTRAPPED:
        _configure_data_paths(app)
        if "admin_pamphlets" not in app.blueprints:
            app.register_blueprint(admin_pamphlets_bp)
        if "admin_rollback" not in app.blueprints:
            app.register_blueprint(admin_rollback_bp)
        pamphlet_search.configure(app.config)
        try:
            pamphlet_search.load_all()
        except Exception:
            app.logger.exception("[pamphlet] initial load failed")
        _bootstrap_files_and_admin(app)
        _APP_BOOTSTRAPPED = True

    return app
