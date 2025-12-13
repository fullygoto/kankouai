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

import coreapp.storage as storage
from services import pamphlet_search
from services.paths import ensure_data_directories, get_data_base_dir

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

MEDIA_ROOT: str | None = None
IMAGES_DIR: str | None = None
MEDIA_DIR: Path | None = None
MAX_UPLOAD_MB: int = 64

# 互換用（他モジュールが参照していても壊れないよう残す）
DATA_BASE_DIR: str = ""
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

ADMIN_INIT_USER = os.environ.get("ADMIN_INIT_USER", "admin")
ADMIN_INIT_PASSWORD = os.environ.get("ADMIN_INIT_PASSWORD")


def _configure_data_paths(flask_app: Flask) -> None:
    """
    テストで monkeypatch される前提なので、create_app() 毎に必ず計算して同期する。
    """
    global DATA_BASE_DIR, ENTRIES_FILE, DATA_DIR, LOG_DIR, LOG_FILE
    global SYNONYM_FILE, USERS_FILE, NOTICES_FILE, SHOP_INFO_FILE
    global PAUSED_NOTICE_FILE, SEND_LOG_FILE, IMAGES_DIR

    base_path = get_data_base_dir(flask_app.config)
    DATA_BASE_DIR = str(base_path)
    flask_app.config["DATA_BASE_DIR"] = DATA_BASE_DIR

    # storage 側も env を優先して読むようにしてあるので、ここで env にも同期して“ズレ”を防ぐ
    os.environ["DATA_BASE_DIR"] = DATA_BASE_DIR
    if flask_app.config.get("PAMPHLET_BASE_DIR"):
        os.environ["PAMPHLET_BASE_DIR"] = str(flask_app.config["PAMPHLET_BASE_DIR"])

    ENTRIES_FILE = os.path.join(DATA_BASE_DIR, "entries.json")
    DATA_DIR = os.path.join(DATA_BASE_DIR, "data")
    LOG_DIR = os.path.join(DATA_BASE_DIR, "logs")
    LOG_FILE = os.path.join(LOG_DIR, "questions_log.jsonl")
    SYNONYM_FILE = os.path.join(DATA_BASE_DIR, "synonyms.json")
    USERS_FILE = os.path.join(DATA_BASE_DIR, "users.json")
    NOTICES_FILE = os.path.join(DATA_BASE_DIR, "notices.json")
    SHOP_INFO_FILE = os.path.join(DATA_BASE_DIR, "shop_infos.json")
    PAUSED_NOTICE_FILE = os.path.join(DATA_BASE_DIR, "paused_notice.json")
    SEND_LOG_FILE = os.path.join(LOG_DIR, "send_log.jsonl")

    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    flask_app.config["IMAGES_DIR"] = IMAGES_DIR

    ensure_data_directories(
        Path(DATA_BASE_DIR),
        pamphlet_dir=flask_app.config.get("PAMPHLET_BASE_DIR"),
    )


def _ensure_json(path: str, default_obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps(default_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _bootstrap_files_and_admin(app: Flask) -> None:
    app.logger.info("[boot] DATA_BASE_DIR=%s", DATA_BASE_DIR)
    app.logger.info("[boot] USERS_FILE=%s", USERS_FILE)

    _ensure_json(ENTRIES_FILE, [])
    _ensure_json(SYNONYM_FILE, {})
    _ensure_json(NOTICES_FILE, [])
    _ensure_json(SHOP_INFO_FILE, {})

    users: list[dict[str, str]] = []
    users_exists = Path(USERS_FILE).exists()
    if users_exists:
        try:
            users = json.loads(Path(USERS_FILE).read_text(encoding="utf-8"))
        except Exception:
            users = []

    if (not users_exists) or (not users):
        if ADMIN_INIT_PASSWORD:
            users = [
                {
                    "user_id": ADMIN_INIT_USER,
                    "name": "管理者",
                    "password_hash": generate_password_hash(ADMIN_INIT_PASSWORD),
                    "role": "admin",
                }
            ]
            Path(USERS_FILE).write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
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
            Path(USERS_FILE).write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

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


def _log_environment_config(app: Flask) -> None:
    important_env_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "COHERE_API_KEY",
        "GOOGLE_API_KEY",
        "RATE_STORAGE_URI",
        "RATE_STORAGE_URL",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "DATABASE_URL",
        "POSTGRES_URL",
        "PORT",
    ]

    present_keys = [key for key in important_env_keys if os.getenv(key)]
    app.logger.info(
        "startup.env_keys_present=%s",
        ",".join(sorted(present_keys)) if present_keys else "none",
    )

    app.logger.info(
        "startup.config PORT=%s DATA_BASE_DIR=%s PAMPHLET_BASE_DIR=%s RATE_STORAGE_URI_set=%s RATE_STORAGE_URL_set=%s UPSTASH_REDIS_REST_URL_set=%s",
        os.getenv("PORT") or app.config.get("PORT") or "5000",
        app.config.get("DATA_BASE_DIR"),
        app.config.get("PAMPHLET_BASE_DIR"),
        bool(os.getenv("RATE_STORAGE_URI") or app.config.get("RATE_STORAGE_URI")),
        bool(os.getenv("RATE_STORAGE_URL") or app.config.get("RATE_STORAGE_URL")),
        bool(os.getenv("UPSTASH_REDIS_REST_URL")),
    )


def _bootstrap_storage_once(app: Flask) -> None:
    """
    pamphlets/backups などの初回seed/legacy移行。
    sentinel書き込み前に親ディレクトリを必ず作る。
    """
    storage.ensure_dirs()

    base_dir = Path(storage.BASE_DIR)
    sentinel = base_dir / ".bootstrap.done"

    if sentinel.exists():
        return

    storage.migrate_from_legacy_paths()
    storage.seed_from_repo_if_empty()

    try:
        if base_dir.exists() and not base_dir.is_dir():
            app.logger.warning("DATA_BASE_DIR exists but is not a directory; skipping bootstrap sentinel: %s", base_dir)
            return
        base_dir.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("ok", encoding="utf-8")
    except OSError:
        app.logger.warning("Failed to write bootstrap sentinel", exc_info=True)


def create_app() -> Flask:
    global MEDIA_ROOT, IMAGES_DIR, MEDIA_DIR, MAX_UPLOAD_MB

    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    # 先に config を入れる（PAMPHLET_BASE_DIR などの値を確定させる）
    app.config.from_object(get_config())

    # ★ テストごとに必ず data paths を再計算（ここが超重要）
    _configure_data_paths(app)

    # storage の seed/legacy（DATA_BASE_DIR を env 同期済みなのでズレない）
    _bootstrap_storage_once(app)

    @app.template_global()
    def safe_url_for(endpoint: str, **values: Any) -> str:
        try:
            return url_for(endpoint, **values)
        except BuildError:
            return ""

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

    # ★ 毎回 create_app で blueprint を確実に登録（モジュールグローバルでスキップしない）
    if admin_pamphlets_bp.name not in app.blueprints:
        app.register_blueprint(admin_pamphlets_bp)
    if admin_rollback_bp.name not in app.blueprints:
        app.register_blueprint(admin_rollback_bp)

    pamphlet_search.configure(app.config)
    try:
        pamphlet_search.load_all()
    except Exception:
        app.logger.exception("[pamphlet] initial load failed")

    # users/entriesなどの初期ファイル
    _bootstrap_files_and_admin(app)

    _log_environment_config(app)
    return app
