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

# 互換用（テスト/他モジュール参照対策：ただし値は create_app() 中で都度同期する）
ADMIN_INIT_USER = "admin"
ADMIN_INIT_PASSWORD: str | None = None


def _is_pytest() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _refresh_admin_init_globals() -> None:
    global ADMIN_INIT_USER, ADMIN_INIT_PASSWORD
    ADMIN_INIT_USER = os.getenv("ADMIN_INIT_USER", "admin")
    ADMIN_INIT_PASSWORD = os.getenv("ADMIN_INIT_PASSWORD")


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

    # storage 側のズレ防止：env にも同期しておく
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

    # ここでディレクトリを作る（ただし壊れてても app 起動を止めない：readyz が検出するため）
    try:
        ensure_data_directories(
            Path(DATA_BASE_DIR),
            pamphlet_dir=flask_app.config.get("PAMPHLET_BASE_DIR"),
        )
    except Exception:
        flask_app.logger.warning("ensure_data_directories failed (continuing)", exc_info=True)


def _ensure_json(path: str, default_obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            json.dumps(default_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _bootstrap_files_and_admin(app: Flask) -> None:
    # env を都度読む（テストごとに変わるため）
    _refresh_admin_init_globals()

    app.logger.info("[boot] DATA_BASE_DIR=%s", DATA_BASE_DIR)
    app.logger.info("[boot] USERS_FILE=%s", USERS_FILE)

    _ensure_json(ENTRIES_FILE, [])
    _ensure_json(SYNONYM_FILE, {})
    _ensure_json(NOTICES_FILE, [])
    _ensure_json(SHOP_INFO_FILE, {})

    users: list[dict[str, str]] = []
    users_path = Path(USERS_FILE)
    users_exists = users_path.exists()

    if users_exists:
        try:
            users = json.loads(users_path.read_text(encoding="utf-8"))
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
            users_path.write_text(
                json.dumps(users, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
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
            users_path.write_text(
                json.dumps(users, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            init_path = users_path.with_suffix(users_path.suffix + ".init")
            init_payload = {
                "user_id": factory_user,
                "password": factory_password,
                "note": "初回ログイン後にこのファイルを削除し、パスワードを変更してください。",
            }
            try:
                init_path.write_text(
                    json.dumps(init_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                app.logger.warning(
                    "failed to write factory credential hint to %s",
                    init_path,
                    exc_info=True,
                )

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
    pytest中は seed/migrate をしない（テストが「空」を期待するため）。
    """
    # storage は env を見て定数化されがちなので、env 同期後に import/reload する
    import importlib
    import coreapp.storage as storage  # noqa

    try:
        importlib.reload(storage)
    except Exception:
        # reload できなくても起動を止めない
        app.logger.warning("failed to reload coreapp.storage (continuing)", exc_info=True)

    base_dir = Path(app.config.get("DATA_BASE_DIR") or os.getenv("DATA_BASE_DIR") or DATA_BASE_DIR or "")
    if not base_dir:
        return

    sentinel = base_dir / ".bootstrap.done"

    # まずベースだけは作る（readyz が見るので）
    try:
        if base_dir.exists() and not base_dir.is_dir():
            app.logger.warning("DATA_BASE_DIR exists but is not a directory: %s", base_dir)
            return
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "backups").mkdir(parents=True, exist_ok=True)
    except Exception:
        app.logger.warning("failed to ensure base storage dirs (continuing)", exc_info=True)
        return

    if sentinel.exists():
        return

    if _is_pytest():
        # pytest 中は migrate/seed をしない（pamphlet_count を増やさない）
        try:
            sentinel.write_text("ok", encoding="utf-8")
        except OSError:
            app.logger.warning("Failed to write bootstrap sentinel", exc_info=True)
        return

    # 本番/通常起動のみ
    try:
        storage.migrate_from_legacy_paths()
        storage.seed_from_repo_if_empty()
    except Exception:
        app.logger.warning("storage migrate/seed failed (continuing)", exc_info=True)

    try:
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

    # 先に config を入れる（PAMPHLET_BASE_DIR などを確定）
    app.config.from_object(get_config())

    # ★ テストごとに必ず data paths を再計算
    _configure_data_paths(app)

    # ★ seed/legacy（pytest中はスキップ）
    _bootstrap_storage_once(app)

    @app.template_global()
    def safe_url_for(endpoint: str, **values: Any) -> str:
        try:
            return url_for(endpoint, **values)
        except BuildError:
            return ""

    # MEDIA / IMAGES は空にならないように DATA_DIR をフォールバックに使う
    # （app.py 側で IMAGES_DIR が空だと画像保存が落ちるため）
    MEDIA_ROOT = os.getenv("MEDIA_ROOT") or app.config.get("MEDIA_ROOT") or app.config.get("IMAGES_DIR") or MEDIA_ROOT
    IMAGES_DIR = os.getenv("IMAGES_DIR") or app.config.get("IMAGES_DIR") or MEDIA_ROOT
    if not IMAGES_DIR:
        IMAGES_DIR = os.path.join(DATA_DIR or DATA_BASE_DIR, "data", "images") if (DATA_DIR or DATA_BASE_DIR) else ""

    app.config["MEDIA_ROOT"] = MEDIA_ROOT or IMAGES_DIR
    app.config["IMAGES_DIR"] = IMAGES_DIR

    app.config["MEDIA_DIR"] = os.getenv("MEDIA_DIR") or app.config.get("MEDIA_DIR") or (MEDIA_ROOT or IMAGES_DIR)
    if app.config.get("MEDIA_DIR"):
        try:
            MEDIA_DIR = Path(app.config["MEDIA_DIR"]).expanduser()
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            app.logger.warning("failed to ensure MEDIA_DIR (continuing)", exc_info=True)

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
    def _no_cache(resp):
        p = (request.path or "")

        # テストが完全一致で見る
        if p in ("/healthz", "/readyz"):
            resp.headers["Cache-Control"] = "no-store"
            return resp

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

        return {"current_app": current_app, "has_endpoint": has_endpoint}

    MAX_UPLOAD_MB = app.config.get("MAX_UPLOAD_MB", 64)

    # ★ 毎回 blueprint を確実に登録（グローバルフラグでスキップしない）
    if admin_pamphlets_bp.name not in app.blueprints:
        app.register_blueprint(admin_pamphlets_bp)
    if admin_rollback_bp.name not in app.blueprints:
        app.register_blueprint(admin_rollback_bp)

    pamphlet_search.configure(app.config)
    try:
        pamphlet_search.load_all()
    except Exception:
        app.logger.exception("[pamphlet] initial load failed")

    _bootstrap_files_and_admin(app)
    _log_environment_config(app)
    return app


__all__ = ["create_app"]
