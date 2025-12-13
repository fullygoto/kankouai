"""Application configuration defaults."""
from __future__ import annotations

import os
from pathlib import Path

from services.paths import default_data_base_dir


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "on", "yes"}


class ConfigValidationError(RuntimeError):
    """Raised when required configuration is missing or unsafe."""


def _infer_service_role() -> str:
    """
    SERVICE_ROLE が未指定の場合、Renderのサービス名から推測して事故を減らす。
    - 例: kankouai-backup / backup / worker / cron など
    """
    explicit = (os.getenv("SERVICE_ROLE") or "").strip().lower()
    if explicit:
        return explicit

    svc = (os.getenv("RENDER_SERVICE_NAME") or "").strip().lower()
    if any(token in svc for token in ("backup", "worker", "cron")):
        return "backup"

    return "api"


_APP_ENV = (os.getenv("APP_ENV") or "staging").strip().lower()
_SERVICE_ROLE = _infer_service_role()

_DEFAULT_BASE_DIR = Path(os.getenv("DATA_BASE_DIR") or default_data_base_dir(_APP_ENV))
_DEFAULT_MEDIA_ROOT = Path(
    os.getenv("MEDIA_ROOT")
    or os.getenv("MEDIA_DIR")
    or os.getenv("IMAGES_DIR")
    or _DEFAULT_BASE_DIR
)
_DEFAULT_PAMPHLET_DIR = Path(os.getenv("PAMPHLET_BASE_DIR") or (_DEFAULT_BASE_DIR / "pamphlets"))


def _default_database_url(env: str) -> str:
    base_dir = Path(os.getenv("DATA_BASE_DIR") or str(_DEFAULT_BASE_DIR) or "/var/data")
    db_name = "kankouai.db" if env == "production" else "kankouai_stg.db"
    return f"sqlite:///{base_dir.joinpath(db_name)}"


def _validate_required_env() -> None:
    """
    Fail fast when required environment variables are missing or unsafe.

    - development は寛容（ローカルで動かしやすくする）
    - staging/production は基本厳格
    - backup/worker/cron 等は、LINE/OpenAIを必須にしない（役割によって不要なため）
    """

    env = _APP_ENV
    role = _SERVICE_ROLE

    if env not in {"development", "staging", "production"}:
        raise ConfigValidationError(f"Unsupported APP_ENV value: {env!r}")

    if env == "development":
        return

    errors: list[str] = []

    def require(key: str, *, unsafe_values: set[str] | None = None) -> None:
        value = os.getenv(key, "").strip()
        if not value:
            errors.append(f"{key} is required in {env} environments")
            return
        if unsafe_values and value in unsafe_values:
            errors.append(f"{key} must not use an unsafe default value ({value})")

    # どの役割でもパスの安全性は担保
    data_base_dir = Path(os.getenv("DATA_BASE_DIR") or str(_DEFAULT_BASE_DIR))
    if not data_base_dir.is_absolute():
        errors.append("DATA_BASE_DIR must be an absolute path")

    backup_dir = Path(os.getenv("BACKUP_DIR", "/var/tmp/backup"))
    if not backup_dir.is_absolute():
        errors.append("BACKUP_DIR must be an absolute path")

    # web/api 役割だけ厳格に必須化（backup等では不要なことが多い）
    if role in {"api", "web"}:
        require("SECRET_KEY", unsafe_values={"dev-secret"})
        require("OPENAI_API_KEY")
        require("LINE_CHANNEL_SECRET")
        require("LINE_CHANNEL_ACCESS_TOKEN")
    else:
        # それ以外の役割でも SECRET_KEY を必須にしたい場合はこれを有効化
        if _truthy(os.getenv("REQUIRE_SECRET_KEY_FOR_WORKERS", "0")):
            require("SECRET_KEY", unsafe_values={"dev-secret"})

    if errors:
        raise ConfigValidationError(
            "Configuration validation failed:\n - " + "\n - ".join(errors)
        )


PAMPHLET_BASE_DIR = str(_DEFAULT_PAMPHLET_DIR)
PAMPHLET_CITIES = {
    "goto": "五島市",
    "shinkamigoto": "新上五島町",
    "ojika": "小値賀町",
    "uku": "宇久町",
}


class BaseConfig:
    # 基本情報
    APP_ENV = _APP_ENV
    SERVICE_ROLE = _SERVICE_ROLE

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    # Flask/SQLAlchemy
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = (
        os.getenv("DATABASE_URL")
        or os.getenv("DB_URL")
        or _default_database_url(_APP_ENV)
    )

    # Data/Media paths
    DATA_BASE_DIR = str(_DEFAULT_BASE_DIR)
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", str(_DEFAULT_MEDIA_ROOT))
    MEDIA_DIR = os.getenv("MEDIA_DIR", MEDIA_ROOT)
    IMAGES_DIR = os.getenv("IMAGES_DIR", MEDIA_DIR)

    USERS_FILE = os.getenv("USERS_FILE", str(_DEFAULT_BASE_DIR / "users.json"))

    # Upload limits
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "64"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

    PAMPHLET_EDIT_MAX_MB = int(os.getenv("PAMPHLET_EDIT_MAX_MB", "2"))
    PAMPHLET_EDIT_MAX_BYTES = PAMPHLET_EDIT_MAX_MB * 1024 * 1024

    # LINE
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    LINE_SIGNATURE_CHECK = _truthy(os.getenv("LINE_SIGNATURE_CHECK", "1"))

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # 表示用
    ENV_NAME = APP_ENV

    # Pamphlet search
    PAMPHLET_BASE_DIR = PAMPHLET_BASE_DIR
    PAMPHLET_TOPK = int(os.getenv("PAMPHLET_TOPK", "12"))
    PAMPHLET_CHUNK_SIZE = int(os.getenv("PAMPHLET_CHUNK_SIZE", "700"))
    PAMPHLET_CHUNK_OVERLAP = int(os.getenv("PAMPHLET_CHUNK_OVERLAP", "150"))
    PAMPHLET_SESSION_TTL = int(os.getenv("PAMPHLET_SESSION_TTL", str(30 * 60)))
    PAMPHLET_MMR_LAMBDA = float(os.getenv("PAMPHLET_MMR_LAMBDA", "0.4"))
    PAMPHLET_MIN_CONFIDENCE = float(os.getenv("PAMPHLET_MIN_CONFIDENCE", "0.42"))

    # Models
    GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
    REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")

    # Backup/Rollback
    BACKUP_DIR = os.getenv("BACKUP_DIR", "/var/tmp/backup")
    BACKUP_RETENTION = int(os.getenv("BACKUP_RETENTION", "14"))
    ROLLBACK_READY_TIMEOUT_SEC = int(os.getenv("ROLLBACK_READY_TIMEOUT_SEC", "90"))
    ROLLBACK_CANARY_ENABLED = os.getenv("ROLLBACK_CANARY_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    ALLOW_ADMIN_ROLLBACK_IPS = os.getenv("ALLOW_ADMIN_ROLLBACK_IPS", "")


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    APP_ENV = "development"


class StagingConfig(BaseConfig):
    DEBUG = False
    APP_ENV = "staging"


class ProductionConfig(BaseConfig):
    DEBUG = False
    APP_ENV = "production"


_validate_required_env()


def get_config():
    env = (os.getenv("APP_ENV") or "staging").lower()
    if env == "production":
        return ProductionConfig
    if env == "staging":
        return StagingConfig
    if env == "development":
        return DevelopmentConfig
    return StagingConfig
