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


_APP_ENV = (os.getenv("APP_ENV") or "staging").lower()
_DEFAULT_BASE_DIR = Path(os.getenv("DATA_BASE_DIR") or default_data_base_dir(_APP_ENV))
_DEFAULT_MEDIA_ROOT = Path(
    os.getenv("MEDIA_ROOT")
    or os.getenv("MEDIA_DIR")
    or os.getenv("IMAGES_DIR")
    or _DEFAULT_BASE_DIR
)
_DEFAULT_PAMPHLET_DIR = Path(
    os.getenv("PAMPHLET_BASE_DIR") or (_DEFAULT_BASE_DIR / "pamphlets")
)


def _default_database_url(env: str) -> str:
    base_dir = Path(os.getenv("DATA_BASE_DIR") or _DEFAULT_BASE_DIR or "/var/data")
    if env == "production":
        db_name = "kankouai.db"
    else:
        db_name = "kankouai_stg.db"
    return f"sqlite:///{base_dir.joinpath(db_name)}"


def _validate_required_env() -> None:
    """Fail fast when required environment variables are missing or unsafe.

    Local development remains lenient, but staging/production/CI must provide
    non-empty values so deployments do not silently start with insecure defaults.
    """

    env = _APP_ENV
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

    require("SECRET_KEY", unsafe_values={"dev-secret"})
    require("OPENAI_API_KEY")
    require("LINE_CHANNEL_SECRET")
    require("LINE_CHANNEL_ACCESS_TOKEN")

    data_base_dir = Path(os.getenv("DATA_BASE_DIR") or _DEFAULT_BASE_DIR)
    if not data_base_dir.is_absolute():
        errors.append("DATA_BASE_DIR must be an absolute path")

    backup_dir = Path(os.getenv("BACKUP_DIR", "/var/tmp/backup"))
    if not backup_dir.is_absolute():
        errors.append("BACKUP_DIR must be an absolute path")

    if errors:
        raise ConfigValidationError("Configuration validation failed:\n - " + "\n - ".join(errors))


PAMPHLET_BASE_DIR = str(_DEFAULT_PAMPHLET_DIR)
PAMPHLET_CITIES = {
    "goto": "五島市",
    "shinkamigoto": "新上五島町",
    "ojika": "小値賀町",
    "uku": "宇久町",
}


class BaseConfig:
    APP_ENV = _APP_ENV
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or _default_database_url(APP_ENV)
    DATA_BASE_DIR = str(_DEFAULT_BASE_DIR)
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", str(_DEFAULT_MEDIA_ROOT))
    MEDIA_DIR = os.getenv("MEDIA_DIR", MEDIA_ROOT)
    IMAGES_DIR = os.getenv("IMAGES_DIR", MEDIA_DIR)
    PAMPHLET_BASE_DIR = PAMPHLET_BASE_DIR
    USERS_FILE = os.getenv("USERS_FILE", str(_DEFAULT_BASE_DIR / "users.json"))
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
    ENV_NAME = os.getenv("APP_ENV", "staging")
    # Pamphlet search
    PAMPHLET_BASE_DIR = PAMPHLET_BASE_DIR
    PAMPHLET_TOPK = int(os.getenv("PAMPHLET_TOPK", "12"))
    PAMPHLET_CHUNK_SIZE = int(os.getenv("PAMPHLET_CHUNK_SIZE", "700"))
    PAMPHLET_CHUNK_OVERLAP = int(os.getenv("PAMPHLET_CHUNK_OVERLAP", "150"))
    PAMPHLET_SESSION_TTL = int(os.getenv("PAMPHLET_SESSION_TTL", str(30 * 60)))
    PAMPHLET_MMR_LAMBDA = float(os.getenv("PAMPHLET_MMR_LAMBDA", "0.4"))
    PAMPHLET_MIN_CONFIDENCE = float(os.getenv("PAMPHLET_MIN_CONFIDENCE", "0.42"))
    GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
    REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
    BACKUP_DIR = os.getenv("BACKUP_DIR", "/var/tmp/backup")
    BACKUP_RETENTION = int(os.getenv("BACKUP_RETENTION", "14"))
    ROLLBACK_READY_TIMEOUT_SEC = int(os.getenv("ROLLBACK_READY_TIMEOUT_SEC", "90"))
    ROLLBACK_CANARY_ENABLED = os.getenv("ROLLBACK_CANARY_ENABLED", "true").lower() in {"1", "true", "yes"}
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
    env = os.getenv("APP_ENV", "staging").lower()
    if env == "production":
        return ProductionConfig
    if env == "staging":
        return StagingConfig
    if env == "development":
        return DevelopmentConfig
    return StagingConfig
