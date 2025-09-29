# config.py
import os
from pathlib import Path

from services.paths import default_data_base_dir

_APP_ENV = os.getenv("APP_ENV", "development")
_DATA_BASE_OVERRIDE = os.getenv("DATA_BASE_DIR")
if _DATA_BASE_OVERRIDE:
    DATA_BASE_DIR = Path(_DATA_BASE_OVERRIDE).expanduser()
else:
    DATA_BASE_DIR = default_data_base_dir(_APP_ENV).expanduser()

PAMPHLET_DIR = Path(os.getenv("PAMPHLET_BASE_DIR", "")).expanduser() if os.getenv("PAMPHLET_BASE_DIR") else DATA_BASE_DIR / "pamphlets"
PAMPHLET_BASE_DIR = str(PAMPHLET_DIR)
ENTRIES_DIR = DATA_BASE_DIR / "entries"
UPLOADS_DIR = DATA_BASE_DIR / "uploads"
IMAGES_DIR = DATA_BASE_DIR / "images"

PAMPHLET_CITIES = {
    "goto": "五島市",
    "shinkamigoto": "新上五島町",
    "ojika": "小値賀町",
    "uku": "宇久町",
}

class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///local.db")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "16"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    PAMPHLET_EDIT_MAX_MB = int(os.getenv("PAMPHLET_EDIT_MAX_MB", "2"))
    PAMPHLET_EDIT_MAX_BYTES = PAMPHLET_EDIT_MAX_MB * 1024 * 1024
    PAMPHLET_UPLOAD_MAX_MB = int(os.getenv("PAMPHLET_UPLOAD_MAX_MB", str(MAX_UPLOAD_MB)))
    PAMPHLET_UPLOAD_MAX_BYTES = PAMPHLET_UPLOAD_MAX_MB * 1024 * 1024
    MEDIA_DIR = os.getenv("MEDIA_DIR", str(IMAGES_DIR))
    IMAGES_DIR = str(IMAGES_DIR)
    ENTRIES_DIR = str(ENTRIES_DIR)
    UPLOADS_DIR = str(UPLOADS_DIR)
    # LINE
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    LINE_SIGNATURE_CHECK = os.getenv("LINE_SIGNATURE_CHECK", "1") == "1"
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # 表示用
    ENV_NAME = os.getenv("APP_ENV", "development")
    # Pamphlet search
    PAMPHLET_BASE_DIR = str(PAMPHLET_DIR)
    PAMPHLET_TOPK = int(os.getenv("PAMPHLET_TOPK", "12"))
    PAMPHLET_CHUNK_SIZE = int(os.getenv("PAMPHLET_CHUNK_SIZE", "700"))
    PAMPHLET_CHUNK_OVERLAP = int(os.getenv("PAMPHLET_CHUNK_OVERLAP", "150"))
    PAMPHLET_SESSION_TTL = int(os.getenv("PAMPHLET_SESSION_TTL", str(30 * 60)))
    PAMPHLET_MMR_LAMBDA = float(os.getenv("PAMPHLET_MMR_LAMBDA", "0.4"))
    PAMPHLET_MIN_CONFIDENCE = float(os.getenv("PAMPHLET_MIN_CONFIDENCE", "0.42"))
    GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
    REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
    DATA_BASE_DIR = str(DATA_BASE_DIR)
    BACKUP_DIR = os.getenv("BACKUP_DIR", "/var/tmp/backup")
    BACKUP_RETENTION = int(os.getenv("BACKUP_RETENTION", "14"))
    ROLLBACK_READY_TIMEOUT_SEC = int(os.getenv("ROLLBACK_READY_TIMEOUT_SEC", "90"))
    ROLLBACK_CANARY_ENABLED = os.getenv("ROLLBACK_CANARY_ENABLED", "true").lower() in {"1", "true", "yes"}
    ALLOW_ADMIN_ROLLBACK_IPS = os.getenv("ALLOW_ADMIN_ROLLBACK_IPS", "")

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    LINE_SIGNATURE_CHECK = os.getenv("LINE_SIGNATURE_CHECK", "0") == "1"

class StagingConfig(BaseConfig):
    DEBUG = False

class ProductionConfig(BaseConfig):
    DEBUG = False

def get_config():
    env = os.getenv("APP_ENV", "development").lower()
    if env == "production":
        return ProductionConfig
    if env == "staging":
        return StagingConfig
    return DevelopmentConfig
