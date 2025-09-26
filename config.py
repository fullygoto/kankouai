# config.py
import os

PAMPHLET_BASE_DIR = os.getenv("PAMPHLET_BASE_DIR", "/var/data/pamphlets")
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
    MEDIA_DIR = os.getenv("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
    # LINE
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    LINE_SIGNATURE_CHECK = os.getenv("LINE_SIGNATURE_CHECK", "1") == "1"
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # 表示用
    ENV_NAME = os.getenv("APP_ENV", "development")
    # Pamphlet search
    PAMPHLET_BASE_DIR = PAMPHLET_BASE_DIR
    PAMPHLET_TOPK = int(os.getenv("PAMPHLET_TOPK", "3"))
    PAMPHLET_CHUNK_SIZE = int(os.getenv("PAMPHLET_CHUNK_SIZE", "1500"))
    PAMPHLET_CHUNK_OVERLAP = int(os.getenv("PAMPHLET_CHUNK_OVERLAP", "200"))
    PAMPHLET_SESSION_TTL = int(os.getenv("PAMPHLET_SESSION_TTL", str(30 * 60)))

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
