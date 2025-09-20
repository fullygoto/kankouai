# config.py
import os

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
