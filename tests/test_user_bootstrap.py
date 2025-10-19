import json
from pathlib import Path

from werkzeug.security import check_password_hash

from tests.utils import load_test_app


def test_bootstrap_uses_admin_init_password(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "ADMIN_INIT_PASSWORD": "s3cret",
        },
    ) as module:
        users_path = Path(module.USERS_FILE)
        assert users_path.exists()
        users = json.loads(users_path.read_text(encoding="utf-8"))
        assert len(users) == 1
        admin = users[0]
        assert admin["user_id"] == module.ADMIN_INIT_USER
        assert admin["role"] == "admin"
        assert check_password_hash(admin["password_hash"], "s3cret")
        assert not users_path.with_suffix(users_path.suffix + ".init").exists()


def test_bootstrap_creates_factory_user(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "ADMIN_INIT_PASSWORD": None,
        },
    ) as module:
        users_path = Path(module.USERS_FILE)
        assert users_path.exists()
        users = json.loads(users_path.read_text(encoding="utf-8"))
        assert len(users) == 1
        admin = users[0]
        assert admin["role"] == "admin"
        init_path = users_path.with_suffix(users_path.suffix + ".init")
        assert init_path.exists()
        credentials = json.loads(init_path.read_text(encoding="utf-8"))
        assert credentials["user_id"] == admin["user_id"]
        assert check_password_hash(admin["password_hash"], credentials["password"])
