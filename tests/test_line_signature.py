from tests.utils import load_test_app


class _VerifyingHandler:
    def __init__(self, error_cls):
        self.error_cls = error_cls
        self.calls = []

    def handle(self, body, signature):
        self.calls.append(signature)
        if signature != "valid-signature":
            raise self.error_cls("bad signature")


def test_callback_rejects_invalid_signature(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "LINE_CHANNEL_ACCESS_TOKEN": "token",
            "LINE_CHANNEL_SECRET": "secret",
        },
    ) as module:
        module.handler = _VerifyingHandler(module.InvalidSignatureError)
        client = module.app.test_client()

        bad = client.post(
            "/callback",
            data="{}",
            headers={"X-Line-Signature": "invalid"},
        )
        assert bad.status_code == 400

        ok = client.post(
            "/callback",
            data="{}",
            headers={"X-Line-Signature": "valid-signature"},
        )
        assert ok.status_code == 200
        assert module.handler.calls == ["invalid", "valid-signature"]
