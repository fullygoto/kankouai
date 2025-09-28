import textwrap
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("werkzeug")
pytest.importorskip("dotenv")
pytest.importorskip("linebot")
pytest.importorskip("PIL")

from tests.utils import load_test_app


def _entries_payload():
    return [
        {
            "title": "高浜海水浴場",
            "desc": "白砂が続く遠浅のビーチ。",
            "areas": ["五島市"],
            "tags": ["ビーチ", "海水浴"],
            "area_checked": True,
        },
        {
            "title": "遣唐使資料館",
            "desc": "遣唐使の歴史資料を展示する施設。",
            "areas": ["五島市"],
            "area_checked": True,
        },
    ]


def _setup_pamphlet(app_module):
    base_dir = Path(app_module.app.config["PAMPHLET_BASE_DIR"])
    city_dir = base_dir / "goto"
    city_dir.mkdir(parents=True, exist_ok=True)
    pamphlet_text = textwrap.dedent(
        """
        遣唐使は630年から約260年にわたり派遣され、五島は最終寄港地として整備された。派遣は約20回行われ、目的は唐の制度や文化を学ぶことだった。
        """
    ).strip()
    (city_dir / "history.txt").write_text(pamphlet_text, encoding="utf-8")
    app_module.pamphlet_search.reindex_all()


def test_city_prompt_only_in_pamphlet(monkeypatch, tmp_path):
    extra_env = {"SUMMARY_STYLE": "terse_short"}
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as app_module:
        app_module._atomic_json_dump(app_module.ENTRIES_FILE, _entries_payload())
        _setup_pamphlet(app_module)

        ans, hit, img = app_module._answer_from_entries_min("高浜", user_id="user1")
        assert hit is True
        assert "高浜海水浴場" in ans
        assert "どの市町" not in ans

        prompt, hit_prompt, _ = app_module._answer_from_entries_min("遣唐使の時代", user_id="user1")
        assert hit_prompt is True
        assert "どの市町の資料ですか？" in prompt

        summary, hit_summary, _ = app_module._answer_from_entries_min("五島市", user_id="user1")
        assert hit_summary is True
        assert summary.startswith("### 要約")
        assert "どの市町" not in summary
