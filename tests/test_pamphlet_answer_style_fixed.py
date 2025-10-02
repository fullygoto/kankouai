import textwrap
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("werkzeug")
pytest.importorskip("dotenv")
pytest.importorskip("linebot")
pytest.importorskip("PIL")

from services.pamphlet_constants import CITY_PROMPT
from tests.utils import load_test_app


def _setup_pamphlet(app_module):
    base_dir = Path(app_module.app.config["PAMPHLET_BASE_DIR"])
    city_dir = base_dir / "goto"
    city_dir.mkdir(parents=True, exist_ok=True)
    pamphlet_text = textwrap.dedent(
        """
        遣唐使は630年から約260年間続き、約20回の派遣が記録されている。五島は最終寄港地として整備され、唐の制度や文化を学ぶことが目的だった。
        五島の港では航海の安全祈願や補給が行われ、遣唐使船はここから大海原へ出帆した。
        """
    ).strip()
    (city_dir / "history.txt").write_text(pamphlet_text, encoding="utf-8")
    app_module.pamphlet_search.reindex_all()


def _write_entries(app_module):
    app_module._atomic_json_dump(
        app_module.ENTRIES_FILE,
        [
            {
                "title": "上五島観光案内所",
                "desc": "上五島を案内する窓口。",
                "areas": ["新上五島町"],
                "area_checked": True,
            }
        ],
    )


def _extract_section(body: str, marker: str) -> str:
    if marker not in body:
        return ""
    tail = body.split(marker, 1)[1]
    for next_marker in ("\n### ", "\n# "):
        if next_marker in tail:
            tail = tail.split(next_marker, 1)[0]
            break
    return tail.strip()


def test_pamphlet_answer_style_fixed(monkeypatch, tmp_path):
    extra_env = {"SUMMARY_STYLE": "terse_short"}
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as app_module:
        _write_entries(app_module)
        _setup_pamphlet(app_module)

        prompt, hit_prompt, _ = app_module._answer_from_entries_min("遣唐使について知りたい", user_id="user2")
        assert hit_prompt is True
        assert CITY_PROMPT in prompt

        answer, hit_answer, _ = app_module._answer_from_entries_min("五島市", user_id="user2")
        assert hit_answer is True
        assert answer.startswith("### 要約"), answer
        assert "### 出典" in answer, answer

        if "### 詳細" in answer:
            assert answer.index("### 要約") < answer.index("### 詳細") < answer.index("### 出典"), answer
        else:
            assert answer.index("### 要約") < answer.index("### 出典"), answer

        summary_section = _extract_section(answer, "### 要約")
        for keyword in ("630年から約260年", "約20回", "最終寄港地", "目的"):
            assert keyword in summary_section

        sources_section = _extract_section(answer, "### 出典")
        assert sources_section
        for line in sources_section.splitlines():
            assert line.startswith("- 五島市/")
            assert ".txt" not in line
            assert "L" not in line
