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
            "title": "遣唐使資料館",
            "desc": "630年から遣唐使が派遣され、五島は最終寄港地として整備された。",
            "areas": ["五島市"],
            "area_checked": True,
            "address": "五島市福江町",
            "tel": "0959-00-0000",
            "open_hours": "9:00-17:00",
            "holiday": "火曜",
            "parking": "あり",
            "payment": ["現金"],
            "map": "https://maps.example.com/museum",
            "category": "観光",
        },
        {
            "title": "巡礼案内所",
            "desc": "教会を巡る旅の案内所です。",
            "areas": ["新上五島町"],
            "area_checked": True,
        },
    ]


def test_city_prompt_then_search(monkeypatch, tmp_path):
    with load_test_app(monkeypatch, tmp_path) as app_module:
        app_module._atomic_json_dump(app_module.ENTRIES_FILE, _entries_payload())

        ans, hit, img = app_module._answer_from_entries_min("遣唐使について知りたい", user_id="user1")
        assert hit is True
        assert "どの市町の資料ですか？" in ans

        ans2, hit2, img2 = app_module._answer_from_entries_min("五島市", user_id="user1")
        assert hit2 is True
        assert "遣唐使" in ans2
        assert "巡礼案内所" not in ans2

        ans3, hit3, img3 = app_module._answer_from_entries_min("遣唐使の目的は？", user_id="user1")
        assert hit3 is True
        assert "どの市町の資料ですか？" not in ans3
        assert "遣唐使" in ans3
