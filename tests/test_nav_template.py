import sys
from pathlib import Path

import pytest
from flask import render_template


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_admin_nav_renders_without_build_error():
    from app import app

    with app.app_context():
        # request.endpoint が None でも動作することを確認
        with app.test_request_context('/'):
            html = render_template('_admin_nav.html')
            assert '[質問・回答ログ]' in html
