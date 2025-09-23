import glob, io

def test_no_main_prefix_in_templates():
    for path in glob.glob("templates/*.html"):
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "url_for('main." not in text
        assert 'url_for("main.' not in text
        assert "safe_url_for('main." not in text
        assert 'safe_url_for("main.' not in text
        assert "has_endpoint('main." not in text
        assert 'has_endpoint("main.' not in text
