"""Ensure the responder skeleton modules are importable."""


def test_import_skeleton_modules() -> None:
    """Verify that the placeholder modules can be imported."""
    import app  # noqa: F401
    from app import (  # noqa: F401
        config,
        intent,
        llm,
        schemas,
    )
    from app.responders import (  # noqa: F401
        entries,
        fallback,
        pamphlet,
        priority,
    )
    from app.search import entries_index, normalize, pamphlet_index  # noqa: F401

    assert app is not None
    assert config is not None
    assert intent is not None
    assert llm is not None
    assert schemas is not None
    assert entries is not None
    assert fallback is not None
    assert pamphlet is not None
    assert priority is not None
    assert entries_index is not None
    assert normalize is not None
    assert pamphlet_index is not None
