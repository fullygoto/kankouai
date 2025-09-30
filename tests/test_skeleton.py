"""Ensure the responder skeleton modules are importable."""


def test_import_skeleton_modules() -> None:
    """Verify that the placeholder modules can be imported."""
    import coreapp  # noqa: F401
    from coreapp import (  # noqa: F401
        config,
        intent,
        llm,
        schemas,
    )
    from coreapp.responders import (  # noqa: F401
        entries,
        fallback,
        pamphlet,
        priority,
    )
    from coreapp.search import entries_index, normalize, pamphlet_index  # noqa: F401

    assert coreapp is not None
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
