You are CPT5-Codex. Apply **minimal, surgical** changes.

### Primary task
Make all **read-side** references to `MEDIA_ROOT`, `IMAGES_DIR`, and `WATERMARK_DIR`
pull from `current_app.config` first, with the existing constants as default.
- Keep write-side constant assignments as-is.
- Avoid mass regex replacements that could touch LHS of assignments.
- If a helper exists (e.g. `__cfg_get`), you may use/adjust it ONLY if it remains tiny and safe.

### Must NOT break
- The no-op LINE decorator bootstrap (`@_line_add`) and its lazy init.
- The already-fixed `IMAGES_SIGNING_KEY` fallback.
- Current routes and admin aliases.
- All existing tests.

### Allowed small improvements
- Add `config.setdefault(...)` entries in `create_app()` for the three keys with current defaults.
- Replace obvious `os.path.join(MEDIA_ROOT, ...)` read-sites with config-first lookups.

### Validation
- `pytest -q` passes.
- Then remove `xfail` marks in `tests/test_codex_read_side.py` and make it pass as well.
