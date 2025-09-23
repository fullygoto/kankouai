# Codex Task: Read-side config fallback & LINE lazy-init

## Goal
- **読み取り側のみ**、`MEDIA_ROOT` / `IMAGES_DIR` / `WATERMARK_DIR` 参照を
  `current_app.config.get(KEY, <既存定数>)` 互換に統一する。
- 既存の **書き込み側（定数定義・代入）には触れない**。
- `LINE` の遅延初期化（`@_line_add` の no-op ブートストラップ）を壊さない。
- 既に導入済みの `IMAGES_SIGNING_KEY` フォールバック実装は**変更しない**。

## Non-goals
- ルーティングの追加/削除や挙動変更は不要（既存 alias は維持）
- LINE SDK の v3 置換は対象外（Deprecation warning は許容）

## Constraints / Guardrails
- **正規表現の一括置換で代入行（= の左辺）を変更しない**。
- `create_app()` の中に `config.setdefault(...)` を足すのはOK。ただし挙動が変わらない初期値のみ。
- 例外：読み取り時の `os.path.join(MEDIA_ROOT, ...)` 等は安全に `os.path.join(cfg("MEDIA_ROOT", MEDIA_ROOT), ...)` のように**個別に**置換する。
- ブートストラップ（SAFE / LINE）は**文法やインデントを壊さない**。改変が必要なら**丸ごと差し替えではなく**最小パッチ。

## Acceptance (tests)
- 既存 `pytest` が **すべて緑**。
- 追加テスト `tests/test_codex_read_side.py` の `xfail` を **Codex が外して緑化**。
- `flask run` → `/healthz` が 200、`/readyz` が 200（未設定時は `errors` フィールドで欠落が判るかは現状維持）。

## How to Run
1. `pytest -q` で緑確認
2. Codex に `codex/prompt.md` を読み込ませ、指示どおり修正
3. 修正後に `pytest -q` が緑になること
4. `tests/test_codex_read_side.py` の `xfail` を外しても緑
