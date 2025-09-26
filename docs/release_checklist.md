# リリース前チェックリスト（develop → main）

## A. 技術的健全性チェック

1. **CI / テスト**
   - `pytest` を実行し、Lint / Unit / E2E をすべてグリーンにする。
   - 不足するテストケース（パンフレット日本語名のアップロード・編集保存・出典整形など）は追加し、テストで担保する。
2. **環境変数 / 設定**
   - `.env.example` に `PAMPHLET_BASE_DIR=/var/data/pamphlets`、`PAMPHLET_EDIT_MAX_MB`、OpenAI/LINE/モデル設定が記載済みか確認。
   - Render の Persistent Disk が `/var/data` にマウントされていることを確認し、`PAMPHLET_BASE_DIR` が `/var/data/pamphlets` を指すよう設定。
3. **ヘルスチェック**
   - `/readyz` が存在し、`status`、`pamphlet_index`、`pamphlet_index_status`、`build(commit / branch)` を返すことを確認する。
   - `pamphlet_search.status()` の結果と `pamphlet_index` が `error` の場合は原因を解消する。

## B. ステージングでの総合動作確認

1. **管理 UI `/admin/pamphlets`**
   - ページ表示が 200 で成功し、「データ管理に戻る」ボタンが表示される。
   - 日本語名 `.txt` のアップロード→一覧反映→再インデックスが成功する。
   - 一覧クリックで右側プレビューが表示され、編集→保存（273KB 程度）が成功し `.bak` が生成される。
   - ナビゲーションから「パンフレット管理」へ遷移でき、旧 `/admin/data_files` はナビから隠れる（直打ちは可）。
2. **RAG 応答**
   - 出典は 1 ブロックのみ、市町/ファイル名（拡張子・行番号なし）で表示される。
   - 「補足」は必要時のみ表示。通常回答では出ない。
   - 同一質問を短時間に連投しても 1 回だけ配信され、回答内容が安定する。
   - 市町のセッションが TTL 内で維持され、再質問で聞き返しが発生しない。
3. **LINE 連携**
   - 返信期限内は `reply` が利用され、期限切れ時のみ `push` にフォールバックする。
   - ログに `Reply token invalid → pushed ...` が出てもユーザー受信は 1 通であることを確認。
4. **抽出データ取り込み**
   - クリーニング済み TXT（例：`五島市_しま山ガイドブック_clean.txt`）をアップロードし、再インデックス後にパンフレット由来の要約が返ること。

## C. 追加の最終手直し

- 出典整形が 2 重適用されていないか確認（本文組み立ては 1 箇所）。
- 編集保存時は `PAMPHLET_EDIT_MAX_MB` に基づく判定となり、「64MB」の上限文言が出ないこと。
- 送信経路は必ず `emit_response` など中央ハンドラを経由し、直接 `reply`/`push` を呼び出さない実装になっていること。

## D. main 反映と運用手順

1. `release/*` ブランチを切り、`CHANGELOG.md` とバージョンを更新して PR を作成（ベース: `develop` → `main`）。
2. Required checks (CI) を必須化し、少なくとも 1 名のレビュー承認を得る。
3. Render 本番が `main` を参照している場合は **Clear build cache → Deploy** を実施。
4. デプロイ後に `/readyz` の `commit` が PR のコミットと一致していることを確認。管理 UI と LINE でスモークテストを行う。
5. ロールバック手順として、前リリースタグへの Revert → Deploy を README / 運用ドキュメントに記載しておく。

---

- リリース後、`docs/release_checklist.md` にある各項目をチェックした記録を残す（Issue や Notion など）。
- 本番環境での確認ポイント: `/readyz` のコミット一致、パンフレット管理 UI 操作、RAG 応答の出典表記、LINE 配信が 1 通で完了すること。
