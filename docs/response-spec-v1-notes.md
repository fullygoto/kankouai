# 応答仕様書 v1.0 差分メモ

## 現状の処理フロー
- **エントリーポイント**: `app.py` が Flask アプリを定義し、LINE Webhook (`/callback`)・Webフォーム・管理UIなど全入出力を集約。パンフレット経路は LINE からのテキストを `_pamphlet_handle_line` が受け取り、都市選択などの対話状態を保持しながら `services.pamphlet_flow.build_response` に処理を委譲している。【F:app.py†L2955-L3031】
- **セッション/状態管理**: `pamphlet_flow` は `services/pamphlet_session.py` ベースのセッションストアを使い、未回答の市町確認や「もっと詳しく」のフォローアップを管理。都市未確定時はクイックリプライで候補を提示する。【F:services/pamphlet_flow.py†L17-L112】
- **検索→要約**: `pamphlet_flow` が都市確定後に `services.pamphlet_rag.answer_from_pamphlets` を呼び出し、`pamphlet_search` の TF-IDF 検索結果と LLM による要約から回答を構築。出典が空のときは `pamphlet_summarize` によるフォールバック要約も試みる。【F:services/pamphlet_flow.py†L114-L189】
- **出力整形**: `build_pamphlet_message` で本文と出典見出しを組み合わせ、LINE へは `_split_for_line` で 5000 文字制限を考慮しながら分割送信。回答や根拠を `save_qa_log` で JSONL ログに残す。【F:services/pamphlet_flow.py†L134-L188】【F:app.py†L2986-L3031】【F:app.py†L6022-L6036】

## 主要ファイル
- `app.py`: Flask ルーティング、LINE/Web 応答、管理機能、ヘルスチェック、ログ保存などほぼ全体のオーケストレーション。【F:app.py†L1-L9898】
- `services/pamphlet_flow.py`: パンフレット経路の会話制御とレスポンス整形。【F:services/pamphlet_flow.py†L1-L199】
- `services/pamphlet_rag.py`: RAG パイプライン（プランニング、検索、要約、引用抽出、再利用キャッシュ）。【F:services/pamphlet_rag.py†L1-L420】
- `services/pamphlet_search.py`: 市町別パンフレットの TF-IDF インデックス構築と検索。再構築時のメタ情報も保持。【F:services/pamphlet_search.py†L1-L210】
- `services/line_handlers.py`: 停止/再開コマンドや TTL の処理など LINE 固有の制御フローを担当。【F:services/line_handlers.py†L1-L120】
- `config.py`: 環境変数ベースのアプリ設定（OpenAI/LINE/APIキー、パンフレット検索パラメータ、バックアップ設定など）。【F:config.py†L1-L63】
- `services/rollback_service.py`: /readyz 監視を前提としたバックアップ・ロールバック基盤とログ設定。【F:services/rollback_service.py†L1-L119】

## LINE アダプタ
- LINE アクセストークンとチャンネルシークレットが設定済みかを起動時に検証し、`handler.handle` へディスパッチ。署名不一致や例外は 400/500 を返し、再試行制御に備えてメモリ上にメトリクスを保持する。未設定時は「LINE disabled」を返して手動リトライを抑制。【F:app.py†L8845-L8920】
- 一時停止・再開、Rate Limit、クイックリプライなどの LINE 特有動作は `_pamphlet_handle_line` と `services.line_handlers` で実装され、管理者 UI からのテスト push も `/admin/line/test_push` で提供。【F:app.py†L2929-L3031】【F:services/line_handlers.py†L1-L120】【F:app.py†L8931-L8974】

## 検索・要約の実装位置
- **検索**: `services.pamphlet_search._PamphletIndex` が TF-IDF ベースでチャンク検索を実装。`overall_state()` / `status()` で readyz 用の状態確認を公開。【F:services/pamphlet_search.py†L1-L210】【F:services/pamphlet_search.py†L216-L379】【F:app.py†L9809-L9836】
- **要約/生成**: `services.pamphlet_rag.answer_from_pamphlets` が `plan_answer` によるクエリプランニング、同義語展開、信頼度に応じた再試行、OpenAI での制約付き要約生成、引用抽出を担当。フォールバック要約は `services.pamphlet_summarize.summarize_with_gpt_nano` を利用。【F:services/pamphlet_rag.py†L320-L420】【F:services/pamphlet_flow.py†L125-L188】

## 環境変数 (ENV)
- `config.BaseConfig` 経由で LINE/OPENAI のキー、パンフレット検索パラメータ (`PAMPHLET_TOPK`, `PAMPHLET_CHUNK_SIZE` など)、バックアップ系 (`DATA_BASE_DIR`, `BACKUP_DIR`, `ROLLBACK_READY_TIMEOUT_SEC` 等) を読み込む。【F:config.py†L1-L63】
- README ではパンフレット運用や RAG で利用する主要変数（`PAMPHLET_BASE_DIR`, `SUMMARY_MODE`, `CITATION_MIN_SCORE`, `CONTROL_CMD_ENABLED` など）を一覧化し、運用時のチューニング指針を示している。【F:README.md†L33-L111】
- `services/line_handlers` や `pamphlet_rag` も個別に `CONTROL_CMD_ENABLED`, `PAUSE_DEFAULT_TTL_SEC`, `GEN_MODEL`, `PAMPHLET_MIN_CONFIDENCE` などの環境変数を読み込む。【F:services/line_handlers.py†L11-L28】【F:services/pamphlet_rag.py†L24-L53】

## /readyz チェック
- `/readyz` は Redis・データベース・メディアディレクトリのヘルスチェックを実施し、パンフレットインデックス状態（`pamphlet_search.overall_state()`）と詳細ステータスを返却する。Git のブランチ/コミット情報、dirty フラグも添付し、問題があれば `status=degraded` で 503 を返す。【F:app.py†L9797-L9879】
- ロールバックカナリアや CLI は `ROLLBACK_READY_URL`/`READYZ_URL` を参照して `/readyz` を監視する前提で設計されている。【F:services/rollback_service.py†L1-L119】

## ログ出力
- Q&A 応答は `save_qa_log` で `logs/questions_log.jsonl`（`DATA_BASE_DIR/logs`）に JSON Lines 追記され、ソース種別や引用情報も保存。送信ログ (`SEND_LOG_FILE`) や LINE 送信エラーはメモリカウンタと `logs/send_log.jsonl` に記録される。【F:app.py†L4086-L4143】【F:app.py†L6022-L6036】
- バックアップ/ロールバック処理は `services.rollback_service.LOGGER` が `logs/rollback.log` に INFO/ERROR を出力するよう初期化される。【F:services/rollback_service.py†L1-L83】

## 仕様①→⑤の到達点と不足
- **① 観光データ登録〜整備**
  - *現状*: `entries.json`/`synonyms.json` を `DATA_BASE_DIR` 配下に保持し、`save_entries` で正規化・重複統合したうえでアトミック保存している。【F:app.py†L4086-L4143】【F:app.py†L6126-L6188】
  - *不足*: 仕様書にある「戦略的な観光データ登録情報の一元/履歴管理」やメタデータ整備の要求（例: 属性統一・未整備検出など）は専用ワークフローがなく、既存 JSON 管理に依存。差分検出や承認フローも未実装。
- **② 利用者入力のルーティング/意図判定**
  - *現状*: LINE 停止コマンドや質問重複ガード、`conversational_router.TourismAI` の簡易ルーティング（LLM JSON 抽出 or ルールベース）でインテント分岐が用意されている。【F:services/line_handlers.py†L1-L120】【F:conversational_router.py†L1-L156】
  - *不足*: 仕様の「LINE/WEB で共通の意図スキーマ・多言語対応での統合ルーティング」や mini GPT での厳密な JSON スキーマ保証は暫定実装（LLM 失敗時はヒューリスティック）で、本番監視指標や高信頼のフローには未到達。
- **③ 検索・要約・根拠提示**
  - *現状*: `pamphlet_rag` がプランニング、同義語再検索、引用抽出、低信頼時の再試行／キャッシュ再利用まで備え、`pamphlet_flow` が要約・出典を整形して返す。【F:services/pamphlet_rag.py†L320-L420】【F:services/pamphlet_flow.py†L125-L188】
  - *不足*: 仕様の「BM25/TF-IDF ハイブリッド」「5 mini テスト」など高度な検索評価・自動チューニング、再ランキングやスコア閾値レポートなどの品質計測は未搭載。OpenAI API エラー時のリカバリーやモデル切替も手動。
- **④ 応答テンプレート/多チャンネル配信**
  - *現状*: LINE 向けの分割送信、クイックリプライ、出典トグル制御、`message_builder` による Markdown 出力を備える。Web UI でも出典付き表示が可能。【F:services/pamphlet_flow.py†L134-L188】【F:app.py†L2929-L3031】
  - *不足*: 仕様で定義されたテンプレート（例: 200〜350 文字本編＋最大 5 行の補足、FullyGOTO サイト導線の優先順位など）の厳格なフォーマット検証や A/B ルールは部分的。多言語回答テンプレートや音声/画像応答は未整備。
- **⑤ 運用監視・ログ・レポート**
  - *現状*: `/readyz` によるヘルスチェック、`logs/questions_log.jsonl`/`rollback.log` のファイルログ、LINE 送信エラーのメモリメトリクスがある。【F:app.py†L9797-L9879】【F:app.py†L6022-L6036】【F:services/rollback_service.py†L1-L83】
  - *不足*: 仕様で想定される「5分インシデントレポート」「検索ゼロ件の自動抽出」「ダッシュボード化された KPI」などの運用レポーティングは未実装。ログはファイル蓄積のみで、集計・通知連携は別途必要。
