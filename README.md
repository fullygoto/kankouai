# GOTO観光AIシステム

## 概要

五島列島の観光・生活情報をAIがLINEやWebで自動回答する、地域特化型のFAQ・サポートシステムです。  
登録された観光データやパンフレット情報、タグ・エリア指定で高精度な回答ができます。  
管理画面からデータ登録・誤答補強・バックアップ・未ヒット抽出・類義語管理までノーコード運用可能！

---

## 特徴

- FlaskベースのWebアプリ（Python3.10以上推奨）
- entries.json, synonyms.json, logs/questions_log.jsonl などファイル型DB
- 管理画面から観光DB登録・FAQ化・タグ・エリア・類義語追加OK
- 未ヒット質問/頻出レポートでAIの回答精度をどんどん進化
- 全体バックアップ＆ZIP復元、Dropboxバックアップ連携も対応可
- Render 本番向けロールバック基盤（/var/data・DB・アプリコードの瞬時復元）
- Github＋Render（無料枠）でノーコスト運用可能

---

## セットアップ

### 1. 必要パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. パンフレット検索フォールバックの準備

市町別パンフレットを以下のディレクトリ構成で配置します（既定パスは`./data/pamphlets`。本番環境では`/var/data/pamphlets`）。

```
{PAMPHLET_BASE_DIR}/
  goto/         # 五島市
  shinkamigoto/ # 新上五島町
  ojika/        # 小値賀町
  uku/          # 宇久町
```

各フォルダに UTF-8 のテキストファイル（1ファイル=1パンフレット）を配置してください。例：`history_guide_2025.txt`、`festivals.txt`。

### 3. 環境変数

パンフレット検索に関する主な環境変数は次のとおりです。

| 変数名 | 既定値 | 説明 |
| ------ | ------ | ---- |
| `PAMPHLET_BASE_DIR` | `./data/pamphlets`（APP_ENV=production は `/var/data/pamphlets`） | パンフレットテキストの格納ルート |
| `PAMPHLET_TOPK` | `3` | 要約に渡す上位チャンク数 |
| `PAMPHLET_CHUNK_SIZE` | `1500` | チャンク長（文字数） |
| `PAMPHLET_CHUNK_OVERLAP` | `200` | チャンク重複幅（文字数） |
| `PAMPHLET_SESSION_TTL` | `1800` | 市町選択の保持時間（秒） |
| `MAX_UPLOAD_MB` | `64` | アップロード可能なファイル上限（Flask `MAX_CONTENT_LENGTH` に反映） |
| `PAMPHLET_EDIT_MAX_MB` | `2` | 管理画面でのテキスト編集時の保存上限（MB） |
| `CITATION_MIN_CHARS` | `80` | 1ドキュメントに紐づく引用テキストの最小合計文字数。未満なら出典から除外 |
| `CITATION_MIN_SCORE` | `0.15` | 検索スコア合算の下限。`CITATION_MIN_CHARS` を満たさなくてもスコアが基準を超えれば出典採用 |
| `ANTIFLOOD_TTL_SEC` | `120` | 入力・出力の短時間重複を抑止するTTL（秒） |
| `REPLAY_GUARD_SEC` | `150` | LINE webhook の遅延イベントを無視するしきい値（秒） |
| `SUMMARY_STYLE` | `polite_long` | パンフレット要約の文体（`SUMMARY_MODE`=legacy用）。|
| `SUMMARY_MIN_CHARS` | `550` | 通常コンテキスト時の要約下限文字数 |
| `SUMMARY_MAX_CHARS` | `800` | 要約の上限文字数（句点で丸める） |
| `SUMMARY_MIN_FALLBACK` | `300` | コンテキストが少ない場合の最低文字数 |
| `SUMMARY_MODE` | `adaptive` | 回答長の自動調整モード。`adaptive`/`terse`/`long` を選択 |
| `CONTROL_CMD_ENABLED` | `true` | 「停止」「解除」コマンドの有効/無効 |
| `ENABLE_EVIDENCE_TOGGLE` | `true` | Web UI で根拠付き本文のトグルを表示するかどうか |

アップロードは 64MB まで受け付けますが、ブラウザからの編集保存は軽量テキスト運用を想定して 2MB で上限判定します。

### 5. 出典ラベル付きRAG応答

- RAGの自動要約は、利用した文脈ごとに `[[1]]` のようなラベルを本文各文末に付与します。
- 表示時はラベルを除去した本文と、トグルで確認できるラベル付き本文を同時に生成します。
- 出典欄には実際に本文で参照したドキュメントだけが並び、`CITATION_MIN_CHARS` / `CITATION_MIN_SCORE` の閾値で自動的にフィルタリングされます。
- しきい値を下げると出典が増え、上げるとノイズを抑えられます。運用環境に合わせて `.env` で調整してください。
- LLMがラベルを生成できなかった場合は本文末尾に注意書きが自動で追加され、出典表示は非表示になります。
- `SUMMARY_MODE=adaptive` にすると質問意図から short/medium/long を自動選択し、2〜4文から最大800字まで出し分けます。

---

## ロールバック / バックアップ運用

Render の本番環境で "失敗したら即時復元" を実現するための自動スナップショットとロールバック機構を搭載しています。詳細な手順・SLA は `docs/runbook.md` を参照してください。ここではサマリのみ記載します。

### コマンドライン

- 事前バックアップ: `scripts/backup_all.sh --notes "deploy YYYYMMDD"`
- 最新スナップショットへ復元: `scripts/restore_all.sh --reason manual`
- カナリアヘルスチェック: `python -m manage.rollback canary --snapshot <ID>`

### 管理UI

`/admin/rollback`（管理IP + ログイン必須）に復元ボタンと履歴一覧を追加しました。二段階確認とCSRF対策を兼ね備えています。

### API / CLI

- CLI: `python -m manage.rollback restore --snapshot <ID>`
- API: `POST /admin/rollback/api/restore {"snapshot_id": "..."}`

### 主要環境変数

| 変数名 | 既定値 | 説明 |
| ------ | ------ | ---- |
| `DATA_BASE_DIR` | `/var/data` | ユーザーデータ領域（tar.gz でスナップショット） |
| `BACKUP_DIR` | `/var/tmp/backup` | スナップショットの保存先 |
| `BACKUP_RETENTION` | `14` | 保存世代数。超過分は自動削除 |
| `ROLLBACK_READY_TIMEOUT_SEC` | `90` | カナリア監視のタイムアウト |
| `ROLLBACK_READY_INTERVAL_SEC` | `10` | ヘルスチェックのポーリング間隔 |
| `ROLLBACK_CANARY_ENABLED` | `true` | カナリア自動実行の有効/無効 |
| `ALLOW_ADMIN_ROLLBACK_IPS` | `` | 管理UI/APIにアクセスできるCIDR（カンマ区切り） |

Render のデプロイ前にはバックアップを取得し、デプロイ後はカナリアが `/readyz` を監視します。NG の場合は自動で直前スナップショットへ巻き戻します。

### 6. 入力ガードと自動返信抑止

- `ANTIFLOOD_TTL_SEC` で指定した期間は、同一ユーザーの空入力・極端に短い入力・直前と同一内容を自動的に無視します。
- 送信済み応答のハッシュも同じTTLでキャッシュし、障害時の重複送信や無限ループを抑止します。
- `/tests/smoke/rag_demo.py` を実行すると、ラベル付き要約から出典抽出までの一連の流れをダミーデータで確認できます。

### 4. インデックス再構築

パンフレットを追加・更新した際は、管理画面から `GET /admin/pamphlet-reindex` を呼び出すと全市町分のインデックスを再構築します。レスポンスには各市町のビルド結果が含まれます。 `/readyz` でも `pamphlet_index` の状態（`ready`/`empty`/`error` など）を確認できます。

### 5. 管理画面でのパンフレット運用

- `/admin/pamphlets` から市町ごとのテキストファイルを一覧・削除・アップロードできます（`.txt` のみ）。
- 「再インデックス」ボタンを押すと `/admin/pamphlet-reindex` が呼び出され、結果がトースト表示されます。
- Render 運用では Persistent Disk（例: 10GB）を `/var/data` にマウントし、環境変数 `PAMPHLET_BASE_DIR=/var/data/pamphlets` を指定してください。`pamphlets` フォルダ配下に各市町フォルダが自動作成されます。デプロイ手順は `docs/release_checklist.md` も参照してください。

### 7. ヘルスチェックエンドポイント

- `/readyz` では次の JSON を返します。CI で green のコミットが本番に反映されているかを確認する際に使用してください。

```json
{
  "status": "ready",
  "pamphlet_index": "ready",
  "pamphlet_index_status": {"goto": {"state": "ready"}, ...},
  "build": {
    "commit": "<Git SHA>",
    "branch": "develop",
    "env": "production"
  }
}
```

- `commit` / `branch` は Render の `RENDER_GIT_*` 環境変数または Git リポジトリから自動取得されます。`status` が `degraded` の場合は `errors` 配列に原因が格納されます。

### 6. LINE 応答での挙動

- 市町が特定できない質問には一度だけ次のクイックリプライを提示します。
  - 「五島市」「新上五島町」「小値賀町」「宇久町」
- 回答末尾には「出典（パンフレット名）」が付与されます。
- 追加情報が必要な場合は、ユーザーが「もっと詳しく」を選ぶと詳細要約を再提示します。
- LINE 停止/再開コマンド（「停止」「停止 60分」「解除」「再開」など）は SQLite でユーザー単位に永続化されます。
- 既定の保存先は `./data/system/user_state.sqlite`（本番は `/var/data/system/user_state.sqlite`）です。存在しない場合は自動作成されます。
  - コントロールの既定 TTL は `PAUSE_DEFAULT_TTL_SEC=86400`（24時間）で、メッセージ内の `60分`、`2h`、`1日` といった表記で上書きできます。
  - 環境変数 `CONTROL_CMD_ENABLED=true` のときに有効化されます。無効化すると旧挙動（ファイルフラグ）のみで動作します。
  - 管理者が `/admin/line/resume` を実行すると、保存済みの停止状態が全ユーザー分クリアされます。
- テストで制御コマンドの挙動を確認する場合は `pytest tests/test_control_commands.py tests/test_control_flow.py -q` を実行してください。
- ログには制御コマンドの状態遷移が `CTRL action=...`、状態確認が `STATE uid=...` 形式で出力されます。`解除` 直後に `STATE ... is_paused=False` が記録されているかで即時復帰を確認できます。

### 8. ルーティングとロギング

- ルーティングは `special → tourism → pamphlet → no_answer` の優先順で判定され、決定後のフォールバック実行は行いません。
- 特殊ハンドラ（天気・運行状況など）がヒットした場合は `ROUTE=special` がログに出力されます。観光DBヒット時は `ROUTE=tourism`、パンフレット回答時は `ROUTE=pamphlet`、いずれも未ヒットの場合は `ROUTE=no_answer` で記録されます。
- パンフレット回答では `CITATIONS=[{doc_id,title}]` が INFO ログに出力され、利用した出典のみが列挙されます。
- AntiFlood の判定結果は `ANTIFLOOD key=<...> hit=<true|false>` としてログに残るため、意図しない抑止が発生していないか確認できます。

## テスト

開発中は最低限下記の Pytest を通してください。パンフレットまわりは OpenAI API キーなしでもフォールバックで動作します。

```bash
pytest tests/test_routing_priority.py \
       tests/test_citations_guard.py \
       tests/test_summary_length.py \
       tests/test_controls_antiflood.py \
       tests/test_admin_smoke.py
```

管理画面や既存の制御コマンドの回帰を一括で確認したい場合は `pytest` を無引数で実行してください。

## デプロイ手順（ステージング→本番）

1. ステージング環境で `pytest` を全件実行し緑化する。
2. ステージングの Render サービスをデプロイし、以下の手動スモークを実施する。
   - `/healthz` と `/readyz` が 200/`ready` を返すこと。
   - LINE 実機で「停止 2分 → 解除 → 通常発話」が即時復帰すること。
   - 「今日の天気」「〇〇（観光地名）」で special/tourism が優先されること。
   - 「もっと詳しく」でパンフレット要約が 300〜800 字で返り、出典が利用分のみ表示されること。
   - 管理画面で観光データ CRUD、パンフレット登録/更新/検索、バックアップ ZIP/URL 復元、設定画面、ログ/未ヒット一覧が正常に動作すること。
3. `develop` を `main` にマージし、本番 Render をカナリア（ワーカー1）でデプロイする。
4. カナリアで上記スモークを再実施し、15分程度ログの `ROUTE`/`CTRL`/`CITATIONS`/`ANTIFLOOD` を監視する。
5. 問題なければワーカー数を通常値に戻し本番リリース完了。

### ロールバック

- Render の直前リリース（前バージョン）へ即時ロールバックできるよう、デプロイ履歴と Git コミットを事前確認しておきます。
- DB スキーマ変更を行う場合は後方互換を維持するか、ダウングレード手順を本 README に追記してください。
- LINE 公式アカウントの応答設定（あいさつ／自動返信 OFF）や `.env` のキー差異も、本番反映前に確認します。
