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
- Github＋Render（無料枠）でノーコスト運用可能

---

## セットアップ

### 1. 必要パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. パンフレット検索フォールバックの準備

市町別パンフレットを以下のディレクトリ構成で配置します（既定パスは`/var/data/pamphlets`）。

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
| `PAMPHLET_BASE_DIR` | `/var/data/pamphlets` | パンフレットテキストの格納ルート |
| `PAMPHLET_TOPK` | `3` | 要約に渡す上位チャンク数 |
| `PAMPHLET_CHUNK_SIZE` | `1500` | チャンク長（文字数） |
| `PAMPHLET_CHUNK_OVERLAP` | `200` | チャンク重複幅（文字数） |
| `PAMPHLET_SESSION_TTL` | `1800` | 市町選択の保持時間（秒） |
| `MAX_UPLOAD_MB` | `64` | アップロード可能なファイル上限（Flask `MAX_CONTENT_LENGTH` に反映） |
| `PAMPHLET_EDIT_MAX_MB` | `2` | 管理画面でのテキスト編集時の保存上限（MB） |
| `CITATION_MIN_CHARS` | `80` | 1ドキュメントに紐づく引用テキストの最小合計文字数。未満なら出典から除外 |
| `CITATION_MIN_SCORE` | `0.15` | 検索スコア合算の下限。`CITATION_MIN_CHARS` を満たさなくてもスコアが基準を超えれば出典採用 |
| `ANTIFLOOD_TTL_SEC` | `120` | 同一ユーザーの重複入力や送信済み応答を抑止するTTL（秒） |
| `ENABLE_EVIDENCE_TOGGLE` | `true` | Web UI で根拠付き本文のトグルを表示するかどうか |

アップロードは 64MB まで受け付けますが、ブラウザからの編集保存は軽量テキスト運用を想定して 2MB で上限判定します。

### 5. 出典ラベル付きRAG応答

- RAGの自動要約は、利用した文脈ごとに `[[1]]` のようなラベルを本文各文末に付与します。
- 表示時はラベルを除去した本文と、トグルで確認できるラベル付き本文を同時に生成します。
- 出典欄には実際に本文で参照したドキュメントだけが並び、`CITATION_MIN_CHARS` / `CITATION_MIN_SCORE` の閾値で自動的にフィルタリングされます。
- しきい値を下げると出典が増え、上げるとノイズを抑えられます。運用環境に合わせて `.env` で調整してください。
- LLMがラベルを生成できなかった場合は本文末尾に注意書きが自動で追加され、出典表示は非表示になります。

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
