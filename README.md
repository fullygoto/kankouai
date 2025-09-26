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

### 4. インデックス再構築

パンフレットを追加・更新した際は、管理画面から `GET /admin/pamphlet-reindex` を呼び出すと全市町分のインデックスを再構築します。レスポンスには各市町のビルド結果が含まれます。 `/readyz` でも `pamphlet_index` の状態（`ready`/`empty`/`error` など）を確認できます。

### 5. 管理画面でのパンフレット運用

- `/admin/pamphlets` から市町ごとのテキストファイルを一覧・削除・アップロードできます（`.txt` のみ）。
- 「再インデックス」ボタンを押すと `/admin/pamphlet-reindex` が呼び出され、結果がトースト表示されます。
- Render 運用では Persistent Disk を `/var/data` にマウントし、環境変数 `PAMPHLET_BASE_DIR=/var/data/pamphlets` を指定してください。該当ディレクトリ配下にファイルが作成されます。

### 6. LINE 応答での挙動

- 市町が特定できない質問には一度だけ次のクイックリプライを提示します。
  - 「五島市」「新上五島町」「小値賀町」「宇久町」
- 回答末尾には「出典（パンフレット名）」が付与されます。
- 追加情報が必要な場合は、ユーザーが「もっと詳しく」を選ぶと詳細要約を再提示します。
