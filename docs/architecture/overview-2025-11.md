# GOTO観光AIシステム - アーキテクチャ概要

## リポジトリ構成

このドキュメントは、GOTO観光AIシステムのリポジトリ構成と各ディレクトリ・ファイルの役割を説明します。

---

## 主要ディレクトリ構成

### 1. `/coreapp` - コアアプリケーションモジュール

システムの中核となるビジネスロジックを格納するディレクトリです。

- **`/coreapp/responders`** - 応答生成ハンドラー
  - `entries.py` - 観光データベース検索からの応答生成
  - `fallback.py` - フォールバック処理（未ヒット時の対応）
  - `pamphlet.py` - パンフレット検索とRAG応答生成
  - `priority.py` - 応答の優先順位判定

- **`/coreapp/search`** - 検索エンジン実装
  - `entries_index.py` - 観光データベースのインデックス構築と検索
  - `pamphlet_index.py` - パンフレットの全文検索インデックス管理
  - `normalize.py` - 検索クエリの正規化処理
  - `query_limits.py` - クエリの制限と最適化

- **その他のコアモジュール**
  - `config.py` - 設定管理
  - `intent.py` - 意図検出（ユーザーの質問意図を判定）
  - `llm.py` - LLM（大規模言語モデル）とのインターフェース
  - `logging_utils.py` - ログ出力のユーティリティ
  - `schemas.py` - データスキーマ定義
  - `state.py` - セッション状態管理
  - `storage.py` - データ永続化層

### 2. `/services` - サービスレイヤー

アプリケーション固有のサービスロジックを実装したモジュール群です。

- `dupe_guard.py` - 重複送信の防止（AntiFlood機能）
- `input_normalizer.py` - 入力データの正規化
- `line_handlers.py` - LINE Messaging APIのイベントハンドラー
- `manifest.py` - システムマニフェスト情報
- `message_builder.py` - メッセージの組み立て
- `pamphlet_*.py` - パンフレット機能群
  - `pamphlet_constants.py` - パンフレット関連の定数定義
  - `pamphlet_flow.py` - パンフレット回答のフロー制御
  - `pamphlet_planner.py` - 回答プラン作成
  - `pamphlet_rag.py` - RAG（Retrieval-Augmented Generation）実装
  - `pamphlet_search.py` - パンフレット検索機能
  - `pamphlet_session.py` - パンフレットセッション管理
  - `pamphlet_store.py` - パンフレットデータストア
  - `pamphlet_summarize.py` - パンフレット要約生成
- `paths.py` - ファイルパス管理
- `rollback_service.py` - ロールバック機能のサービス層
- `route_selector.py` - ルーティング判定（special → tourism → pamphlet → no_answer）
- `sources_fmt.py` - 出典情報のフォーマット
- `state.py` - ユーザー状態管理
- `summary_config.py` - 要約生成の設定管理
- `tourism_search.py` - 観光データ検索

### 3. `/templates` - HTMLテンプレート

Flask/Jinja2で使用するHTMLテンプレートファイル群です。

- **管理画面テンプレート**
  - `admin_dashboard.html` - 管理画面ダッシュボード
  - `admin_entries_edit.html` - 観光データ一括編集
  - `admin_entry.html` - 観光データ個別編集
  - `admin_backup.html` - バックアップ管理
  - `admin_logs.html` - ログ閲覧
  - `admin_pamphlets.html` - パンフレット管理（推定、admin/ディレクトリ内）
  - `admin_synonyms.html` - 類義語管理
  - `admin_unhit.html` / `admin_unhit_questions.html` - 未ヒット質問分析
  - `admin_notices.html` - お知らせ管理
  - `admin_media_*.html` - メディアファイル管理
  - `admin_watermark*.html` - 透かし設定

- **共通パーツ**
  - `_admin_nav.html` - 管理画面ナビゲーション
  - `_back_to_data.html` - データ画面への戻るリンク

- **その他**
  - `login.html` - ログイン画面
  - `notices.html` - お知らせ表示

### 4. `/tests` - テストコード

Pytestベースのテストスイート群です。

- **機能別テスト**
  - `test_routing_*.py` - ルーティングロジックのテスト
  - `test_pamphlet_*.py` - パンフレット機能のテスト
  - `test_control_*.py` - 制御コマンド（停止・再開）のテスト
  - `test_citations_*.py` - 出典表示機能のテスト
  - `test_admin_*.py` - 管理画面のテスト
  - `test_entries*.py` - 観光データ検索のテスト
  - `test_intent_detection.py` - 意図検出のテスト
  - `test_summary_*.py` - 要約生成のテスト

- **サブディレクトリ**
  - `/tests/admin` - 管理機能の詳細テスト
  - `/tests/regression` - リグレッションテスト
  - `/tests/smoke` - スモークテスト

- **ユーティリティ**
  - `conftest.py` - Pytest設定とフィクスチャ定義
  - `utils.py` - テスト用ユーティリティ関数

### 5. `/scripts` - 運用スクリプト

本番環境での運用に使用するシェルスクリプト群です。

- `backup_all.sh` - 全データのバックアップ
- `restore_all.sh` - バックアップからの復元
- `rollback_to_tag.sh` - 指定タグへのロールバック
- `tag_release.sh` - リリースタグ作成
- `verify_restore.sh` - 復元後の検証
- `var_data_metrics.sh` - データメトリクス収集
- `smoke_release.py` - リリース後のスモークテスト

### 6. `/manage` - 管理コマンド

CLIから実行する管理コマンドです。

- `rollback.py` - ロールバック操作のCLIインターフェース

### 7. `/ops` - 運用設定

本番環境での運用に関連する設定ファイル群です。

- `predeploy.sh` - デプロイ前に実行されるスクリプト
- `rollback.sh` - 自動ロールバックスクリプト
- `PROD_COMMIT` - 現在の本番環境コミットハッシュ

### 8. `/docs` - ドキュメント

システムのドキュメント類を格納するディレクトリです。

- `response-spec-v1-notes.md` - 応答仕様のメモ
- `admin_bulk_edit.md` - 管理画面での一括編集仕様
- `pamphlet_fallback.md` - パンフレットフォールバックの仕様
- `release.md` - リリース手順
- `release_checklist.md` - リリース時のチェックリスト
- `runbook.md` - 運用マニュアル
- `/docs/architecture` - アーキテクチャドキュメント（本ディレクトリ）

### 9. `/data` - データファイル

システムで使用するマスターデータです。

- `synonyms.json` - 類義語辞書のシードデータ

### 10. `/codex` - Codex設定

Codex（AI開発支援）関連の設定ファイルです。

- `README.md` - Codex設定の説明
- `prompt.md` - Codexプロンプト定義

### 11. `/logs` - ログディレクトリ

アプリケーションのログファイルを格納するディレクトリです（実行時に生成）。

---

## ルートディレクトリのファイル

### メインアプリケーション

- **`app.py`** - Flaskアプリケーションのメインエントリーポイント
  - ルーティング定義
  - 管理画面の実装
  - LINE Webhook エンドポイント
  - ヘルスチェックエンドポイント（`/healthz`, `/readyz`）

- **`wsgi.py`** - WSGIサーバー（Gunicorn等）用のエントリーポイント

### 設定・管理スクリプト

- **`config.py`** - アプリケーション全体の設定管理
  - 環境変数の読み込み
  - データベース設定
  - パンフレット設定
  - レート制限設定

- **`admin_pamphlets.py`** - パンフレット管理の独立スクリプト
- **`admin_rollback.py`** - ロールバック管理の独立スクリプト

### ルーティング・メッセージング

- **`conversational_router.py`** - 会話型ルーティングの中核ロジック
  - 質問の種別判定（special / tourism / pamphlet / no_answer）
  - 各ハンドラーへのディスパッチ

### 画像処理

- **`watermark_ext.py`** - 透かし処理の拡張実装
- **`watermark_utils.py`** - 透かし処理のユーティリティ関数

### データファイル（JSON）

- **`notices.json`** - お知らせ情報
- **`shop_infos.json`** - 店舗情報
- **`synonyms.json`** - 類義語辞書（実行時に使用）
- **`users.json`** - 管理者ユーザー情報（初回起動時に生成）

### 環境設定・依存関係

- **`.env.example`** - 環境変数のサンプルファイル
- **`requirements.txt`** - Pythonパッケージの依存関係定義
- **`render.yaml`** - Render.comでのデプロイ設定

### テスト設定

- **`pytest.ini`** - Pytestの設定ファイル

### その他設定

- **`.gitignore`** - Gitで追跡しないファイル・ディレクトリの定義
- **`.codexignore`** - Codexで無視するファイルの定義
- **`.github/`** - GitHub Actions等のCI/CD設定

---

## アーキテクチャの特徴

### レイヤー構成

1. **プレゼンテーション層** (`app.py`, `/templates`)
   - Flaskによるウェブインターフェース
   - LINE Messaging APIとの統合

2. **サービス層** (`/services`)
   - ビジネスロジックの実装
   - 各種サービスの調整

3. **コア層** (`/coreapp`)
   - ドメインロジック
   - 検索・応答生成の中核機能

4. **データ層** (`/data`, JSON files)
   - ファイルベースのデータストレージ
   - SQLiteデータベース（実行時に生成）

### 主要な機能モジュール

1. **ルーティングシステム**
   - `conversational_router.py` - メインルーター
   - `services/route_selector.py` - ルート選択ロジック
   - 優先順位: special → tourism → pamphlet → no_answer

2. **検索システム**
   - 観光データ検索（`coreapp/search/entries_index.py`）
   - パンフレット全文検索（`coreapp/search/pamphlet_index.py`）

3. **RAGシステム**
   - パンフレットからの回答生成
   - 出典付き応答の自動生成
   - 要約長の自動調整（adaptive mode）

4. **管理システム**
   - データ登録・編集
   - バックアップ・復元
   - ログ分析
   - 未ヒット質問の分析

5. **運用支援システム**
   - ヘルスチェック（`/healthz`, `/readyz`）
   - ロールバック機能
   - カナリアデプロイ対応

### データフロー

```
ユーザー入力
  ↓
LINE Webhook / Web UI (app.py)
  ↓
入力正規化 (services/input_normalizer.py)
  ↓
ルーティング判定 (conversational_router.py)
  ↓
┌─────────────────────────────────┐
│ 特殊ハンドラー（天気・運行状況等）│ → 応答
├─────────────────────────────────┤
│ 観光DB検索 (responders/entries) │ → 応答
├─────────────────────────────────┤
│ パンフレットRAG (responders/pamphlet) │ → 応答（出典付き）
├─────────────────────────────────┤
│ フォールバック (responders/fallback) │ → 定型応答
└─────────────────────────────────┘
  ↓
メッセージ構築 (services/message_builder.py)
  ↓
LINE返信 / Web表示
```

---

## 開発・運用のポイント

### 開発時

- テストは `/tests` ディレクトリで管理
- `pytest` でテスト実行
- コア機能の変更時は必ず関連テストを実行

### デプロイ

- ステージング環境で全テスト実行
- `develop` → `main` へマージ
- Renderでの自動デプロイ
- `/readyz` エンドポイントで状態確認

### 運用

- `/scripts` のバックアップスクリプトを定期実行
- `/admin` で管理画面にアクセス
- ログは `/logs` または `/var/data/logs` に保存
- 問題発生時は `/ops/rollback.sh` で復元

---

## 更新履歴

- 2025-11: 初版作成
