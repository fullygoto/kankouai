# ロールバック運用 Runbook

fully観光AI (Flask + Render) の本番環境で、デプロイ失敗時に 1 分以内で直前状態へ完全復元するための手順をまとめます。アプリコード・設定・`/var/data`・Postgres のスキーマ/データを対象とします。

## 1. 構成概要

- **バックアップ対象**: アプリ Git リビジョン、`/var/data`（tar.gz + SHA256）、Postgres 論理ダンプ（`pg_dump -Fc`）、Alembic 現在リビジョン。
- **保存場所**: `BACKUP_DIR`（既定: `/var/tmp/backup`）。世代管理は `BACKUP_RETENTION`（既定 14 世代）。
- **マニフェスト**: `BACKUP_DIR/manifest.json` に各スナップショットのメタデータ（ハッシュ、サイズ、アプリ/DBリビジョン）を記録。
- **実行手段**: CLI (`manage/rollback.py`)、シェルスクリプト (`scripts/backup_all.sh` / `scripts/restore_all.sh`)、管理UI (`/admin/rollback`) および HTTP API (`/admin/rollback/api/restore`)。
- **ログ**: `logs/rollback.log` に `backup start/complete`、`rollback start/complete`、カナリア判定などを追記。

## 2. 事前準備

1. Render のサービス設定で以下の環境変数を設定
   - `DATA_BASE_DIR=/var/data`
   - `BACKUP_DIR=/var/tmp/backup`
   - `BACKUP_RETENTION=14`（必要に応じ変更）
   - `ROLLBACK_CANARY_ENABLED=true`
   - `ROLLBACK_READY_TIMEOUT_SEC=90`
   - `ROLLBACK_READY_INTERVAL_SEC=10`
   - `ALLOW_ADMIN_ROLLBACK_IPS=<社内IP>/32,...`
2. Render の Persistent Disk を `/var/data` にマウント。
3. Alembic マイグレーションは必ず `downgrade` が定義されていることを確認。不可逆な変更が必要な場合は代替パスを runbook に明記し、フェイルセーフを設計。
4. CI/CD（Render build hook）でデプロイ前に `scripts/backup_all.sh --notes "deploy $(date -u +%Y%m%dT%H%M%SZ)"` を実行。

## 3. デプロイ手順（通常運用）

1. `scripts/backup_all.sh` を実行し、最新スナップショットを作成。`logs/rollback.log` と `BACKUP_DIR/manifest.json` を確認。
2. Render のステージング環境で `python -m manage.rollback canary` を実行し、/readyz が正常であることを確認。
3. 本番にデプロイ。Render の "Canary deploy" 機能に合わせてワーカー 1 台のみで起動。
4. デプロイ完了後、`python -m manage.rollback canary --snapshot <最新ID>` を Render の deploy hook から実行。
5. カナリアが成功すると自動的に manifest の `ready_check` が更新され、残りのワーカーを起動（Render 側の自動スケールに任せる）。

## 4. 自動ロールバック

- カナリアが `ROLLBACK_READY_TIMEOUT_SEC` 内に `/readyz` の HTTP 200 を確認できない場合、`manager.restore_latest_backup(reason="readyz_fail", auto=True)` を実行。
- 処理順序
  1. Postgres: Alembic `downgrade` → `pg_restore` による論理復元。
  2. `/var/data`: tar.gz 展開 → 既存ディレクトリを `.pre-<txn>` に退避。
  3. アプリコード: Git `checkout <snapshot.app_revision>` （Render では直前デプロイへ戻すのに対応）。
  4. `/readyz` が OK になるまで監視。失敗時は `.pre-<txn>` から手動再実行が可能。
- ログ形式: `ROLLBACK id=<txn> step=db|data|code status=...` を `logs/rollback.log` に追記。

## 5. 手動ロールバック（運用チーム）

### 管理UI

1. 管理IPから `/admin/rollback` にアクセスし、ログイン。
2. 最新スナップショット一覧を確認し、「直前スナップショットへ復元...」ボタンを押す。
3. 二段階確認で「復元を実行する」を選択。
4. 処理中は Render のログと `logs/rollback.log` を監視。`/readyz` が 200 を返したら復旧完了。

### CLI / API

- CLI: `python -m manage.rollback restore --snapshot <ID> --reason manual`
- API: `curl -X POST https://<host>/admin/rollback/api/restore -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' -d '{"snapshot_id": "<ID>"}'`

## 6. DB マイグレーションの安全手順

1. 新規マイグレーションを作成する際は必ず `upgrade` / `downgrade` の両方を定義。
2. 破壊的操作（列削除など）は、事前に `soft-delete` や `rename` と `copy` で段階的に行い、直近 2 リリース分はロールバック可能な状態を維持。
3. マイグレーション本番実行前にステージング環境で `alembic upgrade head` → `alembic downgrade -1` を実施。
4. 自動テスト `pytest -q` に `test_restore_invokes_alembic_downgrade` が含まれており、rollback 経路で `downgrade` が呼び出されることを検証。

## 7. 監視とSLA

- `/readyz` は以下をチェック
  - 必須環境変数の存在（LINE鍵などはマスクログ）
  - `users.json` の読み取り
  - DB 接続および主要テーブル件数サニティ
  - Alembic リビジョンの一致
- SLA 目標: デプロイ失敗から 60 秒以内に直前スナップショットへ復旧し、ユーザー影響を最小化。
- `ROLLBACK_CANARY_ENABLED=false` の場合は自動ロールバックが無効になるため、手動で `/admin/rollback` から復元する。

## 8. よくある落とし穴

- **pg_dump/pg_restore が存在しない**: Render の Docker イメージに含まれていることを確認。無い場合は `apt install postgresql-client` を build hook で追加。
- **バックアップ容量不足**: `BACKUP_RETENTION` を短くするか、S3 等の外部ストレージに複製する。`BACKUP_DIR` の空き容量は監視対象に。
- **/var/data の権限問題**: バックアップ生成時に 600/700 権限を付与。Render の rootless 実行に合わせて chown も検討。
- **Alembic 不整合**: `alembic_version` テーブルと manifest の `alembic_revision` がずれた場合は、手動で `alembic downgrade/upgrade` し整合を取ってからロールバック。

## 9. 事後対応

1. ロールバック後は原因分析を実施し、恒久対応を issue 化。
2. `scripts/backup_all.sh` を再実行して新しいスナップショットを取得。
3. 修正したコードをステージングで再検証し、再デプロイ時も同手順を踏む。

---

本 Runbook は `docs/runbook.md` にバージョン管理されています。更新時は PR で運用チームに通知してください。
