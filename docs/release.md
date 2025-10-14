# 本番リリース運用ガイド

## 概要

本番デプロイは GitHub Actions の `Release to Production` ワークフローが担当します。`main` ブランチまたは `release/**` タグが更新されると、Render へのデプロイとヘルスチェック、ロールバック制御が自動で実行されます。本書では自動リリースと手動作業の両方について手順をまとめます。

## 自動リリース手順

1. `develop` を `main` にマージします。
2. `Release to Production` ワークフローが起動し、以下を順番に実行します。
   - `pytest -q` によるテスト。
   - Render Deploy API を用いた `NEW_SHA` のデプロイ。
   - `/readyz` を 2 秒間隔で最長 10 分間ポーリングし、200 が返るまで待機。
   - `/healthz` によるスモークテスト。
   - ヘルス合格後、`ops/prod-pin` ブランチの `ops/PROD_COMMIT` を最新 SHA に更新。
3. ワークフロー完了後、Render 上のアプリが最新コミットで稼働し、`ops/PROD_COMMIT` にも同じ SHA が記録されます。

### タグリリース

本番リリースが安定したら `main` の最新コミットに `release/YYYYmmdd-HHMM` 形式のタグを付与してください。タグ push でも同じワークフローが実行されます。

## 手動リリース手順

緊急時などで GitHub Actions を使わずにリリースする場合は、以下の手順に従います。

1. `ops/predeploy.sh` を Render の pre-deploy hook で実行し、/var/data のバックアップを取得します。
2. Render ダッシュボードで対象サービスのデプロイをトリガーし、`main` ブランチの対象コミットを選択します。
3. デプロイ完了後、`curl -f $PROD_BASE_URL/readyz` でヘルス確認を行い、成功したら `ops/prod-pin` ブランチの `ops/PROD_COMMIT` を更新します。

## ロールバック手順

### 自動ロールバック

`Release to Production` ワークフローが `/readyz` を 10 分以内に 200 へ復帰できなかった場合、`rollback` ジョブが発火します。直前に保存していた `PREV_SHA` を Render に再デプロイし、`/readyz` と `/healthz` の復旧を待ちます。ロールバック後はワークフローが `failure` で終了しますが、ログにロールバック結果が残ります。

### 手動ロールバック

環境変数 `RENDER_API_KEY`、`RENDER_SERVICE_ID`、`PROD_BASE_URL` を指定して `ops/rollback.sh` を実行すると、`ops/prod-pin` ブランチに保存されている `ops/PROD_COMMIT` の SHA へロールバックします。内部で Render Deploy API を呼び出し、ヘルスチェックまで行います。

```
RENDER_API_KEY=... RENDER_SERVICE_ID=... PROD_BASE_URL=https://example.onrender.com \
  ops/rollback.sh
```

## バックアップ運用

- Render の Persistent Disk は `/var/data` にマウントされます。
- デプロイ前に `ops/predeploy.sh` が `/var/data/_backups` を作成し、`YYYYmmdd-HHMM.tar.gz` 形式でバックアップを保存します。
- `_backups` ディレクトリでは最新 5 世代のみ保持し、それ以前のアーカイブは自動で削除されます。

## Secrets 設定

GitHub Actions で以下のリポジトリシークレットを設定してください。

| Name | 用途 |
| ---- | ---- |
| `RENDER_API_KEY` | Render Deploy API を呼び出すための API Key |
| `RENDER_SERVICE_ID` | 本番サービスの ID |
| `PROD_BASE_URL` | 本番 URL (例: `https://example.onrender.com`) |

## トラブルシューティング

- **GitHub Actions が `ops/prod-pin` を更新できない**: `release-prod.yml` に `permissions: contents: write` を指定しています。Organization 側のポリシーで制限していないか確認してください。
- **Render API が 5xx を返す**: ワークフローやスクリプトでは指数バックオフ付きリトライを実装しています。繰り返し失敗する場合は Render のステータスを確認し、必要に応じてサポートへ問い合わせてください。
- **/readyz が復帰しない**: 自動ロールバックが実行されます。`logs` タブで旧コンテナのログと、Render の Rolling Deploy 状態を確認してください。
- **Persistent Disk の空き容量不足**: `_backups` ディレクトリに古いアーカイブが残っていないか確認し、必要であれば世代数を一時的に増減してください。

## 参考

より詳細な runbook や既存のチェックリストは `docs/runbook.md` および `docs/release_checklist.md` を参照してください。
