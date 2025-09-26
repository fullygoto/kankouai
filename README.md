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

### 2. 主な環境変数（.env）

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `LINE_CHANNEL_ACCESS_TOKEN` / `LINE_CHANNEL_SECRET` | なし | LINE公式アカウントのチャネル情報。必須。 |
| `DATA_BASE_DIR` | `.` | `entries.json` や AntiFlood SQLite を配置するルートディレクトリ。Render では `/var/data` 等を指定。 |
| `ANTIFLOOD_TTL_SEC` | `180` | 「ユーザーID + メッセージID + 正規化テキスト」をキーにしたAntiFloodのTTL。再送イベントはTTL内なら自動で無視されます。 |
| `REPLAY_GUARD_SEC` | `150` | この秒数より古いイベントを再生ガードとして破棄。遅延再送・リトライの暴走を防ぎます。 |
| `RECENT_TEXT_TTL_SEC` | `min(ANTIFLOOD_TTL_SEC, 60)` | 同一本文の短時間連投を抑止するTTL。 |
| `ENABLE_PUSH` | `true` | `false` の場合は push 送信を完全停止し、reply のみで運用します。 |
| `ANTIFLOOD_REDIS_URL` / `REDIS_URL` | なし | Redis を使う場合は指定すると AntiFlood が Redis バックエンドを利用します。未設定時は `DATA_BASE_DIR/system/antiflood.sqlite` が使われます。 |

> **メモ:** SQLite バックエンドは `DATA_BASE_DIR` 配下に `system/antiflood.sqlite` を自動生成します。Render では永続ボリューム上にこのディレクトリを置くことで、再起動後も冪等キーが保持されます。

### 3. テスト実行

AntiFlood や webhook の冪等性を検証する Pytest を同梱しています。デプロイ前に必ず実行してください。

```bash
pytest
```

### 4. Render へのデプロイ手順

1. Render ダッシュボードで `DATA_BASE_DIR=/var/data` など永続ボリューム上のディレクトリを設定します。
2. `ANTIFLOOD_TTL_SEC`・`REPLAY_GUARD_SEC`・`ENABLE_PUSH` を本番と同じ値で登録し、Redis を使う場合は `ANTIFLOOD_REDIS_URL` も追加します。
3. Web サービスは 1 ワーカー運用が推奨です。APScheduler などを追加する場合も `scheduler.start()` は 1 度だけ呼び出す構成にしてください。
4. デプロイ後に Render のログで `/callback` が即時 200 を返していること、重複 push が出ていないことを確認します。
5. ステージング / 本番それぞれで LINE の開発者ツールを使い、同一メッセージ再送や遅延再送が無視されることを実機で確認します。

### 5. ロールバック方法

環境変数のみで制御できるため、緊急時は以下の手順で旧挙動に戻せます。

1. `ENABLE_PUSH=false` にすると push 送信が全停止し、reply のみになります。
2. `ANTIFLOOD_TTL_SEC=0` と `REPLAY_GUARD_SEC=0` を設定すると AntiFlood / 再生ガードが無効化されます（再試行時は重複応答が発生する点に注意）。
3. 変更後に Render を再デプロイすると即座に反映されます。
