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
