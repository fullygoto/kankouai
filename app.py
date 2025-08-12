import os
import json
import re
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, abort

from dotenv import load_dotenv
load_dotenv()

from werkzeug.security import check_password_hash

# LINE Bot関連
from linebot import LineBotApi, WebhookHandler  # ←この1行を追加
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError  # ←追加

from openai import OpenAI
import zipfile
import io

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # ← ここに追加（日本語をJSONでそのまま返す）
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")

from functools import wraps

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("ログインしてください")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


# === 環境変数 / 設定 ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

# ★ モデル切替用（.envで上書き可）
OPENAI_MODEL_PRIMARY = os.environ.get("OPENAI_MODEL_PRIMARY", "gpt-4o-mini")
OPENAI_MODEL_HIGH    = os.environ.get("OPENAI_MODEL_HIGH", "gpt-5-mini")
# 未ヒット時の挙動フラグ
REASK_UNHIT   = os.environ.get("REASK_UNHIT", "1").lower() in {"1", "true", "on", "yes"}
SUGGEST_UNHIT = os.environ.get("SUGGEST_UNHIT", "1").lower() in {"1", "true", "on", "yes"}
# 外国語対応フラグ
ENABLE_FOREIGN_LANG = os.environ.get("ENABLE_FOREIGN_LANG", "1").lower() in {"1", "true", "on", "yes"}

# === OpenAI v1 クライアント（環境変数から自動でAPIキー読込） ===
client = OpenAI()

# === LINE Bot ===
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === データ格納先（Renderの永続ディスクを使うなら DATA_BASE_DIR を設定） ===
BASE_DIR     = os.environ.get("DATA_BASE_DIR", ".")  # 例: /var/appdata
ENTRIES_FILE = os.path.join(BASE_DIR, "entries.json")
DATA_DIR     = os.path.join(BASE_DIR, "data")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
LOG_FILE     = os.path.join(LOG_DIR, "questions_log.jsonl")
SYNONYM_FILE = os.path.join(BASE_DIR, "synonyms.json")
USERS_FILE   = os.path.join(BASE_DIR, "users.json")
NOTICES_FILE   = os.path.join(BASE_DIR, "notices.json")
SHOP_INFO_FILE = os.path.join(BASE_DIR, "shop_infos.json")


# 必要フォルダ作成
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 必須キーが未設定なら警告（起動は継続）
if not OPENAI_API_KEY:
    app.logger.warning("OPENAI_API_KEY が未設定です。OpenAI 呼び出しは失敗します。")
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    app.logger.warning("LINE_CHANNEL_* が未設定です。/callback は正常動作しません。")


def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# === ログ保存用 ===
def save_qa_log(question, answer, source="web", hit_db=False, extra=None):
    os.makedirs(LOG_DIR, exist_ok=True)  # ← LOG_DIR を使う
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": source,
        "question": question,
        "answer": answer,
        "hit_db": hit_db,
        "extra": extra or {},
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

# === OpenAIラッパ（モデル切替を一元管理） ===

def openai_chat(model, messages, **kwargs):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("[OpenAI error]", e)
        return ""

# === 言語検出＆翻訳（繁体字/英語 → 日本語で解析し、元言語で返答） ===
OPENAI_MODEL_TRANSLATE = os.environ.get("OPENAI_MODEL_TRANSLATE", OPENAI_MODEL_HIGH)

def detect_lang_simple(text: str) -> str:
    """ja を既定。zh は明確な中国語マーカーがあるときのみ。en は英字のみっぽいとき。"""
    if not text:
        return "ja"
    t = text.strip()

    # かながあれば日本語
    if re.search(r"[ぁ-んァ-ン]", t):
        return "ja"

    # 明確な中国語マーカー（繁/簡どちらも）
    zh_markers = [
        "今天","天氣","天气","請問","请问","交通","景點","景点","門票","门票",
        "美食","住宿","船班","航班","預約","预约","營業","营业","營業時間","营业时间"
    ]
    if any(m in t for m in zh_markers):
        return "zh-Hant"

    # 英字のみ（数字・記号・空白は許容）なら英語とみなす
    if re.fullmatch(r"[A-Za-z0-9\s\-\.,!?'\"/()]+", t):
        return "en"

    # それ以外は日本語を既定
    return "ja"


def translate_text(text: str, target_lang: str) -> str:
    """翻訳: まず OPENAI_MODEL_TRANSLATE、失敗時は OPENAI_MODEL_PRIMARY で再試行。
    URL・数値・改行は保持する方針でプロンプト化。
    """
    if not text:
        return text

    lang_label = {'ja': '日本語', 'zh-Hant': '繁體中文', 'en': '英語'}.get(target_lang, '日本語')
    prompt = (
        f"次のテキストを{lang_label}に自然に翻訳してください。"
        "意味は変えず、URL・数値・記号・改行はそのまま維持。"
        "箇条書きや見出しの構造も保持してください。\n===\n" + text
    )

    # 1回目: 翻訳用モデル（既に上で決めている OPENAI_MODEL_TRANSLATE を使用）
    model_t = OPENAI_MODEL_TRANSLATE
    out = openai_chat(
        model_t,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200,
    )

    # 失敗時: プライマリで再試行
    if not out and model_t != OPENAI_MODEL_PRIMARY:
        out = openai_chat(
            OPENAI_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )

    return out or text

# === お知らせ・イベント・特売掲示板用 ===
def load_notices():
    if not os.path.exists(NOTICES_FILE):
        return []
    with open(NOTICES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notices(notices):
    with open(NOTICES_FILE, "w", encoding="utf-8") as f:
        json.dump(notices, f, ensure_ascii=False, indent=2)

# === 管理画面: 観光データ登録・編集 ===

def load_entries():
    if not os.path.exists(ENTRIES_FILE):
        return []
    with open(ENTRIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_entries(entries):
    with open(ENTRIES_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def suggest_tags_and_title(question, answer, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    prompt = (
        f"以下は観光案内AIへの質問とそのAI回答例です。\n"
        f"【質問】\n{question}\n"
        f"【AI回答】\n{answer}\n"
        f"---\n"
        f"この内容にふさわしい「登録用タイトル」と「カンマ区切りの日本語タグ案（5～10個）」をそれぞれ1行ずつ、下記の形式で出力してください。\n"
        f"タイトル: ～\n"
        f"タグ: tag1, tag2, tag3, tag4, tag5\n"
        f"---\n"
        f"※タグには質問の意図や主要キーワード、関連ジャンルも含めてください。"
    )
    try:
        content = openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        title, tags = "", ""
        for line in content.splitlines():
            if line.startswith("タイトル:"):
                title = line.replace("タイトル:", "").strip()
            elif line.startswith("タグ:"):
                tags = line.replace("タグ:", "").strip()
        return title, tags
    except Exception as e:
        print("[AIサジェストエラー]", e)
        return "", ""


def ai_suggest_faq(question, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    prompt = (
        f"以下の質問に対し、観光案内AIとして分かりやすいFAQ回答文（最大400文字）を作成してください。\n"
        f"質問: {question}\n"
        f"---\n"
        f"回答:"
    )
    try:
        return openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        ).strip()
    except Exception as e:
        print("[FAQサジェストエラー]", e)
        return ""


def generate_unhit_report(n=7):
    """直近n日分の未ヒット・誤答ログ集計"""
    import datetime as dt
    from collections import Counter

    logs = []
    try:
        with open(LOG_FILE, encoding="utf-8") as f:  # ← LOG_FILE を使用
            for line in f:
                d = json.loads(line)
                logs.append(d)
    except Exception as e:
        print("ログ集計エラー:", e)
        return []

    today = dt.datetime.now()
    recent = []
    for log in logs[::-1]:
        t = log.get("timestamp", "")
        try:
            dtv = dt.datetime.fromisoformat(t[:19])
            if (today - dtv).days <= n and not log.get("hit_db", False):
                recent.append(log)
        except:
            continue

    questions = [l["question"] for l in recent]
    counter = Counter(questions)
    return counter.most_common()

# === 類義語（シノニム）辞書のロード＆保存 ===

def load_synonyms():
    if not os.path.exists(SYNONYM_FILE):
        return {}
    with open(SYNONYM_FILE, encoding="utf-8") as f:
        return json.load(f)

def save_synonyms(synonyms):
    with open(SYNONYM_FILE, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)


def find_tags_by_synonym(question, synonyms):
    tags = set()
    for tag, synlist in synonyms.items():
        for syn in synlist + [tag]:
            if syn and syn in question:
                tags.add(tag)
    return list(tags)

# === AIによるタグ・類義語案サジェスト ===

def ai_suggest_synonyms(question, all_tags, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    prompt = (
        f"以下の質問について、適切な既存タグ（{', '.join(all_tags)}）や新しいタグに対して、日本語の類義語・言い換え案を5つずつタグごとに挙げてください。\n"
        f"質問: {question}\n"
        f"出力例:\nタグ: ビーチ\n類義語: 海水浴場, 砂浜, 泳げる場所, 水遊び, サンビーチ\n---\n"
    )
    try:
        return openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600,
        )
    except Exception as e:
        print("[類義語サジェストエラー]", e)
        return ""

# === 未ヒット時の絞り込み候補 ===
from collections import Counter

def get_top_tags(k=8):
    entries = load_entries()
    counter = Counter()
    for e in entries:
        for t in e.get("tags", []) or []:
            if t:
                counter[t] += 1
    if not counter:
        return ["教会", "五島うどん", "海水浴", "釣り", "温泉", "レンタカー", "カフェ", "体験"]
    return [t for t, _ in counter.most_common(k)]

def build_refine_suggestions(question):
    """未ヒット時に、エリア/タグ/フィルタの絞り込み候補を提示する。
    - タグはユーザーの質問語に応じて優先度を動的に並び替える
    - noticesのイベントから「今週のイベント」有無を検知
    - 雨関連語が含まれる場合は「雨の日OK」フィルタを提示
    """
    areas = ["五島市", "新上五島町", "小値賀町", "宇久町"]

    # 元データから頻出タグを取得
    top_tags = get_top_tags()

    # 内部関数: 質問語に合わせてタグの優先度を並び替え
    def rank_tags(tags, q):
        q = q or ""
        prefs = {
            "海": ["海水浴", "ビーチ", "釣り"],
            "ビーチ": ["海水浴", "ビーチ"],
            "釣り": ["釣り", "船釣り"],
            "教会": ["教会"],
            "うどん": ["五島うどん"],
            "温泉": ["温泉"],
            "雨": ["温泉", "カフェ", "資料館", "美術館", "屋内", "体験"],
            "子連れ": ["体験", "公園", "動物"],
        }
        score = {t: 0 for t in tags}
        for key, pref_list in prefs.items():
            if key in q:
                for w, t in enumerate(pref_list):
                    if t in score:
                        score[t] += (100 - w)
        ranked = sorted(tags, key=lambda t: (-score.get(t, 0), tags.index(t)))
        return ranked

    # 内部関数: 今週のイベント有無
    def has_events_this_week():
        try:
            notices = load_notices()
        except Exception:
            return False
        today = datetime.date.today()
        start_week = today - datetime.timedelta(days=today.weekday())
        end_week = start_week + datetime.timedelta(days=6)
        for n in notices or []:
            if n.get("category") != "イベント":
                continue
            s = n.get("start_date") or n.get("date")
            e = n.get("end_date") or s
            try:
                sd = datetime.date.fromisoformat(s) if s else None
                ed = datetime.date.fromisoformat(e) if e else None
            except Exception:
                continue
            if not sd and not ed:
                continue
            if sd is None:
                sd = ed
            if ed is None:
                ed = sd
            if not (ed < start_week or sd > end_week):
                return True
        return False

    # 並び替え適用
    top_tags = rank_tags(top_tags, question)

    # フィルタ候補
    filters = []
    if "雨" in (question or ""):
        filters.append("雨の日OK（屋内スポット）")
    if has_events_this_week():
        filters.append("今週のイベント")
    if any(w in (question or "") for w in ["子連れ", "家族", "ファミリー"]):
        filters.append("子連れOK")

    # 文面生成
    area_line = " / ".join(areas)
    tag_line = ", ".join(top_tags)
    examples = []
    if areas and top_tags:
        examples.append(f"『{areas[0]}の{top_tags[0]}』")
    if len(areas) > 1 and len(top_tags) > 1:
        examples.append(f"『{areas[1]}の{top_tags[1]}』")
    if len(top_tags) > 2:
        examples.append(f"『{top_tags[2]} の人気スポット』")

    lines = [
        "🔍 絞り込み候補",
        f"- エリア: {area_line}",
        f"- タグ例: {tag_line}",
    ]
    if filters:
        lines.append(f"- フィルタ: {', '.join(filters)}")
    if examples:
        lines.append("- 例: " + " / ".join(examples))

    msg = "\n".join(lines)
    return msg, {"areas": areas, "tags": top_tags, "filters": filters}

@app.route("/admin/entry", methods=["GET", "POST"])
@login_required
def admin_entry():
    if session.get("role") == "shop":
        return redirect(url_for("shop_entry"))
    entries = load_entries()
    edit_id = request.args.get("edit")
    entry_edit = None

    if edit_id is not None and edit_id != "":
        try:
            entry_edit = entries[int(edit_id)]
        except:
            entry_edit = None
    if request.method == "POST":
        # 共通
        category = request.form.get("category", "")
        title = request.form.get("title", "")
        desc = request.form.get("desc", "")
        address = request.form.get("address", "")
        map_url = request.form.get("map", "")
        tags_raw = request.form.get("tags", "")
        tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        areas = request.form.getlist("areas")

        # 店舗系専用
        tel = request.form.get("tel", "")
        holiday = request.form.get("holiday", "")
        open_hours = request.form.get("open_hours", "")
        parking = request.form.get("parking", "")
        parking_num = request.form.get("parking_num", "")
        payment = request.form.getlist("payment")  # チェックボックス
        remark = request.form.get("remark", "")

        # カテゴリで保存構造を分岐
        data = {
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas
        }
        if category != "観光":
            data.update({
                "tel": tel,
                "holiday": holiday,
                "open_hours": open_hours,
                "parking": parking,
                "parking_num": parking_num,
                "payment": payment,
                "remark": remark
            })

        # 編集 or 新規
        if request.form.get("edit_id"):
            idx = int(request.form["edit_id"])
            entries[idx] = data
            flash("編集しました")
        else:
            entries.append(data)
            flash("登録しました")
        save_entries(entries)
        return redirect(url_for("admin_entry"))

    return render_template(
        "admin_entry.html",
        entries=entries,
        entry_edit=entry_edit,
        edit_id=edit_id,
        role=session["role"]
    )


@app.route("/admin/entry/delete/<int:idx>", methods=["POST"])
@login_required
def delete_entry(idx):
    if session.get("role") != "admin":
        abort(403)
    entries = load_entries()
    if 0 <= idx < len(entries):
        entries.pop(idx)
        save_entries(entries)
        flash("削除しました")
    return redirect(url_for("admin_entry"))

import itertools

@app.route("/shop/entry", methods=["GET", "POST"])
@login_required
def shop_entry():
    if session.get("role") != "shop":
        return redirect(url_for("admin_entry"))
    user_id = session["user_id"]

    # POST（登録/更新）
    if request.method == "POST":
        category = request.form.get("category", "")
        title = request.form.get("title", "")
        desc = request.form.get("desc", "")
        address = request.form.get("address", "")
        tel = request.form.get("tel", "")
        holiday = request.form.get("holiday", "")
        open_hours = request.form.get("open_hours", "")
        parking = request.form.get("parking", "")
        parking_num = request.form.get("parking_num", "")
        payment = request.form.getlist("payment")
        remark = request.form.get("remark", "")
        tags_raw = request.form.get("tags", "")
        tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        areas = request.form.getlist("areas")
        map_url = request.form.get("map", "")

        entry_data = {
            "user_id": user_id,
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "tel": tel,
            "holiday": holiday,
            "open_hours": open_hours,
            "parking": parking,
            "parking_num": parking_num,
            "payment": payment,
            "remark": remark,
            "tags": tags,
            "areas": areas,
            "map": map_url
        }

        # entries.jsonに自分の情報を「上書きまたは新規追加」
        entries = load_entries()
        entry_idx = next((i for i, e in enumerate(entries) if e.get("user_id") == user_id), None)
        if entry_idx is not None:
            entries[entry_idx] = entry_data
        else:
            entries.append(entry_data)
        save_entries(entries)

        flash("店舗情報を保存しました")
        return redirect(url_for("shop_entry"))

    # GET時：自分の店舗データのみ
    user_entries = [e for e in load_entries() if e.get("user_id") == user_id]
    shop_entry_data = user_entries[-1] if user_entries else None
    return render_template("shop_entry.html", role="shop", shop_edit=shop_entry_data)

@app.route("/admin/entries_edit", methods=["GET", "POST"])
@login_required
def admin_entries_edit():
    # 管理者のみ
    if session.get("role") != "admin":
        abort(403)

    if request.method == "POST":
        raw_json = request.form.get("entries_raw", "")
        try:
            data = json.loads(raw_json)
            if not isinstance(data, list):
                raise ValueError("ルート要素は配列(list)にしてください")
            # ここで保存（ensure_ascii=False は save_entries 側で設定済み）
            save_entries(data)
            flash("entries.jsonを上書きしました")
            return redirect(url_for("admin_entries_edit"))
        except Exception as e:
            # エラー時は元のJSON文字列をそのまま再表示
            flash("JSONエラー: " + str(e))
            return render_template("admin_entries_edit.html", entries_raw=raw_json), 400

    # GET: 現在の内容を整形して表示（ファイル未作成でも落ちないように）
    try:
        entries = load_entries()  # ファイルが無ければ [] を返す実装
        entries_raw = json.dumps(entries, ensure_ascii=False, indent=2)
    except Exception as e:
        entries_raw = "[]"
        flash("entries.jsonの読み込みに失敗しました: " + str(e))

    return render_template("admin_entries_edit.html", entries_raw=entries_raw)


@app.route("/admin/logs")
@login_required
def admin_logs():
    if session.get("role") != "admin":
        abort(403)
    log_path = LOG_FILE
    logs = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except Exception:
                    pass
    logs = list(itertools.islice(reversed(logs), 300))
    return render_template("admin_logs.html", logs=logs)

@app.route("/admin/unhit_questions")
@login_required
def admin_unhit_questions():
    if session.get("role") != "admin":
        abort(403)
    log_path = LOG_FILE
    unhit_logs = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    if not log.get("hit_db"):
                        unhit_logs.append(log)
                except Exception:
                    pass
    unhit_logs = unhit_logs[-100:]
    return render_template("admin_unhit.html", unhit_logs=unhit_logs)

@app.route("/admin/add_entry", methods=["POST"])
@login_required
def admin_add_entry():
    if session.get("role") != "admin":
        abort(403)
    entries = load_entries()
    title = request.form.get("title", "")
    desc = request.form.get("desc", "")
    tags = [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()]
    areas = [a.strip() for a in request.form.get("areas", "").split(",") if a.strip()]
    if not title or not desc:
        flash("タイトルと説明は必須です")
        return redirect(url_for("admin_unhit_questions"))
    entry = {
        "title": title,
        "desc": desc,
        "address": "",
        "map": "",
        "tags": tags,
        "areas": areas
    }
    entries.append(entry)
    save_entries(entries)
    flash("DBに追加しました")
    return redirect(url_for("admin_entry"))

# === バックアップ機能 ===
@app.route("/admin/backup")
@login_required
def admin_backup():
    if session.get("role") != "admin":
        abort(403)
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 単体ファイル（arcnameは固定名に）
        if os.path.exists(ENTRIES_FILE):
            zf.write(ENTRIES_FILE, arcname="entries.json")
        if os.path.exists(SYNONYM_FILE):
            zf.write(SYNONYM_FILE, arcname="synonyms.json")
        if os.path.exists(NOTICES_FILE):
            zf.write(NOTICES_FILE, arcname="notices.json")
        if os.path.exists(SHOP_INFO_FILE):
            zf.write(SHOP_INFO_FILE, arcname="shop_infos.json")

        # data/ 配下
        if os.path.exists(DATA_DIR):
            for root, dirs, files in os.walk(DATA_DIR):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, BASE_DIR)
                    zf.write(fpath, arcname)

        # logs/ 配下
        if os.path.exists(LOG_DIR):
            for root, dirs, files in os.walk(LOG_DIR):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, BASE_DIR)
                    zf.write(fpath, arcname)

        # .env は意図的に含めない（認証情報のため）

    mem_zip.seek(0)
    return send_file(
        mem_zip,
        as_attachment=True,
        download_name="gotokanko_fullbackup.zip",
        mimetype="application/zip"
    )

# === 復元機能 ===
def _safe_extractall(zf: zipfile.ZipFile, dst):
    base = os.path.abspath(dst)
    for member in zf.namelist():
        target = os.path.abspath(os.path.join(dst, member))
        if not target.startswith(base + os.sep) and target != base:
            raise Exception("Unsafe path found in ZIP (zip slip)")
    os.makedirs(dst, exist_ok=True)
    zf.extractall(dst)

@app.route("/admin/restore", methods=["POST"])
@login_required
def admin_restore():
    if session.get("role") != "admin":
        abort(403)
    file = request.files.get("backup_zip")
    if not file:
        flash("アップロードファイルがありません")
        return redirect(url_for("admin_entry"))
    with zipfile.ZipFile(file, "r") as zf:
        _safe_extractall(zf, BASE_DIR)
    flash("復元が完了しました。データを確認してください。")
    return redirect(url_for("admin_entry"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("username")
        pw = request.form.get("password")
        users = load_users()
        user = next((u for u in users if u["user_id"] == user_id), None)
        if user and check_password_hash(user["password_hash"], pw):
            session["user_id"] = user_id
            session["role"] = user["role"]
            flash("ログインしました")
            if user["role"] == "admin":
                return redirect(url_for("admin_entry"))
            else:
                return redirect(url_for("shop_entry"))
        else:
            flash("ユーザーIDまたはパスワードが違います")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("ログアウトしました")
    return redirect(url_for("login"))

# === 天気・フェリー情報 ===
WEATHER_LINKS = [
    {"area": "五島市", "url": "https://weathernews.jp/onebox/tenki/nagasaki/42211/"},
    {"area": "新上五島町", "url": "https://weathernews.jp/onebox/tenki/nagasaki/42411/"},
    {"area": "小値賀町", "url": "https://tenki.jp/forecast/9/45/8440/42383/"},
    {"area": "宇久町", "url": "https://weathernews.jp/onebox/33.262381/129.131027/q=%E9%95%B7%E5%B4%8E%E7%9C%8C%E4%BD%90%E4%B8%96%E4%BF%9D%E5%B8%82%E5%AE%87%E4%B9%85%E7%94%BA&v=da56215a2617fc2203c6cae4306d5fd8c92e3e26c724245d91160a4b3597570a&lang=ja&type=week"}
]
FERRY_INFO = """【長崎ー五島航路 運行状況】
・野母商船「フェリー太古」運航情報  
http://www.norimono-info.com/frame_set.php?usri=&disp=group&type=ship
・九州商船「フェリー・ジェットフォイル」運航情報  
https://kyusho.co.jp/status
・五島産業汽船「フェリー」運航情報  
https://www.goto-sangyo.co.jp/
その他の航路や詳細は各リンクをご覧ください。
"""

def get_weather_reply(question):
    """天気リンクを返す。ENABLE_FOREIGN_LANG=0 のときは常に日本語で返答。"""
    weather_keywords = [
        "天気", "天候", "気象", "weather", "天気予報", "雨", "晴", "曇", "降水", "気温", "forecast",
        "天氣", "天气", "氣象", "温度", "陰", "晴朗", "預報"
    ]
    if not question:
        return None, False

    # 外国語対応OFFなら日本語固定／ONなら原文言語で
    lang = "ja" if not ENABLE_FOREIGN_LANG else detect_lang_simple(question)

    if not any(kw in question for kw in weather_keywords):
        return None, False

    # エリア別（表記は漢字共通なので包含でOK）
    for entry in WEATHER_LINKS:
        if entry["area"] in question:
            if lang == "zh-Hant":
                return f"【{entry['area']}天氣】\n最新資訊：\n{entry['url']}\n（可查看今日、週預報與降雨雷達）", True
            if lang == "en":
                return f"[{entry['area']} weather]\nLatest forecast:\n{entry['url']}", True
            return f"【{entry['area']}の天気】\n最新の{entry['area']}の天気情報はこちら\n{entry['url']}", True

    # 総合リンク
    if lang == "zh-Hant":
        reply = "【五島列島的主要天氣連結】\n"
        for e in WEATHER_LINKS:
            reply += f"{e['area']}: {e['url']}\n"
        return reply.strip(), True
    if lang == "en":
        reply = "Main weather links for the Goto Islands:\n"
        for e in WEATHER_LINKS:
            reply += f"{e['area']}: {e['url']}\n"
        return reply.strip(), True

    reply = "【五島列島の主な天気情報リンク】\n"
    for e in WEATHER_LINKS:
        reply += f"{e['area']}: {e['url']}\n"
    return reply.strip(), True

# === 観光データ横断検索 ===

def clean_query_for_search(question):
    ignore_patterns = [
        r"(について)?教えて.*$", r"の情報.*$", r"の場所.*$", r"どこ.*$", r"案内.*$", r"を教えて.*$", r"を知りたい.*$", r"アクセス.*$", r"詳細.*$"
    ]
    q = question
    for pat in ignore_patterns:
        q = re.sub(pat, "", q)
    return q.strip()


def find_entry_info(question):
    entries = load_entries()
    synonyms = load_synonyms()
    areas_master = ["五島市", "新上五島町", "宇久町", "小値賀町"]
    target_areas = [area for area in areas_master if area in question]
    tags_from_syn = find_tags_by_synonym(question, synonyms)
    words = set(re.split(r'\s+|　|,|、|。', question))

    cleaned_query = clean_query_for_search(question)
    # 1. タイトル完全一致
    hits = [e for e in entries if cleaned_query and e.get("title", "") == cleaned_query and (not target_areas or any(area in e.get("areas", []) for area in target_areas))]
    if hits:
        return hits
    # 2. 各カラム部分一致（title, tags, desc, address）
    hits = []
    for e in entries:
        haystacks = [
            e.get("title", ""),
            " ".join(e.get("tags", [])),
            e.get("desc", ""),
            e.get("address", "")
        ]
        target_str = " ".join(haystacks)
        if cleaned_query and cleaned_query in target_str:
            if not target_areas or any(area in e.get("areas", []) for area in target_areas):
                hits.append(e)
    if hits:
        return hits
    # 3. 元のtitle, desc部分一致にも対応
    hits = [e for e in entries if question in e.get("title", "") or question in e.get("desc", "")]
    if hits:
        return hits
    # 4. タグ一致・類義語一致
    hits = []
    for e in entries:
        tag_hit = (
            any(tag in question for tag in e.get("tags", [])) or
            any(tag in tags_from_syn for tag in e.get("tags", []))
        )
        if tag_hit and (not target_areas or any(area in e.get("areas", []) for area in target_areas)):
            hits.append(e)
    if hits:
        return hits
    return []

@app.route("/admin/manual")
@login_required
def admin_manual():
    if session.get("role") != "admin":
        abort(403)
    return render_template("admin_manual.html")

@app.route("/admin/synonyms", methods=["GET", "POST"])
@login_required
def admin_synonyms():
    # 管理者のみ許可
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))

    synonyms = load_synonyms()

    if request.method == "POST":
        new_synonyms = {}
        tags = request.form.getlist("tag")
        for i, tag in enumerate(tags):
            syns = request.form.get(f"synonyms_{i}", "")
            syn_list = [s.strip() for s in syns.split(",") if s.strip()]
            if tag:
                new_synonyms[tag] = syn_list

        add_tag = request.form.get("add_tag", "").strip()
        add_synonyms = request.form.get("add_synonyms", "").strip()
        if add_tag:
            new_synonyms[add_tag] = [s.strip() for s in add_synonyms.split(",") if s.strip()]

        save_synonyms(new_synonyms)
        flash("類義語辞書を更新しました。")
        return redirect(url_for("admin_synonyms"))

    return render_template("admin_synonyms.html", synonyms=synonyms)

# === data/配下パンフ全文検索 ===

def search_text_files(question, data_dir=DATA_DIR, max_snippets=5, window=80):
    snippets = []
    words = set(re.split(r'\s+|　|,|、|。', question))
    pattern = '|'.join([re.escape(w) for w in words if w])
    if not pattern:
        return None
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.txt'):
                try:
                    fpath = os.path.join(root, fname)
                    with open(fpath, encoding='utf-8') as f:
                        text = f.read()
                    for m in re.finditer(pattern, text, re.IGNORECASE):
                        start = max(m.start()-window, 0)
                        end = min(m.end()+window, len(text))
                        snippet = text[start:end].replace('\n', ' ').strip()
                        if snippet not in snippets:
                            snippets.append(snippet)
                            if len(snippets) >= max_snippets:
                                return snippets
                except Exception as e:
                    print(f"[全文検索エラー] {fname}: {e}")
    return snippets if snippets else None

# === Webクロール（拡張用フック。初期はダミー返答） ===

def fetch_and_search_web(question):
    return None

# === AI要約（OpenAI GPT利用） ===

def ai_summarize(snippets, question, model=None):
    model = model or OPENAI_MODEL_PRIMARY
    prompt = (
        "以下は五島観光・生活ガイド資料から関連する抜粋です。\n"
        f"質問: 「{question}」\n"
        "抜粋資料を参考に、やさしく正確に回答してください。\n\n"
        "-----\n"
        + "\n---\n".join(snippets)
        + "\n-----"
    )
    return openai_chat(
        model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=512,
    )

# === 総合応答ロジック（API/LINE/WEB共通）＋ヒットフラグ＋メタ ===

def smart_search_answer_with_hitflag(question):
    meta = {"model_primary": OPENAI_MODEL_PRIMARY, "fallback": None}

    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        return weather_reply, True, meta

    # 飛行機・空港・航空便の質問は必ず公式サイトを案内
    if any(word in question for word in ["飛行機", "空港", "航空便", "欠航", "到着", "出発"]):
        return (
            "五島つばき空港の最新の運行状況は、公式Webサイトでご確認ください。\n"
            "▶ https://www.fukuekuko.jp/",
            True,
            meta,
        )

    if any(word in question for word in ["フェリー", "船", "運航", "ジェットフォイル", "太古"]):
        return FERRY_INFO, True, meta

    # DB検索
    entries = find_entry_info(question)
    if entries:
        if len(entries) == 1:
            entry = entries[0]
            msg = f"【{', '.join(entry.get('areas', []))}: {entry.get('title','')}】\n"
            msg += f"説明: {entry.get('desc','')}\n"
            msg += f"住所: {entry.get('address','')}\n"
            if entry.get("map"): msg += f"地図: {entry.get('map')}\n"
            msg += f"タグ: {', '.join(entry.get('tags',[]))}\n"
            return msg, True, meta
        else:
            try:
                snippets = [f"タイトル: {e['title']}\n説明: {e.get('desc','')}\n住所: {e.get('address','')}\n" for e in entries]
                ai_ans = ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY)
                return ai_ans, True, meta
            except Exception as e:
                return "複数スポットが見つかりましたが要約に失敗しました。", False, meta

    # data/全文検索
    snippets = search_text_files(question, data_dir=DATA_DIR)
    if snippets:
        try:
            return ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY), True, meta
        except Exception as e:
            return "AI要約でエラー: " + str(e), False, meta

    # Web検索（未実装フック）
    web_texts = fetch_and_search_web(question)
    if web_texts:
        try:
            return ai_summarize(web_texts, question, model=OPENAI_MODEL_PRIMARY), True, meta
        except Exception as e:
            return "Web要約でエラー: " + str(e), False, meta

    # どのデータにもヒットしない場合のフォールバック＋絞り込み候補
    suggest_text, suggest_meta = ("", {})
    if SUGGEST_UNHIT:
        suggest_text, suggest_meta = build_refine_suggestions(question)
        meta["suggestions"] = suggest_meta

    if REASK_UNHIT:
        fallback_text = ai_suggest_faq(
            question,
            model=OPENAI_MODEL_HIGH,
        )

        # 参考回答が生成できたら、それを返す
        if fallback_text:
            meta["fallback"] = {
                "mode": "high_faq",
                "model": OPENAI_MODEL_HIGH,
            }
            answer = "【参考回答（データ未登録のため要確認）】\n" + fallback_text
            if suggest_text:
                answer += "\n---\n" + suggest_text
            return answer, False, meta

        # 生成に失敗：候補があれば候補だけ返す
        if suggest_text:
            return (
                "申し訳ありません。該当データが見つかりませんでした。\n" + suggest_text,
                False,
                meta,
            )

        # 候補も無い場合の最終フォールバック
        return (
            "申し訳ありません。現在この質問には確実な情報を持っていません。",
            False,
            meta,
        )

    # （REASK_UNHIT=False のとき）厳格モード：候補だけでも返す
    if suggest_text:
        return (
            "申し訳ありません。該当データが見つかりませんでした。\n" + suggest_text,
            False,
            meta,
        )
    return (
        "申し訳ありません。現在この質問には確実な情報を持っていません。",
        False,
        meta,
    )

# === APIエンドポイント（/ask） ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    # 1) 原文で天気を先に判定 → 入力言語で即返信
    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        save_qa_log(question, weather_reply, source="web", hit_db=True, extra={"kind": "weather"})
        return jsonify({"answer": weather_reply, "hit_db": True, "meta": {"kind": "weather"}})

    if not question:
        return jsonify({"error": "質問が空です"}), 400

    # 言語検出→解析は日本語→最終は設定に応じて元言語/日本語
    orig_lang = detect_lang_simple(question)
    lang = orig_lang
    # ★外国語対応OFFなら、常に日本語として返す
    if not ENABLE_FOREIGN_LANG:
        lang = "ja"

    # 解析用は“原文が日本語でなければ”日本語に翻訳
    q_for_logic = question if orig_lang == "ja" else translate_text(question, "ja")

    answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)

    # 最終返信
    if lang == "ja":
        answer = answer_ja
    else:
        target = "zh-Hant" if lang == "zh-Hant" else "en"
        answer = translate_text(answer_ja, target)

    save_qa_log(question, answer, source="web", hit_db=hit_db, extra=meta)
    return jsonify({"answer": answer, "hit_db": hit_db, "meta": meta})

@app.route("/admin/unhit_report")
@login_required
def admin_unhit_report():
    if session.get("role") != "admin":
        abort(403)
    report = generate_unhit_report(7)
    return render_template("admin_unhit_report.html", unhit_report=report)


# === LINE Webhook ===
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError as e:
        app.logger.warning(f"LINE invalid signature: {e}")
        return ("OK", 200)
    except Exception as e:
        app.logger.exception("LINE handler error")
        return ("OK", 200)
    return ("OK", 200)

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text

    # 1) 原文で天気を先に判定 → 入力言語で即返信（二重返信防止のため return）
    weather_reply, weather_hit = get_weather_reply(user_message)
    if weather_hit:
        save_qa_log(user_message, weather_reply, source="line", hit_db=True, extra={"kind": "weather"})
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=weather_reply))
        except LineBotApiError:
            app.logger.exception("LineBotApiError")
        except Exception:
            app.logger.exception("LineBotApi reply error")
        return

    # 言語検出→解析は日本語→最終は設定に応じて元言語/日本語
    orig_lang = detect_lang_simple(user_message)
    lang = orig_lang
    # ★外国語対応OFFなら、常に日本語として返す
    if not ENABLE_FOREIGN_LANG:
        lang = "ja"

    # 解析用は“原文が日本語でなければ”日本語に翻訳
    q_for_logic = user_message if orig_lang == "ja" else translate_text(user_message, "ja")

    answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)

    # 最終返信
    if lang == "ja":
        answer = answer_ja
    else:
        target = "zh-Hant" if lang == "zh-Hant" else "en"
        answer = translate_text(answer_ja, target)

    save_qa_log(user_message, answer, source="line", hit_db=hit_db, extra=meta)
    try:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
    except LineBotApiError:
        app.logger.exception("LineBotApiError")
    except Exception:
        app.logger.exception("LineBotApi reply error")

# === トップ ===
@app.route("/")
def home():
    return "<a href='/admin/entry'>[観光データ管理]</a>"

def load_shop_info(user_id):
    if not os.path.exists(SHOP_INFO_FILE):
        return {}
    with open(SHOP_INFO_FILE, "r", encoding="utf-8") as f:
        infos = json.load(f)
    return infos.get(user_id, {})

def save_shop_info(user_id, info):
    infos = {}
    if os.path.exists(SHOP_INFO_FILE):
        with open(SHOP_INFO_FILE, "r", encoding="utf-8") as f:
            infos = json.load(f)
    infos[user_id] = info
    with open(SHOP_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/admin/notices", methods=["GET", "POST"])
@login_required
def admin_notices():
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))
    notices = load_notices()
    edit_id = request.values.get("edit")  # GET or POST
    edit_notice = None

    if edit_id:
        for n in notices:
            if str(n.get("id")) == str(edit_id):
                edit_notice = n
                break

    if request.method == "POST":
        category = request.form.get("category")
        title = request.form.get("title")
        content = request.form.get("content")
        date = datetime.date.today().isoformat()
        image_url = request.form.get("image_url", "")

        if category in ["イベント", "特売"]:
            start_date = request.form.get("start_date", "")
            end_date = request.form.get("end_date", "")
            expire_date = ""
        else:
            start_date = ""
            end_date = ""
            expire_date = request.form.get("expire_date", "")

        if request.form.get("edit_id"):
            for i, n in enumerate(notices):
                if str(n.get("id")) == str(request.form.get("edit_id")):
                    notices[i].update({
                        "title": title,
                        "content": content,
                        "category": category,
                        "start_date": start_date,
                        "end_date": end_date,
                        "expire_date": expire_date,
                        "image_url": image_url
                    })
                    flash("お知らせを更新しました")
                    break
        else:
            notice = {
                "id": notices[-1]["id"] + 1 if notices else 1,
                "title": title,
                "content": content,
                "category": category,
                "date": date,
                "start_date": start_date,
                "end_date": end_date,
                "expire_date": expire_date,
                "image_url": image_url,
                "created_by": session["user_id"]
            }
            notices.append(notice)
            flash("お知らせを追加しました")
        save_notices(notices)
        return redirect(url_for("admin_notices"))
    return render_template("admin_notices.html", notices=notices, edit_notice=edit_notice)


@app.route("/admin/notices/delete/<int:idx>", methods=["POST"])
@login_required
def delete_notice(idx):
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))
    notices = load_notices()
    notices = [n for n in notices if n.get("id") != idx]
    save_notices(notices)
    flash("お知らせを削除しました")
    return redirect(url_for("admin_notices"))


@app.route("/notices")
def notices():
    notices = load_notices()
    return render_template("notices.html", notices=notices)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
