import os
import json
import re
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file

from dotenv import load_dotenv
load_dotenv()

from werkzeug.security import check_password_hash

# LINE Bot関連
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import openai
import zipfile
import io

app = Flask(__name__)
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


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

ENTRIES_FILE = "entries.json"
DATA_DIR = "data"
LOG_FILE = "logs/questions_log.jsonl"
SYNONYM_FILE = "synonyms.json"
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# === ログ保存用 ===
def save_qa_log(question, answer, source="web", hit_db=False, extra=None):
    os.makedirs("logs", exist_ok=True)
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

# === お知らせ・イベント・特売掲示板用 ===
NOTICES_FILE = "notices.json"

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

def suggest_tags_and_title(question, answer):
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
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",  # 格安モデルを使用
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        content = chat_completion.choices[0].message.content
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

def ai_suggest_faq(question):
    prompt = (
        f"以下の質問に対し、観光案内AIとして分かりやすいFAQ回答文（最大400文字）を作成してください。\n"
        f"質問: {question}\n"
        f"---\n"
        f"回答:"
    )
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print("[FAQサジェストエラー]", e)
        return ""

def generate_unhit_report(n=7):
    """直近n日分の未ヒット・誤答ログ集計（例：logs/questions_log.jsonl使用）"""
    import datetime
    from collections import Counter

    logs = []
    try:
        with open("logs/questions_log.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                logs.append(d)
    except Exception as e:
        print("ログ集計エラー:", e)
        return []

    today = datetime.datetime.now()
    recent = []
    for log in logs[::-1]:
        t = log.get("timestamp", "")
        try:
            dt = datetime.datetime.fromisoformat(t[:19])
            if (today - dt).days <= n and not log.get("hit_db", False):
                recent.append(log)
        except:
            continue

    questions = [l["question"] for l in recent]
    counter = Counter(questions)
    # 件数順リスト
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
def ai_suggest_synonyms(question, all_tags):
    prompt = (
        f"以下の質問について、適切な既存タグ（{', '.join(all_tags)}）や新しいタグに対して、日本語の類義語・言い換え案を5つずつタグごとに挙げてください。\n"
        f"質問: {question}\n"
        f"出力例:\nタグ: ビーチ\n類義語: 海水浴場, 砂浜, 泳げる場所, 水遊び, サンビーチ\n---\n"
    )
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print("[類義語サジェストエラー]", e)
        return ""

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
def delete_entry(idx):
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
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))
    if request.method == "POST":
        raw_json = request.form.get("entries_raw", "")
        try:
            data = json.loads(raw_json)
            if not isinstance(data, list):
                raise Exception("リスト形式で入力してください")
            save_entries(data)
            flash("entries.jsonを上書きしました")
            return redirect(url_for("admin_entries_edit"))
        except Exception as e:
            flash("JSONエラー: " + str(e))
            return render_template("admin_entries_edit.html", entries_raw=raw_json)
    # GET時は現データをjson文字列で
    with open(ENTRIES_FILE, encoding="utf-8") as f:
        entries_raw = f.read()
    return render_template("admin_entries_edit.html", entries_raw=entries_raw)


@app.route("/admin/logs")
def admin_logs():
    log_path = LOG_FILE
    logs = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except Exception:
                    pass
    # 新しい順で最大300件だけ表示
    logs = list(itertools.islice(reversed(logs), 300))
    return render_template("admin_logs.html", logs=logs)

@app.route("/admin/unhit_questions")
def admin_unhit_questions():
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
    # 最新100件だけ表示
    unhit_logs = unhit_logs[-100:]
    return render_template("admin_unhit.html", unhit_logs=unhit_logs)

@app.route("/admin/add_entry", methods=["POST"])
def admin_add_entry():
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
def admin_backup():
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # entries.json
        if os.path.exists(ENTRIES_FILE):
            zf.write(ENTRIES_FILE)
        # dataフォルダ
        for root, dirs, files in os.walk(DATA_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, ".")
                zf.write(fpath, arcname)
        # synonyms.json
        if os.path.exists(SYNONYM_FILE):
            zf.write(SYNONYM_FILE)
        # logsフォルダ
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            for root, dirs, files in os.walk(logs_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, ".")
                    zf.write(fpath, arcname)
        # .envは絶対に含めない！！
        # if os.path.exists(".env"):
        #     zf.write(".env")  # ←この行をコメントアウトまたは削除
    mem_zip.seek(0)
    return send_file(
        mem_zip,
        as_attachment=True,
        download_name="gotokanko_fullbackup.zip",
        mimetype="application/zip"
    )

# === 復元機能 ===
@app.route("/admin/restore", methods=["POST"])
def admin_restore():
    file = request.files.get("backup_zip")
    if not file:
        flash("アップロードファイルがありません")
        return redirect(url_for("admin_entry"))
    with zipfile.ZipFile(file, "r") as zf:
        zf.extractall(".")
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
            # 管理者/店舗で遷移先を切り替え
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
    weather_keywords = ["天気", "天候", "気象", "weather", "天気予報", "雨", "晴", "曇", "降水", "気温", "forecast"]
    if any(kw in question for kw in weather_keywords):
        for entry in WEATHER_LINKS:
            if entry["area"] in question:
                return f"【{entry['area']}の天気】\n最新の{entry['area']}の天気情報はこちら\n{entry['url']}", True
        reply = "【五島列島の主な天気情報リンク】\n"
        for entry in WEATHER_LINKS:
            reply += f"{entry['area']}: {entry['url']}\n"
        return reply, True
    return None, False

# === 観光データ横断検索 ===
def clean_query_for_search(question):
    # よくある日本語の後ろノイズ語句
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

    # ▼ ここから改善
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
    # なければNone
    return []

@app.route("/admin/manual")
def admin_manual():
    return render_template("admin_manual.html")

@app.route("/admin/synonyms", methods=["GET", "POST"])
def admin_synonyms():
    synonyms = load_synonyms()

    if request.method == "POST":
        # 1. 既存の全データをクリアし、フォーム送信内容で再構築
        new_synonyms = {}
        tags = request.form.getlist("tag")
        for i, tag in enumerate(tags):
            syns = request.form.get(f"synonyms_{i}", "")
            syn_list = [s.strip() for s in syns.split(",") if s.strip()]
            if tag:
                new_synonyms[tag] = syn_list
        # 2. 新規追加（未入力分も反映）
        add_tag = request.form.get("add_tag", "").strip()
        add_synonyms = request.form.get("add_synonyms", "").strip()
        if add_tag:
            new_synonyms[add_tag] = [s.strip() for s in add_synonyms.split(",") if s.strip()]
        save_synonyms(new_synonyms)
        flash("類義語辞書を更新しました。")
        return redirect(url_for('admin_synonyms'))

    # GET時は表示用
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
def ai_summarize(snippets, question):
    prompt = (
        "以下は五島観光・生活ガイド資料から関連する抜粋です。\n"
        f"質問: 「{question}」\n"
        "抜粋資料を参考に、やさしく正確に回答してください。\n\n"
        "-----\n"
        + "\n---\n".join(snippets)
        + "\n-----"
    )
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=512,
    )
    return chat_completion.choices[0].message.content

# === 総合応答ロジック（API/LINE/WEB共通）＋ヒットフラグ ===
def smart_search_answer_with_hitflag(question):
    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        return weather_reply, True
        # ▼▼▼ 飛行機・空港・航空便の質問は必ず公式サイトを案内
    if any(word in question for word in ["飛行機", "空港", "航空便", "便", "欠航", "到着", "出発"]):
        return (
            "五島つばき空港の最新の運行状況は、公式Webサイトでご確認いただけます。\n"
            "▶ https://www.fukuekuko.jp/",
            True
        )
    if any(word in question for word in ["フェリー", "船", "運航", "ジェットフォイル", "太古"]):
        return FERRY_INFO, True
    entries = find_entry_info(question)
    if entries:
        # 1件だけならそのまま
        if len(entries) == 1:
            entry = entries[0]
            msg = f"【{', '.join(entry.get('areas', []))}: {entry.get('title','')}】\n"
            msg += f"説明: {entry.get('desc','')}\n"
            msg += f"住所: {entry.get('address','')}\n"
            if entry.get("map"): msg += f"地図: {entry.get('map')}\n"
            msg += f"タグ: {', '.join(entry.get('tags',[]))}\n"
            return msg, True
        else:
            # 複数ヒット時はAIで要約
            try:
                snippets = [f"タイトル: {e['title']}\n説明: {e.get('desc','')}\n住所: {e.get('address','')}\n" for e in entries]
                ai_ans = ai_summarize(snippets, question)
                return ai_ans, True
            except Exception as e:
                return "複数スポットが見つかりましたが要約に失敗しました。", False
    snippets = search_text_files(question, data_dir=DATA_DIR)
    if snippets:
        try:
            return ai_summarize(snippets, question), True
        except Exception as e:
            return "AI要約でエラー: " + str(e), False
    web_texts = fetch_and_search_web(question)
    if web_texts:
        try:
            return ai_summarize(web_texts, question), True
        except Exception as e:
            return "Web要約でエラー: " + str(e), False

    # どのデータにもヒットしない場合、根拠のないAI生成はせず「分かりません」と返す
    return (
        "申し訳ありません。現在この質問には確実な情報を持っていません。",
        False
    )

# === APIエンドポイント（/ask） ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "質問が空です"}), 400
    answer, hit_db = smart_search_answer_with_hitflag(question)
    save_qa_log(question, answer, source="web", hit_db=hit_db)
    return jsonify({"answer": answer})

@app.route("/admin/unhit_report")
def admin_unhit_report():
    report = generate_unhit_report(7)  # 直近7日
    return render_template("admin_unhit_report.html", unhit_report=report)


# === LINE Webhook ===
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"LINE handler error: {e}")
        return "NG"
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    answer, hit_db = smart_search_answer_with_hitflag(user_message)
    save_qa_log(user_message, answer, source="line", hit_db=hit_db)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )

# === トップ ===
@app.route("/")
def home():
    return "<a href='/admin/entry'>[観光データ管理]</a>"

SHOP_INFO_FILE = "shop_infos.json"

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
        # idで一意に取得
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
            # 編集
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
            # 新規
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
    # 有効期限やカテゴリなどで絞り込む場合はここでフィルタ
    return render_template("notices.html", notices=notices)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
