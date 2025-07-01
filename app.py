import os
import json
import re
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file

from dotenv import load_dotenv
load_dotenv()

# LINE Bot関連
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import openai
import zipfile
import io

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")

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
def admin_entry():
    entries = load_entries()
    edit_id = request.args.get("edit")
    entry_edit = None

    if edit_id is not None and edit_id != "":
        try:
            entry_edit = entries[int(edit_id)]
        except:
            entry_edit = None

    # ▼ 未ヒット質問から遷移時のAIサジェスト
    if not entry_edit:
        edit_title = request.args.get("edit_title", "")
        edit_desc = request.args.get("edit_desc", "")
        edit_tags = request.args.get("edit_tags", "")
        ai_title, ai_tags = "", ""
        if edit_title or edit_desc:
            ai_title, ai_tags = suggest_tags_and_title(edit_title, edit_desc)
        entry_edit = {
            "title": ai_title or edit_title,
            "desc": edit_desc,
            "address": "",
            "map": "",
            "tags": [ai_tags] if ai_tags else ([edit_tags] if edit_tags else []),
            "areas": []
        }
    # ...（POST処理やreturnはそのまま）...

    if request.method == "POST":
        # ...（既存の登録・編集処理はそのまま）
        title = request.form.get("title", "")
        desc = request.form.get("desc", "")
        address = request.form.get("address", "")
        map_url = request.form.get("map", "")
        tags_raw = request.form.get("tags", "")
        tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        areas = request.form.getlist("areas")
        if not title or not desc or not tags or not areas:
            flash("タイトル・説明・タグ・エリアは必須です")
            return redirect(url_for("admin_entry"))
        data = {
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas
        }
        if request.form.get("edit_id"):
            idx = int(request.form["edit_id"])
            entries[idx] = data
            flash("編集しました")
        else:
            entries.append(data)
            flash("登録しました")
        save_entries(entries)
        return redirect(url_for("admin_entry"))

    return render_template("admin_entry.html", entries=entries, entry_edit=entry_edit, edit_id=edit_id)

@app.route("/admin/entry/delete/<int:idx>", methods=["POST"])
def delete_entry(idx):
    entries = load_entries()
    if 0 <= idx < len(entries):
        entries.pop(idx)
        save_entries(entries)
        flash("削除しました")
    return redirect(url_for("admin_entry"))

import itertools

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
        if os.path.exists(ENTRIES_FILE):
            zf.write(ENTRIES_FILE)
        for root, dirs, files in os.walk(DATA_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, ".")
                zf.write(fpath, arcname)
    mem_zip.seek(0)
    return send_file(
        mem_zip,
        as_attachment=True,
        download_name="gotokanko_backup.zip",
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

# === ログイン（必要に応じて） ===
def is_logged_in():
    return session.get("logged_in", False)
def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            flash("管理画面にログインしてください。")
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pw = request.form.get("password")
        admin_user = os.environ.get("ADMIN_USER", "admin")
        admin_pass = os.environ.get("ADMIN_PASS", "password")
        if user == admin_user and pw == admin_pass:
            session["logged_in"] = True
            flash("ログインしました")
            next_url = request.args.get("next") or url_for("admin_entry")
            return redirect(next_url)
        else:
            flash("ユーザー名またはパスワードが違います")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
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

# === 観光データ横断検索（タグ・エリア優先） ===
def find_entry_info(question):
    entries = load_entries()
    synonyms = load_synonyms()   # ←追加
    areas_master = ["五島市", "新上五島町", "宇久町", "小値賀町"]
    target_areas = [area for area in areas_master if area in question]

    # ▼ ここでタグ候補を類義語からも抽出！
    tags_from_syn = find_tags_by_synonym(question, synonyms)
    words = set(re.split(r'\s+|　|,|、|。', question))
    for entry in entries:
        area_hit = (not target_areas or any(area in entry.get("areas", []) for area in target_areas))
        tag_hit = (
            any(tag in question for tag in entry.get("tags", [])) or
            any(tag in tags_from_syn for tag in entry.get("tags", []))
        )
        if area_hit and tag_hit:
            return entry
    for entry in entries:
        if any(word in entry.get("title","") or word in entry.get("desc","") for word in words):
            return entry
    return None

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
    if any(word in question for word in ["フェリー", "船", "運航", "ジェットフォイル", "太古"]):
        return FERRY_INFO, True
    entry = find_entry_info(question)
    if entry:
        msg = f"【{', '.join(entry.get('areas', []))}: {entry.get('title','')}】\n"
        msg += f"説明: {entry.get('desc','')}\n"
        msg += f"住所: {entry.get('address','')}\n"
        if entry.get("map"): msg += f"地図: {entry.get('map')}\n"
        msg += f"タグ: {', '.join(entry.get('tags',[]))}\n"
        return msg, True
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
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは五島列島の公式案内AIです。観光・宿・食・生活・移住などあらゆる質問に親切に答えてください。"},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return chat_completion.choices[0].message.content, False
    except Exception as e:
        return str(e), False

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

if __name__ == "__main__":
    app.run(port=10000, debug=True)
