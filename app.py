import os
import json
import re
import datetime
import itertools
import time  # ← 追加
from collections import Counter
from typing import Any, Dict, List

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, abort

from dotenv import load_dotenv
load_dotenv()

from werkzeug.security import check_password_hash, generate_password_hash

# LINE Bot関連
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError

from openai import OpenAI
import zipfile
import io

from functools import wraps


# =========================
#  正規化ヘルパー
# =========================
def _split_lines_commas(val: str) -> List[str]:
    """改行・カンマ両対応で分割 → 余白除去 → 空要素除去"""
    if not val:
        return []
    parts = re.split(r'[\n,]+', str(val))
    return [p.strip() for p in parts if p and p.strip()]

def _norm_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """保存前/表示前にデータ構造を正規化"""
    e = dict(e or {})

    # 文字列 -> 配列に寄せる
    for k in ("tags", "areas", "links", "payment"):
        v = e.get(k)
        if v is None:
            e[k] = []
        elif isinstance(v, str):
            e[k] = _split_lines_commas(v)
        elif isinstance(v, (list, tuple)):
            e[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            e[k] = [str(v).strip()]

    # extras を dict に寄せる
    x = e.get("extras")
    if isinstance(x, dict):
        e["extras"] = {str(k): ("" if v is None else str(v)) for k, v in x.items()}
    elif isinstance(x, list):
        tmp: Dict[str, str] = {}
        for it in x:
            if isinstance(it, dict):
                for k, v in it.items():
                    tmp[str(k)] = "" if v is None else str(v)
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                tmp[str(it[0])] = "" if it[1] is None else str(it[1])
        e["extras"] = tmp
    elif isinstance(x, str) and x.strip():
        e["extras"] = {"備考": x.strip()}
    else:
        e["extras"] = {}

    # 型の下駄（無ければ空文字）
    for k in ("category", "title", "desc", "address", "map", "tel", "holiday", "open_hours", "parking", "parking_num", "remark"):
        if e.get(k) is None:
            e[k] = ""

    # category の既定
    if not e["category"]:
        e["category"] = "観光"

    return e


# =========================
#  Flask / 設定
# =========================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 日本語をJSONでそのまま返す

# 本番では必ず環境変数で設定
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")

# セッション設定
SECURE_COOKIE = os.environ.get("SESSION_COOKIE_SECURE", "1").lower() in {"1", "true", "on", "yes"}
app.config.update(
    SESSION_COOKIE_SECURE=SECURE_COOKIE,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=12),
)

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("ログインしてください")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


# =========================
#  環境変数 / モデル
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") or ""  # 未設定でも起動OK
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET") or ""              # 未設定でも起動OK

OPENAI_MODEL_PRIMARY = os.environ.get("OPENAI_MODEL_PRIMARY", "gpt-4o-mini")
OPENAI_MODEL_HIGH    = os.environ.get("OPENAI_MODEL_HIGH", "gpt-5-mini")

REASK_UNHIT   = os.environ.get("REASK_UNHIT", "1").lower() in {"1", "true", "on", "yes"}
SUGGEST_UNHIT = os.environ.get("SUGGEST_UNHIT", "1").lower() in {"1", "true", "on", "yes"}
ENABLE_FOREIGN_LANG = os.environ.get("ENABLE_FOREIGN_LANG", "1").lower() in {"1", "true", "on", "yes"}

# OpenAI v1 クライアント
client = OpenAI()

# LINE Bot
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# データ格納先
BASE_DIR       = os.environ.get("DATA_BASE_DIR", ".")  # 例: /var/data
ENTRIES_FILE   = os.path.join(BASE_DIR, "entries.json")
DATA_DIR       = os.path.join(BASE_DIR, "data")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
LOG_FILE       = os.path.join(LOG_DIR, "questions_log.jsonl")
SYNONYM_FILE   = os.path.join(BASE_DIR, "synonyms.json")
USERS_FILE     = os.path.join(BASE_DIR, "users.json")
NOTICES_FILE   = os.path.join(BASE_DIR, "notices.json")
SHOP_INFO_FILE = os.path.join(BASE_DIR, "shop_infos.json")

# 必要フォルダ作成
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 便利関数: JSONをアトミックに保存
def _atomic_json_dump(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _safe_read_json(path: str, default_obj):
    """
    JSONを安全に読み込む。壊れていたら .bad-YYYYmmdd-HHMMSS に退避し、
    default_obj を書き戻してから default_obj を返す。
    """
    try:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_obj, f, ensure_ascii=False, indent=2)
            return default_obj
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        app.logger.exception(f"[safe_read_json] JSON read failed: {path}: {e}")
        # 破損ファイルを退避
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            os.replace(path, f"{path}.bad-{ts}")
        except Exception:
            pass
        # 既定値で初期化
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_obj, f, ensure_ascii=False, indent=2)
        except Exception:
            app.logger.exception(f"[safe_read_json] JSON rewrite failed: {path}")
        return default_obj


# 初回ブートストラップ
ADMIN_INIT_USER = os.environ.get("ADMIN_INIT_USER", "admin")
ADMIN_INIT_PASSWORD = os.environ.get("ADMIN_INIT_PASSWORD")  # 初回のみ使用推奨

def _ensure_json(path, default_obj):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f, ensure_ascii=False, indent=2)

def _bootstrap_files_and_admin():
    app.logger.info(f"[boot] BASE_DIR={BASE_DIR}")
    app.logger.info(f"[boot] USERS_FILE path: {USERS_FILE}")

    _ensure_json(ENTRIES_FILE, [])
    _ensure_json(SYNONYM_FILE, {})
    _ensure_json(NOTICES_FILE, [])
    _ensure_json(SHOP_INFO_FILE, {})

    users = []
    users_exists = os.path.exists(USERS_FILE)
    if users_exists:
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = []

    if not users_exists or not users:
        if ADMIN_INIT_PASSWORD:
            users = [{
                "user_id": ADMIN_INIT_USER,
                "name": "管理者",
                "password_hash": generate_password_hash(ADMIN_INIT_PASSWORD, method="scrypt"),
                "role": "admin"
            }]
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            app.logger.warning(
                "users.json を新規作成し、管理者ユーザー '%s' を作成しました。初回ログイン後 ADMIN_INIT_PASSWORD を環境変数から削除してください。",
                ADMIN_INIT_USER,
            )
        else:
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            app.logger.warning(
                "users.json を作成しましたが管理者は未作成です。ADMIN_INIT_PASSWORD を設定して再デプロイするか、手動で users.json を用意してください。"
            )

_bootstrap_files_and_admin()

# 必須キーが未設定なら警告（起動は継続）
if not OPENAI_API_KEY:
    app.logger.warning("OPENAI_API_KEY が未設定です。OpenAI 呼び出しは失敗します。")
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    app.logger.warning("LINE_CHANNEL_* が未設定です。/callback は正常動作しません。")


# =========================
#  基本I/O
# =========================
def load_users():
    return _safe_read_json(USERS_FILE, [])

def save_qa_log(question, answer, source="web", hit_db=False, extra=None):
    os.makedirs(LOG_DIR, exist_ok=True)
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

# OpenAIラッパ（モデル切替を一元管理）
def openai_chat(model, messages, **kwargs):
    params = dict(kwargs)
    mt  = params.pop("max_tokens", None)
    mct = params.pop("max_completion_tokens", None)
    mot = mct if mct is not None else mt  # 統一上限

    # ❶ Chat Completions
    try:
        chat_params = dict(params)
        if mot is not None:
            chat_params["max_tokens"] = mot  # ← 修正
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            **chat_params,
        )
        return resp.choices[0].message.content
    except Exception as e1:
        print("[OpenAI error - chat.completions]", e1)

    # ❷ Responses API（保険）
    try:
        joined = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        rparams = {"model": model, "input": joined, "temperature": params.get("temperature", 0.3)}
        if mot is not None:
            rparams["max_output_tokens"] = mot
        resp = client.responses.create(**rparams)
        return getattr(resp, "output_text", "") or ""
    except Exception as e2:
        print("[OpenAI error - responses]", e2)
        return ""

# 言語検出＆翻訳
OPENAI_MODEL_TRANSLATE = os.environ.get("OPENAI_MODEL_TRANSLATE", OPENAI_MODEL_HIGH)

def detect_lang_simple(text: str) -> str:
    if not text:
        return "ja"
    t = text.strip()
    if re.search(r"[ぁ-んァ-ン]", t):
        return "ja"
    zh_markers = [
        "今天","天氣","天气","請問","请问","交通","景點","景点","門票","门票",
        "美食","住宿","船班","航班","預約","预约","營業","营业","營業時間","营业时间"
    ]
    if any(m in t for m in zh_markers):
        return "zh-Hant"
    if re.fullmatch(r"[A-Za-z0-9\s\-\.,!?'\"/()]+", t):
        return "en"
    return "ja"

def translate_text(text: str, target_lang: str) -> str:
    if not text:
        return text
    lang_label = {'ja': '日本語', 'zh-Hant': '繁體中文', 'en': '英語'}.get(target_lang, '日本語')
    prompt = (
        f"次のテキストを{lang_label}に自然に翻訳してください。"
        "意味は変えず、URL・数値・記号・改行はそのまま維持。"
        "箇条書きや見出しの構造も保持してください。\n===\n" + text
    )
    model_t = OPENAI_MODEL_TRANSLATE
    out = openai_chat(
        model_t,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200,
    )
    if not out and model_t != OPENAI_MODEL_PRIMARY:
        out = openai_chat(
            OPENAI_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )
    return out or text


# =========================
#  お知らせ・データI/O
# =========================
def load_notices():
    return _safe_read_json(NOTICES_FILE, [])

def save_notices(notices):
    _atomic_json_dump(NOTICES_FILE, notices)

def load_entries():
    return _safe_read_json(ENTRIES_FILE, [])

def save_entries(entries):
    _atomic_json_dump(ENTRIES_FILE, entries)

def load_synonyms():
    return _safe_read_json(SYNONYM_FILE, {})

def save_synonyms(synonyms):
    _atomic_json_dump(SYNONYM_FILE, synonyms)

def load_shop_info(user_id):
    infos = _safe_read_json(SHOP_INFO_FILE, {})
    return infos.get(user_id, {})

def save_shop_info(user_id, info):
    infos = {}
    if os.path.exists(SHOP_INFO_FILE):
        with open(SHOP_INFO_FILE, "r", encoding="utf-8") as f:
            infos = json.load(f)
    infos[user_id] = info
    _atomic_json_dump(SHOP_INFO_FILE, infos)


# =========================
#  タグ・サジェスト
# =========================
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

    logs = []
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
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


def find_tags_by_synonym(question, synonyms):
    tags = set()
    for tag, synlist in synonyms.items():
        for syn in synlist + [tag]:
            if syn and syn in question:
                tags.add(tag)
    return list(tags)


# =========================
#  未ヒット時の絞り込み候補
# =========================
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
    areas = ["五島市", "新上五島町", "小値賀町", "宇久町"]
    top_tags = get_top_tags()

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

    top_tags = rank_tags(top_tags, question)
    filters = []
    if "雨" in (question or ""):
        filters.append("雨の日OK（屋内スポット）")
    if has_events_this_week():
        filters.append("今週のイベント")
    if any(w in (question or "") for w in ["子連れ", "家族", "ファミリー"]):
        filters.append("子連れOK")

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


# =========================
#  管理画面: 観光データ登録・編集
# =========================
@app.route("/admin/entry", methods=["GET", "POST"])
@login_required
def admin_entry():
    if session.get("role") == "shop":
        return redirect(url_for("shop_entry"))

    entries = [_norm_entry(x) for x in load_entries()]  # 表示前にも正規化

    edit_id = request.values.get("edit")
    entry_edit = None
    if edit_id not in (None, "", "None"):
        try:
            idx = int(edit_id)
            if 0 <= idx < len(entries):
                entry_edit = entries[idx]
        except Exception:
            entry_edit = None

    if request.method == "POST":
        # 共通
        category = (request.form.get("category") or "").strip() or "観光"
        title    = (request.form.get("title") or "").strip()
        desc     = (request.form.get("desc") or "").strip()
        address  = (request.form.get("address") or "").strip()
        map_url  = (request.form.get("map") or "").strip()

        # リスト系（改行/カンマ両対応）
        tags   = _split_lines_commas(request.form.get("tags", ""))
        areas  = request.form.getlist("areas") or _split_lines_commas(request.form.get("areas",""))
        links  = _split_lines_commas(request.form.get("links", ""))
        pay_in = request.form.getlist("payment") or _split_lines_commas(request.form.get("payment",""))

        # 店舗系
        tel         = (request.form.get("tel") or "").strip()
        holiday     = (request.form.get("holiday") or "").strip()
        open_hours  = (request.form.get("open_hours") or "").strip()
        parking     = (request.form.get("parking") or "").strip()
        parking_num = (request.form.get("parking_num") or "").strip()
        remark      = (request.form.get("remark") or "").strip()

        # 追加情報
        extras_keys = request.form.getlist("extras_key[]")
        extras_vals = request.form.getlist("extras_val[]")
        extras = {}
        for k, v in zip(extras_keys, extras_vals):
            k = (k or "").strip()
            v = (v or "").strip()
            if k:
                extras[k] = v

        if not areas:
            flash("エリアは1つ以上選択してください")
            return redirect(url_for("admin_entry"))

        new_entry = {
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas,
            "links": links,
            "payment": pay_in,
            "tel": tel,
            "holiday": holiday,
            "open_hours": open_hours,
            "parking": parking,
            "parking_num": parking_num,
            "remark": remark,
            "extras": extras,
        }
        new_entry = _norm_entry(new_entry)  # 保存前にも正規化

        edit_hidden = request.form.get("edit_id")
        if edit_hidden not in (None, "", "None"):
            try:
                idx = int(edit_hidden)
                entries[idx] = new_entry
                flash("編集しました")
            except Exception:
                entries.append(new_entry)
                flash("編集ID不正のため新規で追加しました")
        else:
            entries.append(new_entry)
            flash("登録しました")

        save_entries(entries)
        return redirect(url_for("admin_entry"))

    return render_template(
        "admin_entry.html",
        entries=entries,
        entry_edit=entry_edit,
        edit_id=edit_id if edit_id not in (None, "", "None") else None,
        role=session.get("role", "")
    )

@app.route("/admin/entry/delete/", defaults={"idx": None}, methods=["POST"])
@app.route("/admin/entry/delete/<int:idx>", methods=["POST"])
@login_required
def delete_entry(idx):
    if session.get("role") != "admin":
        abort(403)

    if idx is None:
        form_idx = request.form.get("idx")
        if form_idx is not None and str(form_idx).isdigit():
            idx = int(form_idx)

    if idx is None:
        flash("削除対象が指定されていません")
        return redirect(url_for("admin_entry"))

    entries = load_entries()
    if 0 <= idx < len(entries):
        entries.pop(idx)
        save_entries(entries)
        flash("削除しました")
    else:
        flash("指定された項目が見つかりません")
    return redirect(url_for("admin_entry"))


@app.route("/shop/entry", methods=["GET", "POST"])
@login_required
def shop_entry():
    if session.get("role") != "shop":
        return redirect(url_for("admin_entry"))
    user_id = session["user_id"]

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
        map_url = request.form.get("map", "")

        tags = _split_lines_commas(request.form.get("tags", ""))
        areas = request.form.getlist("areas")
        remark = request.form.get("remark", "")

        links = _split_lines_commas(request.form.get("links", ""))

        extras_keys = request.form.getlist("extras_key[]")
        extras_vals = request.form.getlist("extras_val[]")
        extras = {k.strip(): v.strip() for k, v in zip(extras_keys, extras_vals) if k.strip()}

        if not areas:
            flash("エリアは1つ以上選択してください")
            return redirect(url_for("shop_entry"))

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
            "map": map_url,
            "links": links,
            "extras": extras
        }
        entry_data = _norm_entry(entry_data)

        entries = load_entries()
        entry_idx = next((i for i, e in enumerate(entries) if e.get("user_id") == user_id), None)
        if entry_idx is not None:
            entries[entry_idx] = entry_data
        else:
            entries.append(entry_data)
        save_entries(entries)

        flash("店舗情報を保存しました")
        return redirect(url_for("shop_entry"))

    user_entries = [e for e in load_entries() if e.get("user_id") == user_id]
    shop_entry_data = user_entries[-1] if user_entries else None
    shop_entry_data = _norm_entry(shop_entry_data) if shop_entry_data else None
    return render_template("shop_entry.html", role="shop", shop_edit=shop_entry_data)

@app.route("/admin/entries_edit", methods=["GET", "POST"])
@login_required
def admin_entries_edit():
    if session.get("role") != "admin":
        abort(403)

    if request.method == "POST":
        raw_json = request.form.get("entries_raw", "")
        try:
            data = json.loads(raw_json)
            if not isinstance(data, list):
                raise ValueError("ルート要素は配列(list)にしてください")
            data = [_norm_entry(e) for e in data]   # 正規化して保存
            save_entries(data)
            flash("entries.jsonを上書きしました")
            return redirect(url_for("admin_entries_edit"))
        except Exception as e:
            flash("JSONエラー: " + str(e))
            return render_template("admin_entries_edit.html", entries_raw=raw_json), 400

    try:
        entries = load_entries()
        entries_raw = json.dumps(entries, ensure_ascii=False, indent=2)
    except Exception as e:
        entries_raw = "[]"
        flash("entries.jsonの読み込みに失敗しました: " + str(e))

    return render_template("admin_entries_edit.html", entries_raw=entries_raw)


# =========================
#  CSV取り込み（既存に追加）
# =========================
@app.route("/admin/entries_import_csv", methods=["POST"])
@login_required
def admin_entries_import_csv():
    if session.get("role") != "admin":
        abort(403)

    file = request.files.get("csv_file")
    if not file:
        flash("CSVファイルが選択されていません")
        return redirect(url_for("admin_entries_edit"))

    import csv, io as _io, json as _json

    buf = _io.TextIOWrapper(file.stream, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(buf)

    new_entries = []
    for row in reader:
        category = (row.get("category") or "観光").strip()
        title    = (row.get("title") or "").strip()
        desc     = (row.get("desc") or "").strip()
        address  = (row.get("address") or "").strip()
        map_url  = (row.get("map") or "").strip()
        tags     = _split_lines_commas(row.get("tags"))
        areas    = _split_lines_commas(row.get("areas"))
        links    = _split_lines_commas(row.get("links"))

        # 必須チェック
        if not title or not desc or not areas:
            continue  # エリア必須

        # extras: "extras"列(JSON) + extra_*/extra:* 列
        extras = {}
        extras_raw = row.get("extras")
        if extras_raw:
            try:
                obj = _json.loads(extras_raw)
                if isinstance(obj, dict):
                    extras.update({str(k): str(v) for k, v in obj.items()})
            except Exception:
                pass
        for k, v in row.items():
            if k is None:
                continue
            ks = str(k)
            if ks.startswith("extra_") or ks.startswith("extra:"):
                name = ks.split("_", 1)[1] if ks.startswith("extra_") else ks.split(":", 1)[1]
                name = name.strip()
                if name:
                    extras[name] = str(v or "").strip()

        e = {
            "category": category,
            "title": title,
            "desc": desc,
            "address": address,
            "map": map_url,
            "tags": tags,
            "areas": areas,
        }
        if links:
            e["links"] = links
        if extras:
            e["extras"] = extras

        e = _norm_entry(e)
        new_entries.append(e)

    if not new_entries:
        flash("CSVから有効な行が見つかりませんでした（title/desc/areas 必須）")
        return redirect(url_for("admin_entries_edit"))

    entries = load_entries()
    entries.extend(new_entries)
    save_entries(entries)

    flash(f"CSVから {len(new_entries)} 件を追加しました（既存に追記）")
    return redirect(url_for("admin_entries_edit"))


# =========================
#  ログ・未ヒット確認
# =========================
@app.route("/admin/logs")
@login_required
def admin_logs():
    if session.get("role") != "admin":
        abort(403)
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
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
    unhit_logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    if not log.get("hit_db"):
                        unhit_logs.append(log)
                except Exception:
                    pass
    unhit_logs = unhit_logs[-100:]
    return render_template("admin_unhit.html", unhit_logs=unhit_logs)

@app.route("/api/faq_suggest", methods=["POST"])
@login_required
def api_faq_suggest():
    if session.get("role") != "admin":
        abort(403)
    q = (request.form.get("q") or (request.get_json(silent=True) or {}).get("q") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "q is required"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 500
    text = ai_suggest_faq(q, model=OPENAI_MODEL_PRIMARY) or ""
    return jsonify({"ok": True, "text": text})


@app.route("/admin/add_entry", methods=["POST"])
@login_required
def admin_add_entry():
    if session.get("role") != "admin":
        abort(403)
    entries = load_entries()
    title = request.form.get("title", "")
    desc = request.form.get("desc", "")
    tags = _split_lines_commas(request.form.get("tags", ""))
    areas = _split_lines_commas(request.form.get("areas", ""))
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
    entry = _norm_entry(entry)
    entries.append(entry)
    save_entries(entries)
    flash("DBに追加しました")
    return redirect(url_for("admin_entry"))


# =========================
#  バックアップ/復元
# =========================
def write_full_backup_zip(out_dir: str) -> str:
    """アプリ内バックアップを out_dir に保存し、ファイルパスを返す"""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"gotokanko_{ts}.zip"
    out_path = os.path.join(out_dir, filename)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 単体ファイル
        if os.path.exists(ENTRIES_FILE):   zf.write(ENTRIES_FILE,  arcname="entries.json")
        if os.path.exists(SYNONYM_FILE):   zf.write(SYNONYM_FILE,  arcname="synonyms.json")
        if os.path.exists(NOTICES_FILE):   zf.write(NOTICES_FILE,  arcname="notices.json")
        if os.path.exists(SHOP_INFO_FILE): zf.write(SHOP_INFO_FILE,arcname="shop_infos.json")

        # data/
        if os.path.exists(DATA_DIR):
            for root, dirs, files in os.walk(DATA_DIR):
                for fname in files:
                    fpath   = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, BASE_DIR)
                    zf.write(fpath, arcname)

        # logs/
        if os.path.exists(LOG_DIR):
            for root, dirs, files in os.walk(LOG_DIR):
                for fname in files:
                    fpath   = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, BASE_DIR)
                    zf.write(fpath, arcname)

    return out_path

@app.route("/admin/backup")
@login_required
def admin_backup():
    if session.get("role") != "admin":
        abort(403)
    out_dir = os.path.join(BASE_DIR, "manual_backups")
    path = write_full_backup_zip(out_dir)
    app.logger.info(f"[backup] manual saved: {path}")
    return send_file(path, as_attachment=True, download_name=os.path.basename(path), mimetype="application/zip")

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

@app.route("/internal/backup", methods=["POST"])
def internal_backup():
    token = request.headers.get("X-Backup-Token", "")
    if token != os.environ.get("BACKUP_JOB_TOKEN"):
        abort(403)

    out_dir = os.path.join(BASE_DIR, "auto_backups")
    path = write_full_backup_zip(out_dir)

    # ローテーション（最新10個だけ残す）
    try:
        files = sorted(
            [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".zip")],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for old in files[10:]:
            os.remove(old)
    except Exception:
        app.logger.exception("backup rotation failed")

    app.logger.info(f"[backup] saved: {path}")
    return jsonify({"ok": True, "saved": path})


# =========================
#  認証
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("username")
        pw = request.form.get("password")
        users = load_users()
        user = next((u for u in users if u["user_id"] == user_id), None)
        if user and check_password_hash(user["password_hash"], pw):
            session.permanent = True  # ← 追加: 有効期限を効かせる
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


# =========================
#  マスター管理（復活＆強化）
# =========================
@app.route("/admin/synonyms", methods=["GET", "POST"])
@login_required
def admin_synonyms():
    if session.get("role") != "admin":
        abort(403)

    if request.method == "POST":
        tags_in_dom_order = [t.strip() for t in request.form.getlist("tag")]

        syn_keys = [k for k in request.form.keys() if k.startswith("synonyms_")]
        syn_indices = sorted(
            [int(k.split("_", 1)[1]) for k in syn_keys if k.split("_", 1)[1].isdigit()]
        )

        new_synonyms = {}
        for tag, idx in zip(tags_in_dom_order, syn_indices):
            if not tag:
                continue
            syn_str = request.form.get(f"synonyms_{idx}", "") or ""
            syn_list = [s.strip() for s in syn_str.split(",") if s.strip()]
            new_synonyms[tag] = syn_list

        add_tag = (request.form.get("add_tag") or "").strip()
        add_syns_str = request.form.get("add_synonyms", "") or ""
        if add_tag:
            add_list = [s.strip() for s in add_syns_str.split(",") if s.strip()]
            new_synonyms[add_tag] = add_list

        save_synonyms(new_synonyms)
        flash("類義語辞書を保存しました")
        return redirect(url_for("admin_synonyms"))

    synonyms = load_synonyms()
    return render_template("admin_synonyms.html", synonyms=synonyms)

@app.route("/admin/manual")
@login_required
def admin_manual():
    if session.get("role") != "admin":
        abort(403)
    return render_template("admin_manual.html")


# =========================
#  トップ/ヘルスチェック
# =========================
@app.route("/")
def home():
    return "<a href='/admin/entry'>[観光データ管理]</a>"

@app.route("/healthz")
def healthz():
    return "ok", 200


# =========================
#  天気・フェリー情報
# =========================
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

    lang = "ja" if not ENABLE_FOREIGN_LANG else detect_lang_simple(question)
    if not any(kw in question for kw in weather_keywords):
        return None, False

    for entry in WEATHER_LINKS:
        if entry["area"] in question:
            if lang == "zh-Hant":
                return f"【{entry['area']}天氣】\n最新資訊：\n{entry['url']}\n（可查看今日、週預報與降雨雷達）", True
            if lang == "en":
                return f"[{entry['area']} weather]\nLatest forecast:\n{entry['url']}", True
            return f"【{entry['area']}の天気】\n最新の{entry['area']}の天気情報はこちら\n{entry['url']}", True

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


# =========================
#  観光データ横断検索
# =========================
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

    cleaned_query = clean_query_for_search(question)
    # 1. タイトル完全一致
    hits = [e for e in entries if cleaned_query and e.get("title", "") == cleaned_query
            and (not target_areas or any(area in e.get("areas", []) for area in target_areas))]
    if hits:
        return hits

    # 2. 各カラム部分一致（links/extras を含む）
    hits = []
    for e in entries:
        haystacks = [
            e.get("title", ""),
            " ".join(e.get("tags", [])),
            e.get("desc", ""),
            e.get("address", ""),
            " ".join(e.get("links", []) or []),
            " ".join((e.get("extras") or {}).values())
        ]
        target_str = " ".join(haystacks)
        if cleaned_query and cleaned_query in target_str:
            if not target_areas or any(area in e.get("areas", []) for area in target_areas):
                hits.append(e)
    if hits:
        return hits

    # 3. 元のtitle, desc部分一致
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

def fetch_and_search_web(question):
    return None

def smart_search_answer_with_hitflag(question):
    meta = {"model_primary": OPENAI_MODEL_PRIMARY, "fallback": None}

    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        return weather_reply, True, meta

    if any(word in question for word in ["飛行機", "空港", "航空便", "欠航", "到着", "出発"]):
        return (
            "五島つばき空港の最新の運行状況は、公式Webサイトでご確認ください。\n"
            "▶ https://www.fukuekuko.jp/",
            True,
            meta,
        )

    if any(word in question for word in ["フェリー", "船", "運航", "ジェットフォイル", "太古"]):
        return FERRY_INFO, True, meta

    entries = find_entry_info(question)
    if entries:
        if len(entries) == 1:
            e = entries[0]
            lines = []
            if e.get("title"):   lines.append(f"◼︎ {e['title']}")
            if e.get("areas"):   lines.append(f"エリア: {', '.join(e['areas'])}")
            if e.get("desc"):    lines.append(f"説明: {e['desc']}")
            if e.get("address"): lines.append(f"住所: {e['address']}")
            if e.get("map"):     lines.append(f"地図: {e['map']}")
            if e.get("tags"):    lines.append(f"タグ: {', '.join(e['tags'])}")
            if e.get("links"):   lines.append("リンク: " + " / ".join(e['links']))
            return "\n".join(lines), True, meta
        else:
            try:
                snippets = [f"タイトル: {e.get('title','')}\n説明: {e.get('desc','')}\n住所: {e.get('address','')}\n" for e in entries]
                ai_ans = ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY)
                return ai_ans or "複数スポットが見つかりましたが要約に失敗しました。", True, meta
            except Exception as e:
                return "複数スポットが見つかりましたが要約に失敗しました。", False, meta

    snippets = search_text_files(question, data_dir=DATA_DIR)
    if snippets:
        try:
            return ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY), True, meta
        except Exception as e:
            return "AI要約でエラー: " + str(e), False, meta

    web_texts = fetch_and_search_web(question)
    if web_texts:
        try:
            return ai_summarize(web_texts, question, model=OPENAI_MODEL_PRIMARY), True, meta
        except Exception as e:
            return "Web要約でエラー: " + str(e), False, meta

    suggest_text, suggest_meta = ("", {})
    if SUGGEST_UNHIT:
        suggest_text, suggest_meta = build_refine_suggestions(question)
        meta["suggestions"] = suggest_meta

    if REASK_UNHIT:
        fallback_text = ai_suggest_faq(question, model=OPENAI_MODEL_HIGH)
        if fallback_text:
            meta["fallback"] = {"mode": "high_faq", "model": OPENAI_MODEL_HIGH}
            answer = "【参考回答（データ未登録のため要確認）】\n" + fallback_text
            if suggest_text:
                answer += "\n---\n" + suggest_text
            return answer, False, meta

        if suggest_text:
            return ("申し訳ありません。該当データが見つかりませんでした。\n" + suggest_text, False, meta)

        return ("申し訳ありません。現在この質問には確実な情報を持っていません。", False, meta)

    if suggest_text:
        return ("申し訳ありません。該当データが見つかりませんでした。\n" + suggest_text, False, meta)

    return ("申し訳ありません。現在この質問には確実な情報を持っていません。", False, meta)


# =========================
#  API: /ask
# =========================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        save_qa_log(question, weather_reply, source="web", hit_db=True, extra={"kind": "weather"})
        return jsonify({"answer": weather_reply, "hit_db": True, "meta": {"kind": "weather"}})

    if not question:
        return jsonify({"error": "質問が空です"}), 400

    orig_lang = detect_lang_simple(question)
    lang = orig_lang
    if not ENABLE_FOREIGN_LANG:
        lang = "ja"

    q_for_logic = question if orig_lang == "ja" else translate_text(question, "ja")
    answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)

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


# =========================
#  LINE Webhook
# =========================
@app.route("/callback", methods=["POST"])
def callback():
    # 設定未投入なら即200で返す（ログ汚染回避）
    if not (LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET):
        return ("OK", 200)
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

    orig_lang = detect_lang_simple(user_message)
    lang = orig_lang
    if not ENABLE_FOREIGN_LANG:
        lang = "ja"

    q_for_logic = user_message if orig_lang == "ja" else translate_text(user_message, "ja")
    answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)

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


# =========================
#  お知らせ管理
# =========================
@app.route("/admin/notices", methods=["GET", "POST"])
@login_required
def admin_notices():
    if session.get("role") != "admin":
        flash("権限がありません")
        return redirect(url_for("login"))
    notices = load_notices()
    edit_id = request.values.get("edit")
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


# =========================
#  エントリーポイント
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
