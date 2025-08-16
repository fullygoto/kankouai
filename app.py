import os
import json
import re
import datetime
import itertools
import time  
import threading  
from collections import Counter
from typing import Any, Dict, List
from werkzeug.routing import BuildError


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
# --- Jinja2 互換用: 'string' / 'mapping' テストが無い環境向け ---
if 'string' not in app.jinja_env.tests:
    app.jinja_env.tests['string'] = lambda v: isinstance(v, str)
if 'mapping' not in app.jinja_env.tests:
    from collections.abc import Mapping
    app.jinja_env.tests['mapping'] = lambda v: isinstance(v, Mapping)

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

def safe_url_for(endpoint, **values):
    try:
        return url_for(endpoint, **values)
    except BuildError:
        return "#"  # 未実装リンクはダミーへ

app.jinja_env.globals["safe_url_for"] = safe_url_for

# =========================
#  環境変数 / モデル
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") or ""  # 未設定でも起動OK
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET") or ""              # 未設定でも起動OK

# ← ここを gpt-5-mini をデフォルトに
OPENAI_MODEL_PRIMARY = os.environ.get("OPENAI_MODEL_PRIMARY", "gpt-5-mini")
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
    """
    - GPT-5 系: Responses API を使用（temperature は渡さない / max_output_tokens を使用）
    - それ以外: Chat Completions を優先（max_tokens / temperature 可）
    - どちらも同じ呼び出し感覚で使えるよう吸収
    """
    params = dict(kwargs)
    # 呼び出し側がどれで渡しても拾えるようにする
    mot = (
        params.pop("max_output_tokens", None)
        or params.pop("max_completion_tokens", None)
        or params.pop("max_tokens", None)
    )
    temp = params.pop("temperature", None)

    # GPT-5 系は Responses API を使う
    use_responses = model.startswith(("gpt-5",))

    if use_responses:
        try:
            # Responses API: input は messages をそのまま渡せる
            rparams = {"model": model, "input": messages}
            if mot is not None:
                rparams["max_output_tokens"] = mot
            resp = client.responses.create(**rparams)
            return getattr(resp, "output_text", "") or ""
        except Exception as e:
            print("[OpenAI error - responses]", e)
            return ""

    # それ以外のモデルは Chat Completions を優先
    try:
        chat_params = {}
        if mot is not None:
            chat_params["max_tokens"] = mot
        if temp is not None:
            chat_params["temperature"] = temp
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            **chat_params,
        )
        return resp.choices[0].message.content
    except Exception as e1:
        print("[OpenAI error - chat.completions]", e1)
        # 念のため Responses API にもフォールバック（4 系でも通ることがある）
        try:
            rparams = {"model": model, "input": messages}
            if mot is not None:
                rparams["max_output_tokens"] = mot
            resp = client.responses.create(**rparams)
            return getattr(resp, "output_text", "") or ""
        except Exception as e2:
            print("[OpenAI error - responses fallback]", e2)
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
    raw = _safe_read_json(ENTRIES_FILE, [])
    return [_norm_entry(e) for e in raw]

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
#  シノニム ヘルパー & 自動生成
# =========================
def merge_synonyms(base: dict, incoming: dict) -> dict:
    """
    synonyms.json のマージ:
      - 既存のキー（タグ）は残しつつ、新規は追加
      - 値（シノニム配列）はユニーク化・空要素除去
      - 大文字小文字や前後空白も正規化
    """
    base = dict(base or {})
    for tag, syns in (incoming or {}).items():
        if not tag:
            continue
        tag_norm = str(tag).strip()
        cur = set([s.strip() for s in base.get(tag_norm, []) if s and str(s).strip()])
        add = set([s.strip() for s in (syns or []) if s and str(s).strip()])
        base[tag_norm] = sorted(cur.union(add))
    return base

def ai_propose_synonyms(tags, context_text="", model=None) -> dict:
    """
    OpenAIを使って、複数タグに対するシノニム案をJSONで返す。
    返り値の例: {"教会": ["カトリック教会","チャーチ","Church"], "五島うどん": ["うどん","GOTO Udon"]}
    """
    if not OPENAI_API_KEY:
        return {}
    if not tags:
        return {}
    model = model or OPENAI_MODEL_HIGH
    prompt = (
        "次の観光・生活関連の『タグ』ごとに、検索時の取りこぼしを防ぐための日本語/英語/繁体字の同義語・表記ゆれ・通称を出してください。"
        "出力は必ず JSON オブジェクトのみ（整形済み）。キー=元タグ、値=文字列配列。"
        "元タグと完全に同じ語は含めないでください。スラングや不正確な語は避け、1タグあたり1〜6語程度。\n"
        f"【文脈参考】\n{context_text[:1000]}\n"
        f"【タグ一覧】\n{', '.join(sorted(set(tags)))}"
    )
    try:
        content = openai_chat(
            model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        # JSONだけを期待するが、万一先頭/末尾にノイズがあったら抽出
        import json as _json, re as _re
        m = _re.search(r"\{.*\}", content, _re.S)
        j = m.group(0) if m else content
        data = _json.loads(j)
        # 値はリストで揃える
        cleaned = {}
        for k, v in (data or {}).items():
            if not k:
                continue
            if isinstance(v, str):
                v = [v]
            cleaned[k] = [str(x).strip() for x in (v or []) if str(x).strip()]
        return cleaned
    except Exception as e:
        app.logger.warning(f"[ai_propose_synonyms] fail: {e}")
        return {}

def auto_update_synonyms_from_entries(entries_like):
    """
    エントリ群からタグを集めてAIでシノニムを提案→synonyms.json をマージ保存。
    """
    try:
        if not entries_like:
            return
        # まとめて提案（API呼び出し1回）
        all_tags = set()
        contexts = []
        for e in entries_like:
            all_tags.update(e.get("tags", []) or [])
            # タイトル/説明/住所/備考を少し文脈として
            contexts.append(" / ".join([
                e.get("title",""), e.get("desc",""),
                e.get("address",""), " ".join((e.get("extras") or {}).values())
            ]))
        if not all_tags:
            return
        proposal = ai_propose_synonyms(sorted(all_tags), "\n".join(contexts))
        if not proposal:
            return
        syn0 = load_synonyms()
        syn1 = merge_synonyms(syn0, proposal)
        save_synonyms(syn1)
        app.logger.info(f"[auto_update_synonyms] updated for tags: {sorted(all_tags)}")
    except Exception:
        app.logger.exception("[auto_update_synonyms] error")


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

# === 追質問（意図深掘り）ビルダー ===
def build_probe_questions(question: str):
    """
    あいまいな問い合わせのときに、追加で確認したいポイントを返す。
    返り値は「箇条書きに載せる1行文」リスト。
    """
    q = (question or "").strip()
    items = []

    # ドライブ系（モデルコース含む）
    if any(k in q for k in ["ドライブ", "モデルコース", "コース", "ルート"]):
        items.append("どんなドライブが良いですか？（海岸線／教会巡り／展望台／灯台／夕日）")
        items.append("出発エリアはどちらですか？（五島市／新上五島町／小値賀町／宇久町）")
        items.append("所要時間は？（2〜3時間／半日／1日）")
        items.append("お車はありますか？（あり／なし）")

    # 教会系
    if "教会" in q:
        items.append("内部見学の可否は重視しますか？（はい／いいえ）")
        items.append("世界遺産関連を優先しますか？（はい／いいえ）")

    # ビーチ／海水浴
    if any(k in q for k in ["ビーチ", "海水浴"]):
        items.append("遊泳重視ですか？景観重視ですか？")
        items.append("シャワー・更衣室の有無は必要ですか？（必要／不要）")

    # 子連れ
    if any(k in q for k in ["子連れ", "家族", "ファミリー"]):
        items.append("お子さまの年齢層は？（未就学／小学生／中高生）")
        items.append("移動時間はどれくらいが良いですか？（短め／こだわらない）")

    # 雨
    if "雨" in q:
        items.append("屋内中心で探しますか？（はい／いいえ）")

    return items


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
        # ★ここから追記：意図深掘りの追質問を合体
    probe_lines = build_probe_questions(question)
    if probe_lines:
        lines.append("")  # 空行で区切り
        lines.append("❓ もう少し教えてください")
        for p in probe_lines:
            lines.append(f"- {p}")

    msg = "\n".join(lines)
    # 返り値の meta に probe も載せる（後方互換）
    return msg, {"areas": areas, "tags": top_tags, "filters": filters, "probe": probe_lines}

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
        # ① 生JSONでの上書き（従来互換）
        raw_json = request.form.get("entries_raw")
        if raw_json:
            try:
                data = json.loads(raw_json)
                if not isinstance(data, list):
                    raise ValueError("ルート要素は配列(list)にしてください")
                data = [_norm_entry(e) for e in data]
                save_entries(data)
                flash("entries.jsonを上書きしました")
                return redirect(url_for("admin_entries_edit"))
            except Exception as e:
                flash("JSONエラー: " + str(e))
        else:
            # ② UIフォームからの保存（テンプレートに合わせる）
            cats        = request.form.getlist("category[]")
            titles      = request.form.getlist("title[]")
            descs       = request.form.getlist("desc[]")
            addresses   = request.form.getlist("address[]")
            maps        = request.form.getlist("map[]")
            tags_list   = request.form.getlist("tags[]")
            areas_list  = request.form.getlist("areas[]")
            tels        = request.form.getlist("tel[]")
            holidays    = request.form.getlist("holiday[]")
            opens       = request.form.getlist("open_hours[]")
            parkings    = request.form.getlist("parking[]")
            parking_nums= request.form.getlist("parking_num[]")
            payments    = request.form.getlist("payment[]")
            remarks     = request.form.getlist("remark[]")
            links_list  = request.form.getlist("links[]")

            new_entries = []
            n = max(len(titles), len(descs))
            for i in range(n):
                title = (titles[i] if i < len(titles) else "").strip()
                desc  = (descs[i]  if i < len(descs)  else "").strip()
                # 完全空行はスキップ
                if not title and not desc:
                    continue

                e = {
                    "category":   (cats[i] if i < len(cats) else "観光").strip() or "観光",
                    "title":      title,
                    "desc":       desc,
                    "address":    (addresses[i]  if i < len(addresses)  else "").strip(),
                    "map":        (maps[i]       if i < len(maps)       else "").strip(),
                    "tags":       _split_lines_commas(tags_list[i]  if i < len(tags_list)  else ""),
                    "areas":      _split_lines_commas(areas_list[i] if i < len(areas_list) else ""),
                    "links":      _split_lines_commas(links_list[i]  if i < len(links_list)  else ""),
                    "payment":    _split_lines_commas(payments[i]    if i < len(payments)    else ""),
                    "tel":        (tels[i]        if i < len(tels)        else "").strip(),
                    "holiday":    (holidays[i]    if i < len(holidays)    else "").strip(),
                    "open_hours": (opens[i]       if i < len(opens)       else "").strip(),
                    "parking":    (parkings[i]    if i < len(parkings)    else "").strip(),
                    "parking_num":(parking_nums[i]if i < len(parking_nums)else "").strip(),
                    "remark":     (remarks[i]     if i < len(remarks)     else "").strip(),
                    "extras": {},
                }
                e = _norm_entry(e)
                if not e["areas"]:
                    # エリア必須（空なら行ごと無視）
                    continue
                new_entries.append(e)

            save_entries(new_entries)
            flash(f"{len(new_entries)} 件保存しました")
            return redirect(url_for("admin_entries_edit"))

    # GET（またはPOSTエラー後の再表示）
    entries = [_norm_entry(x) for x in load_entries()]
    return render_template("admin_entries_edit.html", entries=entries)


# =========================
#  CSV取り込み（既存に追加）
# =========================
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

    # --- 文字コードを自動判定して読み込む（UTF-8/UTF-8-SIG/CP932対応） ---
    raw = file.read()  # bytes
    if not raw:
        flash("アップロードされたCSVが空です")
        return redirect(url_for("admin_entries_edit"))

    enc_tried = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    decoded = None
    used_enc = None
    for enc in enc_tried:
        try:
            decoded = raw.decode(enc)
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        # 最後の手段：置換ありでCP932として読む（文字化け箇所は□になる）
        decoded = raw.decode("cp932", errors="replace")
        used_enc = "cp932(replace)"

    buf = _io.StringIO(decoded, newline="")
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

        # 必須チェック（title/desc/areas）
        if not title or not desc or not areas:
            continue

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
        flash(f"CSVから有効な行が見つかりませんでした（title/desc/areas 必須）［encoding: {used_enc}］")
        return redirect(url_for("admin_entries_edit"))

    entries = load_entries()
    entries.extend(new_entries)
    save_entries(entries)

    try:
        auto_update_synonyms_from_entries(new_entries)
        flash(f"CSVから {len(new_entries)} 件を追加＋シノニム更新完了（encoding: {used_enc}）")
    except Exception:
        flash(f"CSVから {len(new_entries)} 件を追加（encoding: {used_enc}／シノニム自動更新は失敗）")

    return redirect(url_for("admin_entries_edit"))

# =========================
#  管理: JSONインポート（entries / synonyms）
# =========================
@app.route("/admin/import", methods=["GET", "POST"])
@login_required
def admin_import():
    if session.get("role") != "admin":
        abort(403)

    if request.method == "POST":
        import json as _json
        # entries.json の処理
        if "entries_json" in request.files and request.files["entries_json"].filename:
            mode = request.form.get("entries_mode", "merge")  # merge|replace
            file = request.files["entries_json"]
            try:
                data = _json.load(file.stream)
                if not isinstance(data, list):
                    raise ValueError("entries.json は配列(list)である必要があります")
                data = [_norm_entry(e) for e in data]
                if mode == "replace":
                    save_entries(data)
                    flash(f"entries.json を {len(data)} 件で上書きしました")
                    # 上書き時もシノニム自動生成（重いのでタグだけまとめて）
                    try:
                        auto_update_synonyms_from_entries(data[:80])  # 安全のため上位80件だけ文脈に
                        flash("シノニムを自動更新しました")
                    except Exception:
                        flash("シノニム自動更新に失敗しました")
                else:
                    cur = load_entries()
                    cur.extend(data)
                    save_entries(cur)
                    flash(f"entries.json に {len(data)} 件マージしました（追記）")
                    try:
                        auto_update_synonyms_from_entries(data)
                        flash("シノニムを自動更新しました")
                    except Exception:
                        flash("シノニム自動更新に失敗しました")
            except Exception as e:
                flash(f"entries.json の読み込みに失敗: {e}")

        # synonyms.json の処理
        if "synonyms_json" in request.files and request.files["synonyms_json"].filename:
            mode = request.form.get("synonyms_mode", "merge")  # merge|replace
            file = request.files["synonyms_json"]
            try:
                data = _json.load(file.stream)
                if not isinstance(data, dict):
                    raise ValueError("synonyms.json はオブジェクト(dict)である必要があります")
                if mode == "replace":
                    save_synonyms(data)
                    flash("synonyms.json を上書きしました")
                else:
                    cur = load_synonyms()
                    merged = merge_synonyms(cur, data)
                    save_synonyms(merged)
                    flash("synonyms.json をマージしました（重複は自動でユニーク化）")
            except Exception as e:
                flash(f"synonyms.json の読み込みに失敗: {e}")

        return redirect(url_for("admin_import"))

    # GET: 画面表示
    return render_template("admin_import.html")



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
    return render_template("admin_unhit.html", unhit_logs=unhit_logs, role=session.get("role",""))

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
    try:
        auto_update_synonyms_from_entries([entry])
        flash("DBに追加しました（シノニムも自動更新）")
    except Exception:
        flash("DBに追加しました（シノニム自動更新でエラーが出ました。ログを確認してください）")

    return redirect(url_for("admin_entry"))



# =========================
#  バックアップ/復元
# =========================
def _safe_extractall(zf: zipfile.ZipFile, dst: str):
    base = os.path.abspath(dst)
    for member in zf.namelist():
        target = os.path.abspath(os.path.join(dst, member))
        if not target.startswith(base + os.sep) and target != base:
            raise Exception("Unsafe path found in ZIP (zip slip)")
    os.makedirs(dst, exist_ok=True)
    zf.extractall(dst)

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

    if request.args.get("download"):
        out_dir = os.path.join(BASE_DIR, "manual_backups")
        path = write_full_backup_zip(out_dir)
        app.logger.info(f"[backup] manual saved: {path}")
        return send_file(path, as_attachment=True,
                         download_name=os.path.basename(path),
                         mimetype="application/zip")

    # 画面表示（統計を出すと親切）
    stats = {
        "entries_count": len(load_entries()),
        "synonyms_count": len(load_synonyms()),
        "notices_count": len(load_notices()),
        "logs_count": sum(1 for _ in open(LOG_FILE, encoding="utf-8")) if os.path.exists(LOG_FILE) else 0,
    }
    return render_template("admin_backup.html", stats=stats)

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

# ========== 類義語：インポート（上書き） ==========
@app.route("/admin/synonyms/import", methods=["POST"])
@login_required
def admin_synonyms_import():
    if session.get("role") != "admin":
        abort(403)

    # ファイル or テキストのどちらか
    up = request.files.get("json_file")
    text = (request.form.get("json_text") or "").strip()

    try:
        if up and up.filename:
            raw = up.read().decode("utf-8-sig")
            new_dict = json.loads(raw)
        elif text:
            new_dict = json.loads(text)
        else:
            flash("JSONファイルまたはテキストを指定してください")
            return redirect(url_for("admin_synonyms"))

        if not isinstance(new_dict, dict):
            raise ValueError('JSONは {"タグ": ["別名", ...]} の形にしてください')

        # 正規化：各値を list[str] 化
        norm = {}
        for k, v in new_dict.items():
            key = str(k).strip()
            if isinstance(v, str):
                vals = _split_lines_commas(v)
            elif isinstance(v, (list, tuple)):
                vals = [str(x).strip() for x in v if str(x).strip()]
            elif v:
                vals = [str(v).strip()]
            else:
                vals = []
            norm[key] = vals

        # 上書き保存
        save_synonyms(norm)
        flash(f"類義語辞書を {len(norm)} タグ分で上書きしました")
    except Exception as e:
        app.logger.exception("synonyms import failed")
        flash("インポートに失敗: " + str(e))

    return redirect(url_for("admin_synonyms"))


# ========== 類義語：エクスポート（ダウンロード） ==========
@app.route("/admin/synonyms/export", methods=["GET", "POST"])
@login_required
def admin_synonyms_export():
    if session.get("role") != "admin":
        abort(403)
    syn = load_synonyms()
    buf = io.BytesIO(json.dumps(syn, ensure_ascii=False, indent=2).encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="synonyms.json", mimetype="application/json")


# ========== 類義語：AI自動生成（missing/all × append/overwrite） ==========
@app.route("/admin/synonyms/autogen", methods=["POST"])
@login_required
def admin_synonyms_autogen():
    if session.get("role") != "admin":
        abort(403)
    if not OPENAI_API_KEY:
        flash("OPENAI_API_KEY が未設定です")
        return redirect(url_for("admin_synonyms"))

    target = request.form.get("target", "missing")      # "missing" or "all"
    mode   = request.form.get("mode", "append")         # "append" or "overwrite"

    # 既存エントリからタグ集合を作る
    entries = load_entries()
    all_tags = set()
    for e in entries:
        for t in (e.get("tags") or []):
            t = (t or "").strip()
            if t:
                all_tags.add(t)

    cur = load_synonyms()
    if target == "all":
        target_tags = sorted(all_tags)
    else:
        target_tags = sorted([t for t in all_tags if t not in cur or not cur[t]])

    if not target_tags:
        flash("生成対象のタグがありません")
        return redirect(url_for("admin_synonyms"))

    # トークン対策：分割して順次生成（40件/回 くらい）
    CHUNK = 40
    merged = {}
    for i in range(0, len(target_tags), CHUNK):
        chunk = target_tags[i:i+CHUNK]
        prompt = (
            "次の観光関連タグごとに、日本語の類義語・言い換え・表記揺れを最大5個ずつ返してください。"
            "出力は必ず JSON で、キー=タグ、値=文字列配列の形にしてください。"
            "\nタグ: " + ", ".join(chunk)
        )
        try:
            content = openai_chat(
                OPENAI_MODEL_HIGH,
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("AI出力がJSONオブジェクトではありません")
        except Exception as e:
            app.logger.exception("autogen chunk failed: %s", e)
            flash(f"AI生成に失敗したチャンクがあります: {', '.join(chunk[:3])} …")
            continue

        for k, v in data.items():
            if isinstance(v, str):
                syns = _split_lines_commas(v)
            elif isinstance(v, (list, tuple)):
                syns = [str(x).strip() for x in v if str(x).strip()]
            else:
                syns = []
            merged[str(k).strip()] = syns

    # 反映
    updated = 0
    for tag, syns in merged.items():
        if mode == "overwrite":
            cur[tag] = syns
        else:
            cur.setdefault(tag, [])
            for s in syns:
                if s and s not in cur[tag]:
                    cur[tag].append(s)
        updated += 1

    save_synonyms(cur)
    flash(f"AI生成: {updated} タグ分を{'上書き' if mode=='overwrite' else '追記'}しました")
    return redirect(url_for("admin_synonyms"))


# （互換用：以前のエンドポイント名を使っていた場合のエイリアス）
@app.route("/admin/synonyms/auto", methods=["POST"])
@login_required
def admin_synonyms_auto():
    return admin_synonyms_autogen()


@app.route("/admin/manual")
@login_required
def admin_manual():
    if session.get("role") != "admin":
        abort(403)
    return render_template("admin_manual.html")

# ======== ▼▼▼ ここから追記：テキストファイル管理（/admin/data_files） ▼▼▼ ========
import unicodedata
from flask import send_from_directory

ALLOWED_TXT_EXTS = {".txt", ".md"}  # 必要なら増やせます

def _safe_txt_name(name: str) -> str:
    """
    ディレクトリトラバーサル防止 & 許容文字のみに制限。
    日本語はNFKCに正規化、先頭ドットでの隠しファイル化を禁止。
    拡張子が無ければ .txt を付与。
    """
    if not name:
        return ""
    name = os.path.basename(name)
    name = unicodedata.normalize("NFKC", name)
    # 許可文字のみ（日本語・英数・一部記号・全角/半角スペース）
    name = re.sub(r'[^0-9A-Za-zぁ-んァ-ン一-龥ー_\-\.\(\)\[\] 　]+', '', name)
    name = name.strip()
    # 先頭ドットは禁止（隠しファイル回避）
    name = name.lstrip(".")
    if not name:
        return ""
    root, ext = os.path.splitext(name)
    if not ext:
        name = name + ".txt"
        ext = ".txt"
    if ext.lower() not in ALLOWED_TXT_EXTS:
        return ""
    return (name[:120]).strip()

def _ensure_in_data_dir(path: str) -> bool:
    """DATA_DIR 配下に収まっているかの安全確認"""
    base = os.path.abspath(DATA_DIR)
    target = os.path.abspath(path)
    return target.startswith(base + os.sep) or target == base

def _read_text_any(path: str):
    encs = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "utf-16"]  # ← utf-16 を追加
    with open(path, "rb") as rf:
        raw = rf.read()
    for enc in encs:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # 最後の手段：置換ありでCP932
    return raw.decode("cp932", errors="replace"), "cp932(replace)"

def _write_text(path: str, text: str, encoding: str = "utf-8"):
    """
    テキストを書き出す。CRLF/CR を LF に正規化。
    encoding='cp932' を選ぶと日本語Windows互換保存（文字により失敗時はutf-8にフォールバック）
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path + ".tmp", "w", encoding=encoding, newline="\n") as wf:
            wf.write(text)
        os.replace(path + ".tmp", path)
        return encoding, None
    except UnicodeEncodeError as e:
        if encoding.lower() == "cp932":
            # CP932に載らない文字があった → UTF-8で保存して通知
            with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as wf:
                wf.write(text)
            os.replace(path + ".tmp", path)
            return "utf-8", f"一部の文字がCP932に無いため UTF-8 で保存しました（{e}）"
        else:
            raise

def _list_txt_files():
    files = []
    if not os.path.isdir(DATA_DIR):
        return files
    for fn in os.listdir(DATA_DIR):
        if fn.startswith("."):
            continue
        root, ext = os.path.splitext(fn)
        if ext.lower() not in ALLOWED_TXT_EXTS:
            continue
        fp = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(fp):
            continue
        st = os.stat(fp)
        files.append({
            "name": fn,
            "size": st.st_size,
            "mtime_ts": st.st_mtime,  # 数値
        })
    files.sort(key=lambda x: x["mtime_ts"], reverse=True)
    # 表示用の整形は最後に
    for f in files:
        f["mtime"] = datetime.datetime.fromtimestamp(f["mtime_ts"]).strftime("%Y-%m-%d %H:%M:%S")
        del f["mtime_ts"]
    return files

@app.route("/admin/data_files", methods=["GET"])
@login_required
def admin_data_files():
    if session.get("role") != "admin":
        abort(403)
    files = _list_txt_files()
    edit = (request.args.get("edit") or "").strip()
    content = ""
    used_enc = ""
    if edit:
        safe = _safe_txt_name(edit)
        if not safe:
            flash("不正なファイル名です")
            return redirect(url_for("admin_data_files"))
        path = os.path.join(DATA_DIR, safe)
        if not _ensure_in_data_dir(path) or not os.path.exists(path):
            flash("指定ファイルが見つかりません")
            return redirect(url_for("admin_data_files"))
        content, used_enc = _read_text_any(path)
    return render_template("admin_data_files.html", files=files, edit=edit, content=content, used_enc=used_enc)

@app.route("/admin/data_files/upload", methods=["POST"])
@login_required
def admin_data_files_upload():
    if session.get("role") != "admin":
        abort(403)
    ups = request.files.getlist("txt_files")
    count = 0
    for f in ups:
        if not f or not f.filename:
            continue
        safe = _safe_txt_name(f.filename)
        if not safe:
            flash(f"スキップ: 不正なファイル名 {f.filename}")
            continue
        dst = os.path.join(DATA_DIR, safe)
        if not _ensure_in_data_dir(dst):
            flash(f"スキップ: 保存先エラー {safe}")
            continue

        data = f.read()
        if not data:
            flash(f"スキップ: 空ファイル {safe}")
            continue

        # ★ここから追加：同名回避の連番付与
        base, ext = os.path.splitext(safe)
        candidate = safe
        i = 1
        while os.path.exists(os.path.join(DATA_DIR, candidate)):
            candidate = f"{base} ({i}){ext}"
            i += 1
        dst = os.path.join(DATA_DIR, candidate)
        # ★ここまで追加

        # 文字コードを自動判定
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "utf-16"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                pass
        if text is None:
            text = data.decode("cp932", errors="replace")

        _write_text(dst, text, encoding="utf-8")
        count += 1

    flash(f"アップロード完了：{count} 件")
    return redirect(url_for("admin_data_files"))

@app.route("/admin/data_files/new", methods=["POST"])
@login_required
def admin_data_files_new():
    if session.get("role") != "admin":
        abort(403)
    name = (request.form.get("new_name") or "").strip()
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です（拡張子は .txt / .md のみ）")
        return redirect(url_for("admin_data_files"))
    path = os.path.join(DATA_DIR, safe)
    if not _ensure_in_data_dir(path):
        flash("保存先エラー")
        return redirect(url_for("admin_data_files"))
    if os.path.exists(path):
        flash("同名ファイルが既にあります")
        return redirect(url_for("admin_data_files", edit=safe))
    _write_text(path, "", encoding="utf-8")
    flash("新規作成しました")
    return redirect(url_for("admin_data_files", edit=safe))

@app.route("/admin/data_files/save", methods=["POST"])
@login_required
def admin_data_files_save():
    if session.get("role") != "admin":
        abort(403)
    name = (request.form.get("edit_name") or "").strip()
    enc  = (request.form.get("save_encoding") or "utf-8").strip().lower()
    content = request.form.get("content") or ""
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files"))
    path = os.path.join(DATA_DIR, safe)
    if not _ensure_in_data_dir(path):
        flash("保存先エラー")
        return redirect(url_for("admin_data_files"))

    try:
        used, warn = _write_text(path, content, encoding=("cp932" if enc == "cp932" else "utf-8"))
        msg = f"保存しました（encoding: {used}）"
        if warn:
            msg += " / " + warn
        flash(msg)
    except Exception as e:
        app.logger.exception("save failed")
        flash("保存に失敗しました: " + str(e))
    return redirect(url_for("admin_data_files", edit=safe))

@app.route("/admin/data_files/delete", methods=["POST"])
@login_required
def admin_data_files_delete():
    if session.get("role") != "admin":
        abort(403)
    name = (request.form.get("del_name") or "").strip()
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files"))
    path = os.path.join(DATA_DIR, safe)
    if not _ensure_in_data_dir(path) or not os.path.exists(path):
        flash("ファイルが見つかりません")
        return redirect(url_for("admin_data_files"))
    try:
        os.remove(path)
        flash("削除しました")
    except Exception as e:
        app.logger.exception("delete failed")
        flash("削除に失敗しました: " + str(e))
    return redirect(url_for("admin_data_files"))

@app.route("/admin/data_files/download")
@login_required
def admin_data_files_download():
    if session.get("role") != "admin":
        abort(403)
    name = (request.args.get("name") or "").strip()
    safe = _safe_txt_name(name)
    if not safe:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files"))
    # send_from_directory はパス検証込みで安全
    return send_from_directory(DATA_DIR, safe, as_attachment=True)

@app.route("/admin/data_files/rename", methods=["POST"])
@login_required
def admin_data_files_rename():
    if session.get("role") != "admin":
        abort(403)
    old = (request.form.get("old_name") or "").strip()
    new = (request.form.get("new_name") or "").strip()
    old_s = _safe_txt_name(old)
    new_s = _safe_txt_name(new)
    if not old_s or not new_s:
        flash("ファイル名が不正です")
        return redirect(url_for("admin_data_files", edit=old_s))
    src = os.path.join(DATA_DIR, old_s)
    dst = os.path.join(DATA_DIR, new_s)
    if not (_ensure_in_data_dir(src) and _ensure_in_data_dir(dst)) or not os.path.exists(src):
        flash("リネーム対象が見つかりません")
        return redirect(url_for("admin_data_files"))
    if os.path.exists(dst):
        flash("指定の新ファイル名は既に存在します")
        return redirect(url_for("admin_data_files", edit=old_s))
    os.rename(src, dst)
    flash("名前を変更しました")
    return redirect(url_for("admin_data_files", edit=new_s))
# ======== ▲▲▲ 追記ここまで ▲▲▲


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

def get_transport_reply(question: str):
    """飛行機・船の運行状況を、天気と同様に即答＆多言語で返す。"""
    if not question:
        return None, False

    # 言語判定（多言語を有効にしていない場合は日本語で固定）
    lang = "ja" if not ENABLE_FOREIGN_LANG else detect_lang_simple(question)

    # キーワード
    ferry_kw_ja = ["フェリー", "船", "運航", "ジェットフォイル", "太古", "欠航", "ダイヤ"]
    ferry_kw_zh = ["渡輪", "船", "航線", "噴射船", "停航", "班次"]
    ferry_kw_en = ["ferry", "jetfoil", "ship", "sailing", "service", "cancel", "status", "schedule"]

    flight_kw_ja = ["飛行機", "空港", "航空便", "欠航", "到着", "出発", "フライト", "遅延"]
    flight_kw_zh = ["飛機", "機場", "航班", "延誤", "取消", "起飛", "到達"]
    flight_kw_en = ["flight", "airport", "delay", "cancel", "arrival", "departure", "status"]

    q = question

    ferry_hit = any(k in q for k in ferry_kw_ja) or any(k in q for k in ferry_kw_zh) or any(k in q.lower() for k in ferry_kw_en)
    flight_hit = any(k in q for k in flight_kw_ja) or any(k in q for k in flight_kw_zh) or any(k in q.lower() for k in flight_kw_en)

    # まず船（キーワードが被る「欠航」等は船優先）
    if ferry_hit:
        if lang == "zh-Hant":
            text = (
                "【長崎—五島 航線運行資訊】\n"
                "・野母商船「太古號」運行資訊\n"
                "http://www.norimono-info.com/frame_set.php?usri=&disp=group&type=ship\n"
                "・九州商船（渡輪／噴射船）運行資訊\n"
                "https://kyusho.co.jp/status\n"
                "・五島產業汽船（渡輪）運行資訊\n"
                "https://www.goto-sangyo.co.jp/\n"
                "其他航線請見各連結。"
            )
        elif lang == "en":
            text = (
                "[Nagasaki ⇄ Goto Ferry/Jetfoil Status]\n"
                "• Nomo Shipping “Ferry Taiko” status:\n"
                "http://www.norimono-info.com/frame_set.php?usri=&disp=group&type=ship\n"
                "• Kyushu Shosen (Ferry / Jetfoil) status:\n"
                "https://kyusho.co.jp/status\n"
                "• Goto Sangyo Kisen (Ferry) status:\n"
                "https://www.goto-sangyo.co.jp/\n"
                "For other routes, please check the links above."
            )
        else:
            # 既存の日本語定型をそのまま使用
            text = FERRY_INFO
        return text, True

    # 次に飛行機
    if flight_hit:
        if lang == "zh-Hant":
            text = "五島椿機場的最新航班資訊請見官方網站：\n▶ https://www.fukuekuko.jp/"
        elif lang == "en":
            text = "Latest flight status for Goto Tsubaki Airport (official site):\n▶ https://www.fukuekuko.jp/"
        else:
            text = "五島つばき空港の最新の運行状況は、公式Webサイトでご確認ください。\n▶ https://www.fukuekuko.jp/"
        return text, True

    return None, False


SMALLTALK_PATTERNS = {
    "greet": ["おはよう", "こんにちは", "こんばんは", "やあ", "はじめまして"],
    "howareyou": ["元気", "調子どう", "調子は？"],
    "thanks": ["ありがとう", "助かった", "サンキュー"],
    "capability": ["何ができる", "使い方", "ヘルプ", "help"]
}

def smalltalk_or_help_reply(q_ja: str):
    t = q_ja.strip()
    if any(k in t for k in SMALLTALK_PATTERNS["greet"]):
        return "こんにちは！ご旅行の予定や気分をざっくり教えてくれれば、ぴったりのスポットを探します。"
    if any(k in t for k in SMALLTALK_PATTERNS["howareyou"]):
        return "元気ですよ！今日は海・教会・展望・温泉・グルメなど、気分はどれですか？"
    if any(k in t for k in SMALLTALK_PATTERNS["thanks"]):
        return "どういたしまして！他にも聞きたいことがあればどうぞ。"
    if any(k in t for k in SMALLTALK_PATTERNS["capability"]):
        return (
            "できること：\n"
            "・スポット検索（例：『五島市のビーチ』『教会の見学』）\n"
            "・基本情報の回答（住所／地図／営業時間／駐車場など）\n"
            "・天気・航路リンク案内、未ヒット時の絞り込み提案\n"
            "気軽に話しかけてOK。会話から意図を読み取って探します。"
        )
    return None

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
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {'.txt', '.md'}:
                continue
            try:
                fpath = os.path.join(root, fname)
                # 文字コード自動判定で読む（アップロード時はUTF-8保存だが保険）
                text, _ = _read_text_any(fpath)
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

def format_entry_detail(e: dict) -> str:
    lines = []
    if e.get("title"):      lines.append(f"◼︎ {e['title']}")
    if e.get("areas"):      lines.append(f"エリア: {', '.join(e['areas'])}")
    if e.get("desc"):       lines.append(f"説明: {e['desc']}")
    if e.get("address"):    lines.append(f"住所: {e['address']}")
    if e.get("tel"):        lines.append(f"TEL: {e['tel']}")
    if e.get("open_hours"): lines.append(f"営業時間: {e['open_hours']}")
    if e.get("holiday"):    lines.append(f"定休日: {e['holiday']}")
    if e.get("parking"):    lines.append(f"駐車場: {e['parking']}（台数: {e.get('parking_num','')}）")
    if e.get("payment"):    lines.append("支払い: " + " / ".join(e['payment']))
    if e.get("extras"):
        for k, v in (e.get("extras") or {}).items():
            if v: lines.append(f"{k}: {v}")
    if e.get("map"):        lines.append(f"地図: {e['map']}")
    if e.get("links"):      lines.append("リンク: " + " / ".join(e['links']))
    return "\n".join(lines)


def smart_search_answer_with_hitflag(question):
    meta = {"model_primary": OPENAI_MODEL_PRIMARY, "fallback": None}

    weather_reply, weather_hit = get_weather_reply(question)
    if weather_hit:
        return weather_reply, True, meta
    
    # ★ここから追加：雑談/使い方
    st = smalltalk_or_help_reply(question)
    if st:
        meta["kind"] = "smalltalk"
        return st, True, meta


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
            if e.get("title"):      lines.append(f"◼︎ {e['title']}")
            if e.get("areas"):      lines.append(f"エリア: {', '.join(e['areas'])}")
            if e.get("desc"):       lines.append(f"説明: {e['desc']}")
            if e.get("address"):    lines.append(f"住所: {e['address']}")
            if e.get("tel"):        lines.append(f"TEL: {e['tel']}")
            if e.get("open_hours"): lines.append(f"営業時間: {e['open_hours']}")
            if e.get("holiday"):    lines.append(f"定休日: {e['holiday']}")
            if e.get("parking"):    lines.append(f"駐車場: {e['parking']}（台数: {e.get('parking_num','')}）")
            if e.get("payment"):    lines.append("支払い: " + " / ".join(e['payment']))
            if e.get("extras"):
                for k, v in (e.get("extras") or {}).items():
                    if v: lines.append(f"{k}: {v}")
            if e.get("map"):        lines.append(f"地図: {e['map']}")
            if e.get("links"):      lines.append("リンク: " + " / ".join(e['links']))
            return "\n".join(lines), True, meta
        
        else:
            # 複数ヒット時：まず要約を試す → 失敗/空なら「短い候補一覧＋深掘り質問」にフォールバック
            try:
                snippets = [
                    f"タイトル: {e.get('title','')}\n説明: {e.get('desc','')}\n住所: {e.get('address','')}\n"
                    for e in entries
                ]
                ai_ans = ai_summarize(snippets, question, model=OPENAI_MODEL_PRIMARY)
                # 生成が空/極端に短い場合は失敗扱いにする
                if not ai_ans or len(ai_ans.strip()) < 30:
                    raise RuntimeError("summarize_failed_or_too_short")
                return ai_ans, True, meta
            except Exception:
                # --- フォールバック：短い候補一覧（最大8件）＋意図深掘り ---
                max_list = 8
                short = entries[:max_list]

                lines = ["候補が複数見つかりました。気になるものはありますか？"]
                for i, e in enumerate(short, 1):
                    name = (e.get("title") or "（無題）").strip()
                    areas = ", ".join((e.get("areas") or []))
                    suffix = f"（{areas}）" if areas else ""
                    lines.append(f"{i}. {name}{suffix}")

                remaining = len(entries) - len(short)
                if remaining > 0:
                    lines.append(f"…ほか {remaining} 件")

                # 絞り込み候補＋深掘り質問（build_refine_suggestions は近くで定義済み）
                try:
                    suggest_text, suggest_meta = build_refine_suggestions(question)
                    meta["suggestions"] = suggest_meta
                    lines.append("")
                    lines.append(suggest_text)  # ※probe 質問も含めて返る実装にしていればここで出ます
                except Exception:
                    # build_refine_suggestions が未改修でも最低限の深掘りだけ添える
                    lines.append("")
                    lines.append("❓ もう少し教えてください")
                    lines.append("- 出発エリアは？（五島市／新上五島町／小値賀町／宇久町）")
                    lines.append("- どんな雰囲気？（海岸線／教会巡り／展望台／夕日）")
                    lines.append("- 所要時間は？（2〜3時間／半日／1日）")
                    lines.append("- お車はありますか？（あり／なし）")

                msg = "\n".join(lines)
                return msg, False, meta

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

    trans_reply, trans_hit = get_transport_reply(question)
    if trans_hit:
        save_qa_log(question, trans_reply, source="web", hit_db=True, extra={"kind": "transport"})
        return jsonify({"answer": trans_reply, "hit_db": True, "meta": {"kind": "transport"}})

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
def _reply_or_push(event, text: str):
    """replyが失敗（Invalid reply token等）したらpushに切替える"""
    from linebot.models import TextSendMessage
    from linebot.exceptions import LineBotApiError

    try:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
        return
    except LineBotApiError as e:
        app.logger.exception("LineBotApiError on reply: %s", e)
    except Exception as e:
        app.logger.exception("Unexpected error on reply: %s", e)

    # replyに失敗 → pushで再送
    try:
        target_id = getattr(event.source, "user_id", None) \
                 or getattr(event.source, "group_id", None) \
                 or getattr(event.source, "room_id", None)
        if target_id:
            line_bot_api.push_message(target_id, TextSendMessage(text=text))
        else:
            app.logger.error("No target id found to push message")
    except LineBotApiError as e:
        app.logger.exception("LineBotApiError on push: %s", e)
    except Exception as e:
        app.logger.exception("Unexpected error on push: %s", e)

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

# --- LINE 送信先IDの取得ヘルパー ---
def _target_id_from_event(event):
    return (
        getattr(event.source, "user_id", None)
        or getattr(event.source, "group_id", None)
        or getattr(event.source, "room_id", None)
    )


WAIT_CANDIDATES = [
    "探してみますね。",
    "候補を確認しています…",
    "少々お時間ください、最適な情報を集めています。"
]
def pick_wait_message(q: str) -> str:
    # キーワードで少しだけ言い回しを変える（なくてもOK）
    if any(w in q for w in ["フェリー", "船", "ジェットフォイル", "太古"]):
        return "航路の運行状況を確認しています…"
    if any(w in q for w in ["営業時間", "開店", "閉店", "定休"]):
        return "営業時間を確認しています…"
    # 乱数を使わず、内容に応じて安定した揺らぎ
    return WAIT_CANDIDATES[hash(q) % len(WAIT_CANDIDATES)]

# ===== 送信ユーティリティ（長文分割・複数送信） =====
from linebot.models import TextSendMessage

LINE_MAX_PER_REQUEST = 5         # LINEは1リクエスト最大5メッセージ
LINE_SAFE_CHARS = 1200           # 1通あたり安全な文字数目安（改行・句点で分割）

def _split_for_messaging(text: str, chunk_size: int = LINE_SAFE_CHARS) -> List[str]:
    """長文を自然な区切り（段落→句点）で2〜複数通に分割。最低1通は返す。"""
    if not text:
        return [""]
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    parts: List[str] = []
    rest = text
    while len(rest) > chunk_size:
        cut = rest.rfind("\n\n", 0, chunk_size)
        if cut < int(chunk_size * 0.5):
            cut = rest.rfind("。", 0, chunk_size)
        if cut < int(chunk_size * 0.5):
            cut = chunk_size
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts

def _reply_or_push_multi(event, texts: List[str]):
    """複数メッセージでreply→失敗時pushにフォールバック。"""
    msgs = [TextSendMessage(text=t) for t in texts if t]
    if not msgs:
        msgs = [TextSendMessage(text="（空メッセージ）")]

    try:
        if len(msgs) <= LINE_MAX_PER_REQUEST:
            line_bot_api.reply_message(event.reply_token, msgs)
        else:
            line_bot_api.reply_message(event.reply_token, msgs[:LINE_MAX_PER_REQUEST])
            target_id = _target_id_from_event(event)
            batch = msgs[LINE_MAX_PER_REQUEST:]
            # 以降はpushで残りを送る（5件ずつ）
            for i in range(0, len(batch), LINE_MAX_PER_REQUEST):
                line_bot_api.push_message(target_id, batch[i:i+LINE_MAX_PER_REQUEST])
        return
    except Exception as e:
        app.logger.exception("reply multi failed -> fallback to push: %s", e)

    try:
        target_id = _target_id_from_event(event)
        if not target_id:
            return
        # push（5件ごと）
        for i in range(0, len(msgs), LINE_MAX_PER_REQUEST):
            line_bot_api.push_message(target_id, msgs[i:i+LINE_MAX_PER_REQUEST])
    except Exception as e:
        app.logger.exception("push multi failed: %s", e)

def _push_multi_by_id(target_id: str, texts: List[str]):
    """push専用：複数メッセージを5件ごとに送る。"""
    msgs = [TextSendMessage(text=t) for t in texts if t]
    if not msgs:
        msgs = [TextSendMessage(text="（空メッセージ）")]
    for i in range(0, len(msgs), LINE_MAX_PER_REQUEST):
        line_bot_api.push_message(target_id, msgs[i:i+LINE_MAX_PER_REQUEST])

# --- 最終回答を計算して push する非同期処理 ---
def _compute_and_push_async(event, user_message: str):
    from linebot.models import TextSendMessage
    target_id = _target_id_from_event(event)
    if not target_id:
        return
    try:
        # 言語判定 → ロジックは既存関数をそのまま利用
        orig_lang = detect_lang_simple(user_message)
        lang = orig_lang if ENABLE_FOREIGN_LANG else "ja"
        q_for_logic = user_message if orig_lang == "ja" else translate_text(user_message, "ja")

        answer_ja, hit_db, meta = smart_search_answer_with_hitflag(q_for_logic)
        answer = answer_ja if lang == "ja" else translate_text(answer_ja, "zh-Hant" if lang == "zh-Hant" else "en")

        save_qa_log(user_message, answer, source="line", hit_db=hit_db, extra=meta)
        line_bot_api.push_message(target_id, TextSendMessage(text=answer))
    except Exception as e:
        app.logger.exception("compute/push failed: %s", e)
        try:
            line_bot_api.push_message(target_id, TextSendMessage(text="検索中にエラーが発生しました。もう一度お試しください。"))
        except Exception:
            pass


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text

    # 0) 天気は即返信（既存）
    weather_reply, weather_hit = get_weather_reply(user_message)
    if weather_hit:
        save_qa_log(user_message, weather_reply, source="line", hit_db=True, extra={"kind": "weather"})
        _reply_or_push(event, weather_reply)
        return

    # 0.5) ← ここを追加：飛行機／船も即返信（多言語）
    trans_reply, trans_hit = get_transport_reply(user_message)
    if trans_hit:
        save_qa_log(user_message, trans_reply, source="line", hit_db=True, extra={"kind": "transport"})
        _reply_or_push(event, trans_reply)
        return

    # 言語（日本語以外は重くなりがち＝非同期へ回す）
    orig_lang = detect_lang_simple(user_message)
    is_ja = (orig_lang == "ja") or (not ENABLE_FOREIGN_LANG)

    # 1) 雑談/使い方は即返信（同期）。※smalltalkは日本語ルールベース
    if is_ja:
        st = smalltalk_or_help_reply(user_message)
        if st:
            save_qa_log(user_message, st, source="line", hit_db=True, extra={"kind": "smalltalk"})
            _reply_or_push(event, st)
            return

        # 2) DB単一ヒットは即返信（同期）
        single = None
        try:
            hits = find_entry_info(user_message)
            if hits and len(hits) == 1:
                single = format_entry_detail(hits[0])
        except Exception:
            single = None

        if single:
            save_qa_log(user_message, single, source="line", hit_db=True, extra={"kind": "single_hit"})
            _reply_or_push(event, single)
            return

    # 3) ここからは重め：短い待機メッセージ → 完成文は push
    _reply_or_push(event, pick_wait_message(user_message))

    threading.Thread(
        target=_compute_and_push_async,
        args=(event, user_message),
        daemon=True
    ).start()


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
