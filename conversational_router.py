# -*- coding: utf-8 -*-
import os, json, re, datetime
from typing import Dict, Any, List, Optional

try:
    from openai import OpenAI
    _OPENAI = OpenAI()
    _OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    _OPENAI = None
    _OPENAI_MODEL = None

# --- 使う側：TourismAI(entries_path).handle(text, user_id) を呼ぶだけ ---

CAPABILITIES_JA = (
    "五島の観光・グルメ・アクセス・イベント・営業時間・定休日・モデルコース・"
    "宿やレンタカーの案内（自社分）・雨天/家族連れ向け・台湾人向け情報などに答えられます。"
)

def _now_jst():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=9)

def _safe_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _call_llm_json(user_text: str, user_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    LLMで意図とスロットをJSON抽出。OpenAIが使えない場合はNone。
    """
    if not _OPENAI:
        return None

    system = (
        "You are a tourism AI for Japan's Goto Islands. "
        "Extract the user's INTENT and SLOTS as compact JSON. "
        "Reply ONLY with JSON matching the schema."
    )
    schema_hint = {
        "reply_language": "ja|zh-TW|en",
        "intent": "smalltalk.greeting|smalltalk.status|smalltalk.capabilities|info.search|info.hours|info.access|info.events|booking.guide|fallback",
        "slots": {
            "area": "e.g., 福江, 奈留, 上五島, 若松, 新上五島, 宇久, 小値賀",
            "topic": "spot|food|cafe|beach|onsen|transport|ferry|festival|shopping|modelcourse",
            "tags": ["string"],
            "date": "YYYY-MM-DD or null",
            "time": "HH:MM or null",
            "price_range": "cheap|mid|high|null",
            "people": "solo|couple|family|group|null",
            "weather": "sunny|rain|windy|null",
            "language": "ja|zh-TW|en|null"
        },
        "needs_db_search": True
    }
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":f"USER_TEXT: {user_text}\nSTATE_HINTS: {json.dumps(user_state, ensure_ascii=False)}\nSCHEMA_HINT: {json.dumps(schema_hint, ensure_ascii=False)}"}
    ]
    resp = _OPENAI.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        response_format={"type":"json_object"}
    )
    data = resp.choices[0].message.content
    return _safe_json(data)

def _heuristic_route(user_text: str) -> Dict[str, Any]:
    """
    LLMなしの簡易ルーター（最低限動く保険）。日本語前提。
    """
    t = user_text.strip()
    hiragana = re.search(r"[ぁ-ん]", t) is not None
    katakana = re.search(r"[ァ-ヶ]", t) is not None
    is_jp = hiragana or katakana

    if re.search(r"(おはよう|こんにちは|こんばんは)", t):
        intent = "smalltalk.greeting"
    elif re.search(r"(元気|調子|どう\?|どうですか)", t):
        intent = "smalltalk.status"
    elif re.search(r"(何に答え|何にこたえ|何ができる|できること)", t):
        intent = "smalltalk.capabilities"
    elif re.search(r"(営業時間|開店|閉店|定休|何時から|何時まで)", t):
        intent = "info.hours"
    elif re.search(r"(行き方|アクセス|どう行く|乗り方|フェリー|船|高速船|バス)", t):
        intent = "info.access"
    elif re.search(r"(イベント|祭|フェス|花火)", t):
        intent = "info.events"
    elif re.search(r"(食べ|ランチ|夕食|居酒屋|カフェ|スイーツ|海鮮|丼|ラーメン)", t):
        intent = "info.search"
    elif re.search(r"(見る|観光|スポット|おすすめ|モデルコース)", t):
        intent = "info.search"
    else:
        intent = "fallback"

    # 超ざっくりタグ抽出
    tags = []
    for kw in ["海鮮","丼","カフェ","ラーメン","教会","ビーチ","温泉","子連れ","雨でも","絶景","穴場"]:
        if kw in t: tags.append(kw)

    return {
        "reply_language": "ja" if is_jp else "en",
        "intent": intent,
        "slots": {
            "area": None,
            "topic": "food" if any(k in t for k in ["食","ランチ","夕食","居酒屋","海鮮","丼","ラーメン","カフェ","スイーツ"]) else "spot",
            "tags": tags,
            "date": None, "time": None,
            "price_range": None, "people": None, "weather": None, "language": "ja" if is_jp else "en"
        },
        "needs_db_search": intent.startswith("info.")
    }

def _language_pick(j: Optional[Dict[str,Any]], fallback="ja"):
    if j and j.get("reply_language") in ("ja","zh-TW","en"):
        return j["reply_language"]
    return fallback

def _smalltalk(intent: str, lang: str) -> str:
    if intent == "smalltalk.greeting":
        return {"ja":"おはようございます！今日はどのエリアに行く予定ですか？",
                "zh-TW":"早安！今天想去哪一區？",
                "en":"Good morning! Which area are you thinking to visit today?"}[lang]
    if intent == "smalltalk.status":
        return {"ja":"元気です！そちらはどうですか？今日は何を探しますか？",
                "zh-TW":"我很好，你呢？今天想找什麼資訊？",
                "en":"Doing great—how about you? What would you like to find today?"}[lang]
    if intent == "smalltalk.capabilities":
        return {"ja": f"ありがとうございます。{CAPABILITIES_JA}\nまず、エリアや気分（海鮮/カフェ/景色など）を教えてください。",
                "zh-TW":"感謝提問！五島的景點、美食、交通、活動、營業時間等我都能回答。先告訴我想去的區域或想吃的類型吧！",
                "en":"Thanks for asking! I can help with sights, food, access, events, hours, and sample routes. Tell me your area or craving."}[lang]
    return {"ja":"なるほど！少し詳しく教えてください。",
            "zh-TW":"了解！再多說一點細節吧。",
            "en":"Got it! Tell me a bit more."}[lang]

def _normalize(s):
    if not s: return ""
    return re.sub(r"\s+", " ", str(s)).strip()

class TourismAI:
    def __init__(self, entries_path: str):
        with open(entries_path, "r", encoding="utf-8") as f:
            self.entries: List[Dict[str,Any]] = json.load(f)
        self.user_states: Dict[str, Dict[str,Any]] = {}

    # --- 公開API ---
    def handle(self, user_text: str, user_id: str) -> str:
        state = self.user_states.get(user_id, {})
        parsed = _call_llm_json(user_text, state) or _heuristic_route(user_text)
        lang = _language_pick(parsed, "ja")
        intent = parsed["intent"]

        # 会話の種：小さな返し
        if intent.startswith("smalltalk"):
            reply = _smalltalk(intent, lang)
            self.user_states[user_id] = state
            return reply

        if intent == "fallback":
            return {"ja":"その内容は資料にないかも。エリア（例：福江/奈留/上五島）や、食・観光などのジャンルを教えてください。",
                    "zh-TW":"資料裡可能沒有。區域（例如：福江/奈留/上五島）或類型（吃、景點）告訴我吧！",
                    "en":"I might not have that. Share an area (Fukue/Naru/Kamigoto) or a type (food/sight)."}[lang]

        # ここから情報系
        slots = parsed.get("slots", {}) or {}
        results = self.search(self.entries, slots)

        if intent in ("info.search","info.hours","info.access","info.events"):
            return self.render_answer(lang, intent, slots, results)

        if intent == "booking.guide":
            return {"ja":"ガイドの空き状況と料金をご案内できます。ご希望の日付と人数を教えてください。",
                    "zh-TW":"可以提供嚮導服務的檔期與費用。請告訴我日期與人數。",
                    "en":"I can check guide availability and pricing. What date and how many people?"}[lang]

        return {"ja":"少し整理しますね。エリアや目的をもう少し教えてください。",
                "zh-TW":"再釐清一下，請說說區域或目的。",
                "en":"Let’s narrow it down—area or purpose?"}[lang]

    # --- 検索 ---
    def search(self, entries: List[Dict[str,Any]], slots: Dict[str,Any]) -> List[Dict[str,Any]]:
        area = _normalize(slots.get("area")).lower()
        tags = [t for t in (slots.get("tags") or []) if t]
        topic = slots.get("topic")
        date = slots.get("date")
        time = slots.get("time")
        pr = slots.get("price_range")

        def ok(entry: Dict[str,Any]) -> bool:
            if area and area not in _normalize(entry.get("area","")).lower():
                return False
            if topic and topic not in (entry.get("topic") or topic):
                # 記録がtopicなしなら通し、あれば一致のみ
                if entry.get("topic"):
                    return False
            if pr and pr != (entry.get("price_range") or pr):
                if entry.get("price_range"):
                    return False
            if tags:
                etags = set(entry.get("tags", []))
                if not set(tags) & etags:
                    return False
            # 営業時間フィルタ（ざっくり）
            if date or time:
                # entry["hours"] 例: {"mon":"11:00-15:00,17:00-21:00","closed":["tue"]}
                hours = entry.get("hours", {})
                if isinstance(hours, dict) and date:
                    dt = datetime.datetime.fromisoformat(date)
                    w = ["mon","tue","wed","thu","fri","sat","sun"][dt.weekday()]
                    if hours.get("closed") and w in hours["closed"]:
                        return False
            return True

        filtered = [e for e in entries if ok(e)]
        # シンプルなスコア：タグ一致数＋新しさ優先
        def score(e):
            tag_score = len(set(tags) & set(e.get("tags", [])))
            updated = e.get("updated_at") or "2000-01-01"
            return (tag_score, updated)
        filtered.sort(key=score, reverse=True)
        return filtered[:5]

    # --- 返信生成 ---
    def render_answer(self, lang: str, intent: str, slots: Dict[str,Any], results: List[Dict[str,Any]]) -> str:
        if not results:
            follow = {"ja":"エリア名（例：福江/奈留/上五島）や、タグ（海鮮/カフェ/教会/子連れ）を教えてください。",
                      "zh-TW":"請告訴我區域（如福江/奈留/上五島）或標籤（海鮮/咖啡/教會/親子）。",
                      "en":"Share an area (Fukue/Naru/Kamigoto) or tags (seafood/cafe/church/family)."}[lang]
            head = {"ja":"候補が見つかりませんでした。",
                    "zh-TW":"暫時找不到合適的選項。",
                    "en":"I didn’t find good matches."}[lang]
            return f"{head}\n{follow}"

        lines = []
        lead = {
            "info.search": {"ja":"こちらはいかがでしょう：","zh-TW":"可以參考以下：","en":"Here are a few options:"},
            "info.hours": {"ja":"営業時間の参考です：","zh-TW":"營業時間參考：","en":"Hours info:"},
            "info.access":{"ja":"アクセス情報：","zh-TW":"交通資訊：","en":"Access info:"},
            "info.events":{"ja":"イベント候補：","zh-TW":"活動候選：","en":"Event ideas:"}
        }[intent][lang]
        lines.append(lead)

        for e in results[:3]:
            title = e.get("title","（名称不明）")
            area = e.get("area","")
            one_line = e.get("one_line") or e.get("desc","")[:80]
            tags = " / ".join(e.get("tags", [])[:4])
            hours = e.get("hours_human","")
            link = e.get("link","")
            item = f"・{title}（{area}）— {one_line}"
            if tags: item += f"｜{tags}"
            if hours and intent in ("info.hours","info.search"):
                item += f"｜{hours}"
            if link: item += f"\n  {link}"
            lines.append(item)

        follow = {
            "ja":"もう少し絞りますか？エリア変更や、予算・人数・雨でもOK等の条件もどうぞ。",
            "zh-TW":"要再縮小條件嗎？也可指定預算、人數、雨天OK等。",
            "en":"Want to narrow it further? Share area, budget, people, or rainy-day friendly."}[lang]
        lines.append(follow)
        return "\n".join(lines)
