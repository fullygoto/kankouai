import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Callable, Iterable, Optional

try:  # Optional redis dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis is optional
    redis = None

logger = logging.getLogger(__name__)


def _sha1(data: str) -> str:
    return sha1(data.encode("utf-8")).hexdigest()


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    try:
        import unicodedata

        normalized = unicodedata.normalize("NFKC", str(text))
    except Exception:  # pragma: no cover - extremely rare
        normalized = str(text)
    normalized = normalized.strip()
    if not normalized:
        return ""
    return " ".join(normalized.split()).lower()


def make_key(*parts: Iterable[str | int | float | None]) -> str:
    flat: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple)):
            flat.append(make_key(*part))
            continue
        flat.append(str(part))
    if not flat:
        return ""
    return _sha1("||".join(flat))


def make_event_key(user_id: str | None, message_id: str | None, normalized_text: str | None) -> str:
    return make_key("event", user_id or "", message_id or "", normalized_text or "")


def make_text_key(user_id: str | None, normalized_text: str | None) -> str:
    return make_key("text", user_id or "", normalized_text or "")


def make_push_key(user_id: str | None, payload: str | None) -> str:
    return make_key("push", user_id or "", payload or "")


class _SqliteBackend:
    def __init__(self, path: Path, namespace: str, time_func: Callable[[], float]):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._namespace = namespace
        self._time_func = time_func
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS antiflood (key TEXT PRIMARY KEY, expire_at INTEGER NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expire ON antiflood(expire_at)")
            conn.commit()

    def acquire(self, key: str, ttl: int, now: int) -> bool:
        if not key:
            return True
        expire_at = now + ttl
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM antiflood WHERE expire_at <= ?", (now,))
                try:
                    conn.execute(
                        "INSERT INTO antiflood(key, expire_at) VALUES (?, ?)", (key, expire_at)
                    )
                    conn.commit()
                    return True
                except sqlite3.IntegrityError:
                    return False

    def contains(self, key: str, now: int) -> bool:
        if not key:
            return False
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM antiflood WHERE expire_at <= ?", (now,))
                cur = conn.execute("SELECT 1 FROM antiflood WHERE key = ? LIMIT 1", (key,))
                return cur.fetchone() is not None

    def clear(self) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM antiflood")
                conn.commit()


class _RedisBackend:
    def __init__(self, url: str, namespace: str):
        if not redis:
            raise RuntimeError("redis package is not available")
        self._client = redis.Redis.from_url(url)
        self._prefix = f"{namespace}:"

    def _key(self, key: str) -> str:
        return key

    def acquire(self, key: str, ttl: int, now: int) -> bool:  # noqa: D401 - same API
        if not key:
            return True
        return bool(self._client.set(name=key, value="1", ex=ttl, nx=True))

    def contains(self, key: str, now: int) -> bool:
        if not key:
            return False
        return bool(self._client.exists(key))

    def clear(self) -> None:
        pattern = f"{self._prefix}*"
        for name in self._client.scan_iter(match=pattern):
            self._client.delete(name)


@dataclass
class AntiFlood:
    backend: object
    namespace: str
    time_func: Callable[[], float]

    @classmethod
    def from_env(
        cls,
        *,
        base_dir: str | os.PathLike[str] | None = None,
        redis_url: str | None = None,
        namespace: str = "antiflood",
        time_func: Optional[Callable[[], float]] = None,
    ) -> "AntiFlood":
        base_dir = Path(base_dir or os.environ.get("DATA_BASE_DIR", "."))
        redis_url = redis_url or os.environ.get("ANTIFLOOD_REDIS_URL") or os.environ.get("REDIS_URL")
        if time_func is None:
            time_func = time.time

        if redis_url:
            try:
                backend = _RedisBackend(redis_url, namespace)
                logger.info("AntiFlood using Redis backend: %s", redis_url)
                return cls(backend=backend, namespace=namespace, time_func=time_func)
            except Exception as exc:  # pragma: no cover - depends on runtime redis availability
                logger.warning("Redis backend unavailable (%s), falling back to sqlite", exc)

        sqlite_path = base_dir / "system" / "antiflood.sqlite"
        backend = _SqliteBackend(sqlite_path, namespace, time_func)
        logger.info("AntiFlood using sqlite backend at %s", sqlite_path)
        return cls(backend=backend, namespace=namespace, time_func=time_func)

    def _ns(self, key: str) -> str:
        return f"{self.namespace}:{key}" if key else key

    def acquire(self, key: str, ttl_sec: int) -> bool:
        if not key:
            return True
        ttl = int(ttl_sec)
        if ttl <= 0:
            return True
        try:
            now = int(self.time_func())
            return bool(getattr(self.backend, "acquire")(self._ns(key), ttl, now))
        except Exception as exc:
            logger.warning("AntiFlood.acquire failed for %s: %s", key, exc)
            return True

    def contains(self, key: str) -> bool:
        if not key:
            return False
        try:
            now = int(self.time_func())
            return bool(getattr(self.backend, "contains")(self._ns(key), now))
        except Exception as exc:
            logger.warning("AntiFlood.contains failed for %s: %s", key, exc)
            return False

    def clear(self) -> None:
        try:
            getattr(self.backend, "clear")()
        except Exception as exc:
            logger.warning("AntiFlood.clear failed: %s", exc)


__all__ = [
    "AntiFlood",
    "make_event_key",
    "make_key",
    "make_push_key",
    "make_text_key",
    "normalize_text",
]
