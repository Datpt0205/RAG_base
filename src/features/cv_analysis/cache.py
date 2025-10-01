import hashlib, json, os
from typing import Any, Optional

CACHE_DIR = "/app/cache/cv"
os.makedirs(CACHE_DIR, exist_ok=True)

def cv_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:24]

def load_cache(h: str) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"{h}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cache(h: str, data: dict) -> None:
    path = os.path.join(CACHE_DIR, f"{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
