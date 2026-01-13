import json
import os
import pickle
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from time import time
from typing import Any

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    response: ChatCompletion
    timestamp: float
    hits: int = 0


class PromptCache(BaseModel):
    max_size: int = 100
    time_to_live: int = 3600
    enabled: bool = True
    persist: bool = True
    cache_dir: os.PathLike = "./.kawaicache"
    cache: OrderedDict[str, CacheEntry] = Field(default_factory=OrderedDict)
    _hits: int = 0
    _misses: int = 0

    def model_post_init(self, __context: Any) -> None:
        # Ensure existence of cache dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load cache from cache dir
        cache_file = os.path.join(self.cache_dir, "promptcache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.cache = data.get("cache", OrderedDict())
                    self._hits = data.get("hits", 0)
                    self._misses = data.get("misses", 0)
            except Exception:
                self.cache = OrderedDict()

    def _save_to_disk(self) -> None:
        cache_file = os.path.join(self.cache_dir, "promptcache.pkl")
        try:
            with open(cache_file, "wb") as f:
                data = {
                    "cache": self.cache,
                    "hits": self._hits,
                    "misses": self._misses,
                }
                pickle.dump(data, f)
        except Exception:
            pass

    def _generate_key(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model_id: str,
    ) -> str:
        messages_str = json.dumps(messages, sort_keys=True)
        tools_str = json.dumps(tools, sort_keys=True) if tools else ""
        key_content = f"{model_id}:{messages_str}:{tools_str}"
        return sha256(key_content.encode()).hexdigest()

    def evict_expired_entries(self) -> int:
        current_time = time()
        expired_keys = [
            key
            for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.time_to_live
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys and self.persist:
            self._save_to_disk()
        return len(expired_keys)

    def get(self, key: str) -> ChatCompletion | None:
        if not self.enabled:
            return None

        if key not in self.cache:
            self._misses += 1
            return None

        entry = self.cache[key]

        # Check if expired
        if time() - entry.timestamp > self.time_to_live:
            del self.cache[key]
            self._misses += 1
            if self.persist:
                self._save_to_disk()
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1

        self._save_to_disk()

        return entry.response

    def set(self, key: str, response: ChatCompletion) -> None:
        """Store response in cache with LRU eviction."""
        if not self.enabled:
            return
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)
        # Store new entry
        self.cache[key] = CacheEntry(response=response, timestamp=time())
        # Move to end (most recent)
        self.cache.move_to_end(key)
        self._save_to_disk()

    def clear(self) -> None:
        self.cache.clear()
        self._hits = 0
        self._misses = 0

        self._save_to_disk()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "enabled": self.enabled,
            "persist": self.persist,
        }
