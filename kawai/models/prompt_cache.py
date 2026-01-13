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
    """Represents a single cached API response with metadata.

    Each cache entry stores a ChatCompletion response along with timing
    and usage statistics. Entries are managed by the PromptCache class
    and subject to LRU eviction and TTL expiration.

    Attributes:
        response (ChatCompletion): The cached OpenAI ChatCompletion response
            object containing the model's output, including message content,
            tool calls, and usage statistics.
        timestamp (float): Unix timestamp (seconds since epoch) when this
            entry was created. Used for TTL-based expiration.
        hits (int): Number of times this cached entry has been retrieved.
            Incremented on each cache hit. Defaults to 0.

    !!! example
        ```python
        from openai.types.chat import ChatCompletion
        from time import time

        # Create a cache entry
        entry = CacheEntry(
            response=chat_completion_response,
            timestamp=time(),
            hits=0
        )

        # Access entry data
        print(entry.response.choices[0].message.content)
        print(f"Created at: {entry.timestamp}")
        print(f"Hit count: {entry.hits}")
        ```

    Note:
        - Entries are automatically created by PromptCache.set()
        - The hits counter is managed by PromptCache.get()
        - Timestamps are used for automatic expiration checks
    """

    response: ChatCompletion
    timestamp: float
    hits: int = 0


class PromptCache(BaseModel):
    """Native application-level prompt caching for OpenAI-compatible APIs.

    Implements an exact-match caching strategy with LRU eviction, TTL expiration,
    and persistent storage. Caches API responses based on deterministic hashing
    of messages, tools, and model ID to avoid redundant API calls and reduce
    costs and latency.

    The cache is provider-agnostic and works with any OpenAI-compatible API
    (OpenAI, OpenRouter, Anthropic via OpenAI SDK, etc.). It operates transparently
    at the application level without requiring provider-specific features.

    Attributes:
        max_size (int): Maximum number of cache entries to store. When exceeded,
            the least recently used (LRU) entry is evicted. Defaults to 100.
        time_to_live (int): Time-to-live in seconds for cache entries. Entries
            older than this are considered expired and automatically removed.
            Defaults to 3600 (1 hour).
        enabled (bool): Whether caching is active. When False, all cache
            operations are no-ops. Defaults to True.
        persist (bool): Whether to persist cache to disk. When True, cache is
            saved to disk after modifications and loaded on initialization.
            Defaults to True.
        cache_dir (os.PathLike): Directory path for persistent cache storage.
            Created automatically if it doesn't exist. Defaults to "./.kawaicache".
        cache (OrderedDict[str, CacheEntry]): Internal cache storage mapping
            SHA256 hash keys to CacheEntry objects. Maintains insertion order
            for LRU eviction.
        _hits (int): Internal counter for cache hits. Incremented when a cached
            response is successfully retrieved.
        _misses (int): Internal counter for cache misses. Incremented when a
            requested key is not found or has expired.

    !!! example "Basic Usage"
        ```python
        from kawai.models.openai import OpenAIModel
        from kawai.models.prompt_cache import PromptCache

        # Create model with default caching
        model = OpenAIModel(
            model_id="openai/gpt-4",
            enable_cache=True  # Auto-creates PromptCache with defaults
        )

        # Or create with custom cache configuration
        cache = PromptCache(
            max_size=200,
            time_to_live=7200,  # 2 hours
            cache_dir="./my_cache"
        )

        model = OpenAIModel(
            model_id="openai/gpt-4",
            enable_cache=True,
            cache=cache
        )

        # Use normally - caching is transparent
        response = model.predict_from_messages(messages)

        # Check cache statistics
        stats = model.cache.stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"Cache size: {stats['size']}/{stats['max_size']}")
        ```

    !!! example "Manual Cache Management"
        ```python
        cache = PromptCache()

        # Generate cache key
        key = cache._generate_key(messages, tools, model_id)

        # Check for cached response
        cached = cache.get(key)
        if cached:
            print("Cache hit!")
        else:
            # Make API call and cache result
            response = client.chat.completions.create(...)
            cache.set(key, response)

        # Manually evict expired entries
        evicted_count = cache.evict_expired_entries()

        # Clear entire cache
        cache.clear()
        ```

    !!! example "In-Memory Only Cache"
        ```python
        # Disable persistence for temporary caching
        cache = PromptCache(
            persist=False,
            max_size=50,
            time_to_live=1800  # 30 minutes
        )
        ```

    Note:
        - Cache keys are SHA256 hashes of (model_id, messages, tools)
        - Only exact matches are cached - no semantic similarity
        - Persistence uses pickle format for simplicity
        - Cache is automatically loaded on initialization if persist=True
        - All cache modifications trigger disk saves when persist=True
        - Thread-safety is not guaranteed - use separate instances per thread

    Cache Strategy:
        1. **Exact Match**: Only identical requests (same messages, tools, model)
           result in cache hits
        2. **LRU Eviction**: When max_size is reached, least recently used entry
           is removed
        3. **TTL Expiration**: Entries older than time_to_live are automatically
           removed on access
        4. **Persistent Storage**: Cache survives process restarts when persist=True

    Performance Characteristics:
        - **Get**: O(1) average case for hash lookup
        - **Set**: O(1) average case, O(n) worst case for eviction
        - **Evict Expired**: O(n) where n is cache size
        - **Disk I/O**: Synchronous pickle serialization on every modification
    """

    max_size: int = 100
    time_to_live: int = 3600
    enabled: bool = True
    persist: bool = True
    cache_dir: os.PathLike = "./.kawaicache"
    cache: OrderedDict[str, CacheEntry] = Field(default_factory=OrderedDict)
    _hits: int = 0
    _misses: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.persist:
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

        self.evict_expired_entries()

    def _save_to_disk(self) -> None:
        """Persist cache to disk using pickle serialization.

        Saves the entire cache state (entries, hits, misses) to a pickle file
        in the configured cache directory. Called automatically after cache
        modifications when persist=True.

        Note:
            - Silently fails if disk write fails (permissions, disk full, etc.)
            - Uses pickle format for simplicity (not human-readable)
            - Synchronous I/O - may impact performance for large caches
            - File: {cache_dir}/promptcache.pkl
        """
        if not self.persist:
            return
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
        max_tokens: int | None = None,
    ) -> str:
        """Generate a deterministic cache key from request parameters.

        Creates a SHA256 hash of the serialized request components to uniquely
        identify cache entries. The key is deterministic - identical inputs
        always produce the same key.

        Args:
            messages (list[dict[str, Any]]): OpenAI-format message list containing
                the conversation history.
            tools (list[dict[str, Any]] | None): Optional list of tool/function
                schemas in OpenAI format. None if no tools are used.
            model_id (str): Model identifier (e.g., "gpt-4", "claude-3-sonnet").
            max_tokens (int | None): Maximum tokens for the completion. Different
                values produce different cache keys. Defaults to None.

        Returns:
            str: 64-character hexadecimal SHA256 hash string.

        Note:
            - Messages and tools are JSON-serialized with sorted keys for consistency
            - Different models with same messages/tools produce different keys
            - Even minor differences (whitespace, order) result in different keys
            - Key format: sha256(f"{model_id}:{max_tokens}:{messages_json}:{tools_json}")

        !!! example
            ```python
            cache = PromptCache()
            messages = [{"role": "user", "content": "Hello"}]
            tools = [{"type": "function", "function": {...}}]

            key1 = cache._generate_key(messages, tools, "gpt-4")
            key2 = cache._generate_key(messages, tools, "gpt-4")  # Same key
            key3 = cache._generate_key(messages, tools, "gpt-3.5")  # Different key
            ```
        """
        messages_str = json.dumps(messages, sort_keys=True)
        tools_str = json.dumps(tools, sort_keys=True) if tools else ""
        max_tokens_str = str(max_tokens) if max_tokens is not None else ""
        key_content = f"{model_id}:{max_tokens_str}:{messages_str}:{tools_str}"
        return sha256(key_content.encode()).hexdigest()

    def evict_expired_entries(self) -> int:
        """Remove all expired entries from the cache.

        Iterates through all cache entries and removes those whose age
        exceeds the configured time_to_live.

        Returns:
            int: Number of entries that were evicted.

        Note:
            - Called automatically during cache initialization
            - Saves to disk after eviction if persist=True
        """
        if not self.cache:
            return 0

        current_time = time()
        expired_keys = [
            key
            for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.time_to_live
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self._save_to_disk()

        return len(expired_keys)

    def get(self, key: str) -> ChatCompletion | None:
        """Retrieve a cached response if valid.

        Looks up the cache key and returns the cached ChatCompletion if found
        and not expired. Updates LRU ordering and hit statistics on successful
        retrieval.

        Args:
            key (str): SHA256 hash key generated by _generate_key().

        Returns:
            ChatCompletion | None: The cached response if found and valid,
                None if key not found, expired, or caching is disabled.

        Side Effects:
            - Increments _hits counter on cache hit
            - Increments _misses counter on cache miss or expiration
            - Updates LRU ordering (moves entry to end)
            - Increments entry.hits counter
            - Removes expired entries
            - Saves to disk if persist=True

        Note:
            - Returns None immediately if enabled=False
            - Expired entries are removed during lookup
            - O(1) average case performance

        !!! example
            ```python
            cache = PromptCache()
            key = cache._generate_key(messages, tools, model_id)

            response = cache.get(key)
            if response:
                print("Cache hit!")
                print(response.choices[0].message.content)
            else:
                print("Cache miss - need to call API")
            ```
        """
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
            self._save_to_disk()
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1

        self._save_to_disk()

        return entry.response

    def set(self, key: str, response: ChatCompletion) -> None:
        """Store a response in cache with automatic LRU eviction.

        Adds a new cache entry or updates an existing one. If the cache is at
        capacity, evicts the least recently used entry before adding the new one.

        Args:
            key (str): SHA256 hash key generated by _generate_key().
            response (ChatCompletion): The OpenAI ChatCompletion response to cache.

        Side Effects:
            - Creates new CacheEntry with current timestamp
            - Evicts LRU entry if cache is full
            - Updates LRU ordering (moves entry to end)
            - Saves to disk if persist=True

        Note:
            - No-op if enabled=False
            - Updating an existing key refreshes its timestamp
            - O(1) average case, O(n) worst case for eviction

        !!! example
            ```python
            cache = PromptCache(max_size=100)

            # Make API call
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )

            # Cache the response
            key = cache._generate_key(messages, None, "gpt-4")
            cache.set(key, response)

            # Future identical requests will hit cache
            cached = cache.get(key)  # Returns response instantly
            ```
        """
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
        """Clear all cache entries and reset statistics.

        Removes all cached responses and resets hit/miss counters to zero.
        Persists the empty cache to disk if persist=True.

        Side Effects:
            - Removes all cache entries
            - Resets _hits to 0
            - Resets _misses to 0
            - Saves empty cache to disk if persist=True

        !!! example
            ```python
            cache = PromptCache()

            # ... use cache ...

            # Clear everything
            cache.clear()
            assert cache.stats()["size"] == 0
            assert cache.stats()["hits"] == 0
            ```
        """
        self.cache.clear()
        self._hits = 0
        self._misses = 0

        self._save_to_disk()

    def stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns a dictionary containing cache usage metrics including hit/miss
        counts, current size, and calculated hit rate.

        Returns:
            dict[str, Any]: Dictionary with the following keys:
                - hits (int): Total number of cache hits
                - misses (int): Total number of cache misses
                - size (int): Current number of entries in cache
                - max_size (int): Maximum cache capacity
                - hit_rate (float): Cache hit rate (hits / total requests), 0.0-1.0
                - enabled (bool): Whether caching is currently enabled
                - persist (bool): Whether cache persistence is enabled

        Note:
            - Hit rate is 0.0 if no requests have been made
            - Statistics persist across cache reloads when persist=True
            - Useful for monitoring cache effectiveness

        !!! example
            ```python
            cache = PromptCache()

            # ... use cache ...

            stats = cache.stats()
            print(f"Cache hit rate: {stats['hit_rate']:.2%}")
            print(f"Cache utilization: {stats['size']}/{stats['max_size']}")
            print(f"Total requests: {stats['hits'] + stats['misses']}")

            # Example output:
            # Cache hit rate: 67.50%
            # Cache utilization: 45/100
            # Total requests: 80
            ```
        """
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
