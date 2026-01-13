import json
import os
import pickle
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from time import time
from typing import Any

from pydantic import BaseModel, Field


class ToolCacheEntry(BaseModel):
    """Represents a cached tool execution result with metadata.

    Attributes:
        result (Any): The tool execution result (can be dict, str, int, etc.)
        timestamp (float): Unix timestamp when this entry was created
        hits (int): Number of times this cached entry has been retrieved
    """

    result: Any
    timestamp: float
    hits: int = 0


class ToolCache(BaseModel):
    """Native application-level caching for tool execution results.

    Implements an exact-match caching strategy with LRU eviction, TTL expiration,
    and persistent storage. Caches tool execution results based on deterministic
    hashing of tool name and arguments to avoid redundant tool executions and
    improve agent performance.

    This cache operates independently from the LLM prompt cache and is specifically
    designed for caching deterministic tool outputs. It's particularly useful for
    expensive operations like API calls, web scraping, or database queries.

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
            Created automatically if it doesn't exist. Defaults to "./.kawai_cache".
        cache (OrderedDict[str, ToolCacheEntry]): Internal cache storage mapping
            SHA256 hash keys to ToolCacheEntry objects. Maintains insertion order
            for LRU eviction.
        _hits (int): Internal counter for cache hits. Incremented when a cached
            result is successfully retrieved.
        _misses (int): Internal counter for cache misses. Incremented when a
            requested key is not found or has expired.

    !!! example "Basic Usage with Agent"
        ```python
        from kawai import KawaiReactAgent, OpenAIModel, WebSearchTool

        # Create model with tool caching enabled
        model = OpenAIModel(
            model_id="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key_env_var="OPENROUTER_API_KEY",
            enable_tool_cache=True  # Auto-creates ToolCache with defaults
        )

        agent = KawaiReactAgent(
            model=model,
            tools=[WebSearchTool()],
            max_steps=10
        )

        # First run - tool executes normally
        result1 = agent.run("Search for Python tutorials")

        # Second run with same query - uses cached tool result
        result2 = agent.run("Search for Python tutorials")

        # Check cache statistics
        stats = model.tool_cache.stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"Cache size: {stats['size']}/{stats['max_size']}")
        ```

    !!! example "Custom Cache Configuration"
        ```python
        from kawai.models.tool_cache import ToolCache
        from kawai import OpenAIModel

        # Create cache with custom settings
        tool_cache = ToolCache(
            max_size=200,
            time_to_live=1800,  # 30 minutes for time-sensitive data
            cache_dir="./my_tool_cache"
        )

        model = OpenAIModel(
            model_id="openai/gpt-4",
            enable_tool_cache=True,
            tool_cache=tool_cache
        )

        # Check cache stats
        print(tool_cache.stats())
        ```

    !!! example "In-Memory Only Cache"
        ```python
        # Disable persistence for temporary caching
        cache = ToolCache(
            persist=False,
            max_size=50,
            time_to_live=900  # 15 minutes
        )
        ```

    Note:
        - Cache keys are SHA256 hashes of (tool_name, tool_arguments)
        - Only exact matches are cached - no semantic similarity
        - Persistence uses pickle format for simplicity
        - Cache is automatically loaded on initialization if persist=True
        - All cache modifications trigger disk saves when persist=True
        - Thread-safety is not guaranteed - use separate instances per thread

    Cache Strategy:
        1. **Exact Match**: Only identical tool calls (same name, same arguments)
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

    Warning:
        - Non-deterministic tools (e.g., current_time, random_number) should
          not be cached or should use very short TTLs
        - Tools with side effects (e.g., send_email, create_record) should
          typically not be cached
        - Consider setting cacheable=False on tool definitions for such cases
    """

    max_size: int = 100
    time_to_live: int = 3600
    enabled: bool = True
    persist: bool = True
    cache_dir: os.PathLike = "./.kawai_cache"
    cache: OrderedDict[str, ToolCacheEntry] = Field(default_factory=OrderedDict)
    _hits: int = 0
    _misses: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.persist:
            # Ensure existence of cache dir
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

            # Load cache from cache dir
            cache_file = os.path.join(self.cache_dir, "toolcache.pkl")
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
        """Persist tool cache to disk using pickle serialization.

        Saves the entire cache state (entries, hits, misses) to a pickle file
        in the configured cache directory. Called automatically after cache
        modifications when persist=True.

        Note:
            - Silently fails if disk write fails (permissions, disk full, etc.)
            - Uses pickle format for simplicity (not human-readable)
            - Synchronous I/O - may impact performance for large caches
            - File: {cache_dir}/toolcache.pkl
        """
        if not self.persist:
            return
        cache_file = os.path.join(self.cache_dir, "toolcache.pkl")
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

    def _generate_key(self, tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Generate deterministic cache key from tool call parameters.

        Creates a SHA256 hash of the serialized tool name and arguments to uniquely
        identify cache entries. The key is deterministic - identical inputs always
        produce the same key.

        Args:
            tool_name (str): Name of the tool being called (e.g., "web_search",
                "calculator", "database_query").
            tool_arguments (dict[str, Any]): Arguments passed to the tool as a
                dictionary. Must be JSON-serializable.

        Returns:
            str: 64-character hexadecimal SHA256 hash string.

        Note:
            - Arguments are JSON-serialized with sorted keys for consistency
            - Different tools with same arguments produce different keys
            - Even minor differences (whitespace, case) result in different keys
            - Key format: sha256(f"{tool_name}:{sorted_json_arguments}")

        !!! example
            ```python
            cache = ToolCache()

            # Same inputs produce same keys
            key1 = cache._generate_key("web_search", {"query": "Python"})
            key2 = cache._generate_key("web_search", {"query": "Python"})
            assert key1 == key2

            # Different inputs produce different keys
            key3 = cache._generate_key("web_search", {"query": "Java"})
            assert key1 != key3

            # Different tools produce different keys
            key4 = cache._generate_key("calculator", {"query": "Python"})
            assert key1 != key4
            ```
        """
        args_str = json.dumps(tool_arguments, sort_keys=True)
        key_content = f"{tool_name}:{args_str}"
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
            - O(n) time complexity where n is the number of cache entries
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

    def get(self, key: str) -> Any | None:
        """Retrieve a cached tool result if valid.

        Looks up the cache key and returns the cached tool execution result if
        found and not expired. Updates LRU ordering and hit statistics on
        successful retrieval.

        Args:
            key (str): SHA256 hash key generated by _generate_key().

        Returns:
            Any | None: The cached tool result if found and valid,
                None if key not found, expired, or caching is disabled.
                Result type depends on what the tool returned (dict, str, int, etc.).

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
            cache = ToolCache()

            # Generate key for a tool call
            key = cache._generate_key("web_search", {"query": "AI news"})

            # Try to get cached result
            result = cache.get(key)
            if result:
                print("Cache hit!")
                print(result)
            else:
                print("Cache miss - need to execute tool")
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
        return entry.result

    def set(self, key: str, result: Any) -> None:
        """Store a tool result in cache with automatic LRU eviction.

        Adds a new cache entry or updates an existing one. If the cache is at
        capacity, evicts the least recently used entry before adding the new one.

        Args:
            key (str): SHA256 hash key generated by _generate_key().
            result (Any): The tool execution result to cache. Can be any type
                (dict, str, int, list, etc.) as long as it's pickle-serializable.

        Side Effects:
            - Creates new ToolCacheEntry with current timestamp
            - Evicts LRU entry if cache is full
            - Updates LRU ordering (moves entry to end)
            - Saves to disk if persist=True

        Note:
            - No-op if enabled=False
            - Updating an existing key refreshes its timestamp
            - O(1) average case, O(n) worst case for eviction

        !!! example
            ```python
            cache = ToolCache(max_size=100)

            # Execute a tool
            tool_result = web_search_tool.forward(query="Python tutorials")

            # Cache the result
            key = cache._generate_key("web_search", {"query": "Python tutorials"})
            cache.set(key, tool_result)

            # Future identical calls will hit cache
            cached = cache.get(key)  # Returns tool_result instantly
            ```
        """
        if not self.enabled:
            return
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)
        # Store new entry
        self.cache[key] = ToolCacheEntry(result=result, timestamp=time())
        self.cache.move_to_end(key)
        self._save_to_disk()

    def clear(self) -> None:
        """Clear all cache entries and reset statistics.

        Removes all cached tool results and resets hit/miss counters to zero.
        Persists the empty cache to disk if persist=True.

        Side Effects:
            - Removes all cache entries
            - Resets _hits to 0
            - Resets _misses to 0
            - Saves empty cache to disk if persist=True

        !!! example
            ```python
            cache = ToolCache()

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
            cache = ToolCache()

            # ... use cache ...

            stats = cache.stats()
            print(f"Cache hit rate: {stats['hit_rate']:.2%}")
            print(f"Cache utilization: {stats['size']}/{stats['max_size']}")
            print(f"Total requests: {stats['hits'] + stats['misses']}")

            # Example output:
            # Cache hit rate: 75.00%
            # Cache utilization: 42/100
            # Total requests: 56
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
