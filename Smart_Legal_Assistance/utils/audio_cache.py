"""Smart Legal Assistance – Audio Cache Manager
================================================
Manages the WAV file cache for Piper TTS responses.

Features:
  - Hash-based filename mapping (SHA-256 of voice+text).
  - Cache hit/miss tracking.
  - Cleanup of stale files older than N days.
  - Cache size reporting.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rag_engine")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "utils" / "audio_cache"


class AudioCache:
    """
    SHA-256 hash-based audio cache for Piper TTS.

    Usage:
        cache = AudioCache()
        path = cache.get_cached_path("English Female", "Hello world")
        if path:
            # cache hit
            ...
        else:
            # generate audio and register
            new_path = cache.register("English Female", "Hello world", wav_bytes)
    """

    def __init__(self, cache_dir: Optional[str] = None, max_age_days: int = 7):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.max_age_days = max_age_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------
    def _make_key(self, voice: str, text: str) -> str:
        """Return a 20-char hex SHA-256 key for (voice, text)."""
        raw = f"{voice}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def get_wav_path(self, voice: str, text: str) -> Path:
        """Return the expected WAV path for (voice, text)."""
        key = self._make_key(voice, text)
        return self.cache_dir / f"{key}.wav"

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------
    def get_cached_path(self, voice: str, text: str) -> Optional[str]:
        """
        Return the cached WAV path if it exists and is non-empty, else None.
        Updates hit/miss counters.
        """
        wav_path = self.get_wav_path(voice, text)
        if wav_path.exists() and wav_path.stat().st_size > 0:
            self._hits += 1
            logger.debug(f"[AudioCache] HIT  → {wav_path.name}")
            return str(wav_path)
        self._misses += 1
        logger.debug(f"[AudioCache] MISS → {wav_path.name}")
        return None

    def register(self, voice: str, text: str, wav_bytes: bytes) -> Optional[str]:
        """
        Write *wav_bytes* to the cache and return the file path.
        Returns None on write failure.
        """
        wav_path = self.get_wav_path(voice, text)
        try:
            wav_path.write_bytes(wav_bytes)
            logger.info(f"[AudioCache] Cached {len(wav_bytes):,} bytes → {wav_path.name}")
            return str(wav_path)
        except Exception as exc:
            logger.error(f"[AudioCache] Failed to write cache: {exc}")
            return None

    def invalidate(self, voice: str, text: str) -> None:
        """Remove a specific cache entry."""
        wav_path = self.get_wav_path(voice, text)
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)
            logger.info(f"[AudioCache] Invalidated {wav_path.name}")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def cleanup_old_files(self) -> int:
        """Delete WAV files older than max_age_days. Returns count deleted."""
        cutoff = time.time() - self.max_age_days * 86400
        deleted = 0
        for wav_file in self.cache_dir.glob("*.wav"):
            try:
                if wav_file.stat().st_mtime < cutoff:
                    wav_file.unlink(missing_ok=True)
                    deleted += 1
                    logger.debug(f"[AudioCache] Deleted stale file: {wav_file.name}")
            except Exception:
                pass
        if deleted:
            logger.info(f"[AudioCache] Cleaned up {deleted} stale WAV file(s).")
        return deleted

    def clear_all(self) -> int:
        """Delete all cached WAV files. Returns count deleted."""
        deleted = 0
        for wav_file in self.cache_dir.glob("*.wav"):
            try:
                wav_file.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass
        logger.info(f"[AudioCache] Cleared {deleted} cached WAV file(s).")
        return deleted

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        wav_files = list(self.cache_dir.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in wav_files if f.exists())
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(wav_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1_048_576, 2),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                round(self._hits / (self._hits + self._misses) * 100, 1)
                if (self._hits + self._misses) > 0
                else 0.0
            ),
        }
