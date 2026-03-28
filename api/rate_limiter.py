"""
api/rate_limiter.py
────────────────────
Shared slowapi Limiter instance.

Kept in its own module so server.py and any future routers can
import it without circular-import issues.

Storage: Redis (same instance used for query caching).
Falls back to in-memory if Redis is unavailable at startup.
"""
import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import REDIS_URL

logger = logging.getLogger(__name__)

# slowapi expects the Redis URL without the database path suffix (no /0).
# Strip it so the storage URI is acceptable to the limits library.
_storage_uri = REDIS_URL.rsplit("/", 1)[0] if REDIS_URL.count("/") >= 3 else REDIS_URL

try:
    limiter = Limiter(key_func=get_remote_address, storage_uri=_storage_uri)
    logger.info("Rate limiter initialised with Redis backend: %s", _storage_uri)
except Exception as exc:  # pragma: no cover
    logger.warning("Redis unavailable for rate limiter, falling back to memory: %s", exc)
    limiter = Limiter(key_func=get_remote_address)
