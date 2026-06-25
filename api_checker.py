"""
api_checker.py
Anthropic API health, model freshness, and latency checker for Wemby-GM.

Single check  : python api_checker.py
Watch mode    : python api_checker.py --watch
JSON output   : python api_checker.py --json
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WEMBY_GM_MODEL = "claude-haiku-4-5-20251001"
CHECK_DB = "api_checks.db"
WATCH_INTERVAL_SECONDS = 60
PING_MAX_TOKENS = 10  # minimal tokens — keeps health-check cost near zero

# Known latest model IDs as of knowledge cutoff (2025-08).
# Any model returned by the API that is NOT here = newly released.
# Any model listed here that is NOT returned by the API = deprecated/removed.
KNOWN_LATEST_MODELS: Dict[str, str] = {
    "claude-haiku-4-5-20251001":    "Claude Haiku 4.5 (Wemby-GM target)",
    "claude-sonnet-4-6":            "Claude Sonnet 4.6",
    "claude-opus-4-8":              "Claude Opus 4.8",
    "claude-fable-5":               "Claude Fable 5",
    "claude-opus-4-5-20251101":     "Claude Opus 4.5",
    "claude-opus-4-6":              "Claude Opus 4.6",
    "claude-opus-4-7":              "Claude Opus 4.7",
    "claude-sonnet-4-5-20250929":   "Claude Sonnet 4.5",
}

STATUS_HEALTHY  = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"
STATUS_DOWN     = "DOWN"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_id: str
    display_name: str
    created_at: Optional[str] = None
    is_known_latest: bool = False
    is_wemby_gm_model: bool = False
    is_newly_detected: bool = False


class PingResult(BaseModel):
    success: bool
    latency_ms: float
    resolved_model_id: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None


class APICheckResult(BaseModel):
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    api_key_masked: str
    env_file_present: bool
    key_format_valid: bool
    ping: PingResult
    models_available: List[ModelInfo]
    wemby_gm_model_available: bool
    newly_detected_models: List[str]
    deprecated_known_models: List[str]
    overall_status: str
    status_reason: str


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SQLITE CHECK HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _init_check_db() -> sqlite3.Connection:
    conn = sqlite3.connect(CHECK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_checks (
            check_id   TEXT PRIMARY KEY,
            timestamp  TEXT NOT NULL,
            status     TEXT NOT NULL,
            latency_ms REAL,
            payload    TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_check_result(result: APICheckResult) -> None:
    conn = _init_check_db()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_checks "
                "(check_id, timestamp, status, latency_ms, payload) "
                "VALUES (?,?,?,?,?)",
                (
                    result.check_id,
                    result.timestamp,
                    result.overall_status,
                    result.ping.latency_ms,
                    result.model_dump_json(),
                ),
            )
    finally:
        conn.close()


def load_recent_checks(limit: int = 5) -> List[Dict[str, Any]]:
    conn = _init_check_db()
    try:
        rows = conn.execute(
            "SELECT check_id, timestamp, status, latency_ms "
            "FROM api_checks ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "check_id": r[0],
            "timestamp": r[1],
            "status": r[2],
            "latency_ms": r[3],
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ENVIRONMENT GUARDS
# ─────────────────────────────────────────────────────────────────────────────

def check_env_file() -> bool:
    return os.path.isfile(".env")


def check_key_format(key: str) -> bool:
    """Anthropic API keys start with 'sk-ant-' and are > 40 chars."""
    return key.startswith("sk-ant-") and len(key) > 40


def mask_key(key: str) -> str:
    if len(key) < 12:
        return "***"
    return key[:10] + "..." + key[-4:]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ASYNC API CHECKS
# ─────────────────────────────────────────────────────────────────────────────

async def ping_api(client: anthropic.AsyncAnthropic) -> PingResult:
    """
    Sends a minimal message to measure latency and confirm key validity.
    Uses max_tokens=10 to keep cost near zero.
    """
    t0 = time.perf_counter()
    try:
        response = await client.messages.create(
            model=WEMBY_GM_MODEL,
            max_tokens=PING_MAX_TOKENS,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        resolved = response.model  # actual model ID the alias resolved to
        return PingResult(
            success=True,
            latency_ms=round(latency_ms, 2),
            resolved_model_id=resolved,
        )
    except anthropic.AuthenticationError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return PingResult(
            success=False,
            latency_ms=round(latency_ms, 2),
            http_status=401,
            error=f"AuthenticationError: {exc.message}",
        )
    except anthropic.RateLimitError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return PingResult(
            success=False,
            latency_ms=round(latency_ms, 2),
            http_status=429,
            error=f"RateLimitError: {exc.message}",
        )
    except anthropic.NotFoundError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return PingResult(
            success=False,
            latency_ms=round(latency_ms, 2),
            http_status=404,
            error=f"ModelNotFound: '{WEMBY_GM_MODEL}' is not available on this account (404)",
        )
    except anthropic.APIConnectionError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return PingResult(
            success=False,
            latency_ms=round(latency_ms, 2),
            error=f"ConnectionError: {exc}",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return PingResult(
            success=False,
            latency_ms=round(latency_ms, 2),
            error=f"Unexpected: {type(exc).__name__}: {exc}",
        )


async def fetch_available_models(
    client: anthropic.AsyncAnthropic,
) -> Tuple[List[ModelInfo], List[str], List[str]]:
    """
    Lists all models from the Anthropic API.
    Returns (model_list, newly_detected_ids, deprecated_known_ids).
    """
    try:
        page = await client.models.list()
        raw_models = list(page.data)
    except Exception:
        # Models endpoint unavailable — return empty, don't crash the check
        return [], [], []

    returned_ids = set()
    model_list: List[ModelInfo] = []

    for m in raw_models:
        mid = m.id
        returned_ids.add(mid)
        created = None
        if hasattr(m, "created_at"):
            raw_ts = m.created_at
            if raw_ts:
                created = (
                    raw_ts.isoformat()
                    if hasattr(raw_ts, "isoformat")
                    else str(raw_ts)
                )
        display = getattr(m, "display_name", mid)
        model_list.append(
            ModelInfo(
                model_id=mid,
                display_name=display,
                created_at=created,
                is_known_latest=mid in KNOWN_LATEST_MODELS,
                is_wemby_gm_model=(mid == WEMBY_GM_MODEL),
                is_newly_detected=(mid not in KNOWN_LATEST_MODELS),
            )
        )

    newly_detected = [mid for mid in returned_ids if mid not in KNOWN_LATEST_MODELS]
    deprecated = [
        mid for mid in KNOWN_LATEST_MODELS if mid not in returned_ids
    ]

    model_list.sort(key=lambda m: (not m.is_wemby_gm_model, not m.is_known_latest, m.model_id))
    return model_list, newly_detected, deprecated


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

async def run_full_check() -> APICheckResult:
    env_present = check_env_file()
    key_format_ok = check_key_format(_API_KEY)
    masked = mask_key(_API_KEY) if _API_KEY else "(empty)"

    if not _API_KEY:
        ping = PingResult(
            success=False,
            latency_ms=0.0,
            error="ANTHROPIC_API_KEY is not set in .env",
        )
        return APICheckResult(
            api_key_masked=masked,
            env_file_present=env_present,
            key_format_valid=False,
            ping=ping,
            models_available=[],
            wemby_gm_model_available=False,
            newly_detected_models=[],
            deprecated_known_models=[],
            overall_status=STATUS_DOWN,
            status_reason="API key missing",
        )

    client = anthropic.AsyncAnthropic(api_key=_API_KEY)

    ping, (models, newly_detected, deprecated) = await asyncio.gather(
        ping_api(client),
        fetch_available_models(client),
    )

    wemby_available = any(
        m.model_id == WEMBY_GM_MODEL or m.is_wemby_gm_model for m in models
    )

    # ── Determine overall status ─────────────────────────────────────────────
    if not ping.success:
        if ping.http_status == 401:
            status = STATUS_DOWN
            reason = "API key is invalid or revoked (HTTP 401)"
        elif ping.http_status == 429:
            status = STATUS_DEGRADED
            reason = "Rate-limited — quota may be exhausted (HTTP 429)"
        elif ping.http_status == 404:
            status = STATUS_DEGRADED
            reason = f"Wemby-GM model '{WEMBY_GM_MODEL}' not found — update MODEL in app.py"
        else:
            status = STATUS_DOWN
            reason = ping.error or "Ping failed — API unreachable"
    elif not key_format_ok:
        status = STATUS_DEGRADED
        reason = "Key format unrecognised (expected sk-ant-... prefix)"
    elif not wemby_available and models:
        status = STATUS_DEGRADED
        reason = f"Wemby-GM model '{WEMBY_GM_MODEL}' not found in model list"
    elif newly_detected:
        status = STATUS_HEALTHY
        reason = f"Healthy — {len(newly_detected)} new model(s) detected"
    else:
        status = STATUS_HEALTHY
        reason = "All checks passed"

    result = APICheckResult(
        api_key_masked=masked,
        env_file_present=env_present,
        key_format_valid=key_format_ok,
        ping=ping,
        models_available=models,
        wemby_gm_model_available=wemby_available,
        newly_detected_models=newly_detected,
        deprecated_known_models=deprecated,
        overall_status=status,
        status_reason=reason,
    )

    save_check_result(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {
    STATUS_HEALTHY:  "[OK]",
    STATUS_DEGRADED: "[!!]",
    STATUS_DOWN:     "[XX]",
}

_STATUS_LABEL = {
    STATUS_HEALTHY:  "HEALTHY",
    STATUS_DEGRADED: "DEGRADED",
    STATUS_DOWN:     "DOWN",
}


def _bar(title: str, width: int = 64) -> str:
    pad = width - len(title) - 4
    return f"  -- {title} {'-' * max(0, pad)}"


def print_status_report(result: APICheckResult, history: List[Dict]) -> None:
    icon  = _STATUS_ICON.get(result.overall_status,  "[??]")
    label = _STATUS_LABEL.get(result.overall_status, result.overall_status)

    print()
    print(f"  +{'=' * 62}+")
    print(f"  |{'  Wemby-GM  .  Anthropic API Health Check':^62}|")
    print(f"  +{'=' * 62}+")
    print()

    print(_bar("SUMMARY"))
    print(f"    Check ID     : {result.check_id}")
    print(f"    Timestamp    : {result.timestamp} UTC")
    print(f"    Status       : {icon}  {label}")
    print(f"    Reason       : {result.status_reason}")
    print()

    print(_bar("ENVIRONMENT"))
    env_tag  = "present" if result.env_file_present  else "MISSING"
    fmt_tag  = "valid"   if result.key_format_valid  else "INVALID FORMAT"
    print(f"    .env file    : {env_tag}")
    print(f"    API key      : {result.api_key_masked}")
    print(f"    Key format   : {fmt_tag}")
    print()

    print(_bar("API PING"))
    ping = result.ping
    ping_tag = "SUCCESS" if ping.success else "FAILED"
    print(f"    Result       : {ping_tag}")
    print(f"    Latency      : {ping.latency_ms:.1f} ms")
    if ping.resolved_model_id:
        print(f"    Resolved to  : {ping.resolved_model_id}")
    if ping.error:
        print(f"    Error        : {ping.error}")
    print()

    print(_bar("MODEL FRESHNESS"))
    wemby_tag = "AVAILABLE" if result.wemby_gm_model_available else "NOT FOUND"
    print(f"    Wemby-GM model ({WEMBY_GM_MODEL})")
    print(f"      Status     : {wemby_tag}")
    print(f"    Total models : {len(result.models_available)}")

    if result.newly_detected_models:
        print(f"    NEW models detected ({len(result.newly_detected_models)}) — consider upgrading:")
        for mid in result.newly_detected_models:
            print(f"      + {mid}")
    else:
        print("    New models   : none (all returned models are known)")

    if result.deprecated_known_models:
        print(f"    Removed/deprecated ({len(result.deprecated_known_models)}):")
        for mid in result.deprecated_known_models:
            print(f"      - {mid}")
    print()

    if result.models_available:
        print(_bar("AVAILABLE MODELS"))
        for m in result.models_available:
            flags = []
            if m.is_wemby_gm_model:
                flags.append("* Wemby-GM target")
            if m.is_newly_detected:
                flags.append("NEW")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"    {m.model_id:<45}{flag_str}")
        print()

    if history:
        print(_bar("RECENT CHECK HISTORY"))
        print(f"    {'Check ID':<10}  {'Timestamp':<26}  {'Status':<10}  {'Latency':>10}")
        print(f"    {'-'*8}  {'-'*24}  {'-'*8}  {'-'*10}")
        for h in history:
            lat = f"{h['latency_ms']:.1f} ms" if h["latency_ms"] is not None else "n/a"
            print(
                f"    {h['check_id']:<10}  {h['timestamp']:<26}  "
                f"{h['status']:<10}  {lat:>10}"
            )
        print()

    print(f"  {'-' * 64}")
    print(f"  Results saved to {CHECK_DB}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = sys.argv[1:]
    watch_mode = "--watch" in args
    json_mode  = "--json"  in args

    if watch_mode:
        print(f"  [WATCH MODE] Polling every {WATCH_INTERVAL_SECONDS}s. Ctrl-C to stop.")

    first_run = True
    while True:
        if not first_run:
            print(f"\n  ... next check in {WATCH_INTERVAL_SECONDS}s (Ctrl-C to stop) ...\n")
            await asyncio.sleep(WATCH_INTERVAL_SECONDS)

        result  = await run_full_check()
        history = load_recent_checks(limit=5)

        if json_mode:
            print(result.model_dump_json(indent=2))
        else:
            print_status_report(result, history)

        first_run = False

        if not watch_mode:
            break

    # Exit with non-zero code when the API is DOWN so CI pipelines can react
    if result.overall_status == STATUS_DOWN:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  [STOPPED] API checker terminated.")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  TESTS  (pytest app.py -v  OR  pytest api_checker.py -v)
# ─────────────────────────────────────────────────────────────────────────────

def test_mask_key_hides_middle():
    masked = mask_key("sk-ant-api03-abc1234567890XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    assert masked.startswith("sk-ant-api")
    assert "..." in masked
    assert len(masked) < 50


def test_mask_key_short_input():
    assert mask_key("short") == "***"


def test_key_format_valid():
    valid_key = "sk-ant-api03-" + "a" * 40
    assert check_key_format(valid_key) is True


def test_key_format_invalid_prefix():
    assert check_key_format("sk-openai-badkey") is False


def test_key_format_too_short():
    assert check_key_format("sk-ant-abc") is False


def test_env_file_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-api03-test\n")
    assert check_env_file() is True


def test_env_file_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check_env_file() is False


def test_model_info_schema():
    m = ModelInfo(
        model_id="claude-3-5-haiku-latest",
        display_name="Claude 3.5 Haiku",
        is_known_latest=True,
        is_wemby_gm_model=True,
    )
    assert m.is_wemby_gm_model is True
    assert m.is_newly_detected is False


def test_ping_result_failure_schema():
    p = PingResult(success=False, latency_ms=0.0, http_status=401, error="Unauthorized")
    assert p.success is False
    assert p.http_status == 401


def test_api_check_result_schema():
    ping = PingResult(success=True, latency_ms=123.4, resolved_model_id="claude-3-5-haiku-20241022")
    result = APICheckResult(
        api_key_masked="sk-ant-api...",
        env_file_present=True,
        key_format_valid=True,
        ping=ping,
        models_available=[],
        wemby_gm_model_available=True,
        newly_detected_models=[],
        deprecated_known_models=[],
        overall_status=STATUS_HEALTHY,
        status_reason="All checks passed",
    )
    assert result.overall_status == "HEALTHY"
    assert result.ping.latency_ms == 123.4


def test_save_and_load_check_result(tmp_path, monkeypatch):
    import api_checker as _mod
    original_db = _mod.CHECK_DB
    _mod.CHECK_DB = str(tmp_path / "test_checks.db")
    try:
        ping = PingResult(success=True, latency_ms=88.5, resolved_model_id="claude-3-5-haiku-20241022")
        result = APICheckResult(
            check_id="TESTCHK1",
            api_key_masked="sk-ant-api...",
            env_file_present=True,
            key_format_valid=True,
            ping=ping,
            models_available=[],
            wemby_gm_model_available=True,
            newly_detected_models=[],
            deprecated_known_models=[],
            overall_status=STATUS_HEALTHY,
            status_reason="Test run",
        )
        save_check_result(result)
        history = load_recent_checks(limit=1)
        assert len(history) == 1
        assert history[0]["check_id"] == "TESTCHK1"
        assert history[0]["status"] == STATUS_HEALTHY
    finally:
        _mod.CHECK_DB = original_db


def test_newly_detected_models_classification():
    # A model ID that is NOT in KNOWN_LATEST_MODELS should be flagged as new
    unknown_id = "claude-hypothetical-v99-20991231"
    assert unknown_id not in KNOWN_LATEST_MODELS


def test_known_wemby_gm_model_in_known_list():
    assert WEMBY_GM_MODEL in KNOWN_LATEST_MODELS


def test_status_icons_cover_all_statuses():
    for s in (STATUS_HEALTHY, STATUS_DEGRADED, STATUS_DOWN):
        assert s in _STATUS_ICON
        assert s in _STATUS_LABEL
