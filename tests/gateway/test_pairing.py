"""Tests for gateway/pairing.py — DM pairing security system."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

from gateway.pairing import (
    PairingStore,
    ALPHABET,
    CODE_LENGTH,
    CODE_TTL_SECONDS,
    RATE_LIMIT_SECONDS,
    MAX_PENDING_PER_PLATFORM,
    MAX_FAILED_ATTEMPTS,
    LOCKOUT_SECONDS,
    _secure_write,
)


def _make_store(tmp_path):
    """Create a PairingStore with PAIRING_DIR pointed to tmp_path."""
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        return PairingStore()


# ---------------------------------------------------------------------------
# _secure_write
# ---------------------------------------------------------------------------


class TestSecureWrite:
    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "sub" / "dir" / "file.json"
        _secure_write(target, '{"hello": "world"}')
        assert target.exists()
        assert json.loads(target.read_text()) == {"hello": "world"}

    def test_sets_file_permissions(self, tmp_path):
        target = tmp_path / "secret.json"
        _secure_write(target, "data")
        mode = oct(target.stat().st_mode & 0o777)
        assert mode == "0o600"


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


class TestCodeGeneration:
    def test_code_format(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
        assert isinstance(code, str) and len(code) == CODE_LENGTH
        assert len(code) == CODE_LENGTH
        assert all(c in ALPHABET for c in code)

    def test_code_uniqueness(self, tmp_path):
        """Multiple codes for different users should be distinct."""
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            codes = set()
            for i in range(3):
                code = store.generate_code("telegram", f"user{i}")
                assert isinstance(code, str) and len(code) == CODE_LENGTH
                codes.add(code)
        assert len(codes) == 3

    def test_stores_pending_entry(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            pending = store.list_pending("telegram")
        assert len(pending) == 1
        assert pending[0]["code"] == code
        assert pending[0]["user_id"] == "user1"
        assert pending[0]["user_name"] == "Alice"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_same_user_rate_limited(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            code2 = store.generate_code("telegram", "user1")
        assert isinstance(code1, str) and len(code1) == CODE_LENGTH
        assert code2 is None  # rate limited

    def test_different_users_not_rate_limited(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            code2 = store.generate_code("telegram", "user2")
        assert isinstance(code1, str) and len(code1) == CODE_LENGTH
        assert isinstance(code2, str) and len(code2) == CODE_LENGTH

    def test_rate_limit_expires(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code1 = store.generate_code("telegram", "user1")
            assert isinstance(code1, str) and len(code1) == CODE_LENGTH

            # Simulate rate limit expiry
            limits = store._load_json(store._rate_limit_path())
            limits["telegram:user1"] = time.time() - RATE_LIMIT_SECONDS - 1
            store._save_json(store._rate_limit_path(), limits)

            code2 = store.generate_code("telegram", "user1")
        assert isinstance(code2, str) and len(code2) == CODE_LENGTH
        assert code2 != code1


# ---------------------------------------------------------------------------
# Max pending limit
# ---------------------------------------------------------------------------


class TestMaxPending:
    def test_max_pending_per_platform(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            codes = []
            for i in range(MAX_PENDING_PER_PLATFORM + 1):
                code = store.generate_code("telegram", f"user{i}")
                codes.append(code)

        # First MAX_PENDING_PER_PLATFORM should succeed
        assert all(isinstance(c, str) and len(c) == CODE_LENGTH for c in codes[:MAX_PENDING_PER_PLATFORM])
        # Next one should be blocked
        assert codes[MAX_PENDING_PER_PLATFORM] is None

    def test_different_platforms_independent(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for i in range(MAX_PENDING_PER_PLATFORM):
                store.generate_code("telegram", f"user{i}")
            # Different platform should still work
            code = store.generate_code("discord", "user0")
        assert isinstance(code, str) and len(code) == CODE_LENGTH


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------


class TestApprovalFlow:
    def test_approve_valid_code(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", code)

        assert isinstance(result, dict)
        assert "user_id" in result
        assert "user_name" in result
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"

    def test_approved_user_is_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            store.approve_code("telegram", code)
            assert store.is_approved("telegram", "user1") is True

    def test_unapproved_user_not_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            assert store.is_approved("telegram", "nonexistent") is False

    def test_approve_removes_from_pending(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")
            store.approve_code("telegram", code)
            pending = store.list_pending("telegram")
        assert len(pending) == 0

    def test_approve_case_insensitive(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", code.lower())
        assert isinstance(result, dict)
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"

    def test_approve_strips_whitespace(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            result = store.approve_code("telegram", f"  {code}  ")
        assert isinstance(result, dict)
        assert result["user_id"] == "user1"
        assert result["user_name"] == "Alice"

    def test_invalid_code_returns_none(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            result = store.approve_code("telegram", "INVALIDCODE")
        assert result is None


# ---------------------------------------------------------------------------
# Lockout after failed attempts
# ---------------------------------------------------------------------------


class TestLockout:
    def test_lockout_after_max_failures(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            # Generate a valid code so platform has data
            store.generate_code("telegram", "user1")

            # Exhaust failed attempts
            for _ in range(MAX_FAILED_ATTEMPTS):
                store.approve_code("telegram", "WRONGCODE")

            # Platform should now be locked out — can't generate new codes
            assert store._is_locked_out("telegram") is True

    def test_lockout_blocks_code_generation(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for _ in range(MAX_FAILED_ATTEMPTS):
                store.approve_code("telegram", "WRONG")

            code = store.generate_code("telegram", "newuser")
        assert code is None

    def test_lockout_expires(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            for _ in range(MAX_FAILED_ATTEMPTS):
                store.approve_code("telegram", "WRONG")

            # Simulate lockout expiry
            limits = store._load_json(store._rate_limit_path())
            lockout_key = "_lockout:telegram"
            limits[lockout_key] = time.time() - 1  # expired
            store._save_json(store._rate_limit_path(), limits)

            assert store._is_locked_out("telegram") is False


# ---------------------------------------------------------------------------
# Code expiry
# ---------------------------------------------------------------------------


class TestCodeExpiry:
    def test_expired_codes_cleaned_up(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")

            # Manually expire the code
            pending = store._load_json(store._pending_path("telegram"))
            pending[code]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
            store._save_json(store._pending_path("telegram"), pending)

            # Cleanup happens on next operation
            remaining = store.list_pending("telegram")
        assert len(remaining) == 0

    def test_expired_code_cannot_be_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1")

            # Expire it
            pending = store._load_json(store._pending_path("telegram"))
            pending[code]["created_at"] = time.time() - CODE_TTL_SECONDS - 1
            store._save_json(store._pending_path("telegram"), pending)

            result = store.approve_code("telegram", code)
        assert result is None


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_approved_user(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            store.approve_code("telegram", code)
            assert store.is_approved("telegram", "user1") is True

            revoked = store.revoke("telegram", "user1")
        assert revoked is True
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            assert store.is_approved("telegram", "user1") is False

    def test_revoke_nonexistent_returns_false(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            assert store.revoke("telegram", "nobody") is False


# ---------------------------------------------------------------------------
# List & clear
# ---------------------------------------------------------------------------


class TestListAndClear:
    def test_list_approved(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            code = store.generate_code("telegram", "user1", "Alice")
            store.approve_code("telegram", code)
            approved = store.list_approved("telegram")
        assert len(approved) == 1
        assert approved[0]["user_id"] == "user1"
        assert approved[0]["platform"] == "telegram"

    def test_list_approved_all_platforms(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            c1 = store.generate_code("telegram", "user1")
            store.approve_code("telegram", c1)
            c2 = store.generate_code("discord", "user2")
            store.approve_code("discord", c2)
            approved = store.list_approved()
        assert len(approved) == 2

    def test_clear_pending(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            store.generate_code("telegram", "user1")
            store.generate_code("telegram", "user2")
            count = store.clear_pending("telegram")
            remaining = store.list_pending("telegram")
        assert count == 2
        assert len(remaining) == 0

    def test_clear_pending_all_platforms(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            store = PairingStore()
            store.generate_code("telegram", "user1")
            store.generate_code("discord", "user2")
            count = store.clear_pending()
        assert count == 2


# ---------------------------------------------------------------------------
# Concurrency regression tests
# ---------------------------------------------------------------------------


class TestConcurrentStateUpdates:
    def test_concurrent_rate_limit_updates_are_not_lost(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            stores = [PairingStore() for _ in range(8)]
            start = threading.Event()
            errors = []
            original_load_json = PairingStore._load_json

            def delayed_load(self, path):
                data = original_load_json(self, path)
                if path.name == "_rate_limits.json":
                    time.sleep(0.02)
                return data

            def worker(store, idx):
                start.wait()
                try:
                    store._record_rate_limit("telegram", f"user{idx}")
                except Exception as exc:  # pragma: no cover - diagnostic
                    errors.append(exc)

            with patch.object(PairingStore, "_load_json", delayed_load):
                threads = [
                    threading.Thread(target=worker, args=(store, idx))
                    for idx, store in enumerate(stores)
                ]
                for thread in threads:
                    thread.start()
                start.set()
                for thread in threads:
                    thread.join(timeout=5)

        assert not errors
        assert all(not thread.is_alive() for thread in threads)

        limits = stores[0]._load_json(stores[0]._rate_limit_path())
        tracked_users = {
            key for key in limits
            if key.startswith("telegram:user")
        }
        assert tracked_users == {f"telegram:user{idx}" for idx in range(8)}

    def test_concurrent_code_generation_preserves_all_pending_requests(self, tmp_path):
        with patch("gateway.pairing.PAIRING_DIR", tmp_path):
            stores = [PairingStore() for _ in range(MAX_PENDING_PER_PLATFORM)]
            start = threading.Event()
            errors = []
            codes = [None] * MAX_PENDING_PER_PLATFORM
            original_load_json = PairingStore._load_json

            def delayed_load(self, path):
                data = original_load_json(self, path)
                if path.name in {"telegram-pending.json", "_rate_limits.json"}:
                    time.sleep(0.02)
                return data

            def worker(slot, store):
                start.wait()
                try:
                    codes[slot] = store.generate_code("telegram", f"user{slot}", f"User {slot}")
                except Exception as exc:  # pragma: no cover - diagnostic
                    errors.append(exc)

            with patch.object(PairingStore, "_load_json", delayed_load):
                threads = [
                    threading.Thread(target=worker, args=(idx, store))
                    for idx, store in enumerate(stores)
                ]
                for thread in threads:
                    thread.start()
                start.set()
                for thread in threads:
                    thread.join(timeout=5)

        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert all(isinstance(code, str) and len(code) == CODE_LENGTH for code in codes)

        pending = stores[0].list_pending("telegram")
        pending_users = {entry["user_id"] for entry in pending}
        assert pending_users == {f"user{idx}" for idx in range(MAX_PENDING_PER_PLATFORM)}
