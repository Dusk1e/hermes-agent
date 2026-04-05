import threading
import time
import asyncio
import pytest
from hermes_constants import set_session_env, get_session_env, clear_session_env

def thread_worker(session_id, results):
    set_session_env("HERMES_SESSION_KEY", session_id)
    # Simulate some work and potential concurrency
    time.sleep(0.1)
    # Verify that the session key is still what we set
    results[session_id] = get_session_env("HERMES_SESSION_KEY")

def test_session_env_thread_isolation():
    results = {}
    threads = []
    # Clear any existing state
    clear_session_env()
    
    for i in range(10):
        t = threading.Thread(target=thread_worker, args=(f"session_{i}", results))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    for i in range(10):
        assert results[f"session_{i}"] == f"session_{i}", f"Thread isolation failed for session_{i}"

async def async_worker(session_id, results):
    set_session_env("HERMES_SESSION_KEY", session_id)
    # Yield control to other tasks
    await asyncio.sleep(0.1)
    # Verify that the session key is still what we set
    results[session_id] = get_session_env("HERMES_SESSION_KEY")

@pytest.mark.asyncio
async def test_session_env_async_isolation():
    results = {}
    # Clear any existing state
    clear_session_env()
    
    tasks = []
    for i in range(10):
        tasks.append(async_worker(f"session_{i}", results))
        
    await asyncio.gather(*tasks)
    
    for i in range(10):
        assert results[f"session_{i}"] == f"session_{i}", f"Async isolation failed for session_{i}"

def test_session_env_fallback():
    """Verify that get_session_env falls back to os.environ if contextvar is not set."""
    import os
    import uuid
    test_key = f"TEST_VAR_{uuid.uuid4().hex}"
    test_val = "global_value"
    
    os.environ[test_key] = test_val
    try:
        # Since it's not in _CONTEXT_VAR_MAP, it should go to os.environ
        assert get_session_env(test_key) == test_val
        
        # Now test a key that IS in _CONTEXT_VAR_MAP but not set in context
        # (Should fall back to os.environ if context val is empty string)
        os.environ["HERMES_SESSION_PLATFORM"] = "global_platform"
        clear_session_env() # Ensure it's empty in context
        assert get_session_env("HERMES_SESSION_PLATFORM") == "global_platform"
        
        # Now set it in context and ensure it takes precedence
        set_session_env("HERMES_SESSION_PLATFORM", "local_platform")
        assert get_session_env("HERMES_SESSION_PLATFORM") == "local_platform"
    finally:
        if test_key in os.environ:
            del os.environ[test_key]
        if "HERMES_SESSION_PLATFORM" in os.environ:
            del os.environ["HERMES_SESSION_PLATFORM"]
