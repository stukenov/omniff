import threading

import pytest

from omniff.reliability import VRAM_ESTIMATES, ModelMutex, retry_on_oom


def test_model_mutex_acquire_release():
    mutex = ModelMutex()
    mutex.acquire("test")
    mutex.release("test")


def test_model_mutex_concurrent():
    mutex = ModelMutex()
    results = []

    def worker(n):
        mutex.acquire("shared")
        results.append(n)
        mutex.release("shared")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [0, 1, 2, 3, 4]


def test_retry_on_oom_no_error():
    @retry_on_oom(max_retries=2)
    def ok():
        return 42

    assert ok() == 42


def test_retry_on_oom_non_oom_raises():
    @retry_on_oom(max_retries=2)
    def fail():
        raise ValueError("not oom")

    with pytest.raises(ValueError, match="not oom"):
        fail()


def test_retry_on_oom_retries_oom():
    call_count = 0

    @retry_on_oom(max_retries=2, backoff_base=0.01)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    assert flaky() == "ok"
    assert call_count == 3


def test_vram_estimates_exist():
    assert "Qwen/Qwen3-4B" in VRAM_ESTIMATES
    assert "openai/whisper-large-v3" in VRAM_ESTIMATES


def test_mutex_double_release_safe():
    mutex = ModelMutex()
    mutex.acquire("test")
    mutex.release("test")
    mutex.release("test")
