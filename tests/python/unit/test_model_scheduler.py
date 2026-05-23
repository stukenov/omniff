import time

import pytest

from omniff.scheduler.model_scheduler import LoadPolicy, ModelScheduler


class FakeModel:
    def __init__(self):
        self._loaded = False
        self.load_count = 0

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True
        self.load_count += 1

    def unload(self):
        self._loaded = False


def test_register_and_acquire():
    sched = ModelScheduler(max_loaded=4)
    m = FakeModel()
    sched.register("llm", m, LoadPolicy.WARM)
    result = sched.acquire("llm")
    assert result is m
    assert m.is_loaded


def test_acquire_missing():
    sched = ModelScheduler()
    with pytest.raises(KeyError, match="not registered"):
        sched.acquire("nope")


def test_cold_release_unloads():
    sched = ModelScheduler()
    m = FakeModel()
    sched.register("llm", m, LoadPolicy.COLD)
    sched.acquire("llm")
    assert m.is_loaded
    sched.release("llm")
    assert not m.is_loaded


def test_warm_stays_loaded_after_release():
    sched = ModelScheduler()
    m = FakeModel()
    sched.register("llm", m, LoadPolicy.WARM)
    sched.acquire("llm")
    sched.release("llm")
    assert m.is_loaded


def test_hot_never_evicted():
    sched = ModelScheduler(max_loaded=1)
    hot = FakeModel()
    warm = FakeModel()
    sched.register("hot", hot, LoadPolicy.HOT)
    sched.register("warm", warm, LoadPolicy.WARM)
    sched.acquire("hot")
    sched.acquire("warm")
    assert hot.is_loaded


def test_evict_expired():
    sched = ModelScheduler(default_ttl=0.01)
    m = FakeModel()
    sched.register("llm", m, LoadPolicy.WARM, ttl=0.01)
    sched.acquire("llm")
    time.sleep(0.02)
    evicted = sched.evict_expired()
    assert "llm" in evicted
    assert not m.is_loaded


def test_lru_eviction():
    sched = ModelScheduler(max_loaded=2)
    m1 = FakeModel()
    m2 = FakeModel()
    m3 = FakeModel()
    sched.register("a", m1, LoadPolicy.WARM)
    sched.register("b", m2, LoadPolicy.WARM)
    sched.register("c", m3, LoadPolicy.WARM)
    sched.acquire("a")
    sched.acquire("b")
    sched.acquire("c")
    assert not m1.is_loaded  # LRU evicted
    assert m2.is_loaded or m3.is_loaded


def test_loaded_models():
    sched = ModelScheduler()
    m = FakeModel()
    sched.register("llm", m, LoadPolicy.WARM)
    assert sched.loaded_models() == []
    sched.acquire("llm")
    assert sched.loaded_models() == ["llm"]


def test_status():
    sched = ModelScheduler()
    m = FakeModel()
    sched.register("llm", m, LoadPolicy.WARM)
    s = sched.status()
    assert s["llm"]["policy"] == "warm"
    assert s["llm"]["loaded"] is False


def test_unload_all():
    sched = ModelScheduler()
    m1 = FakeModel()
    m2 = FakeModel()
    sched.register("a", m1, LoadPolicy.HOT)
    sched.register("b", m2, LoadPolicy.WARM)
    sched.acquire("a")
    sched.acquire("b")
    sched.unload_all()
    assert not m1.is_loaded
    assert not m2.is_loaded
