import pytest
from omniff.scaling.device_map import DeviceMap, GPUInfo
from omniff.scaling.queue import RequestQueue, Priority, QueueItem
from omniff.scaling.batch import BatchInference
from omniff.scaling.quantization import detect_quantization, estimate_quantized_size_gb


def test_device_map_empty():
    dm = DeviceMap()
    assert dm.get_device("llm") == "auto"
    assert dm.get_assignments() == {}


def test_device_map_assign():
    dm = DeviceMap()
    dm._gpus = [GPUInfo(0, "A10", 22.0, 20.0), GPUInfo(1, "A10", 22.0, 18.0)]
    dm.assign("llm", 0)
    assert dm.get_device("llm") == "cuda:0"
    assert dm.get_assignments() == {"llm": 0}


def test_device_map_auto_assign():
    dm = DeviceMap()
    dm._gpus = [GPUInfo(0, "A10", 22.0, 10.0), GPUInfo(1, "A10", 22.0, 20.0)]
    idx = dm.auto_assign("vlm", required_gb=5.0)
    assert idx == 1
    assert dm.get_device("vlm") == "cuda:1"


def test_device_map_auto_assign_insufficient():
    dm = DeviceMap()
    dm._gpus = [GPUInfo(0, "A10", 22.0, 2.0)]
    idx = dm.auto_assign("big_model", required_gb=10.0)
    assert idx == -1


def test_queue_enqueue_dequeue():
    q = RequestQueue()
    item = q.enqueue({"input": "hello"})
    assert q.size() == 1
    out = q.dequeue()
    assert out.payload["input"] == "hello"
    assert q.is_empty()


def test_queue_priority_ordering():
    q = RequestQueue()
    q.enqueue({"n": 1}, Priority.LOW)
    q.enqueue({"n": 2}, Priority.CRITICAL)
    q.enqueue({"n": 3}, Priority.NORMAL)
    first = q.dequeue()
    assert first.payload["n"] == 2


def test_queue_max_size():
    q = RequestQueue(max_size=2)
    q.enqueue({"n": 1})
    q.enqueue({"n": 2})
    with pytest.raises(RuntimeError, match="Queue full"):
        q.enqueue({"n": 3})


def test_queue_stats():
    q = RequestQueue()
    q.enqueue({"n": 1})
    q.dequeue()
    stats = q.stats()
    assert stats["processed"] == 1
    assert stats["queue_size"] == 0


def test_batch_add_and_size():
    batch = BatchInference(max_batch_size=4)
    batch.add({"prompt": "a"})
    batch.add({"prompt": "b"})
    assert batch.size() == 2
    assert not batch.is_full()


def test_batch_is_full():
    batch = BatchInference(max_batch_size=2)
    batch.add({"prompt": "a"})
    batch.add({"prompt": "b"})
    assert batch.is_full()


def test_batch_flush():
    class MockModel:
        def infer(self, inputs):
            return {"text": inputs["prompt"].upper()}

    batch = BatchInference(max_batch_size=4)
    batch.add({"prompt": "hello"})
    batch.add({"prompt": "world"})
    results = batch.flush(MockModel())
    assert len(results) == 2
    assert results[0]["text"] == "HELLO"
    assert batch.size() == 0


def test_batch_flush_empty():
    batch = BatchInference()
    results = batch.flush(None)
    assert results == []


def test_detect_quantization_gptq():
    assert detect_quantization("model-gptq-4bit") == "gptq"


def test_detect_quantization_awq():
    assert detect_quantization("model-AWQ") == "awq"


def test_detect_quantization_gguf():
    assert detect_quantization("model.gguf") == "gguf"


def test_detect_quantization_none():
    assert detect_quantization("Qwen/Qwen3-4B") is None


def test_estimate_quantized_size():
    assert estimate_quantized_size_gb(8.0, "gptq") == 2.0
    assert estimate_quantized_size_gb(8.0, "int8") == 4.0
    assert estimate_quantized_size_gb(8.0, None) == 8.0
