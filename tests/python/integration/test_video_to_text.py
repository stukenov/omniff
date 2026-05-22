import pytest
import numpy as np

from omniff.models.video_captioner import VideoCaptionerModel


@pytest.fixture(scope="module")
def captioner():
    model = VideoCaptionerModel(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        device="auto",
        max_new_tokens=128,
        max_frames=4,
    )
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_video(tmp_path):
    import cv2

    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
    for i in range(30):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = min(255, i * 8)  # red gradient over time
        out.write(frame)
    out.release()
    return path


def test_video_caption(captioner, test_video):
    result = captioner.infer({"video_path": test_video})
    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 5


def test_video_caption_with_prompt(captioner, test_video):
    result = captioner.infer({
        "video_path": test_video,
        "prompt": "What colors appear in this video?",
    })
    assert "text" in result
    assert len(result["text"]) > 3
