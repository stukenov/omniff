"""Tests for multi-pipeline methods: audio_translate, audio_dub, video_dub."""

from unittest.mock import MagicMock, patch

import pytest

from omniff.runtime.config import OmniFFConfig, RouterConfig
from omniff.runtime.engine import OmniFFRuntime


@pytest.fixture
def runtime():
    config = OmniFFConfig(
        name="test",
        version="1.0",
        router=RouterConfig(router_type="keyword"),
        experts={},
    )
    return OmniFFRuntime(config)


class TestAudioTranslate:
    def test_chains_asr_then_translation(self, runtime):
        mock_asr = MagicMock()
        mock_asr.infer.return_value = {"text": "Привет, как дела?"}
        mock_asr.is_loaded = True

        mock_llm = MagicMock()
        mock_llm.infer.return_value = {"text": "Hello, how are you?"}
        mock_llm.is_loaded = True

        with patch.object(runtime, "_ensure_model", side_effect=[mock_asr, mock_llm]):
            trace = MagicMock()
            trace.span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            result = runtime._run_audio_translate(
                "test.wav", None, {"target_language": "English"}, trace
            )

        assert result.route == "AUDIO_TRANSLATE"
        assert result.output_text == "Hello, how are you?"
        assert result.metadata["transcript"] == "Привет, как дела?"
        assert result.metadata["target_language"] == "English"

    def test_default_target_language(self, runtime):
        mock_asr = MagicMock()
        mock_asr.infer.return_value = {"text": "test"}
        mock_asr.is_loaded = True

        mock_llm = MagicMock()
        mock_llm.infer.return_value = {"text": "translated"}
        mock_llm.is_loaded = True

        with patch.object(runtime, "_ensure_model", side_effect=[mock_asr, mock_llm]):
            trace = MagicMock()
            trace.span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            result = runtime._run_audio_translate("test.wav", None, {}, trace)

        assert result.metadata["target_language"] == "English"


class TestAudioDub:
    def test_chains_asr_translate_tts(self, runtime):
        mock_asr = MagicMock()
        mock_asr.infer.return_value = {"text": "Оригинальный текст"}
        mock_asr.is_loaded = True

        mock_llm = MagicMock()
        mock_llm.infer.return_value = {"text": "Original text"}
        mock_llm.is_loaded = True

        mock_tts = MagicMock()
        mock_tts.infer.return_value = {
            "audio_path": "dubbed.wav",
            "sample_rate": 24000,
            "duration_s": 2.5,
        }
        mock_tts.is_loaded = True

        with patch.object(
            runtime, "_ensure_model", side_effect=[mock_asr, mock_llm, mock_tts]
        ):
            trace = MagicMock()
            trace.span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            result = runtime._run_audio_dub(
                "input.wav", None, {"target_language": "English"}, "dubbed.wav", trace
            )

        assert result.route == "AUDIO_DUB"
        assert result.output_path == "dubbed.wav"
        assert result.output_text == "Original text"
        assert result.metadata["transcript"] == "Оригинальный текст"
        assert result.metadata["duration_s"] == 2.5

    def test_uses_custom_voice_preset(self, runtime):
        mock_asr = MagicMock()
        mock_asr.infer.return_value = {"text": "test"}
        mock_asr.is_loaded = True
        mock_llm = MagicMock()
        mock_llm.infer.return_value = {"text": "test"}
        mock_llm.is_loaded = True
        mock_tts = MagicMock()
        mock_tts.infer.return_value = {"audio_path": "out.wav", "duration_s": 1}
        mock_tts.is_loaded = True

        with patch.object(
            runtime, "_ensure_model", side_effect=[mock_asr, mock_llm, mock_tts]
        ):
            trace = MagicMock()
            trace.span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            runtime._run_audio_dub(
                "in.wav", None, {"voice_preset": "v2/ru_speaker_0"}, "out.wav", trace
            )

        tts_call = mock_tts.infer.call_args[0][0]
        assert tts_call["voice_preset"] == "v2/ru_speaker_0"


class TestVideoDub:
    @patch("subprocess.run")
    def test_chains_all_steps(self, mock_subprocess, runtime):
        mock_subprocess.return_value = MagicMock(returncode=0)

        mock_asr = MagicMock()
        mock_asr.infer.return_value = {"text": "Видео текст"}
        mock_asr.is_loaded = True

        mock_llm = MagicMock()
        mock_llm.infer.return_value = {"text": "Video text"}
        mock_llm.is_loaded = True

        mock_tts = MagicMock()
        mock_tts.infer.return_value = {
            "audio_path": "/tmp/dubbed.wav",
            "duration_s": 3.0,
        }
        mock_tts.is_loaded = True

        with patch.object(
            runtime, "_ensure_model", side_effect=[mock_asr, mock_llm, mock_tts]
        ), patch("os.unlink"):
            trace = MagicMock()
            trace.span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            result = runtime._run_video_dub(
                "input.mp4", None, {"target_language": "English"}, "output.mp4", trace
            )

        assert result.route == "VIDEO_DUB"
        assert result.output_path == "output.mp4"
        assert result.output_text == "Video text"
        assert mock_subprocess.call_count == 2

    @patch("subprocess.run")
    def test_ffmpeg_extract_params(self, mock_subprocess, runtime):
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_asr = MagicMock()
        mock_asr.infer.return_value = {"text": "t"}
        mock_asr.is_loaded = True
        mock_llm = MagicMock()
        mock_llm.infer.return_value = {"text": "t"}
        mock_llm.is_loaded = True
        mock_tts = MagicMock()
        mock_tts.infer.return_value = {"audio_path": "/tmp/d.wav", "duration_s": 1}
        mock_tts.is_loaded = True

        with patch.object(
            runtime, "_ensure_model", side_effect=[mock_asr, mock_llm, mock_tts]
        ), patch("os.unlink"):
            trace = MagicMock()
            trace.span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
            runtime._run_video_dub("v.mp4", None, {}, "out.mp4", trace)

        first_call = mock_subprocess.call_args_list[0]
        cmd = first_call[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-vn" in cmd
        assert "16000" in cmd


class TestGraphPlannerTemplates:
    def test_audio_translate_template_exists(self):
        from omniff.graph.planner import GraphPlanner

        planner = GraphPlanner()
        routes = planner.available_routes()
        assert "AUDIO_TRANSLATE" in routes

    def test_audio_dub_template_exists(self):
        from omniff.graph.planner import GraphPlanner

        planner = GraphPlanner()
        routes = planner.available_routes()
        assert "AUDIO_DUB" in routes

    def test_video_dub_template_exists(self):
        from omniff.graph.planner import GraphPlanner

        planner = GraphPlanner()
        routes = planner.available_routes()
        assert "VIDEO_DUB" in routes

    def test_audio_translate_plan(self):
        from omniff.graph.planner import GraphPlanner

        planner = GraphPlanner()
        graph = planner.plan("AUDIO_TRANSLATE")
        node_ids = [n.id for n in graph.nodes]
        assert "asr" in node_ids
        assert "translate" in node_ids

    def test_video_dub_plan(self):
        from omniff.graph.planner import GraphPlanner

        planner = GraphPlanner()
        graph = planner.plan("VIDEO_DUB")
        node_ids = [n.id for n in graph.nodes]
        assert "asr" in node_ids
        assert "translate" in node_ids
        assert "tts" in node_ids
