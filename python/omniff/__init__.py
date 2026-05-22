"""Saken OmniFF — FFmpeg-like multimodal AI runtime."""

__version__ = "0.1.0"

from omniff.runtime.engine import OmniFFRuntime
from omniff.runtime.result import RunResult

__all__ = ["OmniFFRuntime", "RunResult"]
