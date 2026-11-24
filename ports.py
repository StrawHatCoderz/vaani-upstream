from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str: ...


class VoiceActivityDetector(Protocol):
    def record_turn(self, stop_requested: Callable[[], bool] | None = None) -> np.ndarray | None: ...


class ActivationProvider(Protocol):
    """Controls when microphone capture begins and ends for one user query."""

    def wait_for_activation(self) -> None: ...

    def is_shutdown_requested(self) -> bool: ...


class ConversationModel(Protocol):
    def reply(self, messages: list[Message]) -> str: ...


class SpeechSynthesizer(Protocol):
    def speak(self, text: str) -> None: ...
