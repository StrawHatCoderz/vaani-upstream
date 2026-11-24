from __future__ import annotations

import queue
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad

from ports import VoiceActivityDetector

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
class SileroTurnDetector(VoiceActivityDetector):
    """Records microphone audio for one conversational turn using Silero VAD."""

    def __init__(self, threshold: float = 0.5, silence_ms: int = 5_000) -> None:
        self._silence_ms = silence_ms
        self._vad = VADIterator(load_silero_vad(onnx=True), sampling_rate=SAMPLE_RATE, threshold=threshold,
                                min_silence_duration_ms=silence_ms)

    def record_turn(self, stop_requested: Callable[[], bool] | None = None) -> np.ndarray | None:
        # The VAD keeps recurrent hidden state and a "triggered" flag across calls;
        # reset them so a previous turn's speech/silence history can't corrupt this one.
        self._vad.reset_states()

        frames: queue.Queue[np.ndarray] = queue.Queue()
        captured: list[np.ndarray] = []
        started = False
        started_waiting_at = time.monotonic()

        def receive(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
            frames.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SAMPLES, callback=receive):
            while True:
                if stop_requested and stop_requested():
                    break
                try:
                    frame = frames.get(timeout=2)
                except queue.Empty:
                    elapsed = (time.monotonic() - started_waiting_at) * 1_000
                    if not started and elapsed >= self._silence_ms:
                        break
                    continue
                elapsed = (time.monotonic() - started_waiting_at) * 1_000
                if not started and elapsed >= self._silence_ms:
                    break
                event = self._vad(torch.from_numpy(frame), return_seconds=False)
                started = started or bool(event and "start" in event)
                if started:
                    captured.append(frame)
                if started and event and "end" in event:
                    break
        return np.concatenate(captured) if captured else None
