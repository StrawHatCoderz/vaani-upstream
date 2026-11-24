import pytest

from vaani.__main__ import build_parser


@pytest.mark.parametrize("value", ["0", "-1"])
def test_silence_timeout_must_be_positive(value):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--voice", "voice.onnx", "--silence-timeout-ms", value])


def test_silence_timeout_defaults_to_five_seconds(monkeypatch):
    monkeypatch.delenv("VAANI_SILENCE_TIMEOUT_MS", raising=False)

    args = build_parser().parse_args(["--voice", "voice.onnx"])

    assert args.silence_timeout_ms == 5_000
