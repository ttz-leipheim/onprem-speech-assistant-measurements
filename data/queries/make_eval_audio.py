#!/usr/bin/env python
"""Speak the German prompt texts into 16 kHz mono WAVs for the timing probes.

Uses the local Piper CLI and a Piper voice model. The output is synthetic and
reproducible, which is what the timing probes need; word error rates measured
against it are not a claim about accuracy on human speech.

Usage:
    python data/queries/make_eval_audio.py --model <voice>.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent          # data/queries/
DEFAULT_PROMPTS = HERE / "prompts.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "audio"

# Voice models are not redistributed. Point --piper-dir at a directory holding
# a Piper .onnx voice, or pass --model directly.
DEFAULT_PIPER_DIR = HERE / "voices"


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        prompts.append({"id": str(raw["id"]), "text": str(raw["text"])})
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def find_model(path: str | None) -> Path:
    if path:
        model = Path(path)
        if not model.is_file():
            raise SystemExit(f"Piper model not found: {model}")
        return model
    candidates = sorted(DEFAULT_PIPER_DIR.rglob("*.onnx"))
    german = [candidate for candidate in candidates if "de_" in candidate.name.lower()]
    if not german and not candidates:
        raise SystemExit(
            f"No Piper .onnx model found under {DEFAULT_PIPER_DIR}. Pass --model to choose one."
        )
    return (german or candidates)[0]


def wav_is_target(path: Path) -> bool:
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getnchannels() == 1
            and handle.getframerate() == 16_000
            and handle.getsampwidth() == 2
        )


def synthesize(piper_bin: str, model: Path, text: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_output = Path(temp_dir) / output.name
        subprocess.run(
            [piper_bin, "--model", str(model), "--output_file", str(raw_output)],
            input=text,
            text=True,
            check=True,
        )
        if wav_is_target(raw_output):
            shutil.copyfile(raw_output, output)
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit(
                f"{raw_output} is not 16 kHz mono PCM16 and ffmpeg is not available for conversion."
            )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(raw_output),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="JSONL prompts with id/text")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output WAV directory")
    parser.add_argument("--model", default=None, help="Piper .onnx model path")
    parser.add_argument("--piper-bin", default="piper", help="Piper executable")
    args = parser.parse_args()

    if shutil.which(args.piper_bin) is None:
        raise SystemExit(f"Piper executable not found: {args.piper_bin}")
    model = find_model(args.model)
    prompts = load_prompts(Path(args.prompts))
    output_dir = Path(args.output_dir)
    for index, prompt in enumerate(prompts, start=1):
        output = output_dir / f"{index:02d}_{prompt['id']}.wav"
        synthesize(args.piper_bin, model, prompt["text"], output)
        print(output)
    print(f"\nGenerated {len(prompts)} WAV files with {model}", file=sys.stderr)


if __name__ == "__main__":
    main()
