#!/usr/bin/env python
"""Measure the F2 phantom-turn rate with the recognizer's hallucination gate off vs on.

Slices the MS-SNSD noise files into utterance-length noise-only windows (no
speech present) and runs each window through ``Transcriber`` twice:

* **gate off** - a raw decode with the same beam size but none of the
  hallucination defenses (no decode thresholds, no segment filter, no
  transcript-level filter); any non-empty transcript would have started an
  assistant turn in the session loop,
* **gate on** - the deployed decode path with its hallucination defenses in
  place: decode thresholds, a segment filter and a transcript-level filter.

Reports the phantom-turn rate per noise type and playback level. Levels are
``recorded`` (file as downloaded) and ``speech`` (RMS-normalized to -26 dBFS,
typical speech loudness, the worst case for the gate).

Requires the assistant's Python package on the import path, so it runs on the
deployment rather than from a clone. Writes raw JSON to ``results/``.

Usage:
    python probes/eval_f2_gate.py
    python probes/eval_f2_gate.py --window-s 4 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "probes"))

from eval_noise import load_wav, rms  # noqa: E402
from deployment import Config, initial_prompt  # noqa: E402

SPEECH_RMS_DBFS = -26.0


def slice_windows(audio: np.ndarray, window_s: float, sample_rate: int) -> list[np.ndarray]:
    step = int(window_s * sample_rate)
    return [audio[i : i + step] for i in range(0, len(audio) - step + 1, step)]


def scale_to_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    current = rms(audio)
    if current == 0.0:
        return audio
    target = 10 ** (target_dbfs / 20)
    return np.clip(audio * (target / current), -1.0, 1.0).astype(np.float32)


def vad_would_form_utterance(vad, audio: np.ndarray, config) -> bool:
    """Model the session's utterance formation: a contiguous VAD-positive run
    of >= vad_onset_ms starts an utterance, which is kept only if total
    VAD-positive speech reaches min_speech_ms, using the assistant's deployed
    thresholds."""
    frame_size = 512 if vad.sample_rate == 16000 else 256
    frame_ms = frame_size / vad.sample_rate * 1000.0
    vad.reset()
    flags = []
    with vad._torch.no_grad():
        for start in range(0, audio.size, frame_size):
            chunk = audio[start : start + frame_size]
            if chunk.size < frame_size:
                chunk = np.pad(chunk, (0, frame_size - chunk.size))
            tensor = vad._torch.from_numpy(chunk.astype(np.float32))
            prob = float(vad.model(tensor, vad.sample_rate).item())
            flags.append(prob >= vad.threshold)
    onset_frames = max(1, int(round(config.vad_onset_ms / frame_ms)))
    run = best_run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best_run = max(best_run, run)
    total_ms = sum(flags) * frame_ms
    return best_run >= onset_frames and total_ms >= config.min_speech_ms


def raw_transcribe(transcriber, audio: np.ndarray) -> str:
    """Gate-off decode: deployed beam size, none of the hallucination defenses."""
    segments, _info = transcriber.model.transcribe(
        audio.astype(np.float32),
        language=None,
        beam_size=transcriber.config.asr_beam_size,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt(transcriber.config),
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-dir", type=Path, default=ROOT / "data" / "noise" / "audio")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None, help="max windows per file")
    args = parser.parse_args()

    # Imported late so --help works without a deployment bound.
    from deployment import Transcriber, VoiceActivity

    config = Config.from_env()
    transcriber = Transcriber(config)
    vad = VoiceActivity(config)
    sample_rate = config.input_sample_rate

    wavs = sorted(args.noise_dir.rglob("*.wav"))
    if not wavs:
        print(f"No noise WAVs under {args.noise_dir}; run probes/fetch_noise.py first.")
        return 1

    rows: list[dict] = []
    for wav in wavs:
        noise_type = wav.parent.name
        audio = load_wav(wav)
        windows = slice_windows(audio, args.window_s, sample_rate)
        if args.limit:
            windows = windows[: args.limit]
        for level in ("recorded", "speech"):
            for index, window in enumerate(windows):
                clip = window if level == "recorded" else scale_to_dbfs(window, SPEECH_RMS_DBFS)
                vad_pass = vad_would_form_utterance(vad, clip, config)
                off_text = raw_transcribe(transcriber, clip)
                on_result = transcriber.transcribe_result(clip, config.language_mode)
                rows.append(
                    {
                        "file": wav.name,
                        "noise_type": noise_type,
                        "level": level,
                        "window": index,
                        "vad_pass": vad_pass,
                        "gate_off_text": off_text,
                        "gate_on_text": on_result.text,
                        "gate_on_reason": on_result.rejection_reason,
                        "phantom_off": bool(vad_pass and off_text),
                        "phantom_on": bool(vad_pass and on_result.text),
                        "asr_only_phantom_off": bool(off_text),
                        "asr_only_phantom_on": bool(on_result.text),
                    }
                )
        print(f"{wav.name}: {len(windows)} windows x 2 levels done", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{stamp}.json"
    payload = {
        "generated_utc": stamp,
        "method": (
            f"noise-only {args.window_s:.0f}s windows from the MS-SNSD subset; "
            "session-modeled: an utterance forms only if a contiguous VAD run >= "
            "vad_onset_ms and total VAD speech >= min_speech_ms (deployed "
            "thresholds); gate off = raw beam decode without thresholds/filters; "
            "gate on = deployed decode path with its filters; phantom = utterance formed "
            "AND non-empty transcript. asr_only_* columns ignore the VAD stage."
        ),
        "vad_threshold": config.vad_threshold,
        "vad_onset_ms": config.vad_onset_ms,
        "min_speech_ms": config.min_speech_ms,
        "asr_model": config.asr_model,
        "asr_compute_type": config.asr_compute_type,
        "asr_beam_size": config.asr_beam_size,
        "language_mode": config.language_mode,
        "speech_level_dbfs": SPEECH_RMS_DBFS,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nRaw artifact: {out_path}\n")

    print("| Noise type | Level | n | VAD-pass | Phantom, gate OFF | Phantom, gate ON |")
    print("|---|---|---:|---:|---:|---:|")
    keys = sorted({(r["noise_type"], r["level"]) for r in rows})
    for noise_type, level in keys:
        cell = [r for r in rows if r["noise_type"] == noise_type and r["level"] == level]
        n = len(cell)
        vp = sum(r["vad_pass"] for r in cell)
        off = sum(r["phantom_off"] for r in cell)
        on = sum(r["phantom_on"] for r in cell)
        print(
            f"| {noise_type} | {level} | {n} | {vp} | {off} ({off / n:.0%}) | {on} ({on / n:.0%}) |"
        )
    total = len(rows)
    vp_total = sum(r["vad_pass"] for r in rows)
    off_total = sum(r["phantom_off"] for r in rows)
    on_total = sum(r["phantom_on"] for r in rows)
    print(
        f"| **all** | both | {total} | {vp_total} | **{off_total} ({off_total / total:.0%})** "
        f"| **{on_total} ({on_total / total:.0%})** |"
    )
    asr_off = sum(r["asr_only_phantom_off"] for r in rows)
    asr_on = sum(r["asr_only_phantom_on"] for r in rows)
    print(
        f"\nTranscriber-only (VAD bypassed): gate OFF {asr_off}/{total} "
        f"({asr_off / total:.0%}), gate ON {asr_on}/{total} ({asr_on / total:.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
