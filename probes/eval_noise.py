#!/usr/bin/env python
"""Evaluate wake-word and recognizer robustness against shop-floor-like noise.

Mixes clean 16 kHz mono PCM16 WAVs with the MS-SNSD noise subset at fixed SNR
levels, runs the wake-word detector and Faster Whisper directly, writes raw JSON
to ``results/``, and prints a Markdown table.

Requires the assistant's Python package on the import path. It measures the
deployed detector and recognizer at their configured thresholds, so it runs on
the deployment rather than from a clone; the result file it produced is in
``results/``.

Usage:
    python probes/fetch_noise.py
    python probes/eval_noise.py
    python probes/eval_noise.py --limit 3 --skip-wake
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The deployment's own components, bound through probes/deployment.py.
from deployment import (  # noqa: E402
    Config,
    MissingSpeechDependencyError,
    Transcriber,
    WakeWord,
    float32_to_pcm16_bytes,
    pcm16_bytes_to_float32,
)
from eval_common import word_error_rate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASR_AUDIO_DIR = ROOT / "data" / "queries" / "audio"
DEFAULT_BOOTSTRAP_AUDIO_DIR = ROOT / "data" / "queries" / "audio"
DEFAULT_REFERENCES = ROOT / "data" / "queries" / "prompts.jsonl"
DEFAULT_WAKE_DIR = ROOT / "data" / "queries" / "wake"
DEFAULT_NOISE_DIR = ROOT / "data" / "noise" / "audio"
DEFAULT_MIXED_DIR = ROOT / "data" / "noise" / "mixed"
DEFAULT_RESULTS_DIR = ROOT / "results"
SAMPLE_RATE = 16_000
SNR_LEVELS = (20, 10, 5, 0)


@dataclass
class CleanCase:
    id: str
    path: Path
    reference: str
    audio: np.ndarray


@dataclass
class NoiseFile:
    label: str
    path: Path
    audio: np.ndarray


@dataclass
class AsrResult:
    clean_id: str
    noise: str
    snr_db: int | None
    mixed_path: str | None
    reference: str
    transcript: str
    wer: float
    transcribe_ms: float
    audio_duration_s: float
    realtime_factor: float | None


@dataclass
class WakeResult:
    wake_file: str
    noise: str
    snr_db: int | None
    detected: bool


@dataclass
class FalseAcceptResult:
    noise: str
    evaluated_seconds: float
    false_accepts: int
    false_accepts_per_hour: float


def _self_check_wer() -> None:
    assert word_error_rate("a b c", "a b c") == 0.0
    assert math.isclose(word_error_rate("a b c", "a c"), 1 / 3)
    assert math.isclose(word_error_rate("a b", "a x b"), 1 / 2)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        pcm = handle.readframes(frames)
    if channels != 1 or sample_rate != SAMPLE_RATE or sample_width != 2:
        raise ValueError(f"{path} must be 16 kHz mono PCM16 WAV")
    return pcm16_bytes_to_float32(pcm)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(float32_to_pcm16_bytes(audio))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def repeat_noise(noise: np.ndarray, length: int, *, offset_seed: str = "") -> np.ndarray:
    if noise.size == 0:
        raise ValueError("noise file is empty")
    if noise.size >= length:
        max_offset = noise.size - length
        offset = 0 if max_offset == 0 else sum(offset_seed.encode("utf-8")) % (max_offset + 1)
        return noise[offset : offset + length]
    repeats = math.ceil(length / noise.size)
    return np.tile(noise, repeats)[:length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if clean_rms == 0.0:
        return clean.copy()
    if noise_rms == 0.0:
        raise ValueError("cannot mix silent noise")
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return np.clip(clean + scaled_noise, -1.0, 1.0).astype(np.float32)


def load_references(path: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    if not path.exists():
        raise SystemExit(f"Reference file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        utterance_id = str(raw.get("utterance_id") or raw.get("id") or raw.get("file") or "")
        text = str(raw.get("reference") or raw.get("text") or raw.get("transcript") or "")
        if utterance_id and text:
            references[utterance_id] = text
    if not references:
        raise SystemExit(f"No references found in {path}")
    return references


def id_from_wav(path: Path, references: dict[str, str]) -> str | None:
    stem = path.stem
    if stem in references:
        return stem
    without_index = re.sub(r"^\d+[_-]", "", stem)
    if without_index in references:
        return without_index
    for reference_id in references:
        if reference_id in stem:
            return reference_id
    return None


def choose_clean_dir(path: Path) -> Path:
    if path.exists() and list(path.glob("**/*.wav")):
        return path
    if DEFAULT_BOOTSTRAP_AUDIO_DIR.exists() and list(DEFAULT_BOOTSTRAP_AUDIO_DIR.glob("*.wav")):
        print(
            f"No WAVs found in {path}; using bootstrap audio in {DEFAULT_BOOTSTRAP_AUDIO_DIR}.",
            file=sys.stderr,
        )
        return DEFAULT_BOOTSTRAP_AUDIO_DIR
    return path


def load_clean_cases(audio_dir: Path, references: dict[str, str], *, limit: int | None) -> list[CleanCase]:
    paths = sorted(audio_dir.glob("**/*.wav"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(
            f"No clean WAV files found in {audio_dir}. Generate bootstrap audio with "
            "`python data/queries/make_eval_audio.py` or pass --clean-audio-dir."
        )
    cases: list[CleanCase] = []
    missing: list[str] = []
    for path in paths:
        clean_id = id_from_wav(path, references)
        if clean_id is None:
            missing.append(path.name)
            continue
        cases.append(CleanCase(clean_id, path, references[clean_id], load_wav(path)))
    if missing:
        raise SystemExit(f"Missing reference transcripts for: {', '.join(missing)}")
    return cases


def load_noise_files(noise_dir: Path, *, limit_per_type: int | None) -> list[NoiseFile]:
    if not noise_dir.exists():
        raise SystemExit(
            f"No noise directory found at {noise_dir}. Run `python probes/fetch_noise.py` first."
        )
    files: list[NoiseFile] = []
    for label_dir in sorted(path for path in noise_dir.iterdir() if path.is_dir()):
        paths = sorted(label_dir.glob("*.wav"))
        if limit_per_type is not None:
            paths = paths[:limit_per_type]
        for path in paths:
            files.append(NoiseFile(label_dir.name, path, load_wav(path)))
    if not files:
        raise SystemExit(
            f"No noise WAVs found in {noise_dir}. Run `python probes/fetch_noise.py` first."
        )
    return files


def cached_mix(clean: CleanCase, noise: NoiseFile, snr_db: int, mixed_dir: Path) -> tuple[np.ndarray, Path]:
    safe_noise = noise.path.stem.replace(" ", "_")
    output = mixed_dir / noise.label / f"{clean.path.stem}__{safe_noise}__snr{snr_db}.wav"
    if output.exists():
        return load_wav(output), output
    noise_audio = repeat_noise(noise.audio, clean.audio.size, offset_seed=clean.id + noise.path.name)
    mixed = mix_at_snr(clean.audio, noise_audio, snr_db)
    write_wav(output, mixed)
    return mixed, output


def transcribe_cases(
    cases: list[CleanCase],
    noises: list[NoiseFile],
    *,
    transcriber: Transcriber,
    mixed_dir: Path,
    snr_levels: tuple[int, ...],
    language: str,
) -> list[AsrResult]:
    results: list[AsrResult] = []
    for case in cases:
        for noise_label, audio, snr_db, mixed_path in [("clean", case.audio, None, None)]:
            start = time.monotonic()
            transcript = transcriber.transcribe(audio, language)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            duration_s = audio.size / SAMPLE_RATE
            results.append(
                AsrResult(
                    clean_id=case.id,
                    noise=noise_label,
                    snr_db=snr_db,
                    mixed_path=mixed_path,
                    reference=case.reference,
                    transcript=transcript,
                    wer=word_error_rate(case.reference, transcript),
                    transcribe_ms=elapsed_ms,
                    audio_duration_s=duration_s,
                    realtime_factor=duration_s / (elapsed_ms / 1000.0) if elapsed_ms > 0 else None,
                )
            )
        for noise in noises:
            for snr_db in snr_levels:
                audio, path = cached_mix(case, noise, snr_db, mixed_dir)
                start = time.monotonic()
                transcript = transcriber.transcribe(audio, language)
                elapsed_ms = (time.monotonic() - start) * 1000.0
                duration_s = audio.size / SAMPLE_RATE
                results.append(
                    AsrResult(
                        clean_id=case.id,
                        noise=noise.label,
                        snr_db=snr_db,
                        mixed_path=str(path.relative_to(ROOT)),
                        reference=case.reference,
                        transcript=transcript,
                        wer=word_error_rate(case.reference, transcript),
                        transcribe_ms=elapsed_ms,
                        audio_duration_s=duration_s,
                        realtime_factor=duration_s / (elapsed_ms / 1000.0) if elapsed_ms > 0 else None,
                    )
                )
    return results


def detect_wake(factory: type[WakeWord], config: Config, audio: np.ndarray) -> bool:
    detector = factory(config)
    frame_size = int(SAMPLE_RATE * 0.08)
    for start in range(0, audio.size, frame_size):
        if detector.is_wake_word(audio[start : start + frame_size]):
            return True
    return False


def evaluate_wake(
    wake_files: list[Path],
    noises: list[NoiseFile],
    *,
    config: Config,
    mixed_dir: Path,
    snr_levels: tuple[int, ...],
) -> list[WakeResult]:
    results: list[WakeResult] = []
    for path in wake_files:
        clean = load_wav(path)
        results.append(WakeResult(path.name, "clean", None, detect_wake(WakeWord, config, clean)))
        for noise in noises:
            clean_case = CleanCase(path.stem, path, "", clean)
            for snr_db in snr_levels:
                audio, _mixed_path = cached_mix(clean_case, noise, snr_db, mixed_dir)
                detected = detect_wake(WakeWord, config, audio)
                results.append(WakeResult(path.name, noise.label, snr_db, detected))
    return results


def evaluate_false_accepts(
    noises: list[NoiseFile],
    *,
    config: Config,
    minutes: float,
) -> list[FalseAcceptResult]:
    target_samples = int(minutes * 60 * SAMPLE_RATE)
    if target_samples <= 0:
        return []
    results: list[FalseAcceptResult] = []
    frame_size = int(SAMPLE_RATE * 0.08)
    for noise in noises:
        audio = repeat_noise(noise.audio, target_samples, offset_seed=noise.path.name)
        detector = WakeWord(config)
        accepts = 0
        for start in range(0, audio.size, frame_size):
            if detector.is_wake_word(audio[start : start + frame_size]):
                accepts += 1
        hours = audio.size / SAMPLE_RATE / 3600
        results.append(FalseAcceptResult(noise.label, audio.size / SAMPLE_RATE, accepts, accepts / hours))
    return results


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}"


def summarize_asr(results: list[AsrResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, list[float]]] = {}
    for result in results:
        key = "clean" if result.snr_db is None else str(result.snr_db)
        summary.setdefault(key, {}).setdefault(result.noise, []).append(result.wer)
    return {
        snr: {noise: mean(values) or 0.0 for noise, values in noise_values.items()}
        for snr, noise_values in summary.items()
    }


def summarize_wake(results: list[WakeResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, list[float]]] = {}
    for result in results:
        key = "clean" if result.snr_db is None else str(result.snr_db)
        bucket = summary.setdefault(key, {}).setdefault(result.noise, [])
        bucket.append(1.0 if result.detected else 0.0)
    return {
        snr: {noise: mean(values) or 0.0 for noise, values in noise_values.items()}
        for snr, noise_values in summary.items()
    }


def markdown_table(asr_summary: dict[str, dict[str, float]], wake_summary: dict[str, dict[str, float]]) -> str:
    noise_labels = sorted(
        {
            label
            for summary in (asr_summary, wake_summary)
            for by_noise in summary.values()
            for label in by_noise
            if label != "clean"
        }
    )
    header = ["SNR dB", "clean WER %"]
    for label in noise_labels:
        header.extend([f"{label} wake %", f"{label} WER %"])
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    row_keys = ["clean", *[str(snr) for snr in SNR_LEVELS]]
    for key in row_keys:
        clean_wer = asr_summary.get("clean", {}).get("clean")
        row = ["clean" if key == "clean" else key, fmt_pct(clean_wer)]
        for label in noise_labels:
            wake_value = wake_summary.get(key, {}).get(label)
            wer_value = asr_summary.get(key, {}).get(label)
            row.extend([fmt_pct(wake_value), fmt_pct(wer_value)])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_results(
    args: argparse.Namespace,
    asr_results: list[AsrResult],
    wake_results: list[WakeResult],
    false_accepts: list[FalseAcceptResult],
    table: str,
) -> Path:
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = output_dir / f"{stamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clean_audio_dir": args.clean_audio_dir,
        "references": args.references,
        "noise_dir": args.noise_dir,
        "wake_dir": args.wake_dir,
        "snr_levels": list(SNR_LEVELS),
        "asr_summary": summarize_asr(asr_results),
        "wake_summary": summarize_wake(wake_results),
        "markdown_table": table,
        "asr_results": [asdict(result) for result in asr_results],
        "wake_results": [asdict(result) for result in wake_results],
        "false_accepts": [asdict(result) for result in false_accepts],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    _self_check_wer()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-audio-dir", default=str(DEFAULT_ASR_AUDIO_DIR), help="clean query WAV directory")
    parser.add_argument("--references", default=str(DEFAULT_REFERENCES), help="JSONL reference transcripts")
    parser.add_argument("--wake-dir", default=str(DEFAULT_WAKE_DIR), help="wake-word WAV directory")
    parser.add_argument("--noise-dir", default=str(DEFAULT_NOISE_DIR), help="noise WAV directory")
    parser.add_argument("--mixed-dir", default=str(DEFAULT_MIXED_DIR), help="cache directory for mixed WAVs")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="directory for raw JSON output")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on clean query WAVs")
    parser.add_argument("--limit-noise-per-type", type=int, default=None, help="optional cap on files per noise type")
    parser.add_argument("--language", default="de", choices=("de", "en", "auto"), help="ASR language mode")
    parser.add_argument("--skip-asr", action="store_true", help="skip ASR scoring")
    parser.add_argument("--skip-wake", action="store_true", help="skip wake-word scoring")
    parser.add_argument(
        "--false-accept-minutes",
        type=float,
        default=30.0,
        help="pure-noise duration per noise file for false-accepts/hour",
    )
    args = parser.parse_args()

    references = load_references(Path(args.references))
    clean_dir = choose_clean_dir(Path(args.clean_audio_dir))
    args.clean_audio_dir = str(clean_dir)
    cases = load_clean_cases(clean_dir, references, limit=args.limit)
    noises = load_noise_files(Path(args.noise_dir), limit_per_type=args.limit_noise_per_type)
    config = Config.from_env()

    asr_results: list[AsrResult] = []
    wake_results: list[WakeResult] = []
    false_accepts: list[FalseAcceptResult] = []
    try:
        if not args.skip_asr:
            transcriber = Transcriber(config)
            asr_results = transcribe_cases(
                cases,
                noises,
                transcriber=transcriber,
                mixed_dir=Path(args.mixed_dir),
                snr_levels=SNR_LEVELS,
                language=args.language,
            )
        if not args.skip_wake:
            wake_files = sorted(Path(args.wake_dir).glob("**/*.wav"))
            if wake_files:
                wake_results = evaluate_wake(
                    wake_files,
                    noises,
                    config=config,
                    mixed_dir=Path(args.mixed_dir),
                    snr_levels=SNR_LEVELS,
                )
                false_accepts = evaluate_false_accepts(
                    noises,
                    config=config,
                    minutes=args.false_accept_minutes,
                )
            else:
                print(f"No wake-word WAVs found in {args.wake_dir}; wake scoring skipped.", file=sys.stderr)
    except MissingSpeechDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    table = markdown_table(summarize_asr(asr_results), summarize_wake(wake_results))
    print(table)
    if false_accepts:
        print("\nFalse accepts per hour:")
        for result in false_accepts:
            print(f"- {result.noise}: {result.false_accepts_per_hour:.2f}")
    path = write_results(args, asr_results, wake_results, false_accepts, table)
    print(f"\nRaw JSON: {path}")


if __name__ == "__main__":
    main()
