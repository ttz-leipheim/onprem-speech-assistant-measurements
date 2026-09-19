#!/usr/bin/env python
"""Measure time from end of speech to first reply audio, against a running daemon.

The harness streams 16 kHz mono PCM16 WAV files into ``/speech`` with realistic
20 ms pacing, records server/client event timings, and reports the stage
breakdown emitted by the server's ``turn_metrics`` event.

Usage:
    python probes/bench_latency.py --runs 3
    python probes/bench_latency.py --no-ptt --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import ssl
import sys
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import websockets

DEFAULT_AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "queries" / "audio"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STAGE_COLUMNS = (
    ("asr_ms", "ASR"),
    ("retrieval_ms", "retrieval"),
    ("llm_first_token_ms", "LLM first token"),
    ("tts_first_audio_ms", "TTS first audio"),
    ("e2e_first_audio_ms", "end-to-end first audio"),
)


@dataclass
class AudioCase:
    path: Path
    duration_ms: int
    pcm: bytes


@dataclass
class TurnResult:
    audio_file: str
    run: int
    duration_ms: int
    over_threshold: bool
    final_metrics: dict[str, Any] | None
    events: list[dict[str, Any]]


def load_wav(path: Path) -> AudioCase:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        pcm = handle.readframes(frames)
    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise ValueError(f"{path} must be 16 kHz mono PCM16 WAV")
    duration_ms = round(frames * 1000 / sample_rate)
    return AudioCase(path=path, duration_ms=duration_ms, pcm=pcm)


def load_audio_cases(audio_dir: Path, *, limit: int | None) -> list[AudioCase]:
    paths = sorted(audio_dir.glob("*.wav"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(
            f"No WAV files found in {audio_dir}. Generate bootstrap audio with "
            "`python data/queries/make_eval_audio.py` or pass --audio-dir."
        )
    return [load_wav(path) for path in paths]


def chunk_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[index : index + chunk_size] for index in range(0, len(data), chunk_size)]


async def stream_pcm(
    websocket: Any,
    pcm: bytes,
    *,
    frame_ms: int,
    silence_ms: int,
) -> None:
    bytes_per_ms = 16_000 * 2 / 1000
    frame_size = max(2, int(bytes_per_ms * frame_ms))
    frame_size += frame_size % 2
    sleep_s = frame_ms / 1000
    for frame in chunk_bytes(pcm, frame_size):
        await websocket.send(frame)
        await asyncio.sleep(sleep_s)
    silence_frame = b"\x00\x00" * int(16_000 * frame_ms / 1000)
    for _ in range(math.ceil(silence_ms / frame_ms)):
        await websocket.send(silence_frame)
        await asyncio.sleep(sleep_s)


def event_record(message: str) -> dict[str, Any] | None:
    received_at = time.time()
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    record = {
        "type": payload.get("type"),
        "server_timestamp": payload.get("timestamp"),
        "client_received_at": received_at,
    }
    if payload.get("type") == "turn_metrics":
        record["turn_metrics"] = {k: v for k, v in payload.items() if k != "type"}
    elif payload.get("type") in {"error", "input_error"}:
        record["message"] = payload.get("message")
    return record


async def receive_events(websocket: Any, stop: asyncio.Event, events: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        if not isinstance(message, str):
            continue
        record = event_record(message)
        if record is None:
            continue
        events.append(record)
        metrics = record.get("turn_metrics")
        if record["type"] == "turn_metrics" and metrics and metrics.get("final"):
            stop.set()
        elif record["type"] in {"error", "input_error"}:
            stop.set()


async def run_turn(
    websocket: Any,
    case: AudioCase,
    *,
    run_index: int,
    ptt: bool,
    frame_ms: int,
    silence_ms: int,
    timeout_s: float,
) -> TurnResult:
    events: list[dict[str, Any]] = []
    stop = asyncio.Event()
    receiver = asyncio.create_task(receive_events(websocket, stop, events))
    try:
        if ptt:
            await websocket.send(json.dumps({"type": "start_recording"}))
        else:
            await websocket.send(json.dumps({"type": "activate"}))
        await stream_pcm(websocket, case.pcm, frame_ms=frame_ms, silence_ms=0 if ptt else silence_ms)
        if ptt:
            await websocket.send(json.dumps({"type": "stop_recording"}))
        await asyncio.wait_for(stop.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        stop.set()
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver

    final_metrics = None
    for record in reversed(events):
        metrics = record.get("turn_metrics")
        if metrics and metrics.get("final"):
            final_metrics = metrics
            break
    e2e_ms = final_metrics.get("e2e_first_audio_ms") if final_metrics else None
    return TurnResult(
        audio_file=case.path.name,
        run=run_index,
        duration_ms=case.duration_ms,
        over_threshold=bool(e2e_ms is not None and e2e_ms > 1500),
        final_metrics=final_metrics,
        events=events,
    )


async def run_benchmark(args: argparse.Namespace) -> list[TurnResult]:
    cases = load_audio_cases(Path(args.audio_dir), limit=args.limit)
    turns: list[TurnResult] = []
    connect_kwargs = websocket_connect_kwargs(args.url)
    async with websockets.connect(args.url, max_size=None, **connect_kwargs) as websocket:
        for case in cases:
            for run_index in range(1, args.runs + 1):
                result = await run_turn(
                    websocket,
                    case,
                    run_index=run_index,
                    ptt=args.ptt,
                    frame_ms=args.frame_ms,
                    silence_ms=args.silence_ms,
                    timeout_s=args.timeout_s,
                )
                turns.append(result)
                await websocket.send(json.dumps({"type": "reset"}))
                await asyncio.sleep(args.reset_pause_s)
    return turns


# Nearest-rank percentile, not interpolated. At n=60 an interpolating
# definition differs by a few milliseconds; stated here so a recomputation from
# the raw file matches.
def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1))
    return ordered[index]


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def websocket_connect_kwargs(url: str) -> dict[str, Any]:
    """Accept a self-signed certificate, on loopback WSS only."""
    parsed = urlparse(url)
    if parsed.scheme != "wss" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return {}
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return {"ssl": context}


def stage_summary(turns: list[TurnResult]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for field, _label in STAGE_COLUMNS:
        values = [
            float(turn.final_metrics[field])
            for turn in turns
            if turn.final_metrics is not None and field in turn.final_metrics
        ]
        summary[field] = {
            "n": len(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "max_ms": max(values) if values else None,
        }
    return summary


def markdown_table(
    turns: list[TurnResult], summary: dict[str, dict[str, float | int | None]]
) -> str:
    lines = [
        "| Stage | n | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, label in STAGE_COLUMNS:
        stats = summary[field]
        lines.append(
            f"| {label} | {stats['n']} | {fmt_ms(stats['p50_ms'])} | "
            f"{fmt_ms(stats['p95_ms'])} | {fmt_ms(stats['max_ms'])} |"
        )
    flagged = sum(1 for turn in turns if turn.over_threshold)
    lines.append("")
    lines.append(f"Turns over 1.5 s end-to-end first audio: {flagged}/{len(turns)}")
    return "\n".join(lines)


def write_results(args: argparse.Namespace, turns: list[TurnResult], table: str) -> Path:
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = output_dir / f"{stamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "ptt": args.ptt,
        "runs": args.runs,
        "frame_ms": args.frame_ms,
        "silence_ms": args.silence_ms,
        "turn_count": len(turns),
        "summary": stage_summary(turns),
        "markdown_table": table,
        "turns": [asdict(turn) for turn in turns],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="wss://127.0.0.1:8765/speech", help="speech WebSocket URL")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="directory of 16 kHz mono PCM16 WAVs")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="directory for raw JSON output")
    parser.add_argument("--runs", type=int, default=3, help="runs per WAV file")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on number of WAV files")
    parser.add_argument(
        "--ptt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use deterministic start_recording/stop_recording framing (default: true)",
    )
    parser.add_argument("--frame-ms", type=int, default=20, help="streaming frame size and pacing")
    parser.add_argument("--silence-ms", type=int, default=1500, help="trailing silence for VAD endpointing")
    parser.add_argument("--timeout-s", type=float, default=45.0, help="per-turn timeout")
    parser.add_argument("--reset-pause-s", type=float, default=0.2, help="pause after reset between turns")
    args = parser.parse_args()

    turns = asyncio.run(run_benchmark(args))
    summary = stage_summary(turns)
    table = markdown_table(turns, summary)
    print(table)
    path = write_results(args, turns, table)
    print(f"\nRaw JSON: {path}")
    missing = [turn for turn in turns if turn.final_metrics is None]
    if missing:
        print(f"\n{len(missing)} turn(s) did not produce final turn_metrics.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
