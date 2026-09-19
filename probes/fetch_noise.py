#!/usr/bin/env python
"""Fetch the MS-SNSD noise subset and record checksums.

The corpus is Microsoft's MS-SNSD and is NOT redistributed with this
repository, because it carries its own licence. This fetches the exact subset
used, from a pinned commit, into ``data/noise/audio/`` which Git ignores. The
checksum manifest beside it is committed, so a clone can verify it has the same
bytes the published results were measured on.

Usage:
    python probes/fetch_noise.py
    python probes/fetch_noise.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "noise" / "audio"
DEFAULT_MANIFEST = ROOT / "data" / "noise" / "noise_manifest.json"
MS_SNSD_COMMIT = "fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742"
RAW_BASE_URL = f"https://raw.githubusercontent.com/microsoft/MS-SNSD/{MS_SNSD_COMMIT}"


@dataclass(frozen=True)
class NoiseSource:
    label: str
    path: str


NOISE_SOURCES = (
    NoiseSource("ventilation", "noise_test/AirConditioner_1.wav"),
    NoiseSource("ventilation", "noise_test/AirConditioner_2.wav"),
    NoiseSource("machinery", "noise_test/VacuumCleaner_1.wav"),
    NoiseSource("copy_machine", "noise_test/CopyMachine_1.wav"),
    NoiseSource("babble", "noise_test/Babble_1.wav"),
    NoiseSource("babble", "noise_test/Babble_2.wav"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        destination.write_bytes(response.read())


def fetch(output_dir: Path, *, force: bool) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for source in NOISE_SOURCES:
        filename = Path(source.path).name
        destination = output_dir / source.label / filename
        url = f"{RAW_BASE_URL}/{source.path}"
        if force or not destination.exists():
            download(url, destination)
        records.append(
            {
                "label": source.label,
                "source_path": source.path,
                "url": url,
                "local_path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        print(destination)
    return records


def write_manifest(path: Path, output_dir: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "MS-SNSD",
        "repository": "https://github.com/microsoft/MS-SNSD",
        "license": "MIT",
        "source_commit": MS_SNSD_COMMIT,
        "output_dir": str(output_dir.relative_to(ROOT)),
        "files": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_manifest(manifest: Path) -> bool:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    ok = True
    for record in raw.get("files", []):
        path = ROOT / record["local_path"]
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            ok = False
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            print(f"checksum mismatch: {path}", file=sys.stderr)
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="ignored local noise directory")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="checksum manifest to write")
    parser.add_argument("--force", action="store_true", help="re-download files even if present")
    parser.add_argument("--verify", action="store_true", help="verify local files against an existing manifest")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if args.verify:
        if not manifest.exists():
            raise SystemExit(f"Manifest not found: {manifest}")
        raise SystemExit(0 if verify_manifest(manifest) else 1)

    output_dir = Path(args.output_dir)
    records = fetch(output_dir, force=args.force)
    write_manifest(manifest, output_dir, records)
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
