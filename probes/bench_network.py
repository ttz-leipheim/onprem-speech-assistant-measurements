#!/usr/bin/env python3
"""Run the timing probes while recording what the assistant sends off site.

Two modes. Neither changes the probes it runs.

  online    uplink up. Records the daemon's off-site sockets while the probes
            run.
  offline   the operator has already closed every path off the host. Refuses to
            start while a default route, an overlay or a wireless interface
            remains, and refuses while the assistant can still reach a known
            off-site address. Reachability is re-checked during the run, because
            a path can come back mid-run.

This script never changes routing. On a shared host that is the operator's call.

    # online
    python probes/bench_network.py --mode online --lan 10.0.0.0/8

    # offline, after the operator has removed the default route and taken
    # down any overlay interface
    python probes/bench_network.py --mode offline --lan 10.0.0.0/8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # probes/
ROOT = HERE.parent                               # repository root
DEFAULT_OUT = ROOT / "results"
WATCHER = HERE / "net_watch.py"

# Deployment-specific. Override on the command line, or edit these three.
DEFAULT_CONTAINER = "speech-assistant"   # container the assistant runs in
CONTAINER_PYTHON = "python3"             # interpreter inside that container
SANDBOX_TABLE = "probe_sandbox"          # nftables table sandbox_egress.sh arms


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# Overlays are matched by name: the kernel exposes no flag meaning "tunnel to
# elsewhere". The prefixes cover Tailscale, WireGuard, tun devices and PPP. An
# unusually named tunnel is missed here, which is why the run also asks the
# assistant directly whether it can reach anything off site.
def wireless_interfaces() -> list[str]:
    """Interface names the kernel reports as wireless."""
    names = []
    base = Path("/sys/class/net")
    if base.exists():
        for iface in sorted(p.name for p in base.iterdir()):
            if (base / iface / "wireless").exists() or (base / iface / "phy80211").exists():
                names.append(iface)
    return names


def network_state() -> dict:
    """Every path off the host: wired ports, wireless adapters, overlay routes.

    Listed in full so an offline run can be checked rather than asserted.
    """
    v4 = run(["ip", "-4", "route", "show", "table", "all"]).stdout
    v6 = run(["ip", "-6", "route", "show", "table", "all"]).stdout
    links = run(["ip", "-br", "addr"]).stdout
    rules = run(["ip", "rule", "show"]).stdout

    def defaults(text):
        return [l.strip() for l in text.splitlines()
                if l.startswith("default") or l.startswith("::/0")]

    up_with_address, overlays, wireless_up = [], [], []
    wireless = set(wireless_interfaces())
    for line in links.splitlines():
        parts = line.split()
        if not parts:
            continue
        name, state, addrs = parts[0], parts[1], parts[2:]
        if state == "DOWN":
            continue
        if name.startswith(("tailscale", "wg", "tun", "ppp")):
            overlays.append(name)
        if name in wireless:
            wireless_up.append(name)
        if addrs and not name.startswith(("docker", "br-", "veth", "lo", "virbr")):
            up_with_address.append(f"{name} {' '.join(addrs)}")

    return {
        "default_routes_v4": defaults(v4),
        "default_routes_v6": defaults(v6),
        "overlay_interfaces_up": overlays,
        "wireless_interfaces": sorted(wireless),
        "wireless_up": wireless_up,
        "external_interfaces_with_address": up_with_address,
        "policy_rules": [l.strip() for l in rules.splitlines() if l.strip()],
        "routes_v4": [l.strip() for l in v4.splitlines() if l.strip()],
    }


def egress_check(container: str, targets: list[str],
                 container_python: str = CONTAINER_PYTHON) -> dict:
    """Ask the assistant's own container whether it can reach anything off site."""
    script = (
        "import socket,json,sys\n"
        "out={}\n"
        "for t in sys.argv[1:]:\n"
        "    h,_,p=t.partition(':')\n"
        "    s=socket.socket(); s.settimeout(4)\n"
        "    try: s.connect((h,int(p or 443))); out[t]=True\n"
        "    except OSError: out[t]=False\n"
        "    finally: s.close()\n"
        "print(json.dumps(out))\n"
    )
    proc = run(["docker", "exec", "-i", container, container_python, "-c",
                script, *targets])
    try:
        return json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"error": proc.stderr[-400:]}


def sandbox_counters(table: str = SANDBOX_TABLE) -> dict:
    """Packets the assistant sent toward an address off site, as counted by the
    firewall rule scoped to its control group. Absent means no rule is armed."""
    proc = run(["nft", "-j", "list", "table", "inet", table])
    if proc.returncode != 0:
        detail = proc.stderr.strip()[-200:]
        if "must be root" in detail or "not permitted" in detail:
            detail = ("reading the ruleset needs root; run the whole experiment "
                      "with: sudo probes/sandbox_egress.sh run")
        return {"armed": False, "detail": detail}
    try:
        items = json.loads(proc.stdout)["nftables"]
    except (json.JSONDecodeError, KeyError):
        return {"armed": False, "detail": "could not parse the ruleset"}
    rules = []
    for item in items:
        rule = item.get("rule")
        if not rule:
            continue
        counter = next((e["counter"] for e in rule.get("expr", []) if "counter" in e), None)
        verdict = next((e for e in rule.get("expr", []) if "drop" in e or "accept" in e), None)
        if counter:
            rules.append({"packets": counter.get("packets"), "bytes": counter.get("bytes"),
                          "drops": verdict is not None and "drop" in verdict})
    return {"armed": True, "rules": rules,
            "packets_total": sum(r["packets"] or 0 for r in rules),
            "enforcing": any(r["drops"] for r in rules)}


def container_env(container: str) -> dict:
    env = run(["docker", "exec", container, "env"]).stdout.splitlines()
    keep = ("HF_", "TRANSFORMERS_", "no_proxy", "NO_PROXY", "http_proxy", "https_proxy")
    return {k: v for k, _, v in (e.partition("=") for e in env) if k.startswith(keep)}


def stop_file_for(path: str) -> str:
    return path + ".stop"


def start_watcher(container: str, seconds: float, lan: list[str], path: str,
                  probes: list[str],
                  container_python: str = CONTAINER_PYTHON) -> None:
    run(["docker", "cp", str(WATCHER), f"{container}:/tmp/net_watch.py"])
    run(["docker", "exec", container, "sh", "-c", f"rm -f {stop_file_for(path)}"])
    cmd = ["docker", "exec", "-d", container, container_python, "/tmp/net_watch.py",
           "--seconds", str(seconds), "--out", path,
           "--stop-file", stop_file_for(path),
           "--allow", "127.0.0.0/8", "--allow", "::1/128"]
    for net in lan:
        cmd += ["--allow", net]
    for target in probes:
        cmd += ["--probe", target]
    run(cmd)


def collect_watcher(container: str, path: str, timeout_s: float = 120.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        size = run(["docker", "exec", container, "stat", "-c", "%s", path]).stdout.strip()
        if size.isdigit() and int(size) > 0:
            return json.loads(run(["docker", "exec", container, "cat", path]).stdout)
        time.sleep(5)
    return {"error": "watcher produced no output"}


def probe(script: str, extra: list[str]) -> dict:
    """Run one of the existing probes unchanged and return how it went."""
    cmd = [sys.executable, str(HERE / script)] + extra
    started = time.time()
    proc = run(cmd, cwd=ROOT)
    return {
        "script": script,
        "args": extra,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("online", "offline", "sandbox"), required=True)
    ap.add_argument("--lan", action="append", default=[],
                    help="subnet that counts as on site; repeatable")
    ap.add_argument("--container", default=DEFAULT_CONTAINER,
                    help="name of the container the assistant runs in")
    ap.add_argument("--container-python", default=CONTAINER_PYTHON,
                    help="interpreter inside that container")
    ap.add_argument("--runs", type=int, default=3, help="latency runs per query file")
    ap.add_argument("--bargein-runs", type=int, default=30)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--watch-seconds", type=float, default=2400.0)
    ap.add_argument("--probe", action="append",
                    default=["1.1.1.1:443", "8.8.8.8:53", "9.9.9.9:443"],
                    help="address off site to try reaching, host:port; repeatable")
    args = ap.parse_args()

    state = network_state()
    before = egress_check(args.container, args.probe, args.container_python)
    counters_before = sandbox_counters()

    if args.mode == "sandbox":
        problems = []
        if not counters_before.get("armed"):
            problems.append("no sandbox rule is armed "
                            f"({counters_before.get('detail', 'unknown')})")
        elif not counters_before.get("enforcing"):
            problems.append("the sandbox is armed but only counting; "
                            "arm it with 'enforce' to make this a test")
        reached = [t for t, ok in before.items() if ok is True]
        if reached:
            problems.append(f"the assistant still reaches: {reached}")
        if problems:
            print("REFUSING: the assistant is not confined.")
            for p in problems:
                print(f"  - {p}")
            print("\nArm it first:")
            print(f"  sudo probes/sandbox_egress.sh enforce {args.container}")
            return 1

    if args.mode == "offline":
        problems = []
        for route in state["default_routes_v4"] + state["default_routes_v6"]:
            problems.append(f"a default route is still present: {route}")
        if state["overlay_interfaces_up"]:
            problems.append(f"overlay interfaces are up: {state['overlay_interfaces_up']}")
        if state["wireless_up"]:
            problems.append(f"wireless interfaces are up: {state['wireless_up']}")
        reached = [t for t, ok in before.items() if ok is True]
        if reached:
            problems.append(f"the assistant still reaches: {reached}")
        if problems:
            print("REFUSING: the host still has a way off site.")
            for p in problems:
                print(f"  - {p}")
            print("\nEvery interface with an address:")
            for line in state["external_interfaces_with_address"]:
                print(f"    {line}")
            print("\nClose them first, for example:")
            print("  sudo ip route del default")
            for name in state["overlay_interfaces_up"] + state["wireless_up"]:
                print(f"  sudo ip link set {name} down")
            return 1

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    watch_path = f"/tmp/net_watch-{stamp}.json"
    print(f"mode={args.mode}")
    print(f"  default routes : {state['default_routes_v4'] + state['default_routes_v6'] or 'none'}")
    print(f"  overlays up    : {state['overlay_interfaces_up'] or 'none'}")
    print(f"  wireless up    : {state['wireless_up'] or 'none'}")
    print(f"  reaches off site: {[t for t, ok in before.items() if ok is True] or 'nothing'}")
    if counters_before.get("armed"):
        print(f"  sandbox        : armed, "
              f"{'dropping' if counters_before.get('enforcing') else 'counting only'}, "
              f"{counters_before.get('packets_total')} packets so far")

    start_watcher(args.container, args.watch_seconds, args.lan, watch_path,
                  args.probe, args.container_python)
    probes = [
        probe("bench_latency.py", ["--runs", str(args.runs)]),
        probe("bench_bargein.py", ["--runs", str(args.bargein_runs)]),
    ]
    # The container has no process tools, so the watcher is asked to stop by
    # creating a file it checks, using a shell redirect rather than any binary.
    run(["docker", "exec", args.container, "sh", "-c",
         f": > {stop_file_for(watch_path)}"])
    watch = collect_watcher(args.container, watch_path)
    after = egress_check(args.container, args.probe, args.container_python)
    state_after = network_state()
    counters_after = sandbox_counters()

    artifact = {
        "kind": "network",
        "mode": args.mode,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "network_state_before": state,
        "network_state_after": state_after,
        "egress_check_before": before,
        "egress_check_after": after,
        "sandbox_counters_before": counters_before,
        "sandbox_counters_after": counters_after,
        "probe_targets": args.probe,
        "container_env": container_env(args.container),
        "on_site_subnets": ["127.0.0.0/8", "::1/128"] + args.lan,
        "sockets": watch,
        "probes": probes,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"network-{args.mode}-{stamp}.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if "error" in watch or "offsite_peers" not in watch:
        print(f"\n  WATCHER FAILED: {watch.get('error', 'no data')}")
        print("  No socket evidence for this run; the absence of peers below "
              "is not a result.")
        offsite = None
    else:
        offsite = watch["offsite_peers"]
        print(f"  reachability probes during the run: "
              f"{watch.get('reachability_successes', '?')} of "
              f"{watch.get('reachability_attempts', '?')} reached off site")
    print(f"\nwrote {out}")
    print(f"  probes: {[ (p['script'], p['returncode']) for p in probes ]}")
    if counters_after.get("armed"):
        sent = (counters_after.get("packets_total") or 0) - (counters_before.get("packets_total") or 0)
        print(f"  packets the assistant aimed off site during this run: {sent}")
    if offsite is None:
        return 1
    print(f"  off-site peers held by the daemon: {len(offsite)}")
    for row in offsite:
        print(f"    {row['peer']}:{row['port']} in {row['samples']} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
