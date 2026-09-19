#!/usr/bin/env python3
"""Sample the sockets held by this process tree and report off-site peers.

Runs inside the daemon's container, where PID 1 is the daemon. Sockets are
attributed by matching the inodes behind the process's open file descriptors
against /proc/net/tcp, which needs no elevated privilege.

Sampling, not packet capture: a connection opened and closed between two
samples is missed. Pair it with the firewall counter in sandbox_egress.sh, which
misses nothing but attributes nothing.

Usage:
    python3 net_watch.py --seconds 1200 --allow 127.0.0.0/8 --allow 10.0.0.0/8
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import struct
import sys
import time


def socket_inodes(pid: int) -> set[int]:
    """Inodes of the sockets this process has open."""
    inodes = set()
    fd_dir = f"/proc/{pid}/fd"
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return inodes
    for name in names:
        try:
            target = os.readlink(f"{fd_dir}/{name}")
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(int(target[8:-1]))
    return inodes


def descendants(pid: int) -> list[int]:
    """The process and its children, read from /proc."""
    found = [pid]
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8") as fh:
                parent = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if parent == pid:
            found.append(int(entry))
    return found


def _addr(hex_addr: str, family: int) -> str:
    """Decode a /proc/net/tcp address field."""
    raw = bytes.fromhex(hex_addr)
    if family == socket.AF_INET:
        return socket.inet_ntop(family, struct.pack("<I", struct.unpack(">I", raw)[0]))
    words = struct.unpack("<4I", raw)
    return socket.inet_ntop(family, struct.pack(">4I", *words))


# TCP only: /proc/net/tcp is what carries a socket inode attributable to a
# process. UDP, a DNS lookup above all, is invisible here. The firewall counter
# in sandbox_egress.sh covers that gap, counting every IP packet regardless of
# protocol. Report the two together.
def connections() -> list[tuple[int, str, int]]:
    """(inode, peer address, peer port) for every TCP connection on the host."""
    out = []
    for path, family in (("/proc/net/tcp", socket.AF_INET),
                         ("/proc/net/tcp6", socket.AF_INET6)):
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            host, port = parts[2].split(":")
            if port == "0000":          # a listening socket has no peer
                continue
            try:
                out.append((int(parts[9]), _addr(host, family), int(port, 16)))
            except (ValueError, OSError):
                continue
    return out


def reachable(target: str, timeout: float = 4.0) -> bool:
    """Can this process open a TCP connection to an address off site?

    Tests the path rather than reading the route table. Called repeatedly, since
    a second path can appear mid-run when an address is handed out.
    """
    host, _, port = target.partition(":")
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, int(port or 443)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=1, help="daemon PID inside this namespace")
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--allow", action="append", default=[],
                    help="subnet whose peers count as on site; repeatable")
    ap.add_argument("--probe", action="append", default=[],
                    help="address off site to try reaching, host:port; repeatable")
    ap.add_argument("--probe-every", type=float, default=30.0,
                    help="seconds between reachability probes")
    ap.add_argument("--stop-file", default="",
                    help="stop as soon as this path exists, and write the output")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    allowed = [ipaddress.ip_network(a) for a in args.allow] or [
        ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128")]
    sink = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")

    started = time.time()
    samples = 0
    offsite: dict[str, dict] = {}
    probes: list[dict] = []
    next_probe = 0.0

    # A stop request breaks the loop rather than killing the process, so the
    # samples taken so far still reach the output file below. The container
    # ships no process tools, so the caller asks for a stop by creating a file
    # rather than by sending a signal.
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while not stopping and time.time() - started < args.seconds:
        if args.stop_file and os.path.exists(args.stop_file):
            stopping = True
            break
        now = time.time()
        if args.probe and now >= next_probe:
            for target in args.probe:
                probes.append({"t": round(now - started, 1), "target": target,
                               "reachable": reachable(target)})
            next_probe = now + args.probe_every
        inodes = set()
        for pid in descendants(args.pid):
            inodes |= socket_inodes(pid)
        for inode, peer, port in connections():
            if inode not in inodes:
                continue
            address = ipaddress.ip_address(peer)
            if any(address in net for net in allowed):
                continue
            key = f"{peer}:{port}"
            row = offsite.setdefault(key, {"peer": peer, "port": port, "samples": 0,
                                           "first_seen": time.time()})
            row["samples"] += 1
            row["last_seen"] = time.time()
        samples += 1
        time.sleep(args.interval)

    succeeded = [p for p in probes if p["reachable"]]
    json.dump({"pid": args.pid, "samples": samples, "interval_s": args.interval,
               "allowed": [str(n) for n in allowed],
               "started_unix": started, "ended_unix": time.time(),
               "offsite_peers": sorted(offsite.values(), key=lambda r: r["peer"]),
               "reachability_probes": probes,
               "reachability_attempts": len(probes),
               "reachability_successes": len(succeeded),
               "reached_off_site": bool(succeeded),
               "stopped_early": stopping},
              sink, indent=2)
    sink.write("\n")
    if sink is not sys.stdout:
        sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
