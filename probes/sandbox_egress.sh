#!/usr/bin/env bash
# Confine the speech assistant to the site and count what it tries to send
# beyond it.
#
# The nftables rule matches the assistant's own cgroup v2 path, so it applies to
# that process tree only: no routes change, no interfaces go down, and other
# services on the host keep their network. On site is loopback plus the private
# ranges; everything else counts as off site, the carrier-grade NAT range
# 100.64.0.0/10 included, so an overlay is not a way around the rule. Deleting the table takes effect at once and leaves nothing to
# restore.
#
#   sudo probes/sandbox_egress.sh observe    # count only, let traffic through
#   sudo probes/sandbox_egress.sh enforce    # count and drop
#   sudo probes/sandbox_egress.sh status     # print the counters
#   sudo probes/sandbox_egress.sh off        # remove the rule
#   sudo probes/sandbox_egress.sh run        # arm, measure, record

set -eu

MODE="${1:-status}"
CONTAINER="${2:-${PROBE_CONTAINER:-speech-assistant}}"
TABLE="probe_sandbox"

ON_SITE_V4="{ 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 }"
ON_SITE_V6="{ ::1/128, fc00::/7, fe80::/10 }"

# The reachability probes run inside the container, so they share the
# assistant's cgroup and the counter cannot distinguish them from the
# assistant's own traffic. They match their own rule first, which drops without
# counting, so the counted packets are the assistant's.
PROBE_TARGETS="{ 1.1.1.1, 8.8.8.8, 9.9.9.9 }"

cgroup_path() {
  local pid
  pid="$(docker inspect "$CONTAINER" --format '{{.State.Pid}}')"
  [ -n "$pid" ] && [ "$pid" != "0" ] || { echo "container $CONTAINER is not running" >&2; exit 1; }
  # /proc/PID/cgroup gives "0::/system.slice/docker-<id>.scope"; nft wants it
  # without the leading slash, and the level is how many components it has.
  cut -d: -f3 "/proc/$pid/cgroup" | sed 's|^/||'
}

remove() {
  nft list table inet "$TABLE" >/dev/null 2>&1 && nft delete table inet "$TABLE"
  echo "sandbox removed; the assistant has its normal network again"
}

arm() {
  local verdict="$1" cg level
  cg="$(cgroup_path)"
  level="$(echo "$cg" | tr -cd '/' | wc -c)"
  level=$((level + 1))
  echo "assistant control group : $cg  (level $level)"
  echo "on site                 : loopback and the private ranges"
  echo "verdict for off-site    : ${verdict:-count only}"

  nft list table inet "$TABLE" >/dev/null 2>&1 && nft delete table inet "$TABLE"
  nft add table inet "$TABLE"
  nft add chain inet "$TABLE" output "{ type filter hook output priority 0; policy accept; }"
  # First: the measurement's own probes, dropped but never counted.
  nft add rule inet "$TABLE" output \
      socket cgroupv2 level "$level" \"$cg\" \
      ip daddr $PROBE_TARGETS drop
  nft add rule inet "$TABLE" output \
      socket cgroupv2 level "$level" \"$cg\" \
      ip daddr != $ON_SITE_V4 counter ${verdict}
  nft add rule inet "$TABLE" output \
      socket cgroupv2 level "$level" \"$cg\" \
      ip6 daddr != $ON_SITE_V6 counter ${verdict}
  echo "armed"
}

# One command for the whole run: the counters need root, and the number that
# matters is the one read after the probes have finished.
measure() {
  local repo owner py out
  # Everything is relative to the repository this script sits in, so the
  # experiment runs from a fresh clone with no paths to edit.
  repo="$(cd "$(dirname "$0")/.." && pwd)"
  owner="${PROBE_OWNER:-$(stat -c %U "$repo")}"
  py="${PROBE_PYTHON:-python3}"
  out="$repo/results"

  arm "drop"
  echo
  echo "--- counters before the run"
  nft list table inet "$TABLE" | grep -E "counter packets" || true
  echo
  # Run as root so the counters can be read before and after and land in the
  # artifact. Ownership of anything written is put back below.
  "$py" "$repo/probes/bench_network.py" \
       --mode sandbox --lan "${PROBE_LAN:-10.0.0.0/8}" --out-dir "$out" \
       --runs "${PROBE_RUNS:-3}" --bargein-runs "${PROBE_BARGEIN_RUNS:-30}" || true
  echo
  echo "--- counters after the run"
  nft list table inet "$TABLE" | grep -E "counter packets" || true
  chown -R "$owner":"$owner" "$out" 2>/dev/null || true
  echo
  echo "the sandbox is still armed; remove it with:  sudo $0 off"
}

case "$MODE" in
  run)     measure ;;
  observe) arm "" ;;
  enforce) arm "drop" ;;
  off)     remove ;;
  status)
    if nft list table inet "$TABLE" >/dev/null 2>&1; then
      nft list table inet "$TABLE"
    else
      echo "no sandbox is armed"
    fi
    ;;
  *) echo "usage: $0 {observe|enforce|run|status|off} [container]" >&2; exit 2 ;;
esac
