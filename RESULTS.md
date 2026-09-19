# Where each printed number comes from

Every value the paper reports, the file it came from, and the command that
produces that file again. Values are shown as the artifact stores them; the
paper rounds them.

Run dates are in the file names: the timing, noise and energy runs are from
27 July 2026, the network runs from 18 September 2026. Section, table and figure
numbers refer to the camera-ready as published.

## Section 5.2, time to first reply audio

`results/latency-2026-07-27T095841Z.json`, 60 turns, 20 queries repeated three
times.

| Stage | p50 | p95 | max |
|---|---|---|---|
| Speech recognition | 182.3 ms | 224.6 ms | 6523.5 ms |
| Retrieval | 94.9 ms | 106.1 ms | 1714.8 ms |
| Language model, first token | 23.4 ms | 44.3 ms | 45.5 ms |
| Speech synthesis, first audio | 579.6 ms | 1141.1 ms | 1243.8 ms |
| End to end, first reply audio | 1119.8 ms | 1682.3 ms | 8865.0 ms |

    python probes/bench_latency.py --runs 3

The 8865 ms maximum is the cold start the paper discusses. It is kept in, not
excluded.

## Section 5.3, interruption

`results/bargein-2026-07-27T100308Z.json`, 30 attempts at offsets from 500 ms to
3000 ms after first reply audio, of which 29 reached their injection point.

| Metric | p50 | p95 | max |
|---|---|---|---|
| Acoustic onset to response cancelled | 603 ms | 760 ms | 916 ms |
| Acoustic onset to last audio chunk sent | 570 ms | 740 ms | — |

    python probes/bench_bargein.py --runs 30

## Section 5.4 and Table 2, word error rate against noise

`results/noise-2026-07-27T141513Z.json`, 20 synthesized German queries mixed with
four noise types at four signal-to-noise ratios. Clean baseline 0.2233.

| SNR | Babble | Copy machine | Machinery | Ventilation |
|---|---|---|---|---|
| 20 dB | 0.2603 | 0.2484 | 0.2542 | 0.2399 |
| 10 dB | 0.2560 | 0.2630 | 0.2580 | 0.2561 |
| 5 dB | 0.3445 | 0.3417 | 0.4371 | 0.3007 |
| 0 dB | 0.7015 | 0.3955 | 0.7127 | 0.5042 |

    python probes/fetch_noise.py          # downloads the corpus, verifies checksums
    python probes/eval_noise.py           # needs the assistant installed, see below

The clean baseline is the harness measured on computer generated German, not
accuracy in use.

## Section 4.3 and Figure 2, the phantom turn defence

`results/f2-gate-2026-07-27T101130Z.json`, 104 four-second windows containing
only noise, transcribed twice, with the segment gate off and on.

    python probes/eval_f2_gate.py         # needs the assistant installed, see below

## Section 5.5, energy

`results/resource-2026-07-27T142935Z.json`, graphics card power sampled at 1 Hz
during a latency run.

| Quantity | Value |
|---|---|
| Energy per query | 0.4019 Wh |

    python probes/profile_resources.py \
        --benchmark-command "python probes/bench_latency.py --runs 3"

Without a benchmark command there are no active-power samples and the energy
figure comes out empty. The command above re-takes the measurement; it does not
reproduce the shipped file byte for byte, because that file was produced against
a latency run of 14:29:34 which is not published here. The 09:58:41 latency file
that is published is the one Section 5.2 reports.

## Sections 3.3, 5.2 and 6.2, what the assistant sends off site

The test confines the assistant rather than disconnecting the machine. An
nftables rule matched to the assistant's cgroup drops and counts everything it
sends to an address outside the site, while the host keeps its own network. The
result is therefore about the assistant, not the machine. On site is loopback
plus the private ranges; the overlay's own range counts as off site.

    sudo probes/sandbox_egress.sh observe     # count only, block nothing
    sudo probes/sandbox_egress.sh run         # arm, measure, record
    sudo probes/sandbox_egress.sh off         # remove

### Results

`results/network-observation-2026-09-18T145000Z.json`, before any change. One
socket to a public content delivery network, in `CLOSE_WAIT`, over a 179-sample
window. Source: the recognizer was named by a model-hub
identifier, so each start asked the hub which commit `main` pointed at. The
request carries a repository name, runs before any session exists, and sends
nothing on the socket. The deployment now resolves models from its local cache,
so the lookup no longer happens.

`results/network-online-2026-09-18T152119Z.json`, the unconfined run. Its
socket sampler produced no output (`sockets.error`), so this file carries no
socket evidence and is not a baseline for the confined runs.

`results/network-sandbox-2026-09-18T165005Z.json`, confined, after the deployment
was set to use its local model cache only. No off-site socket held and none of
the 96 reachability probes escaped. The firewall counted 396 packets in this
run, all of them those probes, which is what the next run corrects. Timing was a
median 1178 ms to first reply audio over the 54 of 60 turns that completed; the
latency probe exited non-zero, with six turns producing no final metrics.

`results/network-sandbox-2026-09-18T214035Z.json`, the clean packet count. Zero
packets aimed off site across the whole run, and none of 105 probes escaped.

### Which fields carry the local-only result

All six are in `network-sandbox-2026-09-18T214035Z.json`. None is affected by
any of the substitutions listed under Redactions below.

| What it establishes | Field | Value in that run |
|---|---|---|
| Nothing left the assistant's process group | `sandbox_counters_before`, `sandbox_counters_after` | 0 packets, 0 bytes, both rules `drops: true`, `enforcing: true` |
| The confinement was in force, not merely a quiet network | `sockets.reachability_successes`, `reached_off_site` | 0 of 105 attempts succeeded, `false` |
| No connection off site was held while turns ran | `sockets.offsite_peers` | empty, over 613 samples spanning 1041 s |
| What "on site" meant | `on_site_subnets` | loopback plus the private ranges |
| The host was not disconnected to achieve this | `network_state_after` list lengths | one default route present, one overlay still up |
| The assistant was serving turns throughout | `probes[]` | the barge-in probe returned 0 |

The fifth row is the one that separates this result from an unplugged machine.
The redaction below replaces the contents of those lists but preserves their
lengths, so the fact that the host kept its own network while the assistant
could reach nothing is still readable.

### Reading the counter

The reachability probes run inside the assistant's container and share its
cgroup, so the firewall cannot distinguish them from the assistant's own
traffic. In the 16:50 run the counter went from 12 to 408, and all 396 of those
packets were probes. The rule was then changed to match the probe targets first
and drop them without counting, which is why the 21:40 run reads 0 before and 0
after.

Timing numbers above come from the 16:50 run, not the 21:40 one: during 21:40
another workload took the graphics card and the assistant fell back to a smaller
model, which its much slower stage times show.


## Two probes cannot run from a fresh clone

`eval_noise.py` and `eval_f2_gate.py` import the assistant's own package: they
measure the deployed voice-activity detector and recognizer at their configured
thresholds. Their result files in `results/` carry every number the paper prints
from them.

## Result file names

The probes write `<timestamp>.json` into `results/`. The files shipped here
carry a prefix (`latency-`, `bargein-`, `noise-`, `f2-gate-`, `resource-`,
`network-`) so one directory can hold all of them, and the probes now write that
prefix themselves. Paths recorded *inside* the older result files still point at
the per-probe subdirectories the deployment used when they were taken.

## Redactions

Measured values, counters, socket observations and timings are unchanged in
every result file. Host network topology is replaced in `network-*.json`, under
`network_state`, `network_state_before` and `network_state_after`:

| Field | Replacement |
|---|---|
| `default_routes_v4`, `default_routes_v6` | `<redacted>` per entry, count preserved |
| `default_route` (online run only) | one `<redacted: a default route was present>` string |
| `external_interfaces_with_address` | `<redacted>` per entry, count preserved |
| `wireless_interfaces`, `wireless_up` | `<redacted>` per entry, count preserved |
| `policy_rules`, `routes`, `routes_v4` | one `<redacted: N entries>` placeholder |
| `overlay_interfaces_up` | `<redacted: overlay>` per entry, count preserved |

The counts are what the offline check turns on: one default route present, no
wireless interface up, one overlay up. Input paths inside the result files refer
to the deployment's own directory layout, not to this repository's.

Three further substitutions are made outside the network files. No measured
value, count or timing is affected by any of them.

| Where | What | Replacement |
|---|---|---|
| `prompts.jsonl`, `noise-*.json` `reference` | the machine's model designation | `<Modell>` |
| `noise-*.json`, `f2-gate-*.json` transcript fields | manufacturer and firmware names the recognizer produced | `<Hersteller>`, `<Firmware>` |
| `latency-*.json` `turns[].events[].type`, `bargein-*.json` `event_log` | event types specific to the deployment and unrelated to the turn cycle | `other` |

The turn-cycle event names are kept, so the stage timings can still be
re-derived from the raw event stream. The word error rates in `noise-*.json`
were computed before the substitution, on the original reference text.

`network-observation-*.json` was taken with an ad-hoc invocation of
`net_watch.py` rather than through `bench_network.py`, so no single command in
this repository reproduces it. Its fields are the sampler's own output.
