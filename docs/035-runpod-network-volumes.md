# RunPod network volumes as a run's store

*Started 2026-10-14 · Applies to v0.2.0 · Report*

*Status (Wednesday 14 October 2026, 7 PM PT): draft. The live test (E2c, §5) runs until about 21:10 PT; its results
and costs are filled in after its final evaluation.*

**RunPod's global volumes cannot hold a run's store, but its network volumes can.** Dan asked whether RunPod volumes
could replace the Hugging Face bucket for distributed actor-learner training. A *global* volume (beta since September
2026) shows a running pod only the files that existed when the pod started, and has no atomic rename, so learners
would never see new shards and actors would never see new versions. A *network* volume is a MooseFS filesystem in one
data center that many Secure Cloud pods mount at once. On 14 October five pods shared one 10 GB volume in EUR-IS-1 as
the store of three runs, with no relay and no token on any pod:
- **Rename race:** 1,100 reads of a file that another pod replaced ten times a second found it missing 0 times and
  half-written 0 times.
- **Visibility:** a new file showed on another pod in 27 ms (median).
- **Smoke run:** two versions in about a minute (`configs/smoke-volume.yml`).
- **E2c:** E2b's two arms continued from their stores (§5).

The catch is price: only Secure Cloud pods mount volumes. The cheapest Secure vCPU-hour is 2–4 times the price of
the Community pods that the relay serves, and a run is limited to one data center's stock.

Numbers are measured unless marked *estimate*.

## 1. What RunPod offers

| | Network volume | Global volume (beta) | Hugging Face bucket | Relay to d1 |
|---|---|---|---|---|
| What it is | MooseFS over FUSE, one data center | Object storage behind a filesystem, any data center | S3-like object storage | rsync over SSH from each pod's own disk |
| Pods that can use it | Secure Cloud, in the volume's data center | GPU pods (the guide; the v2 API says CPU too) | Any machine with the token | Any pod with a public SSH port |
| Sees later writes | Yes (27 ms, §4) | **No**: "a running worker will not pick up a file you upload after it started" | Yes | Yes, each sync (60 s) |
| Atomic rename | Yes (§4) | **No** | Not needed (whole-object puts) | Yes (rsync renames into place) |
| From d1 | S3 API (needs an S3 key, §6) | No | Yes | Yes (the relay runs there) |
| Price | $0.07/GB-month, billed until deleted | $0.09/GB-month plus request fees | Free up to 100 GB private | Free |
| Limits | One volume per pod, set at create | One per pod | 1,000 API calls / 5 min (free account) | Pods need reachable SSH |

Global volumes fail on the first two rows, so the rest of this doc is about network volumes. This answers the
question docs/007 §2.7 left open ("whether one RunPod network volume can be attached to several pods at once"): it
can, and the store's rules (one writer per file, whole files renamed into place, `meta.json` and `latest.json`
written last) are KataGo's shared-directory pattern, which a shared POSIX folder serves as it is.

## 2. How a run uses a volume

Every role opens the same folder on the volume as a `LocalStore`: `--store /workspace/runs/<run>/store`. Nothing
else changes: there is no relay, the pods hold no token, and sync mode counts every actor's shards. The code
(3c8a67e, c096a49):

- **`dz pod volume create|list|rm`.** `create` takes `--name --size --datacenter`, 10–50 GB, and refuses a name in
  use; it prints the monthly cost and asks first. `rm` deletes the volume and its files for good, and asks first.
- **`dz pod up --volume ID --secure`.** It mounts the volume at `/workspace`, pins the pod to the volume's data
  center, and refuses Community Cloud or a mismatched `--datacenter`. The cost guard adds the volume's monthly
  price.
- **`LocalStore` on a shared folder:**
  - each file is fsynced before its rename;
  - a failed put leaves no temp file;
  - a file seen but gone when opened is read again (up to 3 tries) instead of crashing the role;
  - listing uses `os.walk`, with no stat per file.
- **Learners refuse to republish version 0** while a later version is published (`latest.json` lost, or not yet
  visible, would otherwise restart the run).
- **`dz run` on a stopped store** starts the learner, evaluator and actors only after the controller has resumed the
  run. Before c096a49 they read the old stop and exited cleanly, so nothing noticed (§5).

### Running one

```bash
dz pod volume create --name dz-vol --size 10 --datacenter EUR-IS-1
dz pod up --name dz-vh --cpu cpu3c --vcpu 4 --max-price 0.13 --max-hours 4 --secure --volume VOL   # the head
dz pod up --name dz-va --cpu cpu3c --vcpu 4 --max-price 0.13 --max-hours 4 --secure --volume VOL   # an actor pod
# on the head, from /root/dz/draftzero:
dz run configs/smoke-volume.yml --actors 0
# on each actor pod, once the head's controller has written control.json:
python -m draftzero.loop.cli actor --config /workspace/runs/smoke-volume/store/config.yml \
    --store /workspace/runs/smoke-volume/store --machine dz-va
```

**Start remote actors after the run is up.** An actor exits at once on a `control.json` that says stop. That
happens on a fresh store before the controller's first tick, and on a resumed one until the controller clears the
old stop. `dz status STORE` shows when the run is up.

**Seed pods from the volume, not from d1.** `dz pod up` uploads about 340 MB to each pod from d1: the repo, dzgorge,
`.cards` and the deck pool. On 14 October d1's uplink to Iceland gave 2–3 MB/s in all, and one pod got 45 KB/s.
Once one pod had the bundle, it went onto the volume as a single 70 MB tarball. Each other pod unpacked it in
6–11 s and ran `deploy/pod_bootstrap.sh` (`/work/exp/e2c/seed.sh`). *Not in `dz pod up` yet.*

## 3. What it costs

- **Volume:** 10 GB is about $0.70 a month, billed until `dz pod volume rm`, with or without pods.
- **Pods:** only Secure Cloud pods can mount it. Prices per vCPU-hour on 14 October:

| Machine | $/vCPU-h | Used by |
|---|---|---|
| Community RTX 3070 host, 18 vCPU, $0.13/h | 0.0072 | E1's relay actors |
| Secure RTX A4000, 18 vCPU, $0.25/h (if the data center has stock) | 0.0139 | |
| Secure CPU `cpu3c` | 0.030 | E2b's actor pod, E2c |
| Secure CPU `cpu5c` | 0.035 | |

- **Stock moves faster than the catalog.** cpu3c in US-NC-2 went from "HIGH" to absent within minutes. In EUR-IS-1
  two 8-vCPU creates were refused ("no longer any instances available") while the catalog said "MEDIUM", and 4-vCPU
  pods were granted at once. A run on a volume is bound to that one data center's stock.

## 4. Probes

`tools/volume_probe.py` on five pods in EUR-IS-1, 14 October 18:38 PT (raw output in `/work/exp/e2c/probe/`):

| Probe | Result |
|---|---|
| The mount | `mfs#eur-is-1-2.runpod.net:9421[/networkvolumes/<id>]`, `fuse`, `rw,nosuid,nodev` (executables run) |
| Rename race: pod A replaces a 64 KB JSON file 10×/s for 60 s (temp, fsync, rename); pods B and C read it 10×/s | 1,100 reads: **0 missing, 0 half-written**; staleness median 33 ms, 99th percentile 121 ms (the writer's own period is 100 ms) |
| A new file written on pod D, polled on pod H | seen after **27 ms** median, 34 ms worst (20 files) |
| 2,000 small files (temp, fsync, rename each) | 13.7 ms a file to create; `os.walk` lists them in 0.34 s on the writer and on another pod (rglob with a stat per file: 0.49 s) |
| 256 MB with `dd` | written with fsync at 829 MB/s; read on another pod at 1.3 GB/s |
| `statvfs` | reports the cluster (576 TB), not the volume's 10 GB: free space cannot be checked with `df` |

The store needs one writer per file, whole-file renames and close-to-open visibility, and MooseFS gives all three.
The read retries in `LocalStore` never fired in these probes, but they cost nothing on a miss.

## 5. E2c: E2b's arms continued on a volume

*The experiment:*
- **Arms:** E2b's control and PCR 200/40 arms (docs/034 §7), resumed from their stores at versions 4 and 3. Both
  stores were copied to the volume.
- **Head pod** (`cpu3c`, 4 vCPU): both runs' controllers, learners and evaluators.
- **Actor pods:** four, each 4 vCPU, each running one actor per arm with 2 workers. That is 8 workers an arm, and
  both arms share every pod, as in E2b.
- **Settings:** the same as E2b except `actor.workers` 5 → 2, recorded in `events.jsonl`.
- **Run length:** `--max-hours 2.5`, from 18:41 PT.
- **Final evaluation:** each arm's last version against B17, and the two arms head to head, at 100 and 25
  simulations.

*What happened at the start:* `dz run` started each run's learner and evaluator together with its controller. They
read E2b's final `control.json` (stop) before the controller had cleared it, and exited with code 0. The actors,
started 25 s later, played on with no learner for 3 minutes, until the two roles were started by hand. c096a49 fixes
this in `dz run`. The pods ran the earlier commit, so this run kept the hand-started roles.

*Results: to come.*

## 6. Not done yet

- **d1 over S3 (design b).** d1 would keep the learner, controller and evaluator on its own (free) cores and reach
  the volume through RunPod's S3 API, with only actors on pods.
  - It needs an S3 API key, which only Dan can create (console, Settings → S3 API Keys), saved in d1's
    `/root/.runpod/s3_env`. The key reaches every volume on the account.
  - It also needs an `S3Store` (about 150 lines with SigV4 signing).
  - Open questions to measure: S3 list latency on pod-written files (RunPod computes their checksums lazily), and
    whether a PutObject appears atomically on the mount.
- **Seeding in `dz pod up`** (§2): use the volume's bundle when it matches the commit and engine, and otherwise
  upload once and put it there.
- **Several data centers:** one volume each, joined by the relay. Not needed at this scale.
