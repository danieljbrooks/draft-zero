# Reserving RunPod instances: what worked and what didn't

**Read this before reserving a pod.** Every item here cost real time or money to learn.
Written 2026-09-25 while reserving pods for the experiment #2 pilot; the exp #1 lessons come
from the [report](003-fdn-generalist-report.md) §4.2 and the [RUNBOOK](002-RUNBOOK.md).

## Checklist

1. **Choose by vCPU and RAM per dollar**, not by GPU: `dz workers rank --min-vcpu 16`.
2. **Check whether your network allows the pod's SSH port.** If it doesn't, plan on the SSH
   proxy (see *Networks that block high ports*).
3. **Create one pod at a time, and check `runpodctl pod list` by name before any retry**, so
   a retry can't create a duplicate.
4. **Use the CUDA 12.8 image** with `--min-cuda-version 12.8`.
5. **Request 40 GB of container disk**, not 80.
6. **Judge health from `runtimeStatus` / `uptimeSeconds` and `runpodctl pod logs`**, not from
   whether SSH answers.
7. **Arm a self-destruct on the pod first**, then pull results before it fires.
8. **Size threads and heaps from the cgroup**, never from `nproc` or `free`.
9. **End billing with `remove`**, not `stop`. Check with `runpodctl user`.

## Choosing an offer

- **Self-play is CPU-bound, so rank by vCPU per dollar**, and by RAM too once you run several
  JVMs. `dz workers rank --min-vcpu 16` lists in-stock offers that way.
- **Stock is volatile.** "There are no longer any instances available with the requested
  specifications" is routine. The cheap offers sell out between listing and creating, often
  within minutes; exp #1 lost a 2× RTX 4090 twice in two minutes. Retry across GPU types.
- **Community Cloud is cheapest, but check it can be reached.** In exp #1 the best value was a
  Community L40S at $0.79/hr. On 2026-09-25 **no Community host offered a public IP**, with
  or without a CUDA filter, and without a public IP there's no direct SSH.
- **Secure Cloud always gives a public SSH port.** The best value obtained on 2026-09-25 was
  a **Secure RTX 3090 at $0.50/hr**, sold as 32 vCPU / 125 GB, with a real cgroup limit of
  31.1 cores / 116 GB. Network volumes also exist only on Secure Cloud.
- **Container disk:** requesting 80 GB may shrink the pool of eligible hosts. 40 GB was
  enough for the pilot.

## Creating the pod

```bash
runpodctl pod create --name <unique-name> --gpu-id "NVIDIA GeForce RTX 3090" \
  --cloud-type SECURE --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --container-disk-in-gb 40 --ports 22/tcp --min-cuda-version 12.8
```

- **Use a unique `--name` and guard against duplicates.** Before *any* retry, check
  `runpodctl pod list` for that name. A create can fail or time out *after* the pod exists:
  on 2026-09-25 a `--wait` timeout nearly triggered a second pod on the next GPU type.
- **Don't rely on `--wait` from a network that blocks high ports.** It waits for an SSH
  banner on the pod's public port, which never arrives through such a network.
- **Match the image to the GPU.** Blackwell GPUs (for example the RTX PRO 4500) need CUDA 12.8
  builds. A CUDA 12.4 image boots fine, but torch can't use the GPU and inference silently
  falls back to CPU. `deploy/bootstrap.sh` pins the image's torch so pip can't replace it.
- **Pick one image and keep it.** The `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` image
  is the one exp #1 used, and the one that worked here on an RTX 3090 with driver 580.

## Is it running?

- **Missing fields don't mean a stuck pod.** `runpodctl` 2.x reports `runtimeStatus` and
  `uptimeSeconds`. There is no `runtime` or `machineId` field. On 2026-09-25 a healthy pod was
  terminated because those missing fields read as "never got a machine".
- **`runpodctl pod logs <id>` is the ground truth.** It shows sshd starting and "Start
  script(s) finished, Pod is ready to use.", usually within seconds of creation. It works
  over the HTTPS API, so it's reachable from any network.
- **The container's real limits come from the cgroup,** and some hosts are cgroup v1, with
  no `/sys/fs/cgroup/cpu.max`. The pod above reported `nproc=256` and 1,007 GB of host RAM
  against a 31-core, 116 GB limit. Use `magezero.resources.cpu_quota()` and `mem_limit_gb()`,
  which read both cgroup versions.

## Networks that block high ports

RunPod exposes a pod's SSH on a random high public port (for example `213.192.2.92:40066`).
**Some networks block outbound connections to high ports.** The laptop used on 2026-09-25
allowed 22, 80, 443 and 8080 and blocked 2222 and 40066. Symptoms: `ssh` and `nc` time out
(not "connection refused"), and `--wait` never succeeds.

**Test it:** `nc -z -w 8 portquiz.net <port>`. portquiz.net answers on every port, so a
timeout means your network is filtering. Check port 80 too, to be sure portquiz is up.

**Workaround: RunPod's SSH proxy on port 22.**

```bash
# the proxy username is <podId>-<podHostId>; podHostId comes from GraphQL
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { pod(input:{podId:\"<podId>\"}) { machine { podHostId } } }"}'
ssh -tt -i ~/.ssh/id_ed25519 <podId>-<podHostId>@ssh.runpod.io
```

The proxy gives you a PTY shell only:
- **No `scp` or `sftp`.** Bring code in with `git clone` on the pod, and large inputs from a
  private HF repo. Results come back by printing them through the session or by the pod
  uploading them.
- **Send self-contained command lines.** Everything you pipe in arrives before the shell is
  ready. `stty -echo` discarded all the input queued after it, and a `read` mangled what
  followed.
- **Secrets:** start with `unset HISTFILE`, then write the secret in one line, for example
  ` echo <base64> | base64 -d > /root/.token` with a leading space. The terminal echoes it
  back into your local capture, so delete that file. Use a read-only token where one will do.
- **Terminal escape codes prefix output lines.** Filter output with `grep -o 'MARKER .*'`,
  not `^MARKER`.

## While it runs

- **Arm a self-destruct first.** Pods get `RUNPOD_POD_ID` and a `RUNPOD_API_KEY` in their
  environment, and have `runpodctl`, so:

  ```bash
  nohup bash -c 'sleep 10800; runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl pod remove "$RUNPOD_POD_ID"' \
    > /root/selfdestruct.log 2>&1 &
  ```

  It removes the pod after 3 hours even if your session dies. Pull the results before it fires.
- **Don't wait on setup with `pgrep -f <script>`** from a command line that mentions the
  script's name. `pgrep` matches the waiter itself and waits forever. Chain the steps
  sequentially instead (`bash setup.sh && bash run.sh`).
- **Ship big inputs once, through a private HF repo.** The 560 MB engine uploaded once from
  the laptop; each pod downloaded it at datacenter speed.

## Money

- **Billing stops only on `remove`.** A stopped pod still bills for its disk. Orphaned pods
  from aborted grabs kept billing in exp #1.
- **Check the balance** with `runpodctl user` (`clientBalance`, `currentSpendPerHr`).
- **Never add credits.** Spend only the existing balance.

## Auth

- **`runpodctl` reads its key** from `RUNPOD_API_KEY` or `~/.runpod/config.toml` (`apikey`).
- **The GraphQL API wants `?api_key=` in the URL.** A `Bearer` header returned 403.
- **Two SSH keys are registered on the account:** the ed25519 key, and `runpodctl`'s own key
  at `~/.runpod/ssh/runpodctl-ssh-key`.
