"""Workers: SSH targets, actor launch lines, and the RunPod path (requests and commands built against fakes: no pod
is ever created here)."""
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from draftzero.workers import LocalWorker, SSHTarget, SSHWorker, WorkerConfig, actor_command, make_worker
from draftzero.workers import runpod as rp
from draftzero.workers.base import Result, local_run

KEY = "rpa_SECRETSECRETSECRET1234567890"


# ── SSH targets ──
def test_ssh_target_parse_and_spec_round_trip():
    t = SSHTarget.parse("root@213.192.2.92 -p 40066")
    assert (t.user, t.host, t.port, t.key) == ("root", "213.192.2.92", 40066, None)
    t2 = SSHTarget.parse("-i ~/.ssh/k -o ControlMaster=auto dan@r1.lan")
    assert (t2.user, t2.host, t2.port, t2.key, t2.options) == ("dan", "r1.lan", 22, "~/.ssh/k", ("ControlMaster=auto",))
    assert SSHTarget.parse(t2.spec) == t2
    pod = SSHTarget("1.2.3.4", port=40001, key="/k", strict=False)
    assert "StrictHostKeyChecking=no" in pod.spec and SSHTarget.parse(pod.spec) == pod   # a relay host keeps it
    assert SSHTarget.parse("box").user == "root"
    for bad in ("", "-p 22", "a@b c@d", "a@b -L 1:2", "a@b -p"):
        with pytest.raises(ValueError):
            SSHTarget.parse(bad)


def test_ssh_options_are_batch_and_pods_skip_host_keys():
    own = SSHTarget("r1.lan", port=2222, key="/k").ssh_argv("echo ok")
    assert own[0] == "ssh" and own[-2:] == ["root@r1.lan", "echo ok"]
    assert "BatchMode=yes" in own and "StrictHostKeyChecking=accept-new" in own
    assert own[own.index("-p") + 1] == "2222" and own[own.index("-i") + 1] == "/k"
    pod = SSHTarget("1.2.3.4", strict=False).ssh_argv()
    assert "StrictHostKeyChecking=no" in pod and "UserKnownHostsFile=/dev/null" in pod
    assert SSHTarget("h", port=7).rsync_shell().startswith("ssh -o BatchMode=yes")


class Recorder:
    """A fake runner: records argv/input, answers from a list of Results (default ok)."""

    def __init__(self, *results):
        self.calls, self.results = [], list(results)

    def __call__(self, argv, timeout=None, input=None, env=None, **kw):
        self.calls.append({"argv": argv, "input": input, "env": env, "timeout": timeout})
        return self.results.pop(0) if self.results else Result(0, "")


def test_ssh_worker_commands():
    r = Recorder()
    w = SSHWorker(SSHTarget("h", port=9), WorkerConfig(name="h"), runner=r)
    w.run("ls")
    w.push(Path("/src/repo"), "/root/dz/draftzero")
    w.fetch("/root/dz/run/store/shards", Path("/tmp/x/never-created-by-a-fake"))
    ssh, push, fetch = (c["argv"] for c in r.calls)
    assert ssh[-2:] == ["root@h", "ls"]
    assert push[0] == "rsync" and "-a" not in push and push[-2:] == ["/src/repo/", "root@h:/root/dz/draftzero/"]
    assert "/data/" in push and "data" not in push          # anchored: a bare "data" would drop src/draftzero/data
    assert fetch[-2:] == ["root@h:/root/dz/run/store/shards/", "/tmp/x/never-created-by-a-fake/"]


def test_wait_ready_retries_then_gives_up():
    r = Recorder(Result(255, "", "refused"), Result(0, "ok"))
    assert SSHWorker(SSHTarget("h"), runner=r).wait_ready(tries=3, delay=0, sleep=lambda s: None)
    r = Recorder(*[Result(255, "")] * 3)
    assert not SSHWorker(SSHTarget("h"), runner=r).wait_ready(tries=3, delay=0, sleep=lambda s: None)


# ── actors on a worker ──
def test_actor_command_runs_detached_with_the_contract_arguments(tmp_path):
    """Run the real shell line with a stand-in python that records its arguments."""
    rec = tmp_path / "args.txt"
    fake = tmp_path / "fakepy"
    fake.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$PWD\" \"$DZ_X\" \"$@\" > {rec}.tmp && mv {rec}.tmp {rec}\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    store = tmp_path / "run dir" / "store"
    cmd = actor_command(workdir=str(tmp_path), python=str(fake), store=str(store), machine="pod-1", env={"DZ_X": "a b"})
    r = local_run(cmd, timeout=30)
    assert r.ok and r.out.strip().isdigit()                     # the pid
    for _ in range(200):
        if rec.exists():
            break
        subprocess.run(["sleep", "0.05"])
    assert rec.read_text().splitlines() == [
        str(tmp_path), "a b", "-m", "draftzero.loop.cli", "actor", "--config", f"{store}/config.yml",
        "--store", str(store), "--machine", "pod-1"]
    assert (tmp_path / "run dir" / "actor-pod-1.log").exists()


def test_status_parses_heartbeats_and_actor_count():
    out = 'actors=2\nhb=pod-1 {"machine": "pod-1", "time": "t"}\nhb=pod-2 not json\n'
    w = SSHWorker(SSHTarget("h"), runner=Recorder(Result(0, out)))
    s = w.status("/root/dz/run/store")
    assert s["actors"] == 2 and s["heartbeats"]["pod-1"]["machine"] == "pod-1"
    assert "unparsed" in s["heartbeats"]["pod-2"]
    assert "[d]raftzero.loop.cli actor" in w.runner.calls[0]["argv"][-1]


def test_local_worker_and_timeouts(tmp_path):
    w = LocalWorker(WorkerConfig(workdir=str(tmp_path)))
    assert w.run("pwd").out.strip() == str(tmp_path)
    assert w.run("cat", input="hi").out == "hi"
    r = local_run(["sleep", "5"], timeout=0.2)
    assert r.code == 124 and "timed out" in r.err


def test_make_worker():
    assert isinstance(make_worker("local"), LocalWorker)
    w = make_worker("ssh", ssh="dan@r1 -p 2200", workdir="/w")
    assert (w.target.port, w.cfg.workdir, w.cfg.name) == (2200, "/w", "r1")
    pw = make_worker("runpod", gpu="NVIDIA GeForce RTX 3070", max_price=0.2, max_hours=2)
    assert pw.ephemeral and pw.spec.cloud == "COMMUNITY" and pw.max_hours == 2
    with pytest.raises(ValueError):
        make_worker("cloud9")


# ── RunPod: specs and request bodies ──
def test_pod_request_is_community_with_a_public_ip():
    body = rp.pod_request(rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2, min_vcpu=16))
    assert body["cloudType"] == "COMMUNITY" and body["supportPublicIp"] is True
    assert body["computeType"] == "GPU" and body["gpuTypeIds"] == ["NVIDIA GeForce RTX 3070"]
    assert body["minVCPUPerGPU"] == 16 and body["ports"] == ["22/tcp"] and body["imageName"] == rp.IMAGE
    assert "dataCenterIds" not in body and "env" not in body


def test_pod_spec_refuses_secure_cloud_and_bad_cpu_pods():
    with pytest.raises(ValueError, match="allow_secure"):
        rp.PodSpec(name="x", gpu="g", max_price=1, cloud="SECURE").validate()
    with pytest.raises(ValueError, match="Secure Cloud"):
        rp.PodSpec(name="x", cpu="cpu5c", max_price=1, datacenters=("EU-RO-1",)).validate()
    with pytest.raises(ValueError, match="pinned"):
        rp.PodSpec(name="x", cpu="cpu5c", max_price=1, cloud="SECURE", allow_secure=True).validate()
    with pytest.raises(ValueError):
        rp.PodSpec(name="x", cpu="cpu9z", max_price=1, cloud="SECURE", allow_secure=True, datacenters=("A",)).validate()
    with pytest.raises(ValueError):
        rp.PodSpec(name="x", max_price=1).validate()
    with pytest.raises(ValueError):
        rp.PodSpec(name="x", gpu="g", max_price=0).validate()
    cpu = rp.pod_request(rp.PodSpec(name="x", cpu="cpu5c", vcpu=8, max_price=0.3, cloud="SECURE", allow_secure=True,
                                    datacenters=("EU-RO-1",)))
    assert cpu["computeType"] == "CPU" and cpu["cpuFlavorIds"] == ["cpu5c"] and cpu["vcpuCount"] == 8
    assert cpu["dataCenterIds"] == ["EU-RO-1"] and cpu["dataCenterPriority"] == "custom"


def test_ssh_endpoint_and_brief():
    pod = {"id": "p1", "name": "dz", "publicIp": "1.2.3.4", "portMappings": {"22": 40066}, "costPerHr": 0.2}
    assert rp.ssh_endpoint(pod) == ("1.2.3.4", 40066)
    assert rp.ssh_endpoint({"publicIp": "1.2.3.4"}) is None
    assert rp.brief(pod)["ssh"] == "root@1.2.3.4 -p 40066"


# ── RunPod: the API against a fake server ──
GPU_TYPES = {"gpuTypes": [
    {"id": "NVIDIA GeForce RTX 3070", "displayName": "RTX 3070",
     "lowestPrice": {"uninterruptablePrice": 0.13, "minVcpu": 18, "minMemory": 18, "stockStatus": "Low"}},
    {"id": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090",
     "lowestPrice": {"uninterruptablePrice": 0.22, "minVcpu": 16, "minMemory": 62, "stockStatus": "High"}},
    {"id": "NVIDIA A40", "displayName": "A40", "lowestPrice": {"uninterruptablePrice": 0.35, "minVcpu": 9, "stockStatus": None}},
    {"id": "NVIDIA L4", "displayName": "L4", "lowestPrice": None},
]}


class FakeRunPod:
    """The REST + GraphQL endpoints this module uses, in memory."""

    def __init__(self, balance=10.0, pods=(), cost=0.13, fail=None, post_fails=None):
        self.calls, self.balance, self.pods, self.cost, self.fail = [], balance, list(pods), cost, fail
        self.gets = []      # successive GET /pods/<id> answers (the last one repeats)
        self.post_fails = post_fails    # an exception the POST raises after RunPod made the pod (a lost reply)
        self.delete_fails = False

    def __call__(self, method, url, data, headers):
        body = json.loads(data) if data else None
        self.calls.append((method, url, body, headers))
        if self.fail:
            return self.fail
        if url == rp.GQL:
            qy = body["query"]
            if "myself" in qy:
                return 200, json.dumps({"data": {"myself": {"clientBalance": self.balance, "currentSpendPerHr": 0}}})
            if "gpuTypes" in qy:
                return 200, json.dumps({"data": GPU_TYPES})
        if url.startswith(rp.CATALOG_CPU):
            return 200, json.dumps({"price": {"securePerVcpu": 0.035},
                                    "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}, {"id": "US-X", "availability": "NONE"}]})
        if url == f"{rp.REST}/pods" and method == "GET":
            return 200, json.dumps(self.pods)
        if url == f"{rp.REST}/pods" and method == "POST":
            pod = {"id": "pod123", "name": body["name"], "costPerHr": self.cost, "vcpuCount": 18}
            self.pods.append(pod)
            if self.post_fails:
                raise self.post_fails
            return 200, json.dumps(pod)
        if url.startswith(f"{rp.REST}/pods/") and method == "GET":
            g = self.gets.pop(0) if len(self.gets) > 1 else (self.gets[0] if self.gets else {})
            return 200, json.dumps(g)
        if url.startswith(f"{rp.REST}/pods/") and method == "DELETE":
            if self.delete_fails:
                return 500, "internal error"
            self.pods = [p for p in self.pods if p.get("id") != url.rsplit("/", 1)[1]]
            return 200, ""
        return 404, "no route"

    def made(self, method, frag=""):
        return [c for c in self.calls if c[0] == method and frag in c[1]]


def _api(fake):
    return rp.RunPodAPI(key=KEY, http=fake, log=lambda m: None)


def test_the_key_goes_in_a_header_never_a_url_or_an_error():
    fake = FakeRunPod()
    _api(fake).balance()
    method, url, body, headers = fake.calls[0]
    assert headers["Authorization"] == f"Bearer {KEY}" and headers["User-Agent"].startswith("curl/")
    assert KEY not in url and KEY not in json.dumps(body)
    fake.fail = (403, f"denied for key {KEY}")
    with pytest.raises(rp.RunPodError) as e:
        _api(fake).balance()
    assert KEY not in str(e.value) and "rpa_" in str(e.value)
    assert KEY not in repr(_api(fake))


def test_api_key_comes_from_the_env_file_without_exporting(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    f = tmp_path / "env"
    f.write_text(f"export RUNPOD_API_KEY={KEY}\n")
    f.chmod(0o600)
    assert rp.api_key(str(f)) == KEY
    assert "RUNPOD_API_KEY" not in os.environ
    with pytest.raises(rp.RunPodError):
        rp.api_key(str(tmp_path / "none"))
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    assert rp.api_key(str(tmp_path / "none")) == "from-env"


def test_offers_rank_by_vcpu_per_dollar_in_stock_only():
    fake = FakeRunPod()
    rows = _api(fake).offers(min_vcpu=8)
    assert [r["offer"] for r in rows] == ["RTX 3070", "RTX 3090"]           # A40 has no stock, L4 no price
    assert rows[0]["usd_per_vcpu_hr"] == round(0.13 / 18, 4) and rows[0]["cloud"] == "COMMUNITY"
    assert "secureCloud:false" in fake.calls[0][2]["query"] and "minVcpuCount:8" in fake.calls[0][2]["query"]
    assert [r["offer"] for r in _api(fake).offers(max_price=0.2)] == ["RTX 3070"]


def test_create_refuses_before_spending():
    spec = rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3090", max_price=0.2)
    fake = FakeRunPod()
    with pytest.raises(rp.RunPodError, match="above max_price"):
        _api(fake).create(spec)                                          # quote 0.22 > 0.20
    fake = FakeRunPod(balance=0.1)
    with pytest.raises(rp.RunPodError, match="balance"):
        _api(fake).create(rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2))
    fake = FakeRunPod(pods=[{"id": "old", "name": "dz-a1"}])
    with pytest.raises(rp.RunPodError, match="exists"):
        _api(fake).create(rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2))
    fake = FakeRunPod()
    with pytest.raises(rp.RunPodError, match="in stock"):
        _api(fake).create(rp.PodSpec(name="dz-a1", gpu="NVIDIA A40", max_price=1))
    assert not fake.made("POST", "/pods")


def test_create_posts_the_spec_and_removes_a_pod_billing_above_the_cap():
    spec = rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2)
    fake = FakeRunPod()
    pod = _api(fake).create(spec)
    assert pod["id"] == "pod123"
    (post,) = fake.made("POST", f"{rp.REST}/pods")
    assert post[2] == rp.pod_request(spec)
    fake = FakeRunPod(cost=0.5)
    with pytest.raises(rp.RunPodError, match="terminated"):
        _api(fake).create(spec)
    assert fake.made("DELETE", "/pods/pod123")
    fake = FakeRunPod()                                      # the pod reports 18 vCPU
    with pytest.raises(rp.RunPodError, match="below min_vcpu"):
        _api(fake).create(rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2, min_vcpu=24))
    assert fake.made("DELETE", "/pods/pod123")


def test_a_create_that_fails_after_the_pod_exists_removes_it():
    """The POST times out (or the reply is lost) but RunPod made the pod: it must not stay up, unarmed."""
    spec = rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2)
    fake = FakeRunPod(post_fails=TimeoutError("The read operation timed out"))
    logs = []
    api = rp.RunPodAPI(key=KEY, http=fake, log=logs.append, sleep=lambda s: None)
    with pytest.raises(rp.RunPodError, match="removed pod pod123"):
        api.create(spec)
    assert fake.made("DELETE", "/pods/pod123") and fake.pods == []
    # the pod shows up only on a later look
    fake = FakeRunPod(post_fails=ValueError("Expecting value: line 1 column 1 (char 0)"))
    listed = []

    def late(method, url, data, headers):
        if method == "GET" and url == f"{rp.REST}/pods":
            listed.append(1)
            if len(listed) == 2:                      # the create's own name check, then the first look
                return 200, "[]"
        return fake(method, url, data, headers)
    sleeps = []
    with pytest.raises(rp.RunPodError, match="removed pod pod123"):
        rp.RunPodAPI(key=KEY, http=late, log=logs.append, sleep=sleeps.append).create(spec)
    assert sleeps and fake.made("DELETE", "/pods/pod123")
    # removing it fails: say so loudly
    fake = FakeRunPod(post_fails=ConnectionResetError("reset"))
    fake.delete_fails = True
    logs.clear()
    with pytest.raises(rp.RunPodError, match="TERMINATE FAILED"):
        rp.RunPodAPI(key=KEY, http=fake, log=logs.append, sleep=lambda s: None).create(spec)
    assert any("TERMINATE FAILED" in m and "dz pod down" in m for m in logs) and KEY not in " ".join(logs)


def test_cpu_quote_and_terminate_and_list():
    fake = FakeRunPod(pods=[{"id": "a", "name": "x"}])
    api = _api(fake)
    assert api.cpu_quote("cpu5c", 8) == (0.28, ["EU-RO-1"])
    assert "vcpuCount=8" in fake.calls[-1][1]
    assert api.quote(rp.PodSpec(name="x", cpu="cpu5c", vcpu=8, max_price=1, cloud="SECURE", allow_secure=True,
                                datacenters=("EU-RO-1",))) == 0.28
    assert [p["id"] for p in api.list()] == ["a"]
    api.terminate("a")
    assert fake.made("DELETE", f"{rp.REST}/pods/a")


def test_wait_ssh_waits_for_an_endpoint_that_answers():
    fake = FakeRunPod()
    fake.gets = [{}, {"publicIp": "1.2.3.4"}, {"publicIp": "1.2.3.4", "portMappings": {"22": 40001}},
                 {"publicIp": "1.2.3.4", "portMappings": {"22": 40001}}]
    probes = []

    def probe(host, port):
        probes.append((host, port))
        return len(probes) >= 2
    t = _api(fake).wait_ssh("pod123", probe=probe, sleep=lambda s: None, key="/k")
    assert (t.host, t.port, t.user, t.key, t.strict) == ("1.2.3.4", 40001, "root", "/k", False)
    assert probes == [("1.2.3.4", 40001)] * 2

    clock = iter(range(0, 10_000, 100))
    fake.gets = [{}]
    with pytest.raises(rp.RunPodError, match="no SSH banner"):
        _api(fake).wait_ssh("pod123", timeout=300, probe=probe, sleep=lambda s: None, clock=lambda: next(clock))


def test_runpodctl_only_reads_and_its_key_goes_in_the_env():
    """Pods are rented only through RunPodAPI.create, which holds the money guards: runpodctl has no create path."""
    assert not hasattr(rp.RunPodCtl, "create") and not hasattr(rp.RunPodCtl, "argv_create")
    r = Recorder(Result(0, json.dumps({"clientBalance": 3.5})))
    ctl = rp.RunPodCtl(ctl="/bin/runpodctl", runner=r, key=KEY)
    assert ctl.user()["clientBalance"] == 3.5
    argv, env = r.calls[0]["argv"], r.calls[0]["env"]
    assert argv[:2] == ["/bin/runpodctl", "user"] and KEY not in " ".join(argv) and env["RUNPOD_API_KEY"] == KEY
    r = Recorder(Result(1, "", "boom"))
    with pytest.raises(rp.RunPodError):
        rp.RunPodCtl(runner=r).user()


# ── RunPod: on the pod ──
def test_arm_pipes_the_self_destruct_script():
    t = SSHTarget("1.2.3.4", port=40001, strict=False)
    r = Recorder(Result(0, "armed: terminates pod123 at ..."))
    rp.arm(t, 2.5, runner=r)
    c = r.calls[0]
    assert c["argv"][-1] == "bash -s -- 9000" and c["argv"][-2] == "root@1.2.3.4"
    assert c["input"] == rp.ARM_SCRIPT.read_text() and "podTerminate" in c["input"]
    with pytest.raises(rp.RunPodError):
        rp.arm(t, 1, runner=Recorder(Result(0, "")))
    for bad in (0, -1, rp.MAX_ARM_HOURS + 1):
        with pytest.raises(ValueError):
            rp.arm(t, bad, runner=r)


def _bundle(tmp_path, **kw):
    eng = tmp_path / "gorge"
    (eng / "bin").mkdir(parents=True)
    (eng / ".cards").mkdir()
    for f in ("bin/dzgorge", "bin/dzgorge.build.json"):
        (eng / f).write_text("x")
    (tmp_path / "decks").mkdir()
    (tmp_path / "pool.tsv").write_text("deck\tsplit\tcolors\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("")
    return rp.Bundle(repo=repo, engine_root=eng, decks=tmp_path / "decks", pool=tmp_path / "pool.tsv", **kw)


def test_bundle_steps_mirror_the_engine_path_and_unpack_the_repo(tmp_path):
    b = _bundle(tmp_path)
    t = SSHTarget("1.2.3.4", port=40001, strict=False)
    steps = rp.bundle_steps(t, b, tmp_path / "tmp")
    by = {s.what: s.argv for s in steps}
    assert list(by)[0] == "prepare the pod" and "apt-get install -y -qq rsync" in by["prepare the pod"][-1]
    assert by["archive the repo (HEAD)"][:4] == ["git", "-C", str(b.repo), "archive"]
    assert by["send dzgorge"][-1] == f"root@1.2.3.4:{tmp_path}/gorge/bin/"          # same path as here
    assert by["send .cards"][-1] == f"root@1.2.3.4:{tmp_path}/gorge/.cards/" and "--delete" in by["send .cards"]
    assert by["send the decks"][-1] == "root@1.2.3.4:/root/dz/draftzero/data/gorge/decks/"
    assert by["send the pool"][-1] == "root@1.2.3.4:/root/dz/draftzero/data/gorge/pool.tsv"
    assert "-e" in by["send the pool"] and "40001" in by["send the pool"][by["send the pool"].index("-e") + 1]
    assert (b.remote_run, b.remote_python) == ("/root/dz/run", "/root/dz/venv/bin/python")
    b2 = _bundle(tmp_path / "2", repo_mode="rsync", remote_engine_root="/opt/gorge", remote_root="/workspace/dz")
    by2 = {s.what: s.argv for s in rp.bundle_steps(t, b2, tmp_path)}
    assert "/data/" in by2["send the working tree"] and by2["send the working tree"][-1].endswith(":/workspace/dz/draftzero/")
    assert by2["send dzgorge"][-1] == "root@1.2.3.4:/opt/gorge/bin/"


def test_bundle_runs_in_order_and_stops_on_a_failure(tmp_path):
    b = _bundle(tmp_path)
    t = SSHTarget("h", strict=False)
    r = Recorder()
    rp.bundle(t, b, runner=r, log=lambda m: None)
    assert len(r.calls) == len(rp.bundle_steps(t, b, tmp_path))
    r = Recorder(Result(0, ""), Result(128, "", "fatal: not a git repository"))
    with pytest.raises(rp.RunPodError, match="archive"):
        rp.bundle(t, b, runner=r, log=lambda m: None)
    assert len(r.calls) == 2
    (b.engine_root / "bin/dzgorge").unlink()
    with pytest.raises(rp.RunPodError, match="missing"):
        rp.bundle(t, b, runner=Recorder(), log=lambda m: None)


def test_bootstrap_checks_the_ready_line(tmp_path):
    b = _bundle(tmp_path)
    t = SSHTarget("h", strict=False)
    argv = rp.bootstrap_argv(t, b)
    assert argv[-1] == f"bash /root/dz/draftzero/deploy/pod_bootstrap.sh /root/dz {tmp_path}/gorge"
    rp.bootstrap(t, b, runner=Recorder(Result(0, "...\npod_bootstrap: ready\n")))
    with pytest.raises(rp.RunPodError):
        rp.bootstrap(t, b, runner=Recorder(Result(0, "half way")))


def test_pod_bootstrap_script_is_valid_bash():
    script = rp.REPO / rp.BOOTSTRAP
    assert os.access(script, os.X_OK)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "download.pytorch.org/whl/cpu" in text and "pip install -q -c" in text and "dz\" version" in text
    assert text.rstrip().endswith('echo "pod_bootstrap: ready"')


class FakeAPI:
    def __init__(self, fail_wait=False):
        self.events, self.fail_wait = [], fail_wait
        self.log = lambda m: None

    def create(self, spec, balance_hours=1.0):
        self.events.append(("create", spec.name, balance_hours))
        return {"id": "pod123", "name": spec.name}

    def wait_ssh(self, pid, **kw):
        self.events.append(("wait", pid))
        if self.fail_wait:
            raise rp.RunPodError("no banner")
        return SSHTarget("1.2.3.4", port=40001, strict=False, key=kw.get("key"))

    def terminate(self, pid):
        self.events.append(("terminate", pid))


def test_up_runs_the_whole_flow(tmp_path):
    api, r = FakeAPI(), Recorder(Result(0, "armed: ok"))
    b = _bundle(tmp_path)
    r.results += [Result(0, "")] * len(rp.bundle_steps(SSHTarget("x"), b, tmp_path)) + [Result(0, "pod_bootstrap: ready")]
    pod = rp.up(api, rp.PodSpec(name="dz-a1", gpu="g", max_price=0.2), max_hours=2, files=b, runner=r)
    assert pod.id == "pod123" and pod.target.port == 40001
    assert api.events == [("create", "dz-a1", 2), ("wait", "pod123")]
    assert r.calls[0]["argv"][-1] == "bash -s -- 7200"                      # armed before anything else on the pod
    assert "pod_bootstrap.sh" in r.calls[-1]["argv"][-1]
    w = pod.worker(b)
    assert (w.cfg.workdir, w.cfg.python, w.target) == (b.remote_repo, b.remote_python, pod.target)


def test_up_terminates_the_pod_on_any_failure(tmp_path):
    api = FakeAPI()
    with pytest.raises(rp.RunPodError):
        rp.up(api, rp.PodSpec(name="dz-a1", gpu="g", max_price=0.2), max_hours=1, runner=Recorder(Result(1, "", "x")))
    assert api.events[-1] == ("terminate", "pod123")
    api = FakeAPI(fail_wait=True)
    with pytest.raises(rp.RunPodError):
        rp.up(api, rp.PodSpec(name="dz-a1", gpu="g", max_price=0.2), max_hours=1, runner=Recorder())
    assert api.events[-1] == ("terminate", "pod123")
    api = FakeAPI(fail_wait=True)
    with pytest.raises(rp.RunPodError, match="kept"):
        rp.up(api, rp.PodSpec(name="dz-a1", gpu="g", max_price=0.2), max_hours=1, runner=Recorder(), keep_on_failure=True)
    assert ("terminate", "pod123") not in api.events
    with pytest.raises(ValueError):
        rp.up(FakeAPI(), rp.PodSpec(name="dz-a1", gpu="g", max_price=0.2), max_hours=100)


def test_runpod_worker_provision_and_teardown():
    fake = FakeRunPod()
    fake.gets = [{"publicIp": "1.2.3.4", "portMappings": {"22": 40001}}]
    w = rp.RunPodWorker(rp.PodSpec(name="dz-a1", gpu="NVIDIA GeForce RTX 3070", max_price=0.2), max_hours=1,
                        api=_api(fake), runner=Recorder(Result(0, "armed: ok")))
    w.provision(probe=lambda h, p: True, sleep=lambda s: None)
    assert w.pod_id == "pod123" and w.target.host == "1.2.3.4"
    assert w.teardown().ok and fake.made("DELETE", "/pods/pod123") and w.pod_id is None
    assert w.teardown().out == "no pod to tear down"
