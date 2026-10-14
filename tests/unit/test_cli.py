"""The dz command (draftzero.cli): help for every command, argument parsing and dispatch. Games run on the fake
dzgorge (tests/unit/conftest.py); the loop, the relay, RunPod and the Hub are monkeypatched: no pod, no network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from draftzero import cli
from draftzero import config as C
from draftzero.cli import main
from draftzero.loop import control as ctl
from draftzero.loop.store import LocalStore

REPO = Path(__file__).resolve().parents[2]
SAMPLE = [f"data.decks={REPO}/assets/sample/decks", f"data.pool={REPO}/assets/sample/pool.tsv"]

HELP = [["version"], ["manifest"], ["provenance"], ["run"], ["actor"], ["learner"], ["evaluator"], ["controller"],
        ["status"], ["stop"], ["plan"], ["report"], ["models"], ["models", "pull"], ["play"], ["eval"], ["engine"],
        ["engine", "build"], ["engine", "version"], ["engine", "coverage"], ["pod"], ["pod", "up"], ["pod", "down"],
        ["pod", "list"], ["pod", "balance"], ["pod", "offers"], ["relay"]]


def sets(root) -> list[str]:
    out = []
    for s in [f"engine.root={root}", *SAMPLE]:
        out += ["--set", s]
    return out


def write_config(tmp_path, root, **extra) -> Path:
    d = {"extends": "base/player.yml", "run": "t1", "engine": {"root": str(root)},
         "data": {"decks": f"{REPO}/assets/sample/decks", "pool": f"{REPO}/assets/sample/pool.tsv"},
         "store": str(tmp_path / "store"), **extra}
    p = tmp_path / "t1.yml"
    p.write_text(json.dumps(d))                          # JSON is YAML
    return p


# ------------------------------------------------------------------------------------------------ help, version

@pytest.mark.parametrize("words", HELP, ids=" ".join)
def test_every_command_has_help(words, capsys):
    with pytest.raises(SystemExit) as e:
        main([*words, "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: dz {words[0]}") and len(out.splitlines()) > 3


def test_top_level_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for words in HELP:
        assert f"    {words[0]} " in out or f"    {words[0]}\n" in out, words


def test_version(capsys):
    from draftzero import __version__
    assert main(["version"]) == 0 and capsys.readouterr().out.strip() == __version__


def test_manifest_and_its_alias(tmp_path, fake_dzgorge, capsys):
    assert main(["provenance"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["format"] == "draftzero-manifest/1" and m["config"] is None and m["belief"] is None
    assert main(["manifest", str(write_config(tmp_path, fake_dzgorge.root))]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["run"] == "t1" and m["belief"]["corpus_hash"] and m["engine"]["root"] == str(fake_dzgorge.root)


# ------------------------------------------------------------------------------------------------ run and the roles

def test_run_maps_onto_local_mode(tmp_path, fake_dzgorge, monkeypatch, capsys):
    from draftzero.loop import cli as loop_cli
    calls = []
    monkeypatch.setattr(loop_cli, "main", lambda argv, prog=None: calls.append(argv) or 0)
    cfg = write_config(tmp_path, fake_dzgorge.root)
    assert main(["run", str(cfg), "--actors", "2", "--max-hours", "0.1", "--work-dir", str(tmp_path / "w")]) == 0
    assert calls[-1] == ["local", "--config", str(cfg), "--store", str(tmp_path / "store"), "--actors", "2",
                         "--work-dir", str(tmp_path / "w"), "--max-hours", "0.1", "--poll", "10.0"]
    assert main(["run", str(cfg), "--store", str(tmp_path / "s2"), "--set", "actor.sims=8", "--work-dir",
                 str(tmp_path / "w2"), "--poll", "5"]) == 0
    argv = calls[-1]
    used = Path(argv[argv.index("--config") + 1])
    assert used == tmp_path / "w2" / "config.yml" and argv[argv.index("--store") + 1] == str(tmp_path / "s2")
    assert C.load(used).actor.sims == 8 and argv[-2:] == ["--poll", "5.0"]
    assert "dz run: the run stopped" in capsys.readouterr().out
    assert not any("--allow-config-change" in argv for argv in calls)
    assert main(["run", str(cfg), "--work-dir", str(tmp_path / "w"), "--allow-config-change"]) == 0
    assert calls[-1][:2] == ["local", "--config"] and calls[-1].count("--allow-config-change") == 1


def test_run_checks_engine_store_and_overrides(tmp_path, monkeypatch, capsys):
    from draftzero.loop import cli as loop_cli
    monkeypatch.setattr(loop_cli, "main", lambda argv, prog=None: pytest.fail("must not start"))
    cfg = write_config(tmp_path, tmp_path / "no-gorge")
    assert main(["run", str(cfg)]) == 2
    assert "no dzgorge at" in capsys.readouterr().err
    assert main(["run", str(cfg), "--store", "ssh://host/x", "--no-check"]) == 2
    assert "relay" in capsys.readouterr().err
    assert main(["run", str(cfg), "--set", "nonsense"]) == 2
    assert main(["run", str(cfg), "--set", "actor.nope=1"]) == 1              # ConfigError, no traceback
    assert "actor.nope" in capsys.readouterr().err


def test_roles_delegate_to_the_loop_cli(monkeypatch):
    from draftzero.loop import cli as loop_cli
    calls = []
    monkeypatch.setattr(loop_cli, "main", lambda argv, prog=None: calls.append((argv, prog)) or 0)
    for role in ("actor", "learner", "evaluator", "controller"):
        assert main([role, "--config", "c.yml", "--store", "s", "--machine", "m"]) == 0
        assert calls[-1] == ([role, "--config", "c.yml", "--store", "s", "--machine", "m"], "dz")


def test_plan_status_stop_on_a_store(tmp_path, capsys):
    store = tmp_path / "store"
    # a mistyped store path is refused (`dz stop` must not report success while the real run goes on), and no
    # folder is created
    for words in (["status", str(store)], ["stop", str(store)], ["plan", str(store), "list"],
                  ["plan", "--store", str(store), "set", "paused=true"]):
        assert main(words) == 2, words
        assert "not a run's store" in capsys.readouterr().err
    assert not store.exists()
    # changes can be queued before the run starts, on purpose
    assert main(["plan", str(store), "set", "learner.lr=3e-4", "paused=true", "--by", "t", "--before-start"]) == 0
    reqs = [LocalStore(store).read_json(p) for p in LocalStore(store).list(ctl.REQUESTS)]
    assert sorted((r["path"], r["value"]) for r in reqs) == [("learner.lr", 3e-4), ("paused", True)]
    out = capsys.readouterr().out
    assert "at the next version" in out and "now" in out
    LocalStore(store).put(ctl.CONTROL, json.dumps({"format": "dz-control/1", "stop": False}).encode())  # it started
    assert main(["status", str(store)]) == 0 and "no status yet" in capsys.readouterr().out
    assert main(["plan", str(store), "list"]) == 0 and capsys.readouterr().out.count("pending") == 2
    assert main(["stop", str(store)]) == 0 and (store / ctl.STOP).exists()


# ------------------------------------------------------------------------------------------------ report, models

def test_report(tmp_path, monkeypatch, capsys):
    from draftzero.plots import standard
    run = tmp_path / "run"
    run.mkdir()
    seen = {}

    def fake_report(src, out, themes, anchor):
        seen.update(src=src, out=Path(out), themes=themes, anchor=anchor)
        return [Path(out) / "learning-light.png"]
    monkeypatch.setattr(standard, "report", fake_report)
    assert main(["report", str(run), "--theme", "light"]) == 0
    assert seen == {"src": str(run), "out": run / "report", "themes": ("light",), "anchor": "sbtactical"}
    assert capsys.readouterr().out.strip() == str(run / "report" / "learning-light.png")

    def no_mpl(*a, **k):
        raise ImportError("matplotlib is needed for the figures: pip install 'draftzero[plots]'")
    monkeypatch.setattr(standard, "report", no_mpl)
    assert main(["report", str(run), "--out", str(tmp_path / "o")]) == 2
    assert "pip install 'draftzero[plots]'" in capsys.readouterr().err
    monkeypatch.setattr(standard, "report", lambda *a, **k: [])
    assert main(["report", str(run)]) == 1
    assert main(["report", str(tmp_path / "missing")]) == 2


def test_models_pull(monkeypatch, capsys):
    from draftzero import hub
    monkeypatch.setattr(hub, "resolve", lambda uri, cache_dir=None: Path("/cache") / uri.rsplit("/", 1)[-1])
    assert main(["models", "pull", "hf://o/r@v0042/model.dzgw"]) == 0
    assert capsys.readouterr().out.strip() == "/cache/model.dzgw"
    assert main(["models", "pull", "runs/x/model.dzgw"]) == 2


# ------------------------------------------------------------------------------------------------ play and eval

def test_play_on_the_fake_driver(fake_dzgorge, capsys):
    assert main(["play", "--a", "sbtactical", "--b", "bot", "--pairs", "3", "--seed", "5", *sets(fake_dzgorge.root),
                 "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["games"] == 6 and r["score_a"]["n_pairs"] == 3 and r["score_a"]["score"] == 0.5 and r["split"] == "eval"
    argv = fake_dzgorge.calls()[-1]
    assert argv[argv.index("-a") + 1] == "sbtactical" and argv[argv.index("-seed") + 1] == "5"
    assert argv[argv.index("-split") + 1] == "eval"
    # --sims rewrites both search specs; equal specs play one game per pair
    assert main(["play", "--a", "az:sims=100:autopay", "--b", "az:sims=100:autopay", "--sims", "8", "--pairs", "2",
                 *sets(fake_dzgorge.root)]) == 0
    out = capsys.readouterr().out
    argv = fake_dzgorge.calls()[-1]
    assert argv[argv.index("-a") + 1] == argv[argv.index("-b") + 1] == "az:sims=8:autopay"
    assert "2 games" in out and "A's score" in out
    assert main(["play", "--a", "frob:x", "--b", "bot", "--pairs", "1", *sets(fake_dzgorge.root)]) == 2
    assert main(["play", "--a", "bot", "--b", "bot", "--pairs", "1", "--set", "engine.root=/nowhere"]) == 2
    assert "dz engine build" in capsys.readouterr().err


def test_eval_a_dzgw_file(fake_dzgorge, tmp_path, capsys):
    net = tmp_path / "model.dzgw"
    net.write_bytes(b"dzgw")
    out = tmp_path / "ev"
    assert main(["eval", str(net), "--opponents", "sbtactical,nonet", "--pairs", "2", "--sims", "8",
                 "--out", str(out), *sets(fake_dzgorge.root)]) == 0
    text = capsys.readouterr().out
    summary = json.loads(text.strip().splitlines()[-1])
    assert [r["opponent"] for r in summary["results"]] == ["sbtactical", "nonet"]
    assert all(r["n_pairs"] == 2 for r in summary["results"]) and summary["sims"] == 8
    assert "opponent" in text and "95% interval" in text
    assert json.loads((out / "summary.json").read_text()) == summary
    hero, nonet = summary["spec"], summary["results"][1]["spec"]
    assert f"gonet={net}" in hero and ":closedlist" in hero and ":explore" not in hero and ":sims=8" in hero
    assert "gonet" not in nonet and "topk" not in nonet
    argv = fake_dzgorge.calls()[0]
    assert argv[argv.index("-split") + 1] == "eval" and argv[argv.index("-belief-split") + 1] == "train"
    # ad-hoc opponents: random and a raw spec; prev needs a store
    assert main(["eval", str(net), "--opponents", "random,az:sims=4:autopay", "--pairs", "1", "--json",
                 *sets(fake_dzgorge.root)]) == 0
    assert [r["spec"] for r in json.loads(capsys.readouterr().out)["results"]] == ["random", "az:sims=4:autopay"]
    assert main(["eval", str(net), "--opponents", "prev", "--pairs", "1", *sets(fake_dzgorge.root)]) == 2
    assert main(["eval", str(tmp_path / "nothing"), "--opponents", "bot", *sets(fake_dzgorge.root)]) == 2


def test_eval_a_runs_version(fake_dzgorge, tmp_path, capsys):
    cfg = C.load(write_config(tmp_path, fake_dzgorge.root))
    st = LocalStore(tmp_path / "store")
    st.put(ctl.CONFIG, C.dump(cfg).encode())
    net = b"weights"
    st.put_many({ctl.version_file(0, "meta.json"): json.dumps({"version": 0, "net": False}).encode(),
                 ctl.version_file(1, "model.dzgw"): net,
                 ctl.version_file(1, "meta.json"): json.dumps({"version": 1, "net": True,
                                                                "sha256": {"dzgw": hashlib.sha256(net).hexdigest()}
                                                                }).encode(),
                 ctl.latest_path(): json.dumps({"version": 1}).encode()})
    assert main(["eval", str(tmp_path / "store"), "--opponents", "prev,sbtactical", "--pairs", "1", "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["version"] == 1 and "gonet=" in s["spec"] and s["pairs"] == 1
    assert s["results"][0]["spec"].startswith("az:") and "gonet" not in s["results"][0]["spec"]   # v0: no net
    assert main(["eval", f"{tmp_path / 'store'}@v0", "--opponents", "bot", "--pairs", "1", "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["version"] == 0 and "gonet" not in s["spec"]


# ------------------------------------------------------------------------------------------------ engine

def test_engine_build_runs_the_pinned_script(tmp_path, monkeypatch, capsys):
    seen = {}

    class Done:
        returncode = 0

    def fake_run(argv, env):
        seen.update(argv=argv, gorge_dir=env.get("GORGE_DIR"))
        return Done()
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert main(["engine", "build", "--dir", str(tmp_path / "g")]) == 0
    assert seen["argv"] == ["bash", str(C.REPO / "engine" / "gorge" / "build.sh")]
    assert seen["gorge_dir"] == str(tmp_path / "g")
    assert "engine.root" in capsys.readouterr().out                    # not the base config's root: say so
    assert main(["engine", "build", "--config", str(write_config(tmp_path, tmp_path / "from-config"))]) == 0
    assert seen["gorge_dir"] == str(tmp_path / "from-config")


def test_engine_version_and_coverage(fake_dzgorge, tmp_path, monkeypatch, capsys):
    (fake_dzgorge.root / "bin" / "dzgorge.build.json").write_text(json.dumps({"ref": "f" * 40, "repo": "r"}))
    assert main(["engine", "version", "--root", str(fake_dzgorge.root)]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["build"]["ref"] == "f" * 40 and v["binary_sha256"]
    assert main(["engine", "version", "--root", str(tmp_path / "none")]) == 1
    from draftzero.engine.gorge import Gorge
    cov = {"cards": 244, "decks": 80, "unsupported_cards": 1, "failed_decks": 0, "supported_decks": 79,
           "unsupported": {"Some Card": {"decks": 1, "missing": "x"}}, "supported": []}
    monkeypatch.setattr(Gorge, "coverage", lambda self: cov)
    capsys.readouterr()
    assert main(["engine", "coverage", *sets(fake_dzgorge.root)]) == 0
    out = capsys.readouterr().out
    assert "79 of 80 decks" in out and "Some Card" in out
    assert main(["engine", "coverage", "--json", *sets(fake_dzgorge.root)]) == 0
    assert json.loads(capsys.readouterr().out) == cov


# ------------------------------------------------------------------------------------------------ pods (no pod is created)

class FakeAPI:
    terminated: list = []
    fail: set = set()                                    # ids whose terminate fails

    def __init__(self, *a, **k):
        pass

    def balance(self):
        return {"clientBalance": 30.5, "currentSpendPerHr": 0.0, "spendLimit": 80}

    def list(self):
        return [{"id": "p1", "name": "dz-a1", "desiredStatus": "RUNNING", "costPerHr": 0.2, "vcpuCount": 16,
                 "publicIp": "1.2.3.4", "portMappings": {"22": 40100}}]

    def find(self, name):
        return [p for p in self.list() if p["name"] == name]

    def terminate(self, pid):
        if pid in FakeAPI.fail:
            from draftzero.workers.runpod import RunPodError
            raise RunPodError(f"DELETE https://rest.runpod.io/v1/pods/{pid}: HTTP 404: not found")
        FakeAPI.terminated.append(pid)

    def offers(self, *a, **k):
        return [{"cloud": "COMMUNITY", "offer": "RTX 3090", "price": 0.22, "vcpu": 16, "ram": 64, "stock": "High",
                 "usd_per_vcpu_hr": 0.0138, "gpu_id": "NVIDIA GeForce RTX 3090"}]


@pytest.fixture
def pods(monkeypatch):
    from draftzero.workers import runpod as rp
    FakeAPI.terminated, FakeAPI.fail = [], set()
    ups = []

    def fake_up(api, spec, *, max_hours, files=None, setup=True, keep_on_failure=False):
        ups.append({"spec": spec, "max_hours": max_hours, "files": files, "setup": setup})
        from draftzero.workers.ssh import SSHTarget
        return rp.Pod("p9", SSHTarget(host="5.6.7.8", user="root", port=40001, strict=False), {"name": spec.name})
    monkeypatch.setattr(rp, "RunPodAPI", FakeAPI)
    monkeypatch.setattr(rp, "up", fake_up)
    return ups


def test_pod_up_needs_price_hours_and_confirmation(pods, monkeypatch, capsys):
    base = ["pod", "up", "--name", "dz-a1", "--gpu", "NVIDIA GeForce RTX 3090"]
    for missing in (["--max-hours", "1"], ["--max-price", "0.3"]):
        with pytest.raises(SystemExit) as e:
            main(base + missing)
        assert e.value.code == 2
    capsys.readouterr()
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert main(base + ["--max-price", "0.3", "--max-hours", "1.5", "--no-bundle"]) == 2
    out, err = capsys.readouterr()
    assert "cost guard" in out and "$0.300/h" in out and "1.5 h" in out and "$0.45" in out and "$30.50" in out
    assert "1.5 h from its create, the wait for SSH included" in out
    assert "--yes" in err and pods == []
    assert main(base + ["--max-price", "0.3", "--max-hours", "1.5", "--no-bundle", "--yes"]) == 0
    assert len(pods) == 1 and pods[0]["max_hours"] == 1.5 and pods[0]["files"] is None
    s = pods[0]["spec"]
    assert (s.name, s.gpu, s.max_price, s.cloud, s.allow_secure) == ("dz-a1", "NVIDIA GeForce RTX 3090", 0.3,
                                                                     "COMMUNITY", False)
    assert "dz pod down p9" in capsys.readouterr().out
    assert main(["pod", "up", "--name", "c", "--cpu", "cpu5c", "--max-price", "0.3", "--max-hours", "1",
                 "--yes"]) == 2                                          # CPU pods need --secure and a data center
    assert main(base + ["--max-price", "0.3", "--max-hours", "100", "--yes", "--no-bundle"]) == 2
    assert len(pods) == 1


def test_pod_list_balance_offers_down(pods, capsys):
    assert main(["pod", "list"]) == 0 and "root@1.2.3.4 -p 40100" in capsys.readouterr().out
    assert main(["pod", "balance"]) == 0 and "$30.50" in capsys.readouterr().out
    assert main(["pod", "offers"]) == 0 and "RTX 3090" in capsys.readouterr().out
    assert main(["pod", "down", "p7", "--name", "dz-a1"]) == 0 and FakeAPI.terminated == ["p7", "p1"]
    assert main(["pod", "down"]) == 2


def test_pod_down_tries_every_pod_and_reports_the_failures(pods, capsys):
    """One failed terminate (a wrong or already removed id, a 5xx) must not leave the later pods billing."""
    FakeAPI.fail = {"gone1"}
    assert main(["pod", "down", "gone1", "live2", "live2", "--name", "dz-a1"]) == 1
    assert FakeAPI.terminated == ["live2", "p1"]                 # every pod tried, each once
    out, err = capsys.readouterr()
    assert "terminated live2" in out and "1 pod(s) left: p1" in out
    assert "gone1" in err and "HTTP 404" in err and "FAILED" in err


# ------------------------------------------------------------------------------------------------ relay

def test_relay_wraps_relay_main(tmp_path, monkeypatch, capsys):
    from draftzero.loop import relay
    calls = []
    monkeypatch.setattr(relay, "main", lambda argv: calls.append(argv) or 0)
    cfg = write_config(tmp_path, tmp_path / "g")
    assert main(["relay", str(cfg), "--host", "pod1", "/root/dz/run", "root@1.2.3.4 -p 40066", "--once"]) == 0
    assert calls[-1] == ["--store", str(tmp_path / "store"), "--interval", "60", "--host", "pod1", "/root/dz/run",
                         "root@1.2.3.4 -p 40066", "--once"]
    bucket = write_config(tmp_path, tmp_path / "g", store="hf://buckets/o/b")
    assert main(["relay", str(bucket), "--host", "a", "/r"]) == 2
    assert main(["relay", str(bucket), "--store", str(tmp_path / "mirror"), "--hosts", "h.yml"]) == 0
    assert calls[-1] == ["--store", str(tmp_path / "mirror"), "--interval", "60", "--hosts", "h.yml",
                         "--upstream", "hf://buckets/o/b"]


def test_run_warns_when_code_names_a_release_this_checkout_is_not_at(monkeypatch, tmp_path):
    from draftzero import manifest
    cfg = C.from_dict({"run": "w", "code": "v0.2.0"})
    monkeypatch.setattr(manifest, "_checkout", lambda: tmp_path)
    monkeypatch.setattr(manifest, "git_state", lambda repo: {"commit": "a" * 40, "branch": "x", "dirty": False})
    monkeypatch.setattr(manifest, "_git", lambda repo, *args: "v0.2.0")
    assert cli._code_warning(cfg) is None
    monkeypatch.setattr(manifest, "_git", lambda repo, *args: None)
    assert "aaaaaaaaa" in cli._code_warning(cfg) and "not publishable" in cli._code_warning(cfg)
    assert cli._code_warning(C.from_dict({"run": "w"})) is None                       # code: dev
