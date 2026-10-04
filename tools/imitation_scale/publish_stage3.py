"""Publish one of experiment #4's trained networks (docs/018) to the project's Hugging Face model repo, with a model card:
the checkpoints (best_policy, best_value, best, final), the run's config, learning curves, summary,
the 30% checks' choice and the test-split measures. latest.pt (the optimiser's state) and the
30-minute checkpoints stay behind.

Copy the run's directory from the training machine first (runs/exp4/main), then, with a write token
(huggingface-cli login, or HF_TOKEN):

    python tools/imitation_scale/publish_stage3.py --run runs/exp4/main [--repo danbrooks/draftzero-checkpoints]
        [--prefix exp4/stage3] [--dry-run]

Afterwards: ImitationNet.load("hf://danbrooks/draftzero-checkpoints/exp4/stage3/best_policy.pt.gz").
The full-data MLP (run 2): --run runs/exp4/mlp_1ep --prefix exp4/mlp_1ep --name "the full-data MLP".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FILES = ["best_policy.pt.gz", "best_value.pt.gz", "best.pt.gz", "final.pt.gz", "config.json", "evals.jsonl",
         "summary.json", "choice.json", "test_best_policy.json", "test_best_value.json", "train.log", "README.md"]
MEASURES = [("policy/top1_nonpass", "policy: non-Pass top-1"), ("policy/set_nll", "policy: set NLL"),
            ("binary/acc", "attack: accuracy"), ("opp_block/top1", "block: top-1"),
            ("replay_target/top1", "targets: top-1"), ("value/auc", "value: AUC against the result"),
            ("value/logloss", "value: log-loss"), ("opp_priority/pass_top1", "Pass on top, opponent's turn")]


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def network_line(arch: dict, vocab_rows: int | None) -> str:
    if arch.get("type") == "mlp":
        pool = {"max": "max", "mean": "mean", "maxmean": "max + mean"}.get(arch.get("bag_mode"), arch.get("bag_mode"))
        table = f"{vocab_rows:,} " if vocab_rows else ""
        return (f"- MLP: a table of {table}feature embeddings of width {arch.get('width')}, pooled over the state's "
                f"features ({pool}), then {arch.get('layers')} residual {'SwiGLU' if arch.get('ffn', 'swiglu') == 'swiglu' else arch.get('ffn')} blocks with "
                f"{'LayerNorm' if arch.get('mlp_norm', 'layer') == 'layer' else arch.get('mlp_norm')}; five heads with a "
                f"hidden layer of {arch.get('head_hidden')} (the player's priority, the opponent's priority, targets, "
                "yes/no, value).")
    return (f"- {arch.get('type', 'transformer')}, {arch.get('layers')} {'layer' if arch.get('layers') == 1 else 'layers'}, width {arch.get('width')}, "
            f"{'pre-LN' if arch.get('norm_first') else 'post-LN'}, mean pooling over the state's tokens; five heads "
            "(the player's priority, the opponent's priority, targets, yes/no, value).")


def model_card(run: Path, repo: str, prefix: str, name: str = "stage 3") -> str:
    cfg = _json(run / "config.json") or {}
    summ = _json(run / "summary.json") or {}
    choice = _json(run / "choice.json") or {}
    tests = {c: _json(run / f"test_{c}.json") for c in ("best_policy", "best_value")}
    evals = [json.loads(x) for x in (run / "evals.jsonl").read_text().splitlines() if x.strip()] \
        if (run / "evals.jsonl").exists() else []
    last = evals[-1] if evals else {}
    arch = cfg.get("arch", {})
    lines = [
        f"# draft-zero experiment #4, {name}: the imitation network",
        "",
        "A network trained by imitation on 17lands' top players' FDN Premier Draft games, replayed in XMage",
        "(github.com/danieljbrooks/draft-zero, docs/017 and docs/018). It predicts, from a game state as",
        "MageZero's StateEncoder encodes it for the player to act: what the human would do (priority, target",
        "and attack decisions), and the probability that the player wins.",
        "",
        "## Files",
        "",
        "| File | What |",
        "|---|---|",
        "| `best_policy.pt.gz` | the checkpoint with the lowest validation policy NLL: **use this for play and search priors** |",
        "| `best_value.pt.gz` | the lowest validation value log-loss (the value head can peak earlier than the policy) |",
        "| `best.pt.gz`, `final.pt.gz` | the best combined measure; the end of training |",
        "| `config.json`, `evals.jsonl`, `summary.json`, `train.log` | the run's settings, learning curves and log |",
        "| `test_best_policy.json`, `test_best_value.json` | the measures on the held-out test split |",
        "",
        "## Use",
        "",
        "```python",
        "from draftzero.gameplay.imitation_net import ImitationNet",
        f'net = ImitationNet.load("hf://{repo}/{prefix}/best_policy.pt.gz")',
        "out = net.evaluate([features])[0]          # features: the StateEncoder's feature ids for one state",
        "p_win = net.win_probability(out[\"value\"])  # (1 + v) / 2, from the acting player's seat",
        "probs = net.policy_over(out[\"policy_player\"], legal_action_indices)",
        "```",
        "",
        "For search and games, serve it: `tools/search_bench/value_server.py --model best_policy.pt.gz --policy`",
        "(MageZero's inference protocol), then `tools/imitation_scale/play.py --bot1 policy` (no search) or `il_bc`",
        "(IS-MCTS with the policy as priors and the value at the leaves). Action indices are MageZero's",
        "(names: `assets/vocab/FDN_SPG.tsv`, read through `MZ_ACTION_VOCAB`).",
        "",
        "## The network",
        "",
        network_line(arch, (summ.get("data") or {}).get("vocab_rows")),
        f"- Trained on {cfg.get('fraction', 1.0):.0%} of the training games' rows"
        + (f" for {cfg['max_epochs']:g} epoch{'' if cfg['max_epochs'] == 1 else 's'}" if cfg.get("max_epochs") else "") + ": "
        f"lr {cfg.get('lr')} with a {cfg.get('warmup_steps')}-step warm-up, "
        + (f"then a cosine down to {cfg.get('lr_min_frac', 0.1):g}x" if cfg.get("lr_schedule") == "cosine" else "then constant")
        + f"; value targets {'TD(lambda = %s)' % cfg.get('td_lambda') if cfg.get('value_target') == 'td' else 'the game result'}"
        f" on {cfg.get('value_per_game') or 'every'} positions a game, value-loss weight {cfg.get('value_weight')}"
        + (f"; the policy loss weighted on the rows where the human acted: {cfg['act_weights']}" if cfg.get("act_weights") else "")
        + (f"; the embedding table at {cfg['emb_lr_mult']:g}x the learning rate" if cfg.get("emb_lr_mult", 1) != 1 else "")
        + (f"; weight decay {cfg['weight_decay']:g}" if cfg.get("weight_decay") else "")
        + (f"; the saved weights are an average (EMA {cfg['ema_decay']:g})" if cfg.get("ema_decay") else "")
        + (f"; started from `{cfg['init_checkpoint']}`" if cfg.get("init_checkpoint") else "")
        + ".",
    ]
    if choice.get("why"):
        lines += ["- Settled by the 30% checks: " + "; ".join(choice["why"]) + "."]
    if summ:
        lines += [f"- {summ.get('step', 0):,} steps, {summ.get('epochs', 0):.2f} epochs, "
                  f"{summ.get('train_time_s', 0) / 3600:.1f} hours of training."]
    lines += ["", "## Measures", "", "| | " + " | ".join(n for _, n in MEASURES) + " |",
              "|---|" + "---|" * len(MEASURES)]
    if last:
        lines.append("| validation, end of training | " + " | ".join(
            f"{last[k]:.3f}" if isinstance(last.get(k), (int, float)) else "–" for k, _ in MEASURES) + " |")
    for c, t in tests.items():
        if t:
            m = t.get("measures", t)
            lines.append(f"| test, `{c}` | " + " | ".join(
                f"{m[k]:.3f}" if isinstance(m.get(k), (int, float)) else "–" for k, _ in MEASURES) + " |")
    lines += ["", "Humans put Pass on top in the opponent's turn 93.4% of the time.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--repo", default="danbrooks/draftzero-checkpoints")
    ap.add_argument("--prefix", default="exp4/stage3")
    ap.add_argument("--name", default="stage 3", help="the model card's heading: experiment #4, <name>")
    ap.add_argument("--dry-run", action="store_true", help="write README.md and list the files, upload nothing")
    a = ap.parse_args(argv)
    (a.run / "README.md").write_text(model_card(a.run, a.repo, a.prefix, a.name))
    files = [f for f in FILES if (a.run / f).exists()]
    print(f"{len(files)} files for {a.repo}/{a.prefix}: {', '.join(files)}")
    if a.dry_run:
        return 0
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=a.repo, folder_path=str(a.run), path_in_repo=a.prefix, allow_patterns=files,
                          commit_message=f"exp4 {a.name}: {a.prefix}")
    print(f"published: hf://{a.repo}/{a.prefix}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
