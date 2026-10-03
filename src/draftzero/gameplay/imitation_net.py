"""Experiment #4's imitation networks, for use outside training: load a checkpoint and evaluate game
states (docs/018, "Using the stage-3 model").

A state is what MageZero's StateEncoder writes for the player to act: a list of feature ids (the
`indices` of MageZero's inference protocol, and of the imitation tables). The checkpoint's feature
vocabulary maps them to the network's embedding rows; ids it never saw are dropped, as the inference
server drops them. The outputs are the network's five heads:

    value            v in [-1, 1] from the acting player's seat; P(win) = (1 + v) / 2
    policy_player    1,024 logits by MageZero action index: the player's own priority decisions
    policy_opponent  1,024 logits: MageZero's head for the opponent's priority (not trained by imitation)
    policy_target    1,024 logits: target choices
    policy_binary    2 logits: yes / no questions (attack with this creature?)

Action indices are MageZero's ActionEncoder indices; assets/vocab/FDN_SPG.tsv names them (the bridge reads
it through MZ_ACTION_VOCAB). A policy is the softmax of a head's logits over the decision's legal indices
(`policy_over`).

    from draftzero.gameplay.imitation_net import ImitationNet
    net = ImitationNet.load("hf://danbrooks/draftzero-checkpoints/exp4/stage3/best_policy.pt.gz")
    out = net.evaluate([features])[0]
    p_win = (1 + out["value"]) / 2
    probs = net.policy_over(out["policy_player"], legal_indices)

The same checkpoint serves IS-MCTS and play.py's bots through tools/search_bench/value_server.py --policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

HEADS = ("policy_player", "policy_opponent", "policy_target", "policy_binary", "value")
HF_PREFIX = "hf://"


def resolve(path: str | Path) -> Path:
    """A local path as it is; `hf://<owner>/<repo>/<file in the repo>` downloads (and caches) the file
    from that Hugging Face model repo (a private repo needs HF_TOKEN or a `huggingface-cli login`)."""
    s = str(path)
    if not s.startswith(HF_PREFIX):
        return Path(s)
    owner, repo, *rest = s[len(HF_PREFIX):].split("/")
    if not rest:
        raise ValueError(f"{s}: want hf://<owner>/<repo>/<path in the repo>")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(f"{owner}/{repo}", "/".join(rest)))


@dataclass
class ImitationNet:
    model: torch.nn.Module
    vocab: object                 # magezero.vocab.FeatureVocab: feature ids -> embedding rows
    arch: dict
    info: dict | None             # the trainer's record: config, data, the run's validation measures
    device: torch.device

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "ImitationNet":
        """Any checkpoint experiment #4's trainer writes (best_policy / best_value / best / final .pt.gz,
        latest.pt), whatever its shape: the transformer, the MLP, the value-tower variants."""
        from draftzero.gameplay import supervised as sv
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, vocab, meta = sv.load_any_checkpoint(resolve(path), device=dev)
        return cls(model=model, vocab=vocab, arch=meta["arch"], info=meta["info"], device=dev)

    @torch.no_grad()
    def evaluate(self, states: Sequence[Sequence[int]], batch: int = 256) -> list[dict]:
        """The five heads for each state (a list of StateEncoder feature ids), as numpy arrays (value a
        float). A state with no known feature gets the neutral output the server gives (zero logits, v = 0)."""
        out: list[dict | None] = [None] * len(states)
        width = int(self.arch["policy_width"])
        for lo in range(0, len(states), batch):
            part = list(range(lo, min(lo + batch, len(states))))
            idx, off, keep = [], [], []
            for i in part:
                rows, _ = self.vocab.map_bags(list(states[i]), [0])
                if len(rows) == 0:
                    out[i] = {"policy_player": np.zeros(width, np.float32), "policy_opponent": np.zeros(width, np.float32),
                              "policy_target": np.zeros(width, np.float32), "policy_binary": np.zeros(2, np.float32),
                              "value": 0.0}
                    continue
                off.append(sum(len(r) for r in idx))
                idx.append(np.asarray(rows, np.int64))
                keep.append(i)
            if not keep:
                continue
            t_idx = torch.as_tensor(np.concatenate(idx), device=self.device)
            t_off = torch.as_tensor(np.asarray(off, np.int64), device=self.device)
            heads = [h.float().cpu().numpy() for h in self.model(t_idx, t_off)]
            for j, i in enumerate(keep):
                o = {name: heads[k][j] for k, name in enumerate(HEADS[:4])}
                o["value"] = float(heads[4].reshape(-1)[j])
                out[i] = o
        return out  # type: ignore[return-value]

    @staticmethod
    def policy_over(logits: np.ndarray, legal: Sequence[int], temperature: float = 1.0) -> np.ndarray:
        """Probabilities over a decision's legal action indices: softmax(logits[legal] / temperature);
        temperature 0 puts all the mass on the most likely."""
        z = np.asarray(logits, np.float64)[np.asarray(legal, np.int64) % len(logits)]
        if temperature <= 0:
            p = np.zeros(len(z))
            p[int(np.argmax(z))] = 1.0
            return p
        z = (z - z.max()) / temperature
        e = np.exp(z)
        return e / e.sum()

    @staticmethod
    def win_probability(value: float) -> float:
        return (1.0 + float(value)) / 2.0

    def describe(self) -> str:
        cfg = (self.info or {}).get("config") or {}
        n = sum(p.numel() for p in self.model.parameters())
        return (f"{type(self.model).__name__}: {n:,} parameters, arch {self.arch}, "
                f"{len(self.vocab):,} features; lr {cfg.get('lr')}, value target {cfg.get('value_target')} "
                f"(lambda {cfg.get('td_lambda')}), act_weights {cfg.get('act_weights')}")
