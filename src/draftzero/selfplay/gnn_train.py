"""The GNN's self-play training (docs/021 §1.3): fine-tune from the current version on the newest games.

Built on graph_supervised.py (the imitation trainer, which this leaves untouched): its checkpoints, vocabularies,
batches (make_batch), option logits (graph_net.option_logits) and set NLL (row_losses, for the human rows).
What's added, per self-play row:

    policy   cross-entropy against the search's visit shares over the row's options
    value    the value head's cross-entropy against the TD(lambda) target, on at most value_per_game rows a game
    KL       beta x KL(pi_start || pi) over the options, pi_start the starting network (frozen); beta anneals from
             kl_weight to kl_weight_end over kl_anneal_versions versions
    human    a share of each batch from the imitation tables, with graph_supervised's set NLL and the result

The model and its AdamW state live across versions (one continuous trainer); full_state() saves both, so another
machine can resume. Each update trains reuse x (new positions) / batch_rows steps on the window's training rows.

Options for docs/028's transition arms (all off by default, which is the loop's recipe above):
    policy_weight     x the self-play policy loss (0: train the value only, through the whole network)
    policy_target     visits (the search's visit shares, sharpened to visits^(1/visit_temp) when visit_temp < 1:
                      exploration's one- and two-visit options mostly drop out) | cq: the behaviour policy tilted by the search's values,
                      pi'(a) ~ pi_b(a) exp(cq_scale (Q(a) - Q_root)), an unvisited option at Q_root (docs/028 §2)
    train_only        parameter-name prefixes to train (graph_supervised's), the rest frozen; None: everything
    value_head_init   keep | reset (fresh value head) | shrink (shrink and perturb it: 0.4 w + 0.1 std(w) noise)
    kl_mask_use       no KL on yes/no (CHOOSE_USE) rows: the starting network's use head never trained
    human_value_weight  x the value loss on human rows (None: value_weight)
    human_data_cache  graph_supervised's memory-mapped cache of the human tables (fast load, block reads)
    seed              the trainer's random draws (None: the clock)
The offline driver (tools/selfplay_transition/sp_train.py) trains several arms on one fixed set of games through
prepare() and run_steps().
"""
from __future__ import annotations

import copy
import io
import gzip
import math
import time
from pathlib import Path

import numpy as np
import torch

from draftzero.gameplay import graph_net as gn
from draftzero.gameplay import graph_supervised as gs
from draftzero.gameplay import supervised as sv
from draftzero.selfplay import tables


def device_of(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _segment_sum(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, seg, x)


def option_logprobs(model, bd: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(log-probability of each option within its row, the value head's input) for a batch."""
    out = model(bd["graphs"])
    ob = bd["opt"]
    B = bd["graph_type"].shape[0]
    lg = gn.option_logits(out, ob, bd["graph_type"], bd["opt_index"])
    lg = torch.where(torch.isfinite(lg), lg, torch.full_like(lg, -1e4))
    return lg - gn.segment_logsumexp(lg, ob.opt_row, B)[ob.opt_row], out.value_x.float()


def value_mask(game: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """At most `cap` rows per game, chosen at random (all of a game's positions share one result)."""
    if not len(game):
        return np.zeros(0, bool)
    order = np.lexsort((rng.random(len(game)), game))
    g = game[order]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    m = np.zeros(len(game), bool)
    m[order] = rank < cap
    return m


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    ok = np.isfinite(score) & np.isfinite(y)
    s, w = score[ok], y[ok] > 0
    if w.sum() == 0 or (~w).sum() == 0:
        return None
    rk = rankdata(s)
    return float((rk[w].sum() - w.sum() * (w.sum() + 1) / 2) / (w.sum() * (~w).sum()))


class GnnTrainer:
    def __init__(self, init_ckpt: Path, ref_ckpt: Path, cfg: dict, log=print, full_state: Path | None = None):
        self.cfg, self.log = cfg, log
        self.dev = device_of(cfg["device"])
        self.dtype = torch.bfloat16 if self.dev.type == "cuda" else None
        self.model, self.vocab, self.edge_vocab, meta = gs.load_checkpoint(init_ckpt, self.dev)
        self.ref, rv, re_, _ = gs.load_checkpoint(ref_ckpt, self.dev)
        if not (np.array_equal(np.asarray(rv.ids), np.asarray(self.vocab.ids))
                and np.array_equal(np.asarray(re_.ids), np.asarray(self.edge_vocab.ids))):
            raise ValueError("the starting network and the current one have different vocabularies")
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()
        seed = cfg.get("seed")
        self.rng = np.random.default_rng(int(time.time()) if seed is None else int(seed))
        if seed is not None:
            torch.manual_seed(int(seed))
        self.init_value_head(cfg.get("value_head_init", "keep"))
        self.set_trainable(cfg.get("train_only"))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.step = 0
        self.updates = 0
        self.lr_fn = None                    # step within the current run -> lr multiplier (the offline driver's)
        self.behaviour = None                # the policy the games came from (policy_target cq)
        if full_state is not None and Path(full_state).exists():
            self.load_full(full_state)
        self.human = self._load_human() if cfg["human_share"] > 0 else None

    def init_value_head(self, how: str) -> None:
        """keep | reset | shrink: the value head's Linear layers fresh, or shrunk and perturbed (Ash and Adams 2020:
        0.4 w + 0.1 std(w) N(0, 1)), the rest of the network untouched."""
        if how in (None, "keep"):
            return
        lin = [m for m in self.model.value_head if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for m in lin:
                if how == "reset":
                    m.reset_parameters()
                elif how == "shrink":
                    for p in (m.weight, m.bias):
                        sd = float(p.std()) if p.numel() > 1 else 0.0
                        p.mul_(0.4).add_(0.1 * sd * torch.randn_like(p))
                else:
                    raise ValueError(f"value_head_init must be keep, reset or shrink, not {how!r}")
        self.log(f"gnn_train: value head {how}")

    def set_trainable(self, prefixes) -> None:
        """Train only the parameters whose names start with one of `prefixes` (None: all)."""
        keep = tuple(prefixes) if prefixes else None
        for n, p in self.model.named_parameters():
            p.requires_grad_(keep is None or n.startswith(keep))
        if keep is not None:
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if not n_train:
                raise ValueError(f"train_only {list(keep)} matches no parameter")
            self.log(f"gnn_train: training only {list(keep)} ({n_train:,} parameters)")

    # -------------------------------------------------------------------------------- state
    def full_state(self, path: Path, info: dict) -> None:
        buf = io.BytesIO()
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
                    "updates": self.updates, "info": info}, buf)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            f.write(buf.getvalue())
        tmp.replace(path)

    def load_full(self, path: Path) -> None:
        with gzip.open(path, "rb") as f:
            st = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.step, self.updates = int(st["step"]), int(st["updates"])
        self.log(f"gnn_train: resumed the trainer's state at step {self.step} ({self.updates} updates)")

    def save_weights(self, path: Path, info: dict) -> None:
        gs.save_weights(path, self.model, self.vocab, self.edge_vocab, info)

    # -------------------------------------------------------------------------------- data
    def _load_human(self):
        c = self.cfg
        from draftzero.selfplay.config import path as repo_path
        hcfg = {"tables_dir": str(repo_path(c["human_tables_dir"])), "fraction": c["human_fraction"], "val_rows": 0}
        if c.get("human_data_cache"):       # the imitation run's memory-mapped cache: every row, loads in seconds
            hcfg.update(data_cache=str(c["human_data_cache"]), stream_cache=True, fraction=1.0)
        data = gs.load_data(hcfg, vocabs=(self.vocab, self.edge_vocab), splits=("train",), log=self.log)
        self.log(f"gnn_train: human rows {sum(t.n for t in data.train)} from {len(data.train)} tables")
        return data.train

    def _human_batch(self, n: int, block: int = 8) -> dict:
        """n human rows, in runs of `block` consecutive rows from random places (memory-mapped tables read in
        runs, as graph_supervised's block-local sampling does)."""
        sizes = np.asarray([t.n for t in self.human], np.float64)
        nb = max(1, -(-n // block))
        pick = self.rng.choice(len(self.human), size=nb, p=sizes / sizes.sum())
        parts = []
        for ti in range(len(self.human)):
            k = int((pick == ti).sum())
            if not k:
                continue
            tn = self.human[ti].n
            starts = self.rng.integers(0, max(1, tn - block + 1), size=k)
            r = np.unique(np.clip((starts[:, None] + np.arange(block)).ravel(), 0, tn - 1))
            parts.append((ti, r))
        b = gs.make_batch(self.human, parts)
        b["w"] = np.concatenate([self.human[ti].w[r] for ti, r in parts]).astype(np.float32)
        b["z"] = np.concatenate([self.human[ti].z[r] for ti, r in parts]).astype(np.float32)
        return b

    def beta(self) -> float:
        c = self.cfg
        f = min(1.0, self.updates / max(1, c["kl_anneal_versions"]))
        return float(c["kl_weight"] + (c["kl_weight_end"] - c["kl_weight"]) * f)

    # -------------------------------------------------------------------------------- training
    def train(self, rows: dict, new_positions: int) -> dict:
        """One update on the window's training rows (not held out). Returns its metrics."""
        c = self.cfg
        if not self.prepare(rows):
            return {"steps": 0}
        steps = max(1, math.ceil(c["reuse"] * max(new_positions, 1) / c["batch_rows"]))
        out = self.run_steps(steps)
        self.updates += 1
        out.update(new_positions=int(new_positions))
        return out

    def prepare(self, rows: dict, heldout: bool = False) -> int:
        """Make the training rows (not held out, unless `heldout`) this trainer's table; returns their number."""
        c = self.cfg
        train_rows = rows if heldout or not rows["heldout"].any() else tables.select(rows, ~rows["heldout"])
        self.table = t = tables.to_graph_table(train_rows, self.vocab, self.edge_vocab)
        if t.n == 0:
            return 0
        self.vmask = value_mask(t.game, int(c["value_per_game"]), self.rng) & np.isfinite(t.aux["z_td"])
        vt = float(c.get("visit_temp", 1.0))
        p = t.aux["opt_p"].astype(np.float64)
        if vt != 1.0 and len(p):
            row = np.repeat(np.arange(t.n), np.diff(t.row_opt_ptr))
            p = p ** (1.0 / vt)
            p = p / np.maximum(np.bincount(row, weights=p, minlength=t.n)[row], 1e-12)
        t.aux["opt_p_train"] = p.astype(np.float32)
        if c.get("policy_target", "visits") == "cq":
            self.behaviour = copy.deepcopy(self.model).eval()      # the network that played these games
            for p in self.behaviour.parameters():
                p.requires_grad_(False)
        return t.n

    def _cq_target(self, bd: dict, t, r: np.ndarray, ob) -> torch.Tensor:
        """pi'(a) ~ pi_b(a) exp(cq_scale w(a) (Q(a) - Q_root)): the behaviour policy tilted toward the options the
        search valued above the root; an unvisited option (no Q) keeps Q_root (no tilt). w(a) = n(a) / (n(a) + n0)
        discounts values from few visits (cq_visit_shrink = n0; 0: none): a rarely tried option's value is the least
        reliable (Hamrick et al. 2021), and PIMC's one world adds noise."""
        B = len(r)
        with sv._autocast(self.dev, self.dtype):
            logb, _ = option_logprobs(self.behaviour, bd)
        logb = logb.float()
        _, q = gn.gather(t.row_opt_ptr, r, t.aux["opt_q"])
        q = torch.from_numpy(np.ascontiguousarray(q, np.float32)).to(self.dev)
        qr = torch.from_numpy(t.aux["q_root"][r]).to(self.dev)[ob.opt_row]
        adv = torch.where(torch.isfinite(q) & torch.isfinite(qr), q - qr, torch.zeros_like(q))
        n0 = float(self.cfg.get("cq_visit_shrink", 0.0))
        if n0 > 0:
            _, nv = gn.gather(t.row_opt_ptr, r, t.aux["opt_n"])
            nv = torch.from_numpy(np.ascontiguousarray(nv, np.float32)).to(self.dev)
            adv = adv * torch.where(torch.isfinite(nv), nv / (nv + n0), torch.ones_like(nv))
        lg = logb + float(self.cfg.get("cq_scale", 10.0)) * adv
        return (lg - gn.segment_logsumexp(lg, ob.opt_row, B)[ob.opt_row]).exp()

    def run_steps(self, steps: int) -> dict:
        """`steps` training steps on the prepared table."""
        c = self.cfg
        t = self.table
        n = t.n
        n_human = int(round(c["batch_rows"] * c["human_share"])) if self.human else 0
        n_self = max(1, c["batch_rows"] - n_human)
        beta = self.beta()
        pw = float(c.get("policy_weight", 1.0))
        hvw = c.get("human_value_weight")
        hvw = float(c["value_weight"] if hvw is None else hvw)
        mask_use = bool(c.get("kl_mask_use", False))
        cq = c.get("policy_target", "visits") == "cq"
        self.model.train()
        sums = {"loss": 0.0, "policy": 0.0, "value": 0.0, "kl": 0.0, "human": 0.0}
        t0 = time.time()
        for i in range(steps):
            lr = c["lr"] * min(1.0, (self.step + 1) / max(1, c["warmup_steps"]))
            if self.lr_fn is not None:
                lr *= self.lr_fn(i)
            for g in self.opt.param_groups:
                g["lr"] = lr
            r = np.sort(self.rng.choice(n, size=min(n_self, n), replace=n_self > n))
            b = gs.make_batch([t], [(0, r)])
            bd = gs.to_device(b, self.dev)
            ob = bd["opt"]
            B = len(r)
            with sv._autocast(self.dev, self.dtype):
                logp, vx = option_logprobs(self.model, bd)
                logq = None
                if beta > 0:
                    with torch.no_grad():
                        logq, _ = option_logprobs(self.ref, bd)
            if cq:
                with torch.no_grad():
                    p = self._cq_target(bd, t, r, ob)
            else:
                _, p = gn.gather(t.row_opt_ptr, r, t.aux.get("opt_p_train", t.aux["opt_p"]))
                p = torch.from_numpy(np.ascontiguousarray(p, np.float32)).to(self.dev)
            with sv._fp32(self.dev, self.dtype):
                logp = logp.float()
                pol = _segment_sum(-p * logp, ob.opt_row, B).mean()
                if logq is not None:
                    logq = logq.float()
                    q = logq.exp()
                    per = _segment_sum(q * (logq - logp), ob.opt_row, B)
                    if mask_use:
                        per = per * (bd["graph_type"] != 5).float()
                    kl = per.mean()
                else:
                    kl = torch.zeros((), device=self.dev)
                vm = torch.from_numpy(self.vmask[r]).to(self.dev)
                z = torch.from_numpy(t.aux["z_td"][r]).to(self.dev)
                vl = sv.value_bce_from_logit(vx[vm], z[vm]).mean() if vm.any() else vx.sum() * 0
                loss = pw * pol + beta * kl + c["value_weight"] * vl
            hl = torch.zeros((), device=self.dev)
            if n_human:
                hb = self._human_batch(n_human)
                hd = gs.to_device(hb, self.dev)
                with sv._autocast(self.dev, self.dtype):
                    hout = self.model(hd["graphs"])
                with sv._fp32(self.dev, self.dtype):
                    nll, _ = gs.row_losses(hout, hd)
                    w = torch.from_numpy(hb["w"]).to(self.dev)
                    ok = (w > 0) & torch.isfinite(nll)
                    hp = (torch.where(ok, nll, torch.zeros_like(nll)) * w).sum() / ok.sum().clamp(min=1)
                    hz = torch.from_numpy(hb["z"]).to(self.dev)
                    hm = torch.isfinite(hz)
                    hv = sv.value_bce_from_logit(hout.value_x[hm], hz[hm]).mean() if hm.any() and hvw else hp * 0
                    hl = hp + hvw * hv
                share = n_human / (n_human + B)
                loss = (1 - share) * loss + share * hl
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if c["grad_clip"]:
                torch.nn.utils.clip_grad_norm_([p_ for p_ in self.model.parameters() if p_.requires_grad],
                                               c["grad_clip"])
            self.opt.step()
            self.step += 1
            for k, v in (("loss", loss), ("policy", pol), ("value", vl), ("kl", kl), ("human", hl)):
                sums[k] += float(v.detach())
        self.model.eval()
        out = {k: round(v / steps, 5) for k, v in sums.items()}
        out.update(steps=steps, rows=n, beta=round(beta, 4), seconds=round(time.time() - t0, 1), step=self.step)
        return out

    @torch.no_grad()
    def evaluate(self, rows: dict, batch: int = 128, heldout_only: bool = True) -> dict:
        """The offline checks on held-out self-play rows: cross-entropy against the search, agreement with its
        choice, KL to the starting network, the policy's and the targets' entropy, and the value's AUC, log-loss
        and calibration (ECE) against the results."""
        hr = tables.select(rows, rows["heldout"]) if heldout_only and rows["heldout"].any() else rows
        t = tables.to_graph_table(hr, self.vocab, self.edge_vocab, name="heldout")
        if t.n == 0:
            return {"rows": 0}
        self.model.eval()
        ce, agree, kls, ent, tent, vals = [], [], [], [], [], []
        for a in range(0, t.n, batch):
            r = np.arange(a, min(t.n, a + batch))
            bd = gs.to_device(gs.make_batch([t], [(0, r)]), self.dev)
            ob = bd["opt"]
            with sv._autocast(self.dev, self.dtype):
                logp, vx = option_logprobs(self.model, bd)
                logq, _ = option_logprobs(self.ref, bd)
            logp, logq = logp.float(), logq.float()
            _, p = gn.gather(t.row_opt_ptr, r, t.aux["opt_p"])
            p = torch.from_numpy(np.ascontiguousarray(p, np.float32)).to(self.dev)
            ce.append(_segment_sum(-p * logp, ob.opt_row, len(r)).cpu().numpy())
            kls.append(_segment_sum(logq.exp() * (logq - logp), ob.opt_row, len(r)).cpu().numpy())
            ent.append(_segment_sum(-logp.exp() * logp, ob.opt_row, len(r)).cpu().numpy())
            tent.append(_segment_sum(-p * torch.log(p.clamp(min=1e-12)), ob.opt_row, len(r)).cpu().numpy())
            # agreement: the network's top option is the search's most visited
            row = ob.opt_row.cpu().numpy()
            lp, pp = logp.cpu().numpy(), p.cpu().numpy()
            for i in range(len(r)):
                m = row == i
                agree.append(float(np.argmax(lp[m]) == np.argmax(pp[m])))
            vals.append(vx.float().cpu().numpy())
        x = np.concatenate(vals)
        v = np.tanh(x)
        z = t.z
        ok = np.isfinite(z)
        pz = np.clip(1 / (1 + np.exp(-2 * x[ok])), 1e-6, 1 - 1e-6)
        yz = (1 + z[ok]) / 2
        ece = None
        if ok.any():
            bins = np.minimum((pz * 10).astype(int), 9)
            ece = float(sum(abs(pz[bins == k].mean() - yz[bins == k].mean()) * (bins == k).sum()
                            for k in range(10) if (bins == k).any()) / ok.sum())
        return {"rows": int(t.n), "games": int(len(np.unique(t.game))), "ce": round(float(np.concatenate(ce).mean()), 5),
                "agree_top1": round(float(np.mean(agree)), 4), "kl_start": round(float(np.concatenate(kls).mean()), 5),
                "entropy": round(float(np.concatenate(ent).mean()), 4),
                "target_entropy": round(float(np.concatenate(tent).mean()), 4),
                "value_auc": None if ok.sum() < 2 else auc(v[ok], z[ok]),
                "value_logloss": None if not ok.any() else round(float(-(yz * np.log(pz) + (1 - yz) * np.log(1 - pz)).mean()), 5),
                "value_ece": None if ece is None else round(ece, 4),
                "value_mean_p": None if not ok.any() else round(float(pz.mean()), 4)}
