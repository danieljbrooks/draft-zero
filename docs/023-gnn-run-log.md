# MageZero's graph network: the run log

*The stages of [docs/022](022-gnn-imitation-test-plan.md), as they run. Started 4 October 2026, 6:20 PM PT, when
Dan approved the plan ("push to main, use r1 if available, budget is okay"). All times are Pacific.*

## Status (Sunday 4 October, 7:05 PM PT)

| Stage | Status | Where | Spend |
|---|---|---|---|
| 0. Engineering | done, on main (5d1766a, a72792d, c18a727, c646d46) | laptop | – |
| 1. Build | **running**: 13.4 games a second, ~3.3 h; then the tables, the comparison with experiment #4's, the upload | pod `gnn-stage1` | running |
| 2. Sweep | waiting for the tables; split between the pod's 3090 and r1's free GPU | | |
| 3. Scale check, large training | | | |
| 4. Offline evaluation | held-out cards chosen (below); the tooling is on main | | |
| 5. Games | | | |

## Where things run

- **r1** (Dama's box): GPU 0 is free; GPU 1 and 8 CPU workers run another session's MLP games. The home disk has
  25 GB free, too little for the full tables (~30 GB with their flat twins), and the RAM (40 GiB, 16 in use) too
  little for 28 build workers. So the build runs on a pod. r1 takes part of the sweep from the **slim** tables
  (graph files + labels, `build.py slim`), in its own checkout (`~/dz-gnn`, `PYTHONPATH=src`), so the other session's
  running games are untouched.
- **GNN speed on r1's GPU 0** (the planning tables, Will's network): 1,177 states a second at batch 64, no faster
  than the 3090 (1,233): a step at 64 states is overhead-bound (~54 ms). At batch 256 r1 does 3,950 a second (the
  3090: 1,596). Inference at batch 128: 15,600 a second (the 3090: 5,400). The sweep's batch-256 arm decides whether
  the faster batch is free.
- **Pod `gnn-stage1`**: Secure RTX 3090, $0.50 an hour (EU-CZ-1, AMD EPYC 7C13, a 31-core quota, 125 GB RAM, 200 GB
  disk), rented 6:21 PM PT. cgroup v1, so `nproc` shows the host's 256 cores (`gnn_build.sh` now reads the v1
  quota). Self-destruct armed for 6:22 AM PT Tuesday. It runs the build, then training.

## Stage 1: the build

- 6:30 PM PT: `WORKERS=28 bash deploy/gnn_build.sh` started, after the slim bootstrap (~25 minutes, mostly the CUDA
  torch download) and copying `data/17lands/` and the split file from the laptop (~6 minutes).
- **Speed:** 13.4 games a second at 8,000 games (experiment #4's flat build: 17.6 on a different pod), so ~3.3 hours
  for the 161,206 games.
- **Shards:** 1.1 GB per 5,000-game part, ~36 GB in all (the plan guessed 65 GB from the laptop's smaller parts).

## Stage 4, ahead of time: the held-out cards

`heldout_cards.py choose` over the top players' training games (145,903). Five cards, each a different kind, each
sharing its mechanics with other cards: a vanilla creature, a defender, burn, a removal aura and a combat trick.
A game counts as showing a card when it is in the player's decklist or either player is seen with it.

| Card | Type | Rarity | Share of training games |
|---|---|---|---|
| Savannah Lions | Creature (2/1 for W) | uncommon | 6.9% |
| Claws Out | Instant (W) | uncommon | 5.9% |
| Imprisoned in the Moon | Aura (U) | uncommon | 4.6% |
| Gleaming Barrier | Artifact creature, wall | common | 4.4% |
| Goblin Negotiation | Sorcery (R) | uncommon | 3.3% |
| **Any of them** | | | **22.1%** (32,283 games) |

The list is `data/imitation_graph/heldout/{cards.json,exclude_games.npy}` (regenerate with
`heldout_cards.py choose --cards "Savannah Lions,Gleaming Barrier,Goblin Negotiation,Imprisoned in the Moon,Claws Out"`).

## Pods

| Pod | What | Rented | Removed | Hours | $/h | Cost |
|---|---|---|---|---|---|---|
| `gnn-plan` (Community 3090, FR) | docs/022 §2's planning measurements | 4:53 PM PT | 6:07 PM PT | 1.2 | 0.22 | $0.27 |
| `gnn-stage1` (Secure 3090, CZ) | stage 1, then training | 6:21 PM PT | | | 0.50 | |
