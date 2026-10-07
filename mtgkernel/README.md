# mtgkernel: FDN self-play on mtg-kernel

`dzk` plays draft-zero's FDN Limited decks against each other on
[mtg-kernel](https://github.com/jackmaiorino/mtg-kernel) (a Rust rules engine), with simple bots and an
offline search, and writes game records that draft-zero's tools read (`tools/imitation_scale/gih.py`,
`card_stats.py`). Stage 1: engine wrapper, bots, records, bench. Stage 2: a policy/value network on
mtg-kernel's own observations (decision encoder, Rust forward pass, network bots, training data from play,
a PyTorch trainer and an expert-iteration driver; see [Stage 2](#stage-2-policyvalue-network)). No new
cards: only the cards mtg-kernel supports fully on its main branch (38 non-basic FDN cards and the basics).

## Setup

```bash
bash mtgkernel/setup.sh          # or MK_SRC=/path/to/a/local/mtg-kernel/clone bash mtgkernel/setup.sh
cd mtgkernel
cargo build --release            # Rust 1.94.1 (rust-toolchain.toml); first build ~4 min (mtg-kernel)
cargo test --release
```

`setup.sh` clones mtg-kernel at the pinned commit `a4e1474b` into `mtgkernel/.deps/mtg-kernel` (git-ignored)
and applies `patches/` (0001 visibility only, 0002 an opt-in throughput flag and read-only accessors; no
rule changes; see `patches/README.md`). The crate builds mtg-kernel with its `limited-fdn-fixtures` feature
(the FDN cards do not resolve without it).

## Use

```bash
# pairs.tsv: deck1<TAB>deck2 file stems, one pair a line
./target/release/dzk play --decks-dir ../assets/sample/decks --pairs pairs.tsv \
    --bot1 mcts:100 --bot2 random --seed 1 --threads 6 --out runs/x/games.jsonl [--n-pairs N] \
    [--pair-range A:B] [--shard I/N] [--resume] [--data-out DIR] [--max-permanents 100] [--max-game-seconds 0]
./target/release/dzk summarize runs/x/s*/games.jsonl [--out runs/x/summary.json]   # shards of one run
./target/release/dzk bench --decks-dir ../assets/sample/decks --pairs pairs.tsv --bot mcts:100 \
    --seconds 60 --threads 1,6
python3 ../tools/imitation_scale/gih.py runs/x/games.jsonl
```

`play` writes one JSON line per finished game (flushed per game) and `summary.json` next to it
(`summary.shard-I-of-N.json` for a shard); it is killable at any point, and `--resume` skips the (pair,
swap) games already in the file (a torn last line is cut off; a file played with other bots, seeds or decks
is refused). The output is locked (`flock`) for the whole run, so a second `play` on the same file fails
instead of appending duplicates; resume and the summary stream the file and count a repeated (pair, swap)
once. Pair p plays line `p % lines` of the pairs file, for p in `0..--n-pairs` (default: the line count, or
the end of `--pair-range` if larger). `--pair-range A:B` keeps pairs A..B-1, `--shard I/N` pairs with
`p % N == I`; an empty selection is an error. Give each shard its own `--out` and merge with `summarize`.

`bench` reports `*_per_s` over the whole elapsed time (games running at the deadline finish, so threads
idle in the tail count against it; use `--seconds` well above a game's length) and `steady_*_per_s`, the
sum of each thread's own rate. On the 8-core M1 Pro, `--threads 8` beats 6 by 10-20%.

### Bots

| Spec | |
| --- | --- |
| `first` | always the first masked action (mulligans included) |
| `random[,mull=P]` | uniform over the masked legal actions |
| `greedy1[,mull=P]` | one-ply lookahead on one re-dealt world, GameStateEvaluator3 after auto-advancing |
| `mcts:N[,worlds=K][,c=C][,discount=D][,cache=M][,mull=P]` | draft-zero's offline search baseline "heuristic@N" (`heuristic@N` is an alias); defaults K = 1, C = 1, D = 0.99, M = 4096 |
| `net:path=FILE[,t=T][,mull=P]` | stage 2: the network's policy, greedy (argmax logit) at t = 0 (default), sampled at t > 0; `net,path=FILE` is the same |
| `pmcts:N,net=FILE[,pt=T][,leaf=L][,noise=E][,alpha=A][,temp=T][,temp_moves=K][,fpu=R]` + `mcts:N` options | stage 2: `mcts:N` with network priors at every node and the network value at leaves (L = `net` (default), `heuristic`, `mix:λ`); `fpu=R` values an unvisited child at its parent's mean minus R (default: 0, a draw) |

`mcts:N` ports `java/mzbridge/.../BenchSearch.java`'s `searchTree` (PUCT, uniform priors, unvisited
children valued 0, N fresh simulations, 0.99 discount per tree edge, final choice by visits) with
XMage's GameStateEvaluator3 at every leaf (`src/eval.rs`); see `src/bots/mcts.rs` for the differences. A
leaf whose game halts (fail-closed) is dropped from the tree and counted (`halted_leaves`); scored 0 it
used to attract a searcher that was behind, and the record then excluded the game.

Mulligans (`mull=P`, every bot but `first`; default `keep`): the Java baseline never mulligans
(draft-zero's XMage games run in test mode with `allowMulligans = false`), and mtg-kernel asks at every
game start. `keep` always keeps, `xmage` applies `ComputerPlayer.chooseMulligan`'s rule (mulligan a 6+ card
hand with fewer than 2 lands or more than size - 2; XMage's `LondonMulligan` bottoms right after each
redraw, as mtg-kernel does, so the rule sees the hand after bottoming), `own` (also `search` / `random`)
leaves the announcement to the bot. The search applies the policy inside its tree too. Do not use
`mcts:N,mull=own` for evaluation: with a one-turn heuristic horizon a mulligan postpones the opponent's
development past the horizon (`mcts:100,mull=search` mulliganed to zero cards in 8 of its 20 seat-B
games). `random,mull=random` is the old uniform `random` (about half of all seats mulligan, some to zero).

`greedy1` loses to `random` (0.325 over 40 games on the fixture pair, before it moved to re-dealt worlds),
probably because one ply after a cast the spell is usually still on the stack (the opponent holds
priority), so the cast scores as a card lost from hand.

## Stage 2: policy/value network

A new network trained on mtg-kernel's own observations (draft-zero's XMage models take XMage feature strings,
so none ports). Everything runs on the laptop's CPU; the trainer can use a GPU (`--device cuda`).

```bash
cd mtgkernel
# self-play data: every real decision of every seat, encoded, with the search's root visit distribution
./target/release/dzk play --decks-dir decks/gen_v1 --pairs decks/gen_v1/pairs_train.tsv --bot1 mcts:100 \
    --bot2 mcts:100 --seed 100 --threads 8 --pair-range 0:20 --out runs/g0/games.jsonl --data-out runs/g0/data
# train (PyTorch on CPU; uv 0.12.7, from any directory outside .deps/mtg-kernel)
uv run --no-project --python 3.12 --with torch --with numpy python py/train.py --data runs/g0/data \
    --out runs/g0/train --epochs 4                      # -> ckpt_best.pt, ckpt_last.pt, net.dzkn, metrics.json
# play it
./target/release/dzk play --decks-dir decks/gen_v1:decks/fixtures --pairs pairs/eval_v1.tsv --n-pairs 500 \
    --bot1 net:path=runs/g0/train/net.dzkn --bot2 mcts:100 --seed 7 --threads 8 --out runs/eval/games.jsonl
# Rust vs PyTorch on a stratified sample of dumped decisions; per-decision cost of observation, encoding, forward
uv run --no-project --python 3.12 --with torch --with numpy python py/parity.py --net runs/g0/train/net.dzkn \
    --data runs/g0/data --min-decisions 300 --out runs/parity --dzk target/release/dzk
./target/release/dzk nnbench --decks-dir decks/gen_v1:decks/fixtures --pairs pairs/eval_v1.tsv \
    --bot mcts:100 --games 20 --net runs/g0/train/net.dzkn
# all of it, generation after generation (resumable, every step under a hard kill)
uv run --no-project --python 3.12 --with torch --with numpy python py/loop.py --work runs/ei --gens 3 \
    --budget 100 --threads 8          # defaults: 1,500 self-play pairs a generation, 4 epochs, 500-pair evals
```

- **Encoder** (`src/encode.rs`, schema in [docs/encoding.md](docs/encoding.md), version `dzk-enc-v2`): one
  real decision from the acting player's seat, from its redacted observation, the masked actions and static
  card facts only (`encode` never sees the `GameState`; `arena_id` only links actions to object tokens):
  96 global features, one token per visible object (72 features; at most 255 tokens, by per-group quotas so
  every zone group keeps tokens), one token per masked action (32). The version is in every data shard,
  checkpoint and exported network, and loaders refuse a mismatch.
- **Network** (`py/model.py` `DzkNet`, `src/nn.rs`; 485k parameters): DeepSets object MLP (E = 32, H = 128),
  mean ⊕ max pools over my hand / my battlefield / the opponent's battlefield / the rest, state MLP
  (256 → 128), action tokens (kind, cards, the referenced objects' vectors, target player, features) scored
  against the state, value `tanh` from the actor's seat. Exported as `dzk-net-v1` (JSON header + f32 blob,
  sha256-checked). The Rust forward is hand-written (no NN crates): first layers folded into lookup tables,
  identical object tokens computed once, four rows per weight pass, FMA kernels (aarch64 always; x86-64 with
  AVX2+FMA picked at run time, baseline otherwise).
- **Bots**: `net` (greedy by default, as draft-zero evaluates policies) and `pmcts:N` (see the table). The
  network bots keep the mulligan rule (`mull=keep` by default): gen-0 targets for the announcement come from
  the searcher's own `mull=keep` rule, so no head is trained on searched mulligan decisions. A network is
  identified by its content hash, not its path: records carry `bot1_net` / `bot2_net`, `--resume` refuses a
  games file played with another network under the same path, the summary refuses mixed records, and the
  data's game keys include the hashes.
- **Data** (`--data-out DIR`, `src/data.rs`, `py/data.py`): one gzip member per game appended to one shard per
  worker per run (`DIR/<tag>-wNN.dzd.gz`), `DIR/<tag>.meta.json` per run, `DIR/meta.json` the index. Each
  record: the encoded decision, the policy target (search: root visit distribution; net/random/rule: one-hot,
  with the source), the chosen index, seat, turn, the search's root value, and the result `z` from the
  record's seat (flagged: truncated, halted or engine error, draw). About 0.45 MB per 1,000 decisions. A game's
  record line is written before its data, so a killed and resumed run never writes a game's data twice.
- **Trainer** (`py/train.py`): soft cross-entropy against the policy target (`--target visits|top|chosen|sharp:T`;
  no policy term for `mcts:N` decisions with more than N/4 actions, `--flat-k-frac`) + `value_weight` · MSE
  against the value target (`--value-target rootq|z|mix:L`, default `rootq`, the search's root value: training
  on the result z overfits within an epoch or two and drags the policy down); AdamW, cosine schedule with
  warmup, gradient clipping, a 90/10 split by game; per epoch: policy cross-entropy, KL, top-1 agreement (strict
  argmax) with the played move and the target's top move, value MSE against the target and against z, AUC and
  log-loss; the epoch with the lowest validation policy cross-entropy is exported (`--select policy|loss|last`).
  A rerun with the same arguments resumes from `ckpt_last.pt`; `--max-minutes` stops, evaluates and exports.
- **Loop** (`py/loop.py`): generation k self-plays `mcts:N` (k = 0) or `pmcts:N,net=net_k` with Dirichlet root
  noise and visit sampling for the first moves, trains net_{k+1} on the last `--window` generations (warm
  start), checks Rust/PyTorch parity, and evaluates `net` and `pmcts:N` against `mcts:N` and `random` on the
  same seeds every generation. Each step records its inputs (network and checkpoint sha256s, data, arguments):
  a step whose inputs changed has its outputs moved aside (`<dir>.stale-<time>`) and runs again.

Numbers (M1 Pro laptop, other work holding the load average at 5-12; scores are bot1's with 95% intervals, paired
games on `pairs/eval_v1.tsv` with seed 7, cycled past its 62 lines, so every row plays the same games):

| Run | Training decisions | net vs random | net vs mcts:100 | pmcts:100 vs mcts:100 |
| --- | --- | --- | --- | --- |
| gen-0 smoke: 20 pairs of `mcts:100` self-play, 2 epochs, 8 threads | 3,709 | 0.650 (40) | 0.250 (40) | 0.450 (20) |
| `py/loop.py` gen 0: 360 pairs, 4 epochs, 4 threads | 63,695 | 0.830 (200) | 0.269 [0.24, 0.30] (1,000) | 0.626 [0.58, 0.67] (500) |
| gen 1: `pmcts:100` self-play (noise 0.25, 8 sampled moves), window 2, warm start | 130,481 | 0.860 (200) | 0.347 [0.32, 0.38] (998) | 0.660 [0.62, 0.70] (500) |

- Gen 0 is trained on exactly the 720 self-play games of the first stage-2 run (self-play is deterministic); on
  the 123 eval games both played, the old trainer's network (result as value target, best epoch by summed loss,
  which kept epoch 1) scored 0.171 against `mcts:100` and this one 0.309: paired difference +0.139 [0.066, 0.212].
- Gen 1 minus gen 0, paired: `net` +0.077 [0.050, 0.104] (998 games), `pmcts` +0.034 [-0.002, 0.070] (500).
- Value target: on 988 paired games against `mcts:100`, networks trained 4 epochs on 3,240 `mcts:100` games
  scored 0.360 with `rootq`, 0.310 with `mix:0.5` (`mix` - `rootq` = -0.051 [-0.077, -0.024]), and the old
  rule's `z` network 0.245.
- Throughput (8 threads): `mcts:100` self-play 28,000 games an hour, `net` vs `mcts:100` 51,000, `pmcts:100` vs
  `mcts:100` 15,000, `net` vs `random` 1.2 million. Rust cost per decision (`dzk nnbench`, single thread): the
  forward 64 µs in `net` play (47 µs with the weights in cache, 147 µs right after a 100-simulation search),
  encoding 3 µs, the observation build 8 µs.
- Rust vs PyTorch: max |Δlogit| 2.1e-7 and |Δvalue| 2.4e-7 on the committed fixture (575 decisions, every decision
  family, menus up to 5,040 actions); 7.2e-7 / 2.7e-7 or less in every loop parity step.

## Decks, rented machines and analysis

Results and the experiment story: [docs/025](../docs/025-mtg-kernel-fdn-self-play.md).

- `py/gen_decks.py` writes `decks/gen_v1/` (120 two-colour decks from the playable cards, uniform random inclusion;
  `pairs_train.tsv`, `pairs_eval.tsv`). `decks/fixtures/` holds copies of the two human fixture decks;
  `pairs/eval_v1.tsv` is the evaluation set (30 held-out generated decks + the fixture pair), played with
  `--decks-dir decks/gen_v1:decks/fixtures`.
- `deploy/pack.sh` bundles this folder for a RunPod pod, `deploy/pod_bootstrap.sh` installs Rust 1.94.1 (and torch
  with `TORCH=1`), clones and patches mtg-kernel and builds; `deploy/run_jobs.sh JOBS.tsv RUNS` runs a list of
  `dzk play` jobs under hard timeouts (resumable); `deploy/bench_all.sh` and `deploy/eval_jobs.sh NET TAG THREADS`
  write the benchmark and evaluation jobs docs/025 used. On a pod, start long jobs detached:
  `(setsid nohup CMD > log 2>&1 < /dev/null &)`.
- `py/gih_report.py GAMES.jsonl --decks-dir ... --label ... --out-dir ...`: 17lands-style card statistics with a
  per-card matched noise ceiling and the deck-adjusted GIH WR (17lands' data is read from the main checkout).
- `py/jsonl_bench.py`: the same random games through mtg-kernel's own JSON interface, for comparison.
- `py/fig_doc025.py --spec py/fig_doc025_spec.json`: docs/025's figures (paths relative to `runs/`).

## Rules and decision loop

- Rules: schema 4 of `kernel_limited_env` (`--london-mulligans-v1`): engine priority windows,
  Foundations combat (damage assignment, trample), London mulligans, Limited incremental trigger ordering,
  via the same `RlEpisodeSessionV1::reset_with_custom_decks_and_london_v1` call the JSONL server makes.
- Seats: deck1 in seat A = P0, which always plays first; bot1 in A unless `swap`.
- Masking: `activate_mana_ability` actions are dropped whenever another action remains (casting pays
  mana itself through mtg-kernel's exact payment solver). After masking, a decision with one action left
  is stepped without the bot ("forced").
- Caps: 20,000 physical decisions and 20,000 policy steps; a capped game is `truncated` (no winner).
  `dzk play` also ends a game as `truncated` (no winner; `terminal.code` `dzk_permanent_cap` or
  `dzk_game_seconds`, counted in the summary's `terminals.truncated_by_code`) at 100 permanents on the two
  battlefields together (`--max-permanents N`, 0 = none; normal play peaks below 40, a runaway token board
  doubles every turn) or after `--max-game-seconds S` (default 0 = none; a safety net, not reproducible). A
  fail-closed (`halted`) game is recorded as an error with the engine's terminal code and reason, and
  `halt_step` (the seat and role that took the last step, and whether it was forced).

## Paired games and seeds

As `tools/imitation_scale/play.py`: each deck pair is played twice with the bots swapped, both games
with `env_seed = seed * 1000 + pair` (the same shuffles), except a mirror match (bot1 and bot2 parse to the
same bot, `mcts:3` = `heuristic@3` included), whose swapped game adds 500,000. Each bot has its own
SplitMix64 seeded from the game seed and its seat, independent of the engine's shuffle stream.

## Records

One object a line: `pair, swap, deck1, deck2, deck1_hash, deck2_hash (16 hex digits of mtg-kernel's
content identity), botA, botB, bot1, bot2, bot1_net, bot2_net (network bots only: 16 hex digits of the
weights' sha256), winner ("A" | "B" | null), winner_role, winner_bot, first,
turns, rounds, seconds, game_seed, env_seed, method ("mtgkernel"), error, halt_step, terminal {outcome,
classification, code, reason}, decisions {physical, policy_steps, real_A, real_B, forced, max_choices},
mulligans {A, B}, seats {A|B: {inHand, openingHand, keptHand, drawn, tutored}}, search {A|B: bot counters}`.

- Seen in hand follows 17lands' game data: `openingHand` is the player's last seven-card deal, bottomed
  cards included (17lands' `opening_hand` is seven cards whatever the mulligan count), `drawn` the cards
  drawn after the mulligans, `inHand` = openingHand + drawn by name (GIH). `keptHand` is the hand after
  bottoming (7 - mulligans) and `tutored` other library-to-hand moves (not in `inHand`, as 17lands keeps
  `tutored_*` apart; no supported card tutors today).
- `turns` counts individual turns, as XMage's `getTurnNum` (draft-zero's Java records); `rounds` is
  mtg-kernel's own counter (both players' turns are one round).
- Records are deterministic given the seed and the bots (everything but `seconds`).

## Known limits and assumptions

- **Open decklists.** PIMC worlds (`worlds=K`) re-deal the hidden cards from the searcher's view the way
  mtg-kernel's `redeterminize_hidden_zones_v1` does: the searcher knows the opponent's 40-card list, but
  not its order or which cards are in hand. A closed-decklist belief sampler is later work. Each world
  also gets its own future shuffle stream (mtg-kernel's search-only re-seeding), so the search no longer
  sees the real future shuffles. An object a legal action names keeps its definition.
- **Admission check is approximate.** A world must give the actor the same observation; the candidate list
  is not regenerated (the session keeps no regenerable decision), so a root action legal only because of a
  hidden card it does not name would go unnoticed (none known in this pool). A refused world is retried
  with new seeds, then skipped (`world_fallbacks`, totalled in the summary; 0 so far): `mcts`/`pmcts` never
  search the real game (its hidden cards would steer the search, and pmcts's network would encode the
  opponent's real hand). A decision with no searched world is answered by the network's policy (pmcts) or
  uniformly (mcts) and counted (`unsearched`). `greedy1` still falls back to the real game.
- **Hidden information is a convention.** `Bot::choose` gets the whole `Game`; `greedy1` and `mcts:N` only
  look ahead on re-dealt worlds, but nothing stops a new bot from reading hidden cards.
- **Mana payment.** With mana abilities masked, the engine's payment solver picks the payers: the first
  untapped sources in battlefield order, with no preference for lands, so Llanowar Elves can tap for a
  spell while Forests stay untapped (and the Elves then cannot block or attack). 70 of the 120
  `decks/gen_v1` decks run Elves. Fixing it needs either a payer choice for the bots or a lands-first
  rule in the engine; not done.
- Trigger ordering: mtg-kernel offers every order of a simultaneous trigger group of up to 7 as one
  decision (7! = 5,040 actions; larger groups go one trigger at a time). The search gives each order a
  uniform prior like any other option, so such a node soaks up the budget.
- Fail-closed games: about 1 random-vs-random game in 500 to 2,000 halts with
  `fail_closed:observation:legacy observation cannot identify the Ward-bound stack item unambiguously`
  (env_seed 1746, UG vs WG, with `random,mull=random` is one). It comes from mtg-kernel's observation layer
  (`effect.rs` `validate_legacy_ward_observation`), not from the rules: without the observation the game
  plays on. The record carries it as an error with `halt_step`; the summary counts halts by the role of
  the last step.
- Truncation differs from the Java runs: XMage games stop after turn 50 with no winner (`maxTurns`); `dzk`
  only stops at the 20,000-step caps.
- Throughput: patch 0002 removed the observation hash and the response copies (3-4x). What remains is
  mostly the observation build itself and the engine; the next steps are building no observation at all
  (another ~2x, but the Ward fail-close above would then not happen) and a faster allocator than macOS's
  (+35-90% at 6 threads in a prototype; `mimalloc` is not vendored).
- The search tree caps stored game copies by count (`cache=M`, default 4096), not bytes: at a 5,040-order
  trigger decision one copy measured 3.6 MB before patch 0002 (which removed `Game`'s duplicate of the
  decision, roughly half of it), so an `mcts:3000` tree on a trigger-heavy board can grow large.
- Homunculus Horde copies itself whenever its controller draws a second card in a turn, and every copy
  does the same, so with a looter (Strix Lookout) the tokens double each turn: `mcts:10` self-play on the
  fixture pair with env_seed 113 had 236 tokens by round 16, 6,090 physical decisions and took 410 s. The
  rules are right; such a game is slow (the observation grows with the board, and every copy's trigger is a
  decision). The 100-permanent limit now ends it as truncated after 2 s. Eval pair line 47 of
  `pairs/eval_v1.tsv` (`GEN_v1_UG_004` vs `GEN_v1_WU_003`, Horde and Strix on one side, Strix on the other) can
  do the same; it is no longer a game that runs for hours.

### Stage 2 limits

- Gen-0 policy targets are flat by construction (normalized entropy 0.93 for every menu size from 2 to 10),
  not noisy: two seeds of `mcts:100` agree on the top move 93% of the time, and `mcts:400` only reaches 0.85.
  Decisions with many actions get a degenerate target (index 0, PUCT's tie-break) when k exceeds about N/4;
  the trainer drops their policy term (`--flat-k-frac 0.25`). `--target top` (one-hot of the most-visited
  move) and `sharp:T` are alternatives; neither beat the visit targets in play in the review's tests. Storing
  each root action's Q for Q-based targets is not done (a data format change).
- The Rust forward costs about 45 µs per decision with the weights in cache and 55-60 µs in `net` play, but
  about 120 µs right after a 100-simulation search has evicted them (M1 Pro, load average 5-6 from other
  work): a forward reads about 1.7 MB of weights, 1.15 MB of it the 1120 → 256 state layer. Storing that
  layer in bf16 (rounded at export so PyTorch and Rust still agree) would halve it.
- The trainer holds every decision in memory, packed in flat arrays with float16 features (about 4 KB per
  decision; 1M decisions = 4 GB, plus a transient 1.8x while loading). Object tokens repeated across a game's
  decisions are stored each time; deduplicating them, or memory-mapping shards, is the next step past a few
  million decisions.
- Order-trigger decisions encode only the first and last trigger of each order (see docs/encoding.md).
- The FMA kernels (aarch64, and x86-64 with AVX2+FMA) and the baseline x86-64 kernel differ in the last bits,
  so `net`/`pmcts` games are deterministic per machine type, not across them. The x86-64 AVX2+FMA path has
  never run (no x86 machine here; the reviewer forced the non-FMA kernels on the laptop: max |delta| 6.6e-7 vs
  PyTorch). Run `cargo test --release -- --test-threads=1` on the pod before using it; `py/loop.py`'s parity
  step also checks every new network on the machine that runs the loop.
- Unvisited children count as a draw (q = 0) in `pmcts`, so its visits spread out when it is behind (normalized
  entropy 0.90 below a root value of -0.3, 0.72 near 0; `mcts` shows no such gap). `fpu=R` (an unvisited child is
  worth its parent's mean minus R) is available but off by default and untested in play.
