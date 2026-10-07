# dzg: DraftZero networks for gorge

`dzg` trains MLP, Transformer and GNN networks on gorge's searched decisions and serves them to gorge's search
over a socket. It lives on the `claude/gorge-fdn-selfplay` branch, in `gorge/dzg/`, next to the Go driver
(`gorge/cmd/dzgorge`). [SPEC.md](SPEC.md) is the contract with the Go side: the packed arrays, the shard layout
and the socket protocol. The package needs only numpy and torch (written against torch 2.9 APIs, Python 3.11+).

## Modules

| Module | What it does |
|---|---|
| `packfmt.py` | Constants (widths, table sizes, magics), shard directories (`<name>.npy` per array plus `meta.json`, read memory-mapped), wire requests and responses (SPEC §3), per-record views for tests, and a synthetic data generator with learnable value and policy signals |
| `data.py` | `ShardSet` over one or more shard directories (memory-mapped, or `in_ram`, which merges the shards into one pack), the split by game (holdout: `crc32(game id) % 20 == 0`), a vectorised `collate(indices) -> Batch`, `batch_from_wire` (the same `Batch` from a request), and a prefetching `Loader` (collate threads, or worker processes) |
| `models.py` | The three networks, one interface `forward(batch) -> (option scores [no], value logit [B])`; `build`, `save`, `load`; `candidate_logits` |
| `targets.py` | Policy targets per candidate (`visits`, or Gumbel MuZero's completed-Q `cq`), the value target, and the masks |
| `train.py` | Training: curves (JSONL), holdout and extra validation sets, early stopping, checkpoints that carry their architecture and config |
| `serve.py` | The inference server: Unix or TCP socket, one thread per connection, one forward thread that runs every queued request in one forward pass, throughput logging |
| `client.py` | A Python client for the protocol (tests, benchmarks) |
| `bench.py` | Collate, loader, training and inference throughput (direct, through the socket server, and from several client processes at once), written as JSON |
| `tests/` | pytest, CPU only, about 10 seconds |

A `Batch` holds the states' dense features, the sparse bag (rows, values, offsets for `F.embedding_bag`), the cards
(raw features, group, state, position in the state, identity rows, a padded `[B, Cmax]` mask), the options (dense,
bot flag, state, position, `ea`/`eb` resolved to batch-wide card indices with -1 for none, slot and hashed rows),
and for training the candidates (state, visits, prior, q), the candidate-to-option membership as two aligned index
arrays, the root value and the outcome.

## The networks

Every network shares one `nn.Embedding(16384, d_emb)` table between the state's sparse bag, the cards' identity
rows and the options' hashed rows, as gorge's own network does; option slots have their own 128-row table. An
option's input vector is `bag(slots) + proj(bag(hashed rows)) + Linear(opt_dense) + bot x a learned vector`, plus
the vectors of its `ea` and `eb` cards (a learned "none" vector when there is no card). **Options never attend to
each other**: an option's score depends only on its state and itself (a test checks it), because gorge scores all
of a decision's options while the corpus stores only the options some candidate uses. A candidate's logit is the
sum of its options' scores (0 for an empty candidate), softmaxed within the state.

| Arch | Analogue | Shape | Defaults | Parameters (table) |
|---|---|---|---|---|
| `mlp` | DraftZero's `BagMLPNet`; a DeepSets model, no attention | card vector `relu(W relu(Linear(raw) + proj(bag(identity))))`; per group the 0.25-scaled sum and the max; state `Linear(dense) + proj(bag(sparse)) + Linear(pools)`, residual MLP blocks; score `MLP([s, o])`, value `MLP(s)` | d 384, ff 1024, 3 blocks, card width 256, d_emb 128 | 6,341,122 (2,097,152) |
| `transformer` | an entity transformer | tokens `[CLS, DENSE, SPARSE, one per card]`, pre-norm encoder layers with a padding mask; value `MLP(h_CLS)`; per-state option queries cross-attend once over the final tokens; score `MLP([h_CLS, q', h_ea, h_eb])` | d 192, 6 heads, 4 layers, ff 768 | 4,749,634 (2,097,152) |
| `gnn` | MageZero's hierarchical graph network (`graph_net.NetGraph`) | root, 2 players, 4 zones (my battlefield and hand under me, the opponent's battlefield under them, the stack under the root), cards, leaves (one per identity row, one `Linear(raw)`; the root's DENSE and sparse-row leaves); typed node and edge embeddings; one bottom-up pass of local attention (card, zone, player, root), then global layers over the internal nodes; value `MLP(root)`; options as in `transformer` | d 128, 4 heads, 2 global layers, ff 512 | 3,689,986 (2,097,152) |

Override any default with `--config '{"d": 256}'` (the keys are in `models.DEFAULTS`). The heads' last layers start
at zero, so a new network has a uniform policy and a 50% value.

## Targets and losses

- `--policy-target visits`: `visits^(1/T)` normalised (`--visits-temp`, default 1).
- `--policy-target cq`: completed q is the candidate's q when visited, else `v_mix = (root_value + sum N / (sum
  over visited of prior) x sum over visited of prior x q) / (1 + sum N)`; it is min-max normalised per state (a
  constant gives 0.5), and the target is `softmax(log(prior + 1e-8) + (c_visit + max N) x c_scale x qnorm)` with
  `--cq-c-visit 50`, `--cq-c-scale 0.1`.
- The policy loss is the cross-entropy against the candidate log-softmax, over states with at least 2 candidates
  and some visits. The value loss is the log loss of the value logit against the outcome (draws 0.5), optionally
  `(1 - x) outcome + x root_value` (`--value-blend x`), over states whose outcome is known (not -1).

At each evaluation `curves.jsonl` gets, for the holdout and each `--val` set: the policy cross-entropy, the
uniform-policy baseline, the targets' own entropy, top-1 agreement with the search's most-visited candidate and
with the bot's candidate (candidate 0), the value log loss, AUC (decisive games), Brier score and the base-rate log
loss. Early stopping (`--patience K`) and `best.pt` follow the holdout's policy cross-entropy plus value log loss.

## Throughput

`python -m dzg.bench` on synthetic data (per state: 29 sparse rows, 19.5 cards, 37 identity rows, 5.5 options),
batch 512, on an Apple M1 Pro laptop. These are development numbers; the GPU numbers come from running the same
command on the training machine.

| | `mlp` | `transformer` | `gnn` |
|---|---|---|---|
| Training, CPU (8 cores), states/s | 5,450 | 850 | 1,250 |
| Training, MPS, states/s | 6,100 | 1,220 | 2,030 |
| Inference through the socket, MPS, 256 states per request | 15 ms | 92 ms | 61 ms |
| Inference through the socket, MPS, 1 state per request | 3.1 ms | 5.0 ms | 7.3 ms |

The server's own work (decode, validate, batch, reply) adds 1 to 2 ms per request to the forward pass. Each
forward also has a fixed cost, mostly launching its operators (on MPS about 1, 2 and 3 ms for mlp, transformer and
gnn), so the server runs every request waiting in its queue as one forward. From client processes sending
requests back to back (`python -m dzg.bench --conns 1,4,8 --conn-modes nomerge,merge`), states/s with 8
connections, one forward per request then merged:

| | `mlp` | `transformer` | `gnn` |
|---|---|---|---|
| CPU, 32-256 states per request | 9,290 then 14,540 | 1,930 then 2,220 | 2,290 then 2,990 |
| MPS, 32-256 states per request | 12,960 then 14,200 | 2,690 then 2,490 | 3,660 then 3,000 |
| MPS, 4-16 states per request | 2,660 then 5,480 | 1,400 then 1,970 | 1,150 then 1,650 |

On this laptop the processor running the model is the limit: forward time grows in proportion to the states, so
merging gains least at the larger sizes. MPS also compiles a kernel set for each new batch size, and merged
forwards keep producing new sizes, which costs the transformer and gnn there (the gnn took about 40% longer per
state when every forward had a new size). A CUDA GPU computes far faster, so the fixed cost per forward weighs more
and merging should matter more; that is not measured yet. The log's `busy` (the forward thread's share of the wall
time) and `states_per_forward` show whether a server process is the limit: if `busy` stays near 1 while the GPU is
not saturated, run two processes per GPU (`SERVERS=2`). `--no-merge` restores one forward per request.

Collate runs at about 270,000 states/s on one core for a 20,000-record set in RAM, about 100,000 states/s for
random batches over a 400,000-record (2.6 GB) set, and about 26,000 states/s memory-mapped with a cold page cache.
The training loader's collate threads share the GIL, so more than two rarely help. Over many shards a batch needs
one gather per shard. With 200,000 records of about 145 sparse rows each, as 25 shards of 8,000, two threads gave
91,000 states/s in RAM before `--in-ram` merged the shards and 279,000 after, and about 70,000 memory-mapped
(unchanged). `--loader-procs 4` (worker processes) gave 211,000 memory-mapped over the 25 shards and 326,000 over
one shard. Worker processes fork on Linux and pass batches through `/dev/shm` (about 11 KB a state). On macOS they
spawn, and torch then takes about 3 s to start them and 5 s per worker to stop them, at every epoch and evaluation.

## Commands

From `gorge/` (or with `PYTHONPATH=gorge` from the repo root):

```bash
# synthetic shards, laid out as `dzgorge pack` writes them (DIR/00000, DIR/00001, ...)
python -c "from dzg import packfmt as pf; pf.write_synthetic('/tmp/syn/train', 100000, seed=1); pf.write_synthetic('/tmp/syn/eval', 5000, seed=2)"

# train
python -m dzg.train --arch mlp --train /tmp/syn/train --val evaldecks=/tmp/syn/eval --out runs/mlp0 --epochs 3
python -m dzg.train --arch gnn --train DATA/gen1 DATA/gen2 --out runs/gnn1 --policy-target cq --patience 4 --in-ram
python -m dzg.train --arch mlp --train DATA/gen* --out runs/mlp2 --loader-procs 8   # collate in 8 processes (Linux)
python -m dzg.train --init runs/mlp0/best.pt --train DATA/gen3 --out runs/mlp1          # warm start a generation

# serve to gorge's search
python -m dzg.serve --ckpt runs/mlp0/best.pt --socket /tmp/dzg.sock
python -m dzg.serve --arch transformer --random-init --tcp 127.0.0.1:7070               # speed tests

# benchmark (JSON with --out)
python -m dzg.bench --arch transformer --device cuda --out bench-transformer.json
python -m dzg.bench --arch mlp --shards DATA/gen1 --sizes 1,16,64,256,1024

# tests
python -m pytest dzg/tests -q
```

`train.py` writes `OUT/config.json`, `OUT/curves.jsonl` (lines of `"type": "train"` every `--log-every` steps and
`"type": "eval"` every `--eval-every` steps, by default four times an epoch), `OUT/best.pt` and `OUT/last.pt`, and
prints a summary JSON line. A checkpoint is a dict with `arch`, `config`, `state_dict` and `meta` (step, metrics,
arguments); `models.load(path)` rebuilds the network from it.

## Choices where SPEC.md leaves room

- Responses: `score` is the last field and is not padded. The decoder accepts a request whose last array
  (`oh_val`) lacks its padding, and rejects any other size mismatch.
- The server validates each request (`--no-validate` turns it off): rows outside their tables, offsets that do
  not start at 0 or decrease, groups above 3, and `ea`/`eb` outside the state's cards are logged and close that
  connection, since each is a bug on the sending side. Without validation an out-of-range `ea`/`eb` is treated as
  none.
- The server drops the options of `want = 0` states before the forward pass (scores are per option, so the others
  are unchanged) and returns 0 for them.
- The server runs the requests waiting in its queue as one forward (up to `--max-merge` states, default 4,096; a
  larger request runs alone). Each state's outputs depend only on that state, so merging changes an answer only by
  float rounding. If a merged forward fails, its requests run again one at a time, so a bad request (possible with
  `--no-validate`) closes only its own connection.
- With several memory-mapped shards a batch's states come grouped by shard; within a shard, in the order given.
  `--in-ram` merges the shards into one pack (int64 offsets, `game` re-indexed into the concatenated games list),
  so a batch is one gather in the given order, however many shards there are.
- `--limit-records N` takes the first N records before the holdout split; `--eval-max` (default 100,000) caps each
  evaluation set with a fixed random subsample.
