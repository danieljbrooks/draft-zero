# dzg: DraftZero-style networks for gorge

`dzg` trains MLP, Transformer and GNN networks on gorge's searched decisions and serves them to gorge's
search over a socket. The Go half lives in `gorge/cmd/dzgorge` (`remote.go`, `pack.go`); the Python half
is this package. This file is the contract between them.

## 1. What a position looks like

gorge encodes the searching seat's redacted view under its `entity` feature set
(`internal/policynet/entity.go`, `features.go`, `option.go`). One searched decision is:

- **State**
  - `dense`: 68 floats (turn, step and phase one-hots, priority, stack depth, mana pool, per-seat blocks
    such as life, hand and library size).
  - `sparse`: a bag of `(row, value)` pairs, `row < 16384`: gorge's hashed "mz" features (per-zone card
    properties, available mana, the stack's spell APIs).
  - `cards`: one entity per visible card, in a fixed order (my battlefield, the opponent's battlefield, my
    hand, the stack). Each card has a `group` (0 my battlefield, 1 opponent's battlefield, 2 my hand,
    3 stack), `raw` (68 floats: types, power, toughness, keywords, tapped, counters, attached, mine,
    zone, castable, mana value, ...) and identity rows `(row, value)`, `row < 16384` (its name and
    spell API, hashed into the same table as `sparse`).
- **Options**: every option of the decision. Each has `slots` `(row, value)` with `row < 128` (one-hot
  categorical blocks), `hashed` `(row, value)` with `row < 16384` (card identity, card x target,
  option kind; the same table as `sparse`), `dense` (24 floats), `bot` (1 when the option is part of
  gorge's heuristic bot's answer, else 0), and `ea`, `eb`: the option's own card and its related card
  (a blocker's attacker) as **1 + an index into this state's cards**, 0 for none.
- **Candidates** (training only): the search's root candidates, each a set of option indices. The
  network scores every option independently; a candidate's logit is **the sum of its options'
  scores** (an empty candidate scores 0). This equals gorge's `policynet.CandidateScore` up to a
  per-decision constant (for attackers and blockers gorge adds `sum over all options of
  log(1 - sigmoid(s))`, which cancels in the softmax), so training and gorge's prior agree.

A network maps (state, options) to one score (a logit) per option and one value logit per state.
`sigmoid(value)` is the probability that the searching seat wins. **Options must not attend to each
other**: an option's score may depend only on the state and that option, because gorge scores all of a
decision's options at search time, while the training corpus stores only the options some candidate
uses.

## 2. Packed arrays

Both the training shards and the socket messages carry the same columnar arrays. `B` states,
`nc` cards, `no` options in total. Offsets are CSR: rows of state `i` are `sp_row[sp_off[i]:sp_off[i+1]]`.

| Array | dtype | shape | |
|---|---|---|---|
| `dense` | f32 | [B, 68] | |
| `sp_off` | i32 | [B+1] | |
| `sp_row`, `sp_val` | u16, f32 | [nsp] | state sparse bag |
| `card_off` | i32 | [B+1] | cards of state i: `card_off[i]:card_off[i+1]` |
| `card_group` | u8 | [nc] | 0..3 |
| `card_raw` | f32 | [nc, 68] | |
| `cr_off` | i32 | [nc+1] | |
| `cr_row`, `cr_val` | u16, f32 | [ncr] | card identity rows |
| `opt_off` | i32 | [B+1] | options of state i |
| `opt_dense` | f32 | [no, 24] | |
| `opt_bot` | u8 | [no] | |
| `opt_ea`, `opt_eb` | i32 | [no] | 1 + card index **within the state** (0 = none) |
| `os_off` | i32 | [no+1] | |
| `os_row`, `os_val` | u16, f32 | [nos] | option slots, row < 128 |
| `oh_off` | i32 | [no+1] | |
| `oh_row`, `oh_val` | u16, f32 | [noh] | option hashed rows |

Training shards add, per state (one record = one searched decision):

| Array | dtype | shape | |
|---|---|---|---|
| `outcome` | f32 | [B] | 1 the seat won, 0 lost, 0.5 draw, -1 unknown |
| `root_value` | f32 | [B] | the search's root value, in [0, 1] |
| `turn` | i32 | [B] | |
| `seat` | u8 | [B] | |
| `subset` | u8 | [B] | 1 for attackers and blockers |
| `kind` | u8 | [B] | gorge's `decision.Kind` |
| `sims` | i32 | [B] | |
| `choice` | i32 | [B] | the candidate the search played |
| `game` | i32 | [B] | index into `meta.json`'s `games` |
| `cand_off` | i32 | [B+1] | candidates of state i |
| `cand_visits`, `cand_prior`, `cand_q` | f32 | [ncand] | visits, the prior the search used (before root noise), mean value (0 when unvisited) |
| `co_off` | i32 | [ncand+1] | |
| `co_opt` | i32 | [nco] | option index **within the state** (0-based) |

**On disk** a shard is a directory: one `<name>.npy` per array (standard `.npy`, little-endian, C
order; read with `np.load(path, mmap_mode="r")`) plus `meta.json`:
`{"format": "dzg-pack-1", "n": B, "games": ["<game id>", ...], "source": "<corpus path>"}`.
`dzgorge pack -in corpus.visits.jsonl.gz -out DIR -shard 200000` writes `DIR/00000/`, `DIR/00001/`, ...

## 3. Socket protocol

The server listens on a Unix socket (`--socket PATH`) or TCP (`--tcp HOST:PORT`). A client opens one or
more connections and sends requests one at a time per connection; each request gets one response, in
order. All integers little-endian.

**Request**: `u32 nbytes` (bytes after this field), then

```
u32 magic = 0x31425A47      ("GZB1")
u32 B, nsp, nc, ncr, no, nos, noh
then the arrays in this order, each zero-padded to a multiple of 8 bytes:
  want   u8[B]             1: return this state's option scores
  dense  f32[B*68]
  sp_off i32[B+1]  sp_row u16[nsp]  sp_val f32[nsp]
  card_off i32[B+1]  card_group u8[nc]  card_raw f32[nc*68]  cr_off i32[nc+1]  cr_row u16[ncr]  cr_val f32[ncr]
  opt_off i32[B+1]  opt_dense f32[no*24]  opt_bot u8[no]  opt_ea i32[no]  opt_eb i32[no]
  os_off i32[no+1]  os_row u16[nos]  os_val f32[nos]
  oh_off i32[no+1]  oh_row u16[noh]  oh_val f32[noh]
```

The header is 32 bytes, so every array starts 8-byte aligned. Offsets inside a request start at 0.

**Response**: `u32 nbytes`, then `u32 magic = 0x31525A47 ("GZR1")`, `u32 B`, `u32 no`, `u32 0`, then
`value f32[B]` (the **probability** the searching seat wins, i.e. `sigmoid(value logit)`), zero-padded
to 8 bytes, then `score f32[no]` (option logits; 0 for states whose `want` is 0).

A state with `want = 1` and zero options is legal (value only).

## 4. The Python package

- `dzg/packfmt.py`: read and write shards and wire messages (numpy), with a Python writer so tests can
  build synthetic data.
- `dzg/data.py`: a dataset over shard directories (memmapped), a split by game, and a collate that turns
  a list of record indices into a `Batch` of torch tensors. The server builds the same `Batch` from a
  wire message.
- `dzg/models.py`: `mlp`, `transformer` and `gnn` (below), one interface: `forward(batch) -> (scores
  [no], value_logit [B])`.
- `dzg/train.py`: training with curves (JSONL), holdout and early stopping, checkpoints that carry
  their architecture config.
- `dzg/serve.py`: the batching inference server.
