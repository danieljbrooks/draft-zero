# MageZero's graph network, part 2: local layers, a full run on Dama's GPUs, and games

*Follows [docs/023](023-gnn-run-log.md), where the GNN at width 256 beat experiment #4's MLP and transformer on
every offline measure (test set NLL 0.214 against 0.229 and 0.234). Started Monday 5 October 2026, 8:30 AM PT, at
Dan's request after Will's review ("this looks great! also try to ablate GNN local layers ... the num local layers is
what im most interested in though at high data scale"). All times are Pacific.*

## Status (Monday 5 October, 9:00 AM PT)

| Stage | Status | Where | Spend |
|---|---|---|---|
| 1. Code: `local_depth`, the streamed cache | done, on main (160eca4) | laptop | – |
| 2. Round 7: local layers at 10% | running | r1, both GPUs | free |
| 3. Round 8: the best at 30% ("at high data scale") | | r1 | free |
| 4. The full run | the cache for all the rows building | r1 | free |
| 5. Games: the PIMC ladder and 10,000 self-play games | the pipeline on a Community 3070 | RunPod | |
| 6. 17lands analysis (docs/019 §4.4) | | laptop | – |

## The plan (Dan, 5 October)

1. **Small-scale experiments, Will's question first:** the number of local layers. In MageZero's network each of
   the 2 bottom-up passes runs one `LocalLayer` per node type. docs/023 varied the passes (1 was worse) and the
   global layers, but not the layers per type. `local_depth` (graph_net.py) stacks that many LocalLayers per type
   within a pass, each re-attending from the node's updated embedding to the same children; depth 1 is Will's
   network, with his checkpoint layout. Width stays 256 (Dan: for speed).
2. **"At high data scale":** the most promising shapes again on 30% of the games.
3. **A full-scale run** of the best recipe on all the training games, **on Dama's RTX 6000s** (r1).
4. **Evaluation**, on cheap rented GPUs (RTX 3070/3090, Community): 100 paired games at 0, 100, 300, 1,000 and
   3,000 simulations against the baseline (MageZero's heuristic bot at 100), with PIMC as in docs/019 §4.6; and
   10,000 policy-only self-play games for docs/019 §4.4's 17lands analysis. Games an hour and games per dollar.

## 1. Fitting all the rows on r1

r1's container has 40 GiB of RAM. All the training rows hold 2.50 billion graph nodes and 5.10 billion edges (the
six tables), ~58 GB in the trainer's arrays: docs/023's full run went to a rented 188 GB pod for that reason.

- **Compact dtypes after mapping** (`_compact`): node rows as int16 (1,932 leaves) and edge labels as int8 (22
  labels), 2.5 of a node's 7 bytes and 3 of an edge's 8: ~38 GB for all the rows. Every run gets this, in RAM or not.
- **`stream_cache`**: the vocab is counted a chunk of rows at a time (300,000), then each table is mapped and
  written a chunk at a time into preallocated `.npy` files, memory-mapped for training. No step holds a whole
  table. Same vocabs and arrays as the in-memory path (`test_stream_cache_matches_the_in_memory_path`). The cache
  sits on the container's scratch disk (`/var/tmp`, 153 GB free; lost on a restart), and the page cache keeps most of
  it resident.
