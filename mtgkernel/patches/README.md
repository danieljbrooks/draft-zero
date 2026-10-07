# Patches to the pinned mtg-kernel

`setup.sh` applies every `*.patch` here, in name order, with `git apply` to the clone of mtg-kernel at
`a4e1474b492f2f14ce1ed9ef8e40309fb70e4e8e` in `mtgkernel/.deps/mtg-kernel`. Neither patch changes a rule,
card, action encoding, stable id, RNG stream or serialized identity, and the JSONL server
(`kernel_limited_env`) behaves exactly as before: 0001 changes visibility only, and 0002 adds an opt-in
session flag (off by default) and read-only accessors. The patched tree is left uncommitted; mtg-kernel's
`build.rs` accepts a dirty tree (it records `MTG_KERNEL_BUILD_GIT_CLEAN=false` and hashes the committed HEAD
tree, so the build identity is still the pinned commit's).

## `0001-dzk-session-visibility.patch` (visibility only)

| Line | Why |
| --- | --- |
| `rl_session.rs`: `pub(crate) fn reset_with_custom_decks_and_london_v1` -> `pub fn` | The one constructor `limited_session_v1::LimitedJsonlServerV1::new_with_london_mulligans_v1` (the `kernel_limited_env --london-mulligans-v1`, schema 4) uses to reset a game; `dzk` calls it with the same arguments (`PriorityModeV1::EngineWindowsV1`, Foundations combat on, London mulligans on). Without it a custom-deck game with these rules can only be driven through the JSONL server (serialization on every step). |
| `rl_session.rs`: `pub fn game_state_mut_v1(&mut self) -> &mut GameState` | Re-dealing the hidden cards of a search copy (PIMC determinization), as `kernel_native_search_opponent_v1::redeterminize_hidden_zones_v1` does on the FastActor session (that function is `pub(crate)`; `dzk` re-implements it). Only ever called on a clone. |
| `rl_session.rs`: `pub fn policy_surface_v1(&self) -> &PolicySurfaceV5` | Read-only. The determinization admission check re-observes the re-dealt copy and requires the acting player's observation to be unchanged (the kernel's own FastActor redeterminization performs the same rebuild-and-compare). |
| `state.rs`: `pub(crate) fn resample_future_randomness_for_search_v3` -> `pub fn` | mtg-kernel's own search-only re-seeding of a disposable clone's future shuffle stream. `dzk` calls it on every determinized world (seeded from the bot's generator, never from the real stream), so a search no longer sees the real future shuffles (at a London mulligan they told the searcher exactly which cards come back). |

## `0002-dzk-session-throughput.patch` (opt-in flag and borrows; not visibility-only)

The engine built and hashed a full observation at every policy step (forced steps included) and `step`
returned a deep copy of it and of every legal action: 77-81% of a worker's CPU in a profile, none of it
used by `dzk`. This patch removes both costs for `dzk` and nothing else. Measured on the fixture pair, one
thread, back to back: random self-play 22.3 -> 81.8 games/s, `mcts:100` 3,426 -> 10,807 simulations/s;
records byte-identical to the unpatched build (200 random, 120 random-vs-first, 24 `mcts:30`, 24
`mcts:40,worlds=2` vs `mcts:20,cache=8`, 24 `greedy1` games and the Ward fail-close game; only the search
counter `replay_steps` changes, from `dzk`'s own copy saving).

| Line | Why |
| --- | --- |
| `rl.rs`: `pub fn observe_policy_v5_without_projection_hash_v1(...)` | `observe_policy_v5` minus its last line: the same builder, `FullArtifact` text mode and fail-closed checks (the Ward observation fail-close still happens), with `visible_projection_hash` left 0. That hash is serde_json of the whole observation, then hashed (two thirds of all CPU); it is a pure function of the other fields. The crate's existing unhashed builder (`observe_policy_v5_unhashed_for_flat_policy`) uses another text mode, so it is not the same observation. |
| `rl_session.rs`: field `skip_visible_projection_hash_v1: bool` (false in all three constructors) and `pub fn set_skip_visible_projection_hash_v1` | Selects that builder at the session's one observe call. Off unless set; only `dzk` sets it (`Game::new`), right after the reset (so a game's first observation still carries the hash). The flag changes no transition: the hash is read only by the JSONL environment-hash envelope, which `dzk` never computes. |
| `rl_session.rs`: `pub fn step_quiet_v1(...)` | `step` without its `current_response()` deep copy (exactly `apply_step_profiled(.., None)`, which `step` calls first). |
| `rl_session.rs`: `current_actor_v1`, `current_legal_action_count_v1`, `current_legal_action_v1(i)`, `current_observation_v1`, `current_substep_v1`, `terminal_v1` (and a private helper) | Read-only borrows of exactly what `current_response()` copies (none at a terminal, as it reports). `Game` reads the decision through them instead of keeping its own copy, which also halves a search copy's size at a large decision (a 5,040-order trigger decision). |

Not done (would change behavior, so left for a decision): building no observation at all (a further ~2x;
the Ward fail-close lives in the observation layer, so those games would play on instead of halting).

## Regenerating

After editing the clone (`mtgkernel/.deps/mtg-kernel`):

```bash
cd mtgkernel/.deps/mtg-kernel
git apply --cached ../../patches/0001-dzk-session-visibility.patch   # stage 0001's lines only
git diff > ../../patches/0002-dzk-session-throughput.patch           # the rest, on top of 0001
git reset -q
```

(To change 0001 itself: stage its lines by hand, `git diff --cached > ../../patches/0001-...`, then the above.)
