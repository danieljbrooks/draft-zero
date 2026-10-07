//! `dzk play` (paired games to a JSONL file, resumable), `dzk summarize` and `dzk bench` (self-play
//! throughput).
//!
//! Paired-game protocol (draft-zero `tools/imitation_scale/play.py`): deck pair p keeps its seats (deck1 in
//! A, which plays first); game 1 has bot1 in A, game 2 (`swap`) bot1 in B; both use
//! `env_seed = seed * 1000 + p`, so the shuffles match and deck luck cancels over the pair, except in a
//! mirror (bot1 and bot2 parse to the same bot, aliases included), where the swapped game uses
//! `env_seed + 500000` (it would replay game 1 otherwise). `game_seed` = `env_seed`; the bots' generators
//! are derived from it with a salt (`rng::bot_seed`).
//!
//! Pairs: `--pairs FILE.tsv` lists `deck1<TAB>deck2` stems; pair p plays line `p % lines`, for p in
//! `0..--n-pairs` (default: the file's line count, or the end of `--pair-range` if that is larger), so a
//! short file can be cycled with new shuffles.
//!
//! The output file is locked (`flock`) for the whole run, so two runs cannot append to it at once. Resume
//! and the summary stream the file (one record in memory at a time); a duplicate (pair, swap) record counts
//! once (the first).

use crate::bots::BotSpec;
use crate::data;
use crate::deck::{load_deck, Deck};
use crate::record::{
    play_game, play_game_data, DecisionsRec, GameLimits, GameRecord, GameTask, GameTotals, HaltStep, TerminalRec,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// Progress to stderr that never panics: a closed stderr (a run piped into `head`, a lost terminal) must not
/// stop a run whose records are already safe on disk.
macro_rules! progress {
    ($($arg:tt)*) => {{
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

pub const MIRROR_SWAP_OFFSET: u64 = 500_000;

pub fn read_pairs(path: &Path) -> Result<Vec<(String, String)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t').map(str::trim).filter(|c| !c.is_empty());
        match (cols.next(), cols.next(), cols.next()) {
            (Some(a), Some(b), None) => out.push((a.to_string(), b.to_string())),
            _ => return Err(format!("{}:{}: expected deck1<TAB>deck2", path.display(), n + 1)),
        }
    }
    if out.is_empty() {
        return Err(format!("{}: no deck pairs", path.display()));
    }
    Ok(out)
}

pub fn load_decks(dir: &Path, stems: impl IntoIterator<Item = String>) -> Result<BTreeMap<String, Arc<Deck>>, String> {
    let mut decks = BTreeMap::new();
    for stem in stems {
        if let std::collections::btree_map::Entry::Vacant(e) = decks.entry(stem) {
            let d = load_deck(dir, e.key())?;
            e.insert(Arc::new(d));
        }
    }
    Ok(decks)
}

#[derive(Debug, Clone)]
pub struct PlayArgs {
    pub decks_dir: PathBuf,
    pub pairs_file: PathBuf,
    pub bot1: String,
    pub bot2: String,
    pub seed: u64,
    pub threads: usize,
    pub out: PathBuf,
    pub n_pairs: Option<usize>,
    pub pair_range: Option<(usize, usize)>,
    pub shard: Option<(usize, usize)>,
    pub resume: bool,
    /// `--data-out DIR`: training data of every real decision (`data.rs`), one shard per worker per run.
    pub data_out: Option<PathBuf>,
    /// `--max-permanents N`, `--max-game-seconds S`: a game reaching one ends as truncated.
    pub limits: GameLimits,
}

/// The run's games, in pair order.
pub fn game_tasks(pairs: &[(String, String)], n_pairs: usize, seed: u64, mirror: bool) -> Vec<GameTask> {
    let mut tasks = Vec::with_capacity(2 * n_pairs);
    for p in 0..n_pairs {
        let (d1, d2) = &pairs[p % pairs.len()];
        for swap in [false, true] {
            let env_seed = seed.wrapping_mul(1000).wrapping_add(p as u64).wrapping_add(if mirror && swap {
                MIRROR_SWAP_OFFSET
            } else {
                0
            });
            tasks.push(GameTask { pair: p, swap, deck1: d1.clone(), deck2: d2.clone(), game_seed: env_seed, env_seed });
        }
    }
    tasks
}

/// What resume reads of a finished game (unknown fields ignored).
#[derive(Debug, Deserialize)]
struct KeyRec {
    pair: usize,
    swap: bool,
    #[serde(default)]
    bot1: String,
    #[serde(default)]
    bot2: String,
    #[serde(default)]
    bot1_net: Option<String>,
    #[serde(default)]
    bot2_net: Option<String>,
    #[serde(default)]
    env_seed: Option<u64>,
    #[serde(default)]
    deck1: String,
    #[serde(default)]
    deck2: String,
    #[serde(default)]
    deck1_hash: Option<String>,
    #[serde(default)]
    deck2_hash: Option<String>,
}

/// Streams the complete lines of a games file: `f(line)` for each non-blank one. Returns the byte length of
/// the complete lines; a last line without its newline (a run killed mid-write) is left out (and reported).
fn for_each_line(path: &Path, mut f: impl FnMut(&[u8]) -> Result<(), String>) -> Result<(u64, u64), String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut buf = Vec::new();
    let (mut good, mut torn) = (0u64, 0u64);
    loop {
        buf.clear();
        let n = r.read_until(b'\n', &mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        if buf.last() != Some(&b'\n') {
            torn = n as u64;
            break;
        }
        if !buf.iter().all(|c| c.is_ascii_whitespace()) {
            f(&buf).map_err(|e| format!("{}: line at byte {good}: {e}", path.display()))?;
        }
        good += n as u64;
    }
    Ok((good, torn))
}

/// Reads an existing games file for --resume: the (pair, swap) keys present (each record is first passed to
/// `check`), the byte length of its complete lines, and how many records repeat a key.
fn scan_existing(
    path: &Path,
    mut check: impl FnMut(&KeyRec) -> Result<(), String>,
) -> Result<(BTreeSet<(usize, bool)>, u64, usize), String> {
    let mut done = BTreeSet::new();
    let mut dups = 0usize;
    let (good, _) = for_each_line(path, |line| {
        let k: KeyRec = serde_json::from_slice(line).map_err(|e| format!("unreadable record: {e}"))?;
        check(&k)?;
        if !done.insert((k.pair, k.swap)) {
            dups += 1;
        }
        Ok(())
    })?;
    Ok((done, good, dups))
}

fn wilson(k: f64, n: usize) -> (Option<f64>, Option<f64>) {
    if n == 0 {
        return (None, None);
    }
    let z = 1.96f64;
    let n = n as f64;
    let p = k / n;
    let d = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / d;
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / d;
    (Some(c - h), Some(c + h))
}

pub fn load_average() -> Option<String> {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return Some(s.split_whitespace().take(3).collect::<Vec<_>>().join(" "));
    }
    let out = std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    Some(s.trim().trim_matches(|c| c == '{' || c == '}').trim().to_string())
}

/// What the summary reads of a record (unknown fields ignored, so older records still parse).
#[derive(Debug, Deserialize)]
pub struct SummaryRec {
    pub pair: usize,
    pub swap: bool,
    #[serde(default)]
    pub bot1: String,
    #[serde(default)]
    pub bot2: String,
    #[serde(default)]
    pub bot1_net: Option<String>,
    #[serde(default)]
    pub bot2_net: Option<String>,
    #[serde(default)]
    pub winner: Option<String>,
    #[serde(default)]
    pub winner_role: Option<String>,
    #[serde(default)]
    pub turns: u32,
    #[serde(default)]
    pub rounds: Option<u32>,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub halt_step: Option<HaltStep>,
    #[serde(default)]
    pub terminal: TerminalRec,
    #[serde(default)]
    pub decisions: DecisionsRec,
    #[serde(default)]
    pub mulligans: BTreeMap<String, u8>,
    #[serde(default)]
    pub search: BTreeMap<String, Option<SearchTotals>>,
}

/// The search counters the summary totals (per seat in a record).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchTotals {
    #[serde(default)]
    pub simulations: u64,
    #[serde(default)]
    pub engine_errors: u64,
    #[serde(default)]
    pub halted_leaves: u64,
    #[serde(default)]
    pub world_refusals: u64,
    #[serde(default)]
    pub world_fallbacks: u64,
}

impl From<&GameRecord> for SummaryRec {
    fn from(g: &GameRecord) -> Self {
        serde_json::from_value(serde_json::to_value(g).expect("records serialize")).expect("a record summarizes")
    }
}

/// A running summary of games files: bot1's score (win 1, draw or truncated 0.5, loss 0; errors excluded),
/// folded record by record. A repeated (pair, swap) counts once (the first).
#[derive(Debug, Default)]
pub struct Summary {
    seen: BTreeSet<(usize, bool)>,
    duplicates: usize,
    /// bot1, bot2 and their networks' hashes (every record of a summary must agree)
    bots: Option<(String, String, Option<String>, Option<String>)>,
    games: usize,
    valid: usize,
    score: f64,
    bot1_wins: usize,
    bot2_wins: usize,
    no_winner: usize,
    seat_a_wins: usize,
    classes: BTreeMap<String, usize>,
    truncated_by_code: BTreeMap<String, usize>,
    errors_by_kind: BTreeMap<String, usize>,
    halts_by_role: BTreeMap<String, usize>,
    halts_forced: usize,
    by_pair: BTreeMap<usize, Vec<f64>>,
    /// Sums over the valid games: seconds, turns, rounds (and how many records had them), mulligans,
    /// physical, policy steps, real A, real B, forced.
    sums: [f64; 9],
    rounds_n: usize,
    max_choices: Option<u64>,
    max_policy_steps: Option<u64>,
    search: SearchTotals,
}

impl Summary {
    /// Fold one record in; false (and nothing counted) for a repeated (pair, swap).
    pub fn add(&mut self, g: &SummaryRec) -> bool {
        if !self.seen.insert((g.pair, g.swap)) {
            self.duplicates += 1;
            return false;
        }
        self.bots.get_or_insert_with(|| (g.bot1.clone(), g.bot2.clone(), g.bot1_net.clone(), g.bot2_net.clone()));
        self.games += 1;
        if let Some(c) = &g.terminal.classification {
            *self.classes.entry(c.clone()).or_default() += 1;
            if c == "truncated" {
                *self.truncated_by_code.entry(g.terminal.code.clone().unwrap_or_default()).or_default() += 1;
            }
        }
        for s in g.search.values().flatten() {
            self.search.simulations += s.simulations;
            self.search.engine_errors += s.engine_errors;
            self.search.halted_leaves += s.halted_leaves;
            self.search.world_refusals += s.world_refusals;
            self.search.world_fallbacks += s.world_fallbacks;
        }
        if let Some(e) = &g.error {
            let kind = e.split(':').take(2).collect::<Vec<_>>().join(":");
            *self.errors_by_kind.entry(kind).or_default() += 1;
            if let Some(h) = &g.halt_step {
                *self.halts_by_role.entry(h.role.clone()).or_default() += 1;
                self.halts_forced += usize::from(h.forced);
            }
            return true;
        }
        self.valid += 1;
        let pts = match g.winner_role.as_deref() {
            Some("bot1") => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        };
        self.score += pts;
        match g.winner_role.as_deref() {
            Some("bot1") => self.bot1_wins += 1,
            Some(_) => self.bot2_wins += 1,
            None => self.no_winner += 1,
        }
        self.seat_a_wins += usize::from(g.winner.as_deref() == Some("A"));
        self.by_pair.entry(g.pair).or_default().push(pts);
        let d = &g.decisions;
        let mulligans = g.mulligans.values().map(|&x| f64::from(x)).sum::<f64>() / 2.0;
        for (s, x) in self.sums.iter_mut().zip([
            g.seconds,
            f64::from(g.turns),
            f64::from(g.rounds.unwrap_or(0)),
            mulligans,
            d.physical as f64,
            d.policy_steps as f64,
            d.real_a as f64,
            d.real_b as f64,
            d.forced as f64,
        ]) {
            *s += x;
        }
        self.rounds_n += usize::from(g.rounds.is_some());
        self.max_choices = self.max_choices.max(Some(d.max_choices));
        self.max_policy_steps = self.max_policy_steps.max(Some(d.policy_steps));
        true
    }

    /// Stream every record of a games file in (a torn last line is skipped and reported).
    pub fn add_file(&mut self, path: &Path) -> Result<(), String> {
        let (_, torn) = for_each_line(path, |line| {
            let g: SummaryRec = serde_json::from_slice(line).map_err(|e| format!("unreadable record: {e}"))?;
            if let Some((b1, b2, n1, n2)) = &self.bots {
                if (b1.as_str(), b2.as_str()) != (g.bot1.as_str(), g.bot2.as_str()) {
                    return Err(format!("{} vs {} among {b1} vs {b2} games", g.bot1, g.bot2));
                }
                if (n1, n2) != (&g.bot1_net, &g.bot2_net) {
                    return Err(format!(
                        "pair {} swap {}: networks {:?} vs {:?} among {n1:?} vs {n2:?} games (a network file was \
                         replaced under the same path)",
                        g.pair, g.swap, g.bot1_net, g.bot2_net
                    ));
                }
            }
            self.add(&g);
            Ok(())
        })?;
        if torn > 0 {
            progress!("{}: skipped a torn last line ({torn} bytes)", path.display());
        }
        Ok(())
    }

    pub fn finish(&self) -> Value {
        let (bot1, bot2, bot1_net, bot2_net) = self.bots.clone().unwrap_or_default();
        let (lo, hi) = wilson(self.score, self.valid);
        let n = self.valid as f64;
        let mean = |i: usize| (self.valid > 0).then(|| self.sums[i] / n);
        let class = |c: &str| self.classes.get(c).copied().unwrap_or(0);
        let pairs = |f: &dyn Fn(&[f64]) -> bool| self.by_pair.values().filter(|v| v.len() == 2 && f(v)).count();
        let mean_seconds = mean(0);
        json!({
            "bot1": bot1,
            "bot2": bot2,
            "bot1_net": bot1_net,
            "bot2_net": bot2_net,
            "method": crate::record::METHOD,
            "games": self.games,
            "duplicates_ignored": self.duplicates,
            "valid_games": self.valid,
            "errors": self.games - self.valid,
            "errors_by_kind": self.errors_by_kind,
            "halts": {
                "by_role_of_last_step": self.halts_by_role,
                "last_step_forced": self.halts_forced,
                "note": "halted (fail-closed) games are errors, excluded from bot1_score",
            },
            "bot1_score": (self.valid > 0).then(|| self.score / n),
            "ci95": [lo, hi],
            "bot1_wins": self.bot1_wins,
            "bot2_wins": self.bot2_wins,
            "no_winner": self.no_winner,
            "seat_A_wins": self.seat_a_wins,
            "terminals": {"natural": class("natural"), "truncated": class("truncated"), "halted": class("halted"),
                          "truncated_by_code": self.truncated_by_code},
            "pairs_both_won_by_bot1": pairs(&|v| v.iter().all(|&x| x == 1.0)),
            "pairs_both_won_by_bot2": pairs(&|v| v.iter().all(|&x| x == 0.0)),
            "pairs_split": pairs(&|v| v.contains(&1.0) && v.contains(&0.0)),
            "mean_turns": mean(1),
            "mean_rounds": (self.rounds_n > 0).then(|| self.sums[2] / self.rounds_n as f64),
            "mean_seconds": mean_seconds,
            "games_per_hour_per_thread": mean_seconds.filter(|&s| s > 0.0).map(|s| 3600.0 / s),
            "game_seconds": self.sums[0],
            "mean_mulligans": mean(3),
            "decisions": {
                "physical": mean(4),
                "policy_steps": mean(5),
                "real_A": mean(6),
                "real_B": mean(7),
                "forced": mean(8),
                "max_choices": self.max_choices,
                "max_policy_steps": self.max_policy_steps,
            },
            "search": {
                "simulations": self.search.simulations,
                "engine_errors": self.search.engine_errors,
                "halted_leaves": self.search.halted_leaves,
                "world_refusals": self.search.world_refusals,
                "world_fallbacks": self.search.world_fallbacks,
            },
        })
    }
}

/// The summary of in-memory records (tests, tools).
pub fn summarize(records: &[GameRecord]) -> Value {
    let mut s = Summary::default();
    for g in records {
        s.add(&SummaryRec::from(g));
    }
    s.finish()
}

/// `summary.json` next to the output; a shard's is `summary.shard-I-of-N.json`.
pub fn summary_path(out: &Path, shard: Option<(usize, usize)>) -> PathBuf {
    match shard {
        None => out.with_file_name("summary.json"),
        Some((i, n)) => out.with_file_name(format!("summary.shard-{i}-of-{n}.json")),
    }
}

/// Lock `file` for this process (advisory `flock`); another holder is an error, a filesystem without locks
/// only a warning.
fn lock_exclusive(file: &File, path: &Path) -> Result<(), String> {
    match file.try_lock() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => {
            Err(format!("{}: another dzk play is writing this file (it is locked)", path.display()))
        }
        Err(std::fs::TryLockError::Error(e)) => {
            progress!("warning: {}: cannot lock ({e}); continuing without a lock", path.display());
            Ok(())
        }
    }
}

pub fn run_play(a: &PlayArgs) -> Result<Value, String> {
    let spec1 = BotSpec::parse(&a.bot1)?;
    let spec2 = BotSpec::parse(&a.bot2)?;
    let pairs = read_pairs(&a.pairs_file)?;
    let n_pairs = a.n_pairs.unwrap_or_else(|| pairs.len().max(a.pair_range.map_or(0, |(_, hi)| hi)));
    let mirror = spec1 == spec2;
    let mut tasks = game_tasks(&pairs, n_pairs, a.seed, mirror);
    if let Some((lo, hi)) = a.pair_range {
        tasks.retain(|t| t.pair >= lo && t.pair < hi);
    }
    if let Some((i, n)) = a.shard {
        tasks.retain(|t| t.pair % n == i);
    }
    if tasks.is_empty() {
        return Err(format!(
            "no games in this selection (--n-pairs {n_pairs}, --pair-range {:?}, --shard {:?})",
            a.pair_range, a.shard
        ));
    }
    let used: BTreeSet<usize> = tasks.iter().map(|t| t.pair % pairs.len()).collect();
    let decks = load_decks(&a.decks_dir, used.iter().flat_map(|&i| [pairs[i].0.clone(), pairs[i].1.clone()]))?;

    if let Some(dir) = a.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file =
        OpenOptions::new().create(true).append(true).open(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    lock_exclusive(&file, &a.out)?;
    let len = file.metadata().map(|m| m.len()).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let mut done = BTreeSet::new();
    if len > 0 {
        if !a.resume {
            return Err(format!("{} exists: pass --resume to continue it", a.out.display()));
        }
        // the file must be this run's: same bots (networks by content hash, not path), seeds and decks for the
        // games it shares with this selection
        let by_key: BTreeMap<(usize, bool), &GameTask> = tasks.iter().map(|t| ((t.pair, t.swap), t)).collect();
        let (net1, net2) = (spec1.net_hash(), spec2.net_hash());
        let (keys, good, dups) = scan_existing(&a.out, |k| {
            let Some(t) = by_key.get(&(k.pair, k.swap)) else { return Ok(()) };
            let (h1, h2) = (decks[&t.deck1].short_hash(), decks[&t.deck2].short_hash());
            let same_decks = k.deck1 == t.deck1
                && k.deck2 == t.deck2
                && k.deck1_hash.as_ref().is_none_or(|h| *h == h1)
                && k.deck2_hash.as_ref().is_none_or(|h| *h == h2);
            if k.bot1 != a.bot1 || k.bot2 != a.bot2 || k.env_seed != Some(t.env_seed) || !same_decks {
                return Err(format!(
                    "pair {} swap {} was played as {} vs {} with env_seed {:?} on {} ({:?}) vs {} ({:?}), \
                     not {} vs {} with {} on {} ({h1}) vs {} ({h2})",
                    t.pair,
                    t.swap,
                    k.bot1,
                    k.bot2,
                    k.env_seed,
                    k.deck1,
                    k.deck1_hash,
                    k.deck2,
                    k.deck2_hash,
                    a.bot1,
                    a.bot2,
                    t.env_seed,
                    t.deck1,
                    t.deck2
                ));
            }
            if (&k.bot1_net, &k.bot2_net) != (&net1, &net2) {
                return Err(format!(
                    "pair {} swap {} was played with networks {:?} vs {:?}, not {net1:?} vs {net2:?}: a network \
                     file changed under the same path (start a new --out, or restore the network)",
                    t.pair, t.swap, k.bot1_net, k.bot2_net
                ));
            }
            Ok(())
        })?;
        if good < len {
            progress!("resume: cutting a torn last line ({} bytes)", len - good);
            file.set_len(good).map_err(|e| format!("{}: {e}", a.out.display()))?;
        }
        if dups > 0 {
            progress!("resume: {dups} records repeat a (pair, swap) already in the file; the first of each counts");
        }
        done = keys;
    }
    let todo: Vec<GameTask> = tasks.iter().filter(|t| !done.contains(&(t.pair, t.swap))).cloned().collect();
    progress!(
        "dzk play: {} games in this selection, {} done, {} to play on {} threads ({} vs {}{})",
        tasks.len(),
        tasks.len() - todo.len(),
        todo.len(),
        a.threads,
        a.bot1,
        a.bot2,
        if mirror { ", mirror" } else { "" }
    );
    let load_before = load_average();
    let mut out = BufWriter::new(&file);
    let t0 = Instant::now();
    let mut data_run = match &a.data_out {
        Some(dir) => {
            Some(DataRun::start(dir, a, n_pairs, mirror, &used.iter().map(|&i| pairs[i].clone()).collect::<Vec<_>>())?)
        }
        None => None,
    };
    let collect = data_run.is_some();
    let todo = Arc::new(todo);
    let next = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<(GameRecord, Option<(Vec<u8>, data::GameData)>, usize)>();
    let mut handles = Vec::new();
    for worker in 0..a.threads.max(1) {
        let (todo, next, tx, decks, limits) = (todo.clone(), next.clone(), tx.clone(), decks.clone(), a.limits);
        let (b1, b2, s1, s2) = (a.bot1.clone(), a.bot2.clone(), spec1.clone(), spec2.clone());
        handles.push(std::thread::spawn(move || loop {
            let i = next.fetch_add(1, Ordering::SeqCst);
            let Some(t) = todo.get(i) else { break };
            let (rec, _, d) =
                play_game_data(t, &decks[&t.deck1], &decks[&t.deck2], (&b1, &s1), (&b2, &s2), collect, &limits);
            // compressed here, written by the main thread before the game's record line
            let d = d.map(|mut d| {
                let member = d.to_member(&rec);
                d.records.clear();
                (member, d)
            });
            if tx.send((rec, d, worker)).is_err() {
                break;
            }
        }));
    }
    drop(tx);
    let mut n = 0usize;
    let mut game_seconds = 0.0;
    for (rec, d, worker) in rx {
        // the record line first, then the game's data: a run killed in between loses that game's data, where the
        // other order would replay the game on resume and write its data twice
        let line = serde_json::to_string(&rec).map_err(|e| e.to_string())?;
        writeln!(out, "{line}").and_then(|_| out.flush()).map_err(|e| format!("{}: {e}", a.out.display()))?;
        if let (Some(run), Some((member, d))) = (data_run.as_mut(), d) {
            run.write(worker, &member, &d)?;
        }
        n += 1;
        game_seconds += rec.seconds;
        let what =
            rec.error.clone().map(|e| format!("ERROR {}", e.chars().take(120).collect::<String>())).unwrap_or_else(
                || {
                    rec.winner_role.clone().unwrap_or_else(|| {
                        format!("no winner ({})", rec.terminal.classification.clone().unwrap_or_default())
                    })
                },
            );
        progress!(
            "  {}/{} games  pair {}{}: {}  ({:.1} s, {} turns)  elapsed {:.0} s",
            n + tasks.len() - todo.len(),
            tasks.len(),
            rec.pair,
            if rec.swap { " swapped" } else { "" },
            what,
            rec.seconds,
            rec.turns,
            t0.elapsed().as_secs_f64()
        );
    }
    for h in handles {
        let _ = h.join();
    }
    drop(out);
    if let Some(run) = data_run.as_mut() {
        run.finish()?;
    }
    let wall = t0.elapsed().as_secs_f64();
    let mut summary = Summary::default();
    summary.add_file(&a.out)?;
    let mut s = summary.finish();
    s["bot1"] = json!(a.bot1);
    s["bot2"] = json!(a.bot2);
    s["this_run"] = json!({
        "games": n,
        "threads": a.threads,
        "wall_seconds": (wall * 10.0).round() / 10.0,
        "games_per_hour": if wall > 0.0 { Some(n as f64 * 3600.0 / wall) } else { None },
        "games_per_hour_per_thread": if wall > 0.0 { Some(n as f64 * 3600.0 / wall / a.threads.max(1) as f64) } else { None },
        "mean_game_seconds": if n > 0 { Some(game_seconds / n as f64) } else { None },
        "load_average_before": load_before,
        "load_average_after": load_average(),
    });
    s["config"] = json!({
        "decks_dir": a.decks_dir, "pairs_file": a.pairs_file, "seed": a.seed, "n_pairs": n_pairs,
        "pair_range": a.pair_range, "shard": a.shard.map(|(i, n)| format!("{i}/{n}")),
        "mirror": mirror, "bot1_spec": format!("{spec1:?}"), "bot2_spec": format!("{spec2:?}"),
        "max_physical_decisions": crate::game::MAX_PHYSICAL_DECISIONS,
        "max_policy_steps": crate::game::MAX_POLICY_STEPS,
        "max_permanents": a.limits.max_permanents,
        "max_game_seconds": a.limits.max_seconds,
        "rules": "kernel_limited_env schema 4 (engine priority windows, Foundations combat, London mulligans)",
        "mtg_kernel_commit": crate::MTG_KERNEL_COMMIT,
        "data_out": a.data_out,
    });
    if let Some(run) = &data_run {
        s["data"] = run.meta.clone();
    }
    let summary_path = summary_path(&a.out, a.shard);
    let mut f = File::create(&summary_path).map_err(|e| format!("{}: {e}", summary_path.display()))?;
    writeln!(f, "{}", serde_json::to_string_pretty(&s).unwrap()).map_err(|e| e.to_string())?;
    drop(file); // releases the lock
    Ok(s)
}

/// One `dzk play --data-out DIR` run: a shard per worker (`DIR/<tag>-wNN.dzd.gz`, created on its first game)
/// and the run's `DIR/<tag>.meta.json`, indexed in `DIR/meta.json`.
struct DataRun {
    dir: PathBuf,
    tag: String,
    shards: Vec<Option<(File, PathBuf, u64, u64)>>,
    meta: Value,
    by_source: [u64; 5],
    encode_failures: u64,
}

impl DataRun {
    fn start(
        dir: &Path,
        a: &PlayArgs,
        n_pairs: usize,
        mirror: bool,
        pairs: &[(String, String)],
    ) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        // a killed earlier run saved its meta every 25 games only: count its shards
        data::recount_incomplete(dir)?;
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let tag = format!("r{secs}-p{}", std::process::id());
        let mut decks: Vec<&String> = pairs.iter().flat_map(|(x, y)| [x, y]).collect();
        decks.sort();
        decks.dedup();
        let meta = json!({
            "format": data::DATA_FORMAT,
            "encoder": crate::encode::ENC_VERSION,
            "tag": tag,
            "complete": false,
            "started_unix": secs,
            "bot1": a.bot1, "bot2": a.bot2, "mirror": mirror,
            "seed": a.seed, "n_pairs": n_pairs, "pair_range": a.pair_range,
            "shard": a.shard.map(|(i, n)| format!("{i}/{n}")),
            "decks_dir": a.decks_dir, "pairs_file": a.pairs_file, "decks": decks,
            "games_file": a.out,
            "mtg_kernel_commit": crate::MTG_KERNEL_COMMIT,
            "games": 0, "records": 0, "shards": [],
        });
        let mut run = DataRun {
            dir: dir.to_path_buf(),
            tag,
            shards: (0..a.threads.max(1)).map(|_| None).collect(),
            meta,
            by_source: [0; 5],
            encode_failures: 0,
        };
        run.save()?;
        Ok(run)
    }

    fn write(&mut self, worker: usize, member: &[u8], d: &data::GameData) -> Result<(), String> {
        if self.shards[worker].is_none() {
            let path = self.dir.join(format!("{}-w{worker:02}.dzd.gz", self.tag));
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            self.shards[worker] = Some((f, path, 0, 0));
        }
        let (f, path, games, records) = self.shards[worker].as_mut().unwrap();
        f.write_all(member).and_then(|_| f.flush()).map_err(|e| format!("{}: {e}", path.display()))?;
        *games += 1;
        let n: u64 = d.by_source.iter().sum();
        *records += n;
        for (t, x) in self.by_source.iter_mut().zip(d.by_source) {
            *t += x;
        }
        self.encode_failures += d.encode_failures;
        let games: u64 = self.shards.iter().flatten().map(|s| s.2).sum();
        if games.is_multiple_of(25) {
            self.save()?; // a killed run's meta stays roughly current
        }
        Ok(())
    }

    fn save(&mut self) -> Result<(), String> {
        let shards: Vec<Value> = self
            .shards
            .iter()
            .flatten()
            .map(|(_, p, g, r)| {
                json!({"file": p.file_name().map(|x| x.to_string_lossy().to_string()), "games": g, "records": r})
            })
            .collect();
        self.meta["games"] = json!(shards.iter().map(|s| s["games"].as_u64().unwrap_or(0)).sum::<u64>());
        self.meta["records"] = json!(shards.iter().map(|s| s["records"].as_u64().unwrap_or(0)).sum::<u64>());
        self.meta["shards"] = json!(shards);
        let by: serde_json::Map<String, Value> =
            data::SOURCE_NAMES.iter().zip(self.by_source).map(|(k, v)| (k.to_string(), json!(v))).collect();
        self.meta["records_by_source"] = Value::Object(by);
        self.meta["encode_failures"] = json!(self.encode_failures);
        let p = self.dir.join(format!("{}.meta.json", self.tag));
        std::fs::write(&p, serde_json::to_string_pretty(&self.meta).unwrap() + "\n")
            .map_err(|e| format!("{}: {e}", p.display()))?;
        data::write_index(&self.dir)
    }

    fn finish(&mut self) -> Result<(), String> {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        self.meta["complete"] = json!(true);
        self.meta["finished_unix"] = json!(secs);
        self.save()
    }
}

/// `dzk summarize FILE...`: one summary over games files (shards of one run), duplicates counted once.
pub fn run_summarize(files: &[PathBuf]) -> Result<Value, String> {
    let mut summary = Summary::default();
    for f in files {
        summary.add_file(f)?;
    }
    let mut s = summary.finish();
    s["files"] = json!(files);
    Ok(s)
}

#[derive(Debug, Clone)]
pub struct BenchArgs {
    pub decks_dir: PathBuf,
    pub pairs_file: PathBuf,
    pub bot: String,
    pub seconds: f64,
    pub threads: Vec<usize>,
    pub seed: u64,
}

/// Self-play throughput: for each thread count, every thread plays games (the bot on both seats) until the
/// deadline; games running at the deadline finish and count. `*_per_s` divide by the time until the last
/// thread is done (so threads idle in the tail count against them: use `--seconds` well above a game's
/// length); `steady_*_per_s` add each thread's own rate (its work over its own busy time).
pub fn run_bench(a: &BenchArgs) -> Result<Value, String> {
    let spec = BotSpec::parse(&a.bot)?;
    let pairs = read_pairs(&a.pairs_file)?;
    let decks = load_decks(&a.decks_dir, pairs.iter().flat_map(|(x, y)| [x.clone(), y.clone()]))?;
    let mut results = Vec::new();
    for &threads in &a.threads {
        let load_before = load_average();
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs_f64(a.seconds);
        let next = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..threads.max(1) {
            let (next, decks, pairs, spec, name) =
                (next.clone(), decks.clone(), pairs.clone(), spec.clone(), a.bot.clone());
            let seed = a.seed;
            handles.push(std::thread::spawn(move || {
                let mut acc = (0u64, Vec::<Value>::new(), GameTotals::default(), 0.0f64, 0.0f64);
                while Instant::now() < deadline {
                    let g = next.fetch_add(1, Ordering::SeqCst);
                    let (d1, d2) = &pairs[g % pairs.len()];
                    let env_seed = seed.wrapping_mul(1000).wrapping_add(g as u64);
                    let t = GameTask {
                        pair: g,
                        swap: false,
                        deck1: d1.clone(),
                        deck2: d2.clone(),
                        game_seed: env_seed,
                        env_seed,
                    };
                    let (rec, tot) = play_game(&t, &decks[d1], &decks[d2], (&name, &spec), (&name, &spec));
                    acc.0 += 1;
                    if let Some(e) = &rec.error {
                        // enough to replay it: `dzk play` with --seed S, the same pairs file and --pair-range g:g+1
                        acc.1.push(json!({"game": g, "deck1": d1, "deck2": d2, "env_seed": env_seed, "error": e}));
                    }
                    acc.2.policy_steps += tot.policy_steps;
                    acc.2.physical += tot.physical;
                    acc.2.real += tot.real;
                    acc.2.simulations += tot.simulations;
                    acc.3 += rec.seconds;
                }
                acc.4 = t0.elapsed().as_secs_f64();
                acc
            }));
        }
        let mut games = 0u64;
        let mut errors = Vec::new();
        let mut tot = GameTotals::default();
        let mut game_seconds = 0.0;
        let mut steady = [0.0f64; 4];
        for h in handles {
            let (g, e, t, s, busy) = h.join().map_err(|_| "bench thread panicked".to_string())?;
            if busy > 0.0 {
                for (x, w) in steady.iter_mut().zip([g, t.policy_steps, t.real, t.simulations]) {
                    *x += w as f64 / busy;
                }
            }
            games += g;
            errors.extend(e);
            tot.policy_steps += t.policy_steps;
            tot.physical += t.physical;
            tot.real += t.real;
            tot.simulations += t.simulations;
            game_seconds += s;
        }
        let el = t0.elapsed().as_secs_f64();
        let per = |x: u64| x as f64 / el;
        let per_game = |x: u64| if games > 0 { Some(x as f64 / games as f64) } else { None };
        let r = json!({
            "threads": threads,
            "elapsed_seconds": (el * 100.0).round() / 100.0,
            "games": games,
            "errors": errors.len(),
            "error_games": errors,
            "games_per_s": per(games),
            "games_per_hour": per(games) * 3600.0,
            "policy_steps_per_s": per(tot.policy_steps),
            "physical_decisions_per_s": per(tot.physical),
            "real_decisions_per_s": per(tot.real),
            "search_simulations_per_s": per(tot.simulations),
            "steady_games_per_s": steady[0],
            "steady_policy_steps_per_s": steady[1],
            "steady_real_decisions_per_s": steady[2],
            "steady_search_simulations_per_s": steady[3],
            "deadline_overrun_seconds": ((el - a.seconds) * 100.0).round() / 100.0,
            "mean_policy_steps_per_game": per_game(tot.policy_steps),
            "mean_physical_decisions_per_game": per_game(tot.physical),
            "mean_real_decisions_per_game": per_game(tot.real),
            "mean_game_seconds": if games > 0 { Some(game_seconds / games as f64) } else { None },
            "load_average_before": load_before,
            "load_average_after": load_average(),
        });
        progress!("{}", serde_json::to_string(&r).unwrap());
        results.push(r);
    }
    Ok(json!({
        "bot": a.bot,
        "pairs_file": a.pairs_file,
        "seconds": a.seconds,
        "seed": a.seed,
        "mtg_kernel_commit": crate::MTG_KERNEL_COMMIT,
        "results": results,
    }))
}
