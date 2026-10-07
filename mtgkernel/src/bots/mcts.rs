//! `mcts:N`: a port of draft-zero's offline search baseline "heuristic@N" (java/mzbridge BenchSearch's
//! `searchTree` with its Config defaults, the leaf `heuristic`), on mtg-kernel.
//!
//! - PUCT with c = 1 and uniform priors (1/k); an unvisited child's value is 0; ties to the lower index.
//! - Values are always the searcher's seat; selection maximizes `q` at the searcher's nodes and `-q` at the
//!   opponent's: `sign * q + c * prior * sqrt(N) / (1 + n)`.
//! - The budget is N fresh simulations per decision (no tree reuse). The root is expanded first, not a
//!   simulation. One simulation walks down by PUCT to a child not yet computed, computes it (a copy of the
//!   parent's game, the chosen action, then every forced decision up to the next real decision or a
//!   terminal), scores it with GameStateEvaluator3 (`eval.rs`), expands it, and backs the value up,
//!   multiplied by 0.99 per tree edge ("ply").
//! - The final choice: most root visits, ties to the higher mean, then the lower index.
//! - The single-child shortcut is kept, but never fires: forced decisions are already auto-stepped.
//!
//! Differences from the Java baseline, by construction: a tree edge is one real decision (2+ masked
//! actions) of either player; forced decisions and mana activations never appear in the tree, so the
//! discount counts real decisions rather than XMage's plies. Every leaf is scored by GSE3 (mtg-kernel has no
//! priority vs. micro decision split, so there is no "inherit the parent's score"). A draw scores 0 (the
//! Java scores every non-win terminal -1), and so does a truncated game inside the search. A halted
//! (fail-closed) leaf is removed from the tree like an engine error and counted (`halted_leaves`): scored 0,
//! it attracted a searcher that was behind, and the game record then excluded the game as an error.
//! mtg-kernel offers all orders of a simultaneous trigger group of up to 7 as one decision, so a node can
//! have up to 7! = 5040 children, each with a uniform prior.
//!
//! Hidden information (PIMC): `worlds` (default 1) determinized copies of the root from the searcher's view
//! (`determinize.rs`: open decklists, hidden cards re-dealt, the engine's future shuffle stream re-seeded),
//! the budget split across them, one tree each; root visits are summed by action `stable_id`. A refused
//! re-deal is retried with new seeds (`world_refusals`); after `WORLD_ATTEMPTS` refusals that world is not
//! searched (`world_fallbacks`, also totalled in `summary.json`; none seen so far): searching the real game
//! would read the hidden cards (and pmcts's network would encode the opponent's real hand at its nodes). A
//! decision with no searched world (or every root option failed in the engine) is answered without a search
//! (`unsearched`): pmcts by the network's own policy on the actor's observation (argmax), mcts uniformly at
//! random; its data record is one-hot with source `net` / `random`.
//!
//! Mulligans: the bot's own London mulligan announcements follow `mull=` (`MulliganPolicy`, default keep, as
//! the Java baseline's games never mulligan) instead of a search, at the live decision and inside the tree
//! (a node of the searcher's own announcement gets the policy's answer as its only child); bottom choices
//! and the opponent's announcements are searched.
//!
//! Memory: every computed node keeps its game copy until all its children are computed and hold their
//! own (the last child that needs a non-root parent's copy takes it instead of cloning it). Past `max_cached` stored copies (default 4096) new nodes stop storing one, and a child below such a
//! node is computed by replaying the path from the nearest ancestor that still has its copy.
//!
//! `pmcts:N,net=FILE` (stage 2, `cfg.net` set): the same search with the network in it. Every computed
//! node (both seats) is encoded from its acting player's seat in its world and run through the network
//! once: the priors of its children are `softmax(logits / pt)` (default pt = 1) and its leaf value is the
//! network's value, converted to the searcher's seat (negated at the opponent's nodes), or GSE3
//! (`leaf=heuristic`), or `λ · net + (1 - λ) · GSE3` (`leaf=mix:λ`). Terminals score ±1/0 as before. The
//! root's priors come from the same network on each world's root (the actor's observation is the real
//! one). Differences from `mcts:N`: the exploration term uses `sqrt(max(N, 1))` so a fresh root's first
//! simulation follows the prior (with `sqrt(0)` every child scores 0 and the lowest index wins); optional
//! Dirichlet root noise for self-play data (`noise=ε,alpha=α`: priors `(1 - ε) p + ε Dir(α)` at each
//! world's root; off by default, keep it off for evaluation); optional move sampling for the bot's first
//! `temp_moves` searched decisions of a game (`temp=τ`: probability ∝ visits^(1/τ); default 0 moves, i.e.
//! always the most-visited move); optional first-play urgency (`fpu=R`: an unvisited child is valued at its
//! parent's mean value minus R from the deciding player's seat, instead of 0, a draw; off by default). A node
//! whose decision cannot be encoded (no observation; not seen in practice) gets uniform priors and the GSE3
//! value (`encode_failures`).

use super::{mulligan_override, world_for, Bot, MulliganPolicy, MulliganStats};
use crate::data::{PolicyTarget, Source};
use crate::encode::{encode_game, Encoded};
use crate::eval;
use crate::game::Game;
use crate::nn::{softmax, NetRef, Scratch};
use crate::rng::SplitMix64;
use mtg_kernel::ids::PlayerId;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct MctsConfig {
    pub budget: u32,
    pub worlds: u32,
    pub c_puct: f64,
    pub discount: f64,
    pub max_cached: usize,
    pub mulligan: MulliganPolicy,
    /// `pmcts:N`: network priors and leaf values (None: `mcts:N`).
    pub net: Option<NetSearch>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            budget: 100,
            worlds: 1,
            c_puct: 1.0,
            discount: 0.99,
            max_cached: 4096,
            mulligan: MulliganPolicy::Keep,
            net: None,
        }
    }
}

/// Leaf values of `pmcts:N`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Leaf {
    Net,
    Heuristic,
    /// `λ · net + (1 - λ) · GSE3`.
    Mix(f64),
}

/// The network part of `pmcts:N`.
#[derive(Debug, Clone, PartialEq)]
pub struct NetSearch {
    pub net: NetRef,
    pub prior_temp: f32,
    pub leaf: Leaf,
    pub noise_eps: f64,
    pub noise_alpha: f64,
    pub temp: f64,
    pub temp_moves: u32,
    /// First-play urgency reduction (None: an unvisited child is valued 0).
    pub fpu: Option<f64>,
}

/// `pmcts` options as parsed (the network is loaded by `finish`).
#[derive(Debug, Clone)]
pub struct NetSearchOptions {
    path: Option<String>,
    prior_temp: f32,
    leaf: Leaf,
    noise_eps: f64,
    noise_alpha: f64,
    temp: f64,
    temp_moves: u32,
    fpu: Option<f64>,
}

impl Default for NetSearchOptions {
    fn default() -> Self {
        NetSearchOptions {
            path: None,
            prior_temp: 1.0,
            leaf: Leaf::Net,
            noise_eps: 0.0,
            noise_alpha: 0.3,
            temp: 1.0,
            temp_moves: 0,
            fpu: None,
        }
    }
}

impl NetSearchOptions {
    /// Take one `key=value` option; false if the key is not a network option.
    pub fn set(&mut self, k: &str, v: &str) -> Result<bool, String> {
        let bad = || format!("bad value for {k}");
        let num = |v: &str| v.parse::<f64>().ok().filter(|x| x.is_finite());
        match k {
            "net" => self.path = Some(v.to_string()),
            "pt" => self.prior_temp = num(v).filter(|&x| x > 0.0).ok_or_else(bad)? as f32,
            "leaf" => {
                self.leaf = match v {
                    "net" => Leaf::Net,
                    "heuristic" => Leaf::Heuristic,
                    _ => Leaf::Mix(
                        v.strip_prefix("mix:").and_then(num).filter(|l| (0.0..=1.0).contains(l)).ok_or_else(bad)?,
                    ),
                }
            }
            "noise" => self.noise_eps = num(v).filter(|x| (0.0..=1.0).contains(x)).ok_or_else(bad)?,
            "alpha" => self.noise_alpha = num(v).filter(|&x| x > 0.0).ok_or_else(bad)?,
            "temp" => self.temp = num(v).filter(|&x| x >= 0.0).ok_or_else(bad)?,
            "temp_moves" => self.temp_moves = v.parse().map_err(|_| bad())?,
            "fpu" => self.fpu = Some(num(v).filter(|&x| x >= 0.0).ok_or_else(bad)?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub fn finish(self) -> Result<NetSearch, String> {
        let path = self.path.ok_or("pmcts needs net=FILE")?;
        Ok(NetSearch {
            net: NetRef::load(&path)?,
            prior_temp: self.prior_temp,
            leaf: self.leaf,
            noise_eps: self.noise_eps,
            noise_alpha: self.noise_alpha,
            temp: self.temp,
            temp_moves: self.temp_moves,
            fpu: self.fpu,
        })
    }
}

/// The network in a search: buffers and counters.
pub struct NetEval {
    pub cfg: NetSearch,
    enc: Encoded,
    sc: Scratch,
    probs: Vec<f32>,
    pub evals: u64,
    pub encode_failures: u64,
}

impl NetEval {
    pub fn new(cfg: NetSearch) -> Self {
        NetEval {
            cfg,
            enc: Encoded::default(),
            sc: Scratch::default(),
            probs: Vec::new(),
            evals: 0,
            encode_failures: 0,
        }
    }

    /// Priors over `g`'s masked actions and the value from `me`'s seat (None: not encodable).
    pub fn eval(&mut self, g: &Game, me: PlayerId) -> Option<(Vec<f64>, f64)> {
        if encode_game(g, &mut self.enc).is_err() {
            self.encode_failures += 1;
            return None;
        }
        self.cfg.net.model.forward(&self.enc, &mut self.sc);
        self.evals += 1;
        softmax(&self.sc.logits, self.cfg.prior_temp, &mut self.probs);
        let v = f64::from(self.sc.value);
        let v = if g.actor() == Some(me) { v } else { -v };
        Some((self.probs.iter().map(|&p| f64::from(p)).collect(), v))
    }

    /// The leaf value of a computed nonterminal node from `me`'s seat, and its children's priors.
    fn leaf(&mut self, g: &Game, me: PlayerId) -> (Option<Vec<f64>>, f64) {
        match self.eval(g, me) {
            None => (None, eval::evaluate(g, me)),
            Some((p, v)) => {
                let v = match self.cfg.leaf {
                    Leaf::Net => v,
                    Leaf::Heuristic => eval::evaluate(g, me),
                    Leaf::Mix(l) => l * v + (1.0 - l) * eval::evaluate(g, me),
                };
                (Some(p), v)
            }
        }
    }
}

/// A standard normal draw (Box-Muller) from the bot's generator.
fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn uniform(rng: &mut SplitMix64) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Gamma(alpha, 1) (Marsaglia-Tsang; alpha < 1 boosted).
pub fn gamma(alpha: f64, rng: &mut SplitMix64) -> f64 {
    if alpha < 1.0 {
        return gamma(alpha + 1.0, rng) * uniform(rng).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = uniform(rng);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// `(1 - eps) p + eps Dir(alpha)`.
pub fn add_dirichlet(p: &mut [f64], eps: f64, alpha: f64, rng: &mut SplitMix64) {
    let g: Vec<f64> = (0..p.len()).map(|_| gamma(alpha, rng)).collect();
    let s: f64 = g.iter().sum();
    if s <= 0.0 || !s.is_finite() {
        return;
    }
    for (x, n) in p.iter_mut().zip(g) {
        *x = (1.0 - eps) * *x + eps * n / s;
    }
}

const NONE: u32 = u32::MAX;

struct Node {
    parent: u32,
    /// Masked index of the action that leads here from the parent.
    choice: u32,
    game: Option<Box<Game>>,
    validated: bool,
    expanded: bool,
    terminal: bool,
    actor: Option<PlayerId>,
    kids: Vec<u32>,
    n: u32,
    w: f64,
    prior: f64,
    /// A terminal's value (its game is not kept).
    tvalue: f64,
}

impl Node {
    fn new(parent: u32, choice: u32, prior: f64) -> Self {
        Node {
            parent,
            choice,
            game: None,
            validated: false,
            expanded: false,
            terminal: false,
            actor: None,
            kids: Vec::new(),
            n: 0,
            w: 0.0,
            prior,
            tvalue: 0.0,
        }
    }

    fn q(&self) -> f64 {
        if self.n > 0 {
            self.w / f64::from(self.n)
        } else {
            0.0
        }
    }
}

/// Per-decision search counters, summed over a game.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub searches: u64,
    pub simulations: u64,
    pub nodes: u64,
    /// Engine steps spent re-deriving uncached nodes.
    pub replay_steps: u64,
    /// Children whose computation failed in the engine (removed from the tree).
    pub engine_errors: u64,
    /// Children whose game halted (fail-closed; removed from the tree) or was truncated (scored 0).
    pub halted_leaves: u64,
    pub truncated_leaves: u64,
    pub worlds: u64,
    /// Re-deals refused by the admission check (each retried with a new seed).
    pub world_refusals: u64,
    /// Worlds given up after `WORLD_ATTEMPTS` refused re-deals (not searched).
    pub world_fallbacks: u64,
    pub world_fallback_reasons: BTreeMap<String, u64>,
    pub max_tree_nodes: u64,
    /// The most children any tree node had.
    pub max_children: u64,
    /// Live mulligan announcements answered by the `mull` policy instead of a search.
    pub mulligan: MulliganStats,
    /// pmcts: decisions whose move was sampled from the visits (`temp_moves`).
    pub sampled_moves: u64,
    /// Decisions answered without a search (no world admitted, or every root option failed).
    pub unsearched: u64,
}

/// The visits and value sum of each root option, by masked index.
struct RootResult {
    visits: Vec<u32>,
    w: Vec<f64>,
}

struct Tree<'a> {
    cfg: &'a MctsConfig,
    me: PlayerId,
    nodes: Vec<Node>,
    cached: usize,
    stats: &'a mut SearchStats,
    net: Option<&'a mut NetEval>,
}

impl<'a> Tree<'a> {
    /// `root_priors`: the root children's priors (pmcts; None = uniform).
    fn new(
        root: Game,
        cfg: &'a MctsConfig,
        stats: &'a mut SearchStats,
        net: Option<&'a mut NetEval>,
        root_priors: Option<Vec<f64>>,
    ) -> Self {
        let me = root.actor().expect("search root is a decision");
        let mut t = Tree { cfg, me, nodes: Vec::with_capacity(cfg.budget as usize + 64), cached: 0, stats, net };
        let mut r = Node::new(NONE, 0, 1.0);
        r.validated = true;
        r.actor = root.actor();
        let k = root.num_choices();
        let only = mulligan_override(cfg.mulligan, &root, me);
        r.game = Some(Box::new(root));
        t.cached = 1;
        t.nodes.push(r);
        t.expand(0, k, only, root_priors);
        t
    }

    /// Children for every masked action of node `i` (`k` of them), or the one action `only`; `priors`
    /// (pmcts) over the `k` actions, else uniform.
    fn expand(&mut self, i: usize, k: usize, only: Option<usize>, priors: Option<Vec<f64>>) {
        debug_assert!(k > 0);
        let first = self.nodes.len() as u32;
        let choices: Vec<u32> = match only {
            Some(c) => vec![c as u32],
            None => (0..k as u32).collect(),
        };
        let uniform = 1.0 / choices.len() as f64;
        let priors = priors.filter(|p| p.len() == k && only.is_none());
        self.stats.max_children = self.stats.max_children.max(choices.len() as u64);
        for &c in &choices {
            let prior = priors.as_ref().map_or(uniform, |p| p[c as usize]);
            self.nodes.push(Node::new(i as u32, c, prior));
        }
        self.nodes[i].kids = (first..first + choices.len() as u32).collect();
        self.nodes[i].expanded = true;
        self.stats.nodes += choices.len() as u64;
    }

    fn select(&self, i: usize) -> usize {
        let node = &self.nodes[i];
        if node.kids.len() == 1 {
            return node.kids[0] as usize;
        }
        let sign = if node.actor == Some(self.me) { 1.0 } else { -1.0 };
        let sqrt_n = if self.net.is_some() { f64::from(node.n.max(1)).sqrt() } else { f64::from(node.n).sqrt() };
        // an unvisited child's value from the deciding seat: 0, or (fpu=R) the node's own mean minus R
        let unvisited = self.net.as_ref().and_then(|n| n.cfg.fpu).map_or(0.0, |r| sign * node.q() - r);
        let mut best = node.kids[0] as usize;
        let mut best_val = f64::NEG_INFINITY;
        for &k in &node.kids {
            let kid = &self.nodes[k as usize];
            let q = if kid.n == 0 { unvisited } else { sign * kid.q() };
            let val = q + self.cfg.c_puct * kid.prior * sqrt_n / (1.0 + f64::from(kid.n));
            if val > best_val {
                best_val = val;
                best = k as usize;
            }
        }
        best
    }

    /// The game at node `i` (unvalidated: its parent is validated): a copy of the nearest stored ancestor's
    /// game with the path replayed.
    fn compute(&mut self, i: usize) -> Result<Game, String> {
        // The last child that needs a stored (non-root) parent's game takes it instead of a copy: every other
        // child is terminal or holds its own, so `maybe_release` would drop it right after this one anyway.
        let p = self.nodes[i].parent as usize;
        if p != 0
            && self.nodes[p].game.is_some()
            && self.nodes[p].kids.iter().all(|&k| {
                let kid = &self.nodes[k as usize];
                k as usize == i || (kid.validated && (kid.terminal || kid.game.is_some()))
            })
        {
            let mut g = *self.nodes[p].game.take().unwrap();
            self.cached -= 1;
            g.play(self.nodes[i].choice as usize)?;
            return Ok(g);
        }
        let mut path = vec![self.nodes[i].choice];
        let mut a = self.nodes[i].parent as usize;
        while self.nodes[a].game.is_none() {
            path.push(self.nodes[a].choice);
            a = self.nodes[a].parent as usize;
        }
        let mut g = self.nodes[a].game.as_ref().unwrap().clone_untracked();
        let replays = path.len() - 1;
        for &c in path.iter().rev() {
            g.play(c as usize)?;
        }
        self.stats.replay_steps += replays as u64;
        Ok(g)
    }

    fn remove_dead(&mut self, i: usize) {
        let p = self.nodes[i].parent;
        if p == NONE {
            return;
        }
        let p = p as usize;
        self.nodes[p].kids.retain(|&k| k as usize != i);
        if self.nodes[p].kids.is_empty() && self.nodes[p].parent != NONE {
            self.remove_dead(p);
        }
    }

    fn backprop(&mut self, leaf: usize, v: f64) {
        let mut cur = leaf;
        let mut val = v;
        loop {
            let node = &mut self.nodes[cur];
            node.n += 1;
            node.w += val;
            if node.parent == NONE {
                break;
            }
            cur = node.parent as usize;
            val *= self.cfg.discount;
        }
    }

    /// Drop a node's game once every child is computed and holds its own copy or is terminal (never the
    /// root's): nothing below it needs it any more.
    fn maybe_release(&mut self, i: usize) {
        if i == 0 || self.nodes[i].game.is_none() {
            return;
        }
        let done = self.nodes[i].kids.iter().all(|&k| {
            let kid = &self.nodes[k as usize];
            kid.validated && (kid.terminal || kid.game.is_some())
        });
        if done {
            self.nodes[i].game = None;
            self.cached -= 1;
        }
    }

    fn run(&mut self, budget: u32) -> RootResult {
        let mut sims = 0u32;
        let mut iterations = 0u64;
        let max_iterations = 4 * u64::from(budget) + 200;
        while sims < budget && iterations < max_iterations && !self.nodes[0].kids.is_empty() {
            iterations += 1;
            let mut cur = 0usize;
            while self.nodes[cur].expanded && !self.nodes[cur].kids.is_empty() && !self.nodes[cur].terminal {
                cur = self.select(cur);
            }
            if self.nodes[cur].expanded && self.nodes[cur].kids.is_empty() && !self.nodes[cur].terminal {
                self.remove_dead(cur); // every option below it failed
                continue;
            }
            let v = if self.nodes[cur].validated {
                // only a terminal leaf is revisited (a computed nonterminal is expanded at once)
                self.nodes[cur].tvalue
            } else {
                let g = match self.compute(cur) {
                    Ok(g) => g,
                    Err(_) => {
                        self.stats.engine_errors += 1;
                        self.remove_dead(cur);
                        continue;
                    }
                };
                if g.is_halted() {
                    // a fail-closed game is no outcome: never steer into one
                    self.stats.halted_leaves += 1;
                    self.remove_dead(cur);
                    continue;
                }
                self.stats.truncated_leaves += u64::from(g.is_truncated());
                let me = self.me;
                let (priors, v) = match self.net.as_deref_mut() {
                    Some(net) if !g.is_terminal() => net.leaf(&g, me),
                    _ => (None, eval::evaluate(&g, me)),
                };
                let node = &mut self.nodes[cur];
                node.validated = true;
                node.terminal = g.is_terminal();
                node.actor = g.actor();
                if node.terminal {
                    node.tvalue = v; // a terminal never needs its game again
                } else {
                    let k = g.num_choices();
                    let only = mulligan_override(self.cfg.mulligan, &g, self.me);
                    if self.cached < self.cfg.max_cached {
                        self.nodes[cur].game = Some(Box::new(g));
                        self.cached += 1;
                    }
                    self.expand(cur, k, only, priors);
                }
                let p = self.nodes[cur].parent;
                if p != NONE {
                    self.maybe_release(p as usize);
                }
                v
            };
            self.backprop(cur, v);
            sims += 1;
        }
        self.stats.simulations += u64::from(sims);
        self.stats.max_tree_nodes = self.stats.max_tree_nodes.max(self.nodes.len() as u64);
        let root = &self.nodes[0];
        let k = root.game.as_ref().unwrap().num_choices();
        let mut res = RootResult { visits: vec![0; k], w: vec![0.0; k] };
        for &kid in &root.kids {
            let kid = &self.nodes[kid as usize];
            res.visits[kid.choice as usize] = kid.n;
            res.w[kid.choice as usize] = kid.w;
        }
        res
    }
}

pub struct MctsBot {
    cfg: MctsConfig,
    rng: SplitMix64,
    stats: SearchStats,
    net: Option<NetEval>,
    /// Searched decisions so far this game (pmcts `temp_moves`).
    searched: u32,
    last: Option<PolicyTarget>,
}

impl MctsBot {
    pub fn new(cfg: MctsConfig, seed: u64) -> Self {
        let net = cfg.net.clone().map(NetEval::new);
        MctsBot { cfg, rng: SplitMix64::new(seed), stats: SearchStats::default(), net, searched: 0, last: None }
    }

    pub fn search_stats(&self) -> &SearchStats {
        &self.stats
    }

    /// Root visits and value sums by masked index, summed over the worlds (matched by `stable_id`).
    pub fn search(&mut self, game: &Game) -> (Vec<u32>, Vec<f64>) {
        let k = game.num_choices();
        let ids: Vec<&str> = (0..k).map(|i| game.choice(i).stable_id.as_str()).collect();
        let mut visits = vec![0u32; k];
        let mut w = vec![0.0f64; k];
        let worlds = self.cfg.worlds.max(1);
        self.stats.searches += 1;
        for wi in 0..worlds {
            let budget = self.cfg.budget / worlds + u32::from(wi < self.cfg.budget % worlds);
            if budget == 0 {
                continue;
            }
            let world = match world_for(game, &mut self.rng) {
                Ok(world) => world,
                Err((reason, refused)) => {
                    // never the real game: its hidden cards would steer the search
                    self.stats.world_refusals += u64::from(refused);
                    self.stats.world_fallbacks += 1;
                    *self.stats.world_fallback_reasons.entry(format!("{reason:?}")).or_default() += 1;
                    continue;
                }
            };
            self.stats.worlds += 1;
            // A world is a copy of the root's session with only hidden card definitions (and the future RNG)
            // changed, so its legal list is the root's own stored list (the session does not regenerate it);
            // the admission check keeps every object an action names, so the ids match. Matched by stable id
            // all the same.
            let world_ids: Vec<String> = (0..world.num_choices()).map(|i| world.choice(i).stable_id.clone()).collect();
            let me = world.actor().expect("search root is a decision");
            let root_priors = match self.net.as_mut() {
                Some(net) => net.eval(&world, me).map(|(mut p, _)| {
                    if net.cfg.noise_eps > 0.0 {
                        add_dirichlet(&mut p, net.cfg.noise_eps, net.cfg.noise_alpha, &mut self.rng);
                    }
                    p
                }),
                None => None,
            };
            let mut tree = Tree::new(world, &self.cfg, &mut self.stats, self.net.as_mut(), root_priors);
            let res = tree.run(budget);
            for (j, id) in world_ids.iter().enumerate() {
                if let Some(i) = ids.iter().position(|x| x == id) {
                    visits[i] += res.visits[j];
                    w[i] += res.w[j];
                }
            }
        }
        (visits, w)
    }
}

/// Most visits; ties to the higher mean value, then the lower index.
pub fn final_choice(visits: &[u32], w: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..visits.len() {
        let (vi, vb) = (visits[i], visits[best]);
        if vi > vb {
            best = i;
        } else if vi == vb && vi > 0 {
            let (qi, qb) = (w[i] / f64::from(vi), w[best] / f64::from(vb));
            if qi > qb {
                best = i;
            }
        }
    }
    best
}

/// Sample an index with probability ∝ visits^(1/temp) (temp > 0; all-zero visits: index 0).
pub fn sample_visits(visits: &[u32], temp: f64, rng: &mut SplitMix64) -> usize {
    let m = visits.iter().copied().max().unwrap_or(0);
    if m == 0 {
        return 0;
    }
    let ws: Vec<f64> = visits.iter().map(|&v| (f64::from(v) / f64::from(m)).powf(1.0 / temp)).collect();
    let total: f64 = ws.iter().sum();
    let u = uniform(rng) * total;
    let mut acc = 0.0;
    for (i, w) in ws.iter().enumerate() {
        acc += w;
        if u < acc {
            return i;
        }
    }
    ws.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

impl Bot for MctsBot {
    fn choose(&mut self, game: &Game) -> usize {
        if let Some(k) = self.stats.mulligan.apply(self.cfg.mulligan, game) {
            self.last = Some(PolicyTarget::onehot(Source::Rule));
            return k;
        }
        let (visits, w) = self.search(game);
        let total: u32 = visits.iter().sum();
        if total == 0 {
            // nothing searched: the network's policy on the actor's own observation (pmcts), else uniform
            self.stats.unsearched += 1;
            self.searched += 1;
            let me = game.actor().expect("a decision");
            if let Some((p, _)) = self.net.as_mut().and_then(|n| n.eval(game, me)) {
                self.last = Some(PolicyTarget::onehot(Source::Net));
                let mut best = 0;
                for (i, &x) in p.iter().enumerate() {
                    if x > p[best] {
                        best = i;
                    }
                }
                return best;
            }
            self.last = Some(PolicyTarget::onehot(Source::Random));
            return self.rng.below(game.num_choices() as u64) as usize;
        }
        let root_q = if total > 0 { Some((w.iter().sum::<f64>() / f64::from(total)) as f32) } else { None };
        self.last = Some(PolicyTarget {
            source: Source::Search,
            probs: (total > 0).then(|| visits.iter().map(|&v| v as f32 / total as f32).collect()),
            root_q,
        });
        let sampled = self.cfg.net.as_ref().filter(|n| self.searched < n.temp_moves && n.temp > 0.0).map(|n| n.temp);
        self.searched += 1;
        match sampled {
            Some(t) => {
                self.stats.sampled_moves += 1;
                sample_visits(&visits, t, &mut self.rng)
            }
            None => final_choice(&visits, &w),
        }
    }

    fn take_target(&mut self) -> Option<PolicyTarget> {
        self.last.take()
    }

    fn stats(&self) -> Value {
        let s = &self.stats;
        let mut v = json!({
            "searches": s.searches,
            "simulations": s.simulations,
            "nodes": s.nodes,
            "replay_steps": s.replay_steps,
            "engine_errors": s.engine_errors,
            "halted_leaves": s.halted_leaves,
            "truncated_leaves": s.truncated_leaves,
            "worlds": s.worlds,
            "world_refusals": s.world_refusals,
            "world_fallbacks": s.world_fallbacks,
            "world_fallback_reasons": s.world_fallback_reasons,
            "unsearched": s.unsearched,
            "max_tree_nodes": s.max_tree_nodes,
            "max_children": s.max_children,
            "mulligan_decisions": s.mulligan.decisions,
            "mulligans_taken": s.mulligan.taken,
        });
        if let Some(n) = &self.net {
            v["net"] = json!(n.cfg.net.model.hash);
            v["net_evals"] = json!(n.evals);
            v["encode_failures"] = json!(n.encode_failures);
            v["sampled_moves"] = json!(s.sampled_moves);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::final_choice;

    #[test]
    fn final_choice_ties() {
        assert_eq!(final_choice(&[3, 5, 5], &[0.0, 1.0, 2.0]), 2); // tie on visits: higher mean
        assert_eq!(final_choice(&[5, 5], &[1.0, 1.0]), 0); // full tie: lower index
        assert_eq!(final_choice(&[0, 0], &[0.0, 0.0]), 0);
    }
}
