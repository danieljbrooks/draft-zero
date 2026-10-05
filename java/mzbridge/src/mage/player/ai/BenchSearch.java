package mage.player.ai;

import mage.MageObject;
import mage.abilities.Ability;
import mage.cards.Card;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.GameState;
import mage.game.permanent.Permanent;
import mage.player.ai.encoder.ActionEncoder;
import mage.player.ai.score.GameStateEvaluator3;
import mage.players.Player;
import mage.players.PlayerScript;
import mage.target.TargetImpl;

import java.util.*;

/**
 * The search benchmark's driver (docs/012 §2.1). It lives in MageZero's package so it can read
 * MCTSNode's package-private fields, and it uses MageZero's nodes only to step the engine
 * (validateState: copy the last priority state, replay the scripted path, run to the next
 * decision) and to list options (expand). Statistics live in its own tree, so that:
 *
 *  - the backprop discount can be counted per ply, per logical action or per turn (E2b);
 *  - every simulation is fresh, and the budget counts simulations run, not root visits;
 *  - nothing walks the whole tree per iteration (MageZero checks size() and maxDepth() every
 *    iteration, and searches for duplicate states even with pruning off: docs/012 §2.9);
 *  - network evaluations are synchronous, so there is no virtual loss (whose sign is wrong at
 *    opponent nodes in MageZero) and a seeded network search is deterministic.
 *
 * Everything else follows ComputerPlayerMCTS2: PUCT with c = 1, unvisited children valued 0,
 * uniform priors (priors off, as in experiment #2), the single-child shortcut, offline leaves
 * scored by GameStateEvaluator3 at priority decisions and inheriting the parent's score at micro
 * decisions, and the final choice by visits.
 *
 * Experiment #4 crosses the prior with the leaf evaluator: Config.leaf (net | heuristic | mix,
 * independent of Config.priors) and Config.opponentPriors (net | uniform at the opponent's nodes).
 * Values are always the searcher's: the network encodes every state from the searcher's seat, and
 * backprop never flips signs (selection does, at the opponent's nodes).
 *
 * Two methods:
 *  - searchTree: one MageZero-style tree on one world (clairvoyant MCTS on the real world; one
 *    of PIMC's K worlds).
 *  - searchIS: single-observer IS-MCTS (Cowling, Powley and Whitehouse 2012). One tree whose
 *    edges are world-independent keys. Every iteration picks one of the caller's belief worlds,
 *    re-deals it (the opponent's hand from that world's unseen cards, both libraries shuffled),
 *    and replays its path from the root in that world, following only options legal there.
 *    Selection uses availability counts. A node is scored once, in the world of the iteration
 *    that creates it.
 */
public final class BenchSearch {

    private BenchSearch() {
    }

    public static final class Config {
        public int budget = 1000;
        public double discount = 0.99;
        /** ply (MageZero today) | action (only edges out of priority decisions) | turn */
        public String unit = "ply";
        public double cPuct = 1.0;
        /** null: offline search (the heuristic) */
        public RemoteModelEvaluator nn;
        /** a graph network instead of nn (docs/022): MageZero's graph encoder at every node, priors from
         *  the scores of the options' graph nodes (GraphNet) */
        public GraphNet gnn;
        public long seed = 0;
        /** IS-MCTS: re-deal the hidden cards of the chosen world every iteration */
        public boolean redeal = true;
        /** use the network's policy heads as PUCT priors (MageZero's setPriors: softmax at priorTemp, plus priorBonus off Pass) */
        public boolean priors = false;
        public double priorTemp = 1.5;
        public double priorBonus = 0.1;
        /**
         * What scores a leaf (experiment #4 crosses it with the prior):
         *   net        the network's value head (needs nn)
         *   heuristic  offline search's: GameStateEvaluator3 at priority decisions, micro decisions
         *              inherit their parent's score. With nn and priors on, the network is still
         *              called for the policy, and its value is ignored.
         *   mix        leafMix x net + (1 - leafMix) x heuristic, the heuristic part as above (needs nn)
         * null: net with a network, heuristic without one.
         */
        public String leaf = null;
        public double leafMix = 0.5;
        /**
         * Priors at nodes where the opponent acts (priority, target and binary decisions alike):
         * net (the opponent's priority head, the target and binary heads: experiment #3) or uniform
         * (no network prior, as MageZero's noPolicyOpponent). Only matters with priors on.
         */
        public String opponentPriors = "net";
        /**
         * IS-MCTS only, with priors on. A shared node's actor and decision type can differ between
         * worlds. true: a node keeps one policy per (actor, decision type), read from the network
         * in the first world that meets that pair there. false (experiment #3): the policy read in
         * the world that created the node is mapped onto every world's options.
         */
        public boolean isPolicyPerWorld = false;
        public double timeoutSec = 900;
        /** 0: 4 x budget + 200 */
        public int maxIterations = 0;
        /**
         * Policy-only play (BenchPlayer, docs/018): every decision with a policy head plays the network's
         * own policy over the options, no search, no belief worlds: sampled from softmax(logit / policyTemp),
         * the most likely option at policyTemp 0. Decisions without a head are searched as configured.
         */
        public boolean policyOnly = false;
        public double policyTemp = 1.0;

        /** A copy with another seed (the game player searches each decision with its own). */
        public Config copyWithSeed(long newSeed) {
            Config c = new Config();
            c.budget = budget;
            c.discount = discount;
            c.unit = unit;
            c.cPuct = cPuct;
            c.nn = nn;
            c.gnn = gnn;
            c.seed = newSeed;
            c.redeal = redeal;
            c.timeoutSec = timeoutSec;
            c.maxIterations = maxIterations;
            c.policyOnly = policyOnly;
            c.policyTemp = policyTemp;
            c.priors = priors;
            c.priorTemp = priorTemp;
            c.priorBonus = priorBonus;
            c.leaf = leaf;
            c.leafMix = leafMix;
            c.opponentPriors = opponentPriors;
            c.isPolicyPerWorld = isPolicyPerWorld;
            return c;
        }

        public String leafMode() {
            return leaf != null ? leaf : hasNet() ? "net" : "heuristic";
        }

        /** a network of either kind (flat or graph) */
        public boolean hasNet() {
            return nn != null || gnn != null;
        }

        /** Throws on an option combination the search can't run. */
        public void check() {
            String l = leafMode();
            if (!List.of("net", "heuristic", "mix").contains(l)) throw new IllegalArgumentException("leaf must be net, heuristic or mix, got '" + l + "'");
            if (!hasNet() && !l.equals("heuristic")) throw new IllegalArgumentException("leaf " + l + " needs a network (evaluator.type remote or graph)");
            if (nn != null && gnn != null) throw new IllegalArgumentException("a flat network and a graph network: give one");
            if (!(leafMix >= 0.0 && leafMix <= 1.0)) throw new IllegalArgumentException("leafMix must be in [0, 1], got " + leafMix);
            if (!List.of("net", "uniform").contains(opponentPriors)) throw new IllegalArgumentException("opponentPriors must be net or uniform, got '" + opponentPriors + "'");
            if (priors && !hasNet()) throw new IllegalArgumentException("priors need a network (evaluator.type remote or graph)");
            if (policyOnly && !hasNet()) throw new IllegalArgumentException("policyOnly needs a network (evaluator.type remote or graph)");
            if (!(policyTemp >= 0.0)) throw new IllegalArgumentException("policyTemp must be >= 0, got " + policyTemp);
        }
    }

    /** One world at the decision: MageZero's root (validated and expanded) and how to rebuild it. */
    public static final class World {
        public final MCTSNode2 root;
        public final ComputerPlayerMCTS2 player;
        public final PlayerScript prefixA;
        public final PlayerScript prefixB;
        public final ActionEncoder.ActionType type;
        /** the live game paused at the decision (labels) */
        public final Game live;
        final GameState anchor;

        public World(MCTSNode2 root, ComputerPlayerMCTS2 player, PlayerScript prefixA, PlayerScript prefixB,
                     ActionEncoder.ActionType type, Game live) {
            this.root = root;
            this.player = player;
            this.prefixA = prefixA;
            this.prefixB = prefixB;
            this.type = type;
            this.live = live;
            this.anchor = root.state.copy();
        }
    }

    public static final class Stats {
        public long sims, iterations, scriptFailures, engineSteps, evals, nodes, redeals, redealFailures;
        /** network calls (leaf values, priors, the root's reported value) */
        public long netEvals;
        /** network priors applied to a node's options (tree: once per expanded node; IS-MCTS: per selection), and those at nodes where the opponent acts */
        public long netPriors, oppNetPriors;
        /** IS-MCTS with isPolicyPerWorld: network calls for a shared node's other (actor, decision type) */
        public long policyRefreshes;
        /** IS-MCTS without isPolicyPerWorld: priors applied from a policy read for another actor or decision type */
        public long policyMismatches;
        /** graph network: options with no node in the node's policy (another world's objects), and the
         *  network calls that read the policy again in the iteration's world */
        public long graphPolicyMisses;
        public int maxDepth;
        public long engineNanos, evalNanos, searchNanos;
        /** per edge traversed in backprop: all, out of a priority decision, and turns crossed */
        public long edgeVisits, priorityEdgeVisits, turnEdgeSum;
        public boolean timedOut;

        public void add(Stats o) {
            sims += o.sims;
            iterations += o.iterations;
            scriptFailures += o.scriptFailures;
            engineSteps += o.engineSteps;
            evals += o.evals;
            nodes += o.nodes;
            redeals += o.redeals;
            redealFailures += o.redealFailures;
            netEvals += o.netEvals;
            netPriors += o.netPriors;
            oppNetPriors += o.oppNetPriors;
            policyRefreshes += o.policyRefreshes;
            policyMismatches += o.policyMismatches;
            graphPolicyMisses += o.graphPolicyMisses;
            maxDepth = Math.max(maxDepth, o.maxDepth);
            engineNanos += o.engineNanos;
            evalNanos += o.evalNanos;
            searchNanos += o.searchNanos;
            edgeVisits += o.edgeVisits;
            priorityEdgeVisits += o.priorityEdgeVisits;
            turnEdgeSum += o.turnEdgeSum;
            timedOut |= o.timedOut;
        }
    }

    public static final class RootChild {
        public String label;
        public String key;
        public int visits;
        public Double q;
        public double prior;
        /** decisions after this option before the next priority decision, along the most visited path; -1 unknown */
        public int subDecisions = -1;
        /** IS-MCTS: iterations in which the option was legal */
        public int avail;
        /** CHOOSE_NUM: the option's index (the caller adds the minimum) */
        public Integer amount;
    }

    public static final class Result {
        public final List<RootChild> children = new ArrayList<>();
        public int rootVisits;
        /** the search's backed-up root value, root.w / root.n, the searcher's perspective (null: no simulation, or no search) */
        public Double rootQ;
        /** root.w and root.n, so several worlds' roots can be pooled (visit-weighted) */
        public double rootW;
        public int rootN;
        /**
         * The root's static evaluation by the leaf evaluator, before any search (MageZero scores the
         * root first), the searcher's perspective. Policy only: the network's value. Offline, a
         * micro-decision root (attack, block) has no parent to inherit from, so it is 0.
         */
        public Double rootValue;
        /** the network's value at the root, whatever the leaf evaluator (null without a network) */
        public Double rootNet;
        public final Stats stats = new Stats();
    }

    static final class Node {
        final Node parent;
        final String key;
        final int depth;
        String label;
        Integer amount;
        MCTSNode2 eng;            // tree method: the engine node (MageZero caches its state)
        List<Node> kids;          // tree method: null until expanded
        Map<String, Node> byKey;  // IS-MCTS
        int n;
        int avail;
        double w;
        double value;             // the leaf evaluator's score
        double heur;              // its heuristic part (leaf heuristic or mix): what micro decisions below inherit
        double net;               // the network's value, when the network was called here
        double prior = 1.0;
        float[] policy;           // the network's policy head for this node's decision (priors on)
        String policyKey;         // the (actor, decision type) the policy was read for
        Map<String, float[]> policyByKey;  // IS-MCTS with isPolicyPerWorld: one policy per (actor, type)
        GraphNet.Policy gpolicy;           // the same two for a graph network
        Map<String, GraphNet.Policy> gpolicyByKey;
        boolean hasValue, hasHeur, hasNet, validated, terminal, win;
        ActionEncoder.ActionType type;
        UUID actor;
        int turn;

        Node(Node parent, String key) {
            this.parent = parent;
            this.key = key;
            this.depth = parent == null ? 0 : parent.depth + 1;
        }

        double q() {
            return n > 0 ? w / n : 0.0;
        }
    }

    // ============================================================================ tree method

    public static Result searchTree(World world, Config cfg) {
        cfg.check();
        Result res = new Result();
        Stats st = res.stats;
        long t0 = System.nanoTime();
        long deadline = t0 + (long) (cfg.timeoutSec * 1e9);
        UUID me = world.player.getId();
        Node root = new Node(null, null);
        root.eng = world.root;
        root.validated = true;
        describe(root, world.root);
        evaluate(root, world.root, cfg, st); // MageZero scores the root before searching; not a simulation
        rootNet(root, world.root, cfg, st);
        expandTree(root, world, cfg, st);
        int maxIt = cfg.maxIterations > 0 ? cfg.maxIterations : 4 * cfg.budget + 200;
        while (st.sims < cfg.budget && st.iterations < maxIt && !root.kids.isEmpty()) {
            if (System.nanoTime() > deadline) {
                st.timedOut = true;
                break;
            }
            st.iterations++;
            Node cur = root;
            while (cur.kids != null && !cur.kids.isEmpty() && !cur.terminal) cur = selectTree(cur, me, cfg);
            if (cur.kids != null && cur.kids.isEmpty() && !cur.terminal) {
                removeDead(cur); // every option below it failed
                continue;
            }
            if (!cur.validated) {
                long te = System.nanoTime();
                cur.eng.validateState();
                st.engineSteps++;
                st.engineNanos += System.nanoTime() - te;
                cur.validated = true;
                describe(cur, cur.eng);
                if (!cur.terminal && cur.eng.getPlayer().scriptFailed) {
                    st.scriptFailures++;
                    removeDead(cur);
                    continue;
                }
            }
            double v;
            if (cur.terminal) {
                v = cur.win ? 1.0 : -1.0;
            } else {
                v = evaluate(cur, cur.eng, cfg, st);
                expandTree(cur, world, cfg, st);
            }
            backprop(cur, v, cfg, st);
            st.sims++;
        }
        st.searchNanos = System.nanoTime() - t0;
        for (Node k : root.kids) {
            RootChild c = new RootChild();
            c.label = k.label;
            c.key = k.key;
            c.visits = k.n;
            c.q = k.n > 0 ? k.q() : null;
            c.prior = k.prior;
            c.amount = k.amount;
            c.subDecisions = subDecisions(k);
            res.children.add(c);
            res.rootVisits += k.n;
        }
        res.rootQ = root.n > 0 ? root.q() : null;
        res.rootW = root.w;
        res.rootN = root.n;
        res.rootValue = root.value;
        res.rootNet = root.hasNet ? root.net : null;
        return res;
    }

    private static void expandTree(Node node, World world, Config cfg, Stats st) {
        long te = System.nanoTime();
        node.eng.expand();
        st.engineNanos += System.nanoTime() - te;
        List<MCTSNode> ch = node.eng.getChildren();
        node.kids = new ArrayList<>(ch.size());
        for (MCTSNode c : ch) {
            Node k = new Node(node, null);
            k.eng = (MCTSNode2) c;
            k.prior = 1.0 / ch.size();
            if (node.parent == null) {
                k.label = label(c, node.type, world.live, world.player.getId());
                if (node.type == ActionEncoder.ActionType.CHOOSE_NUM) k.amount = c.getAmountAction();
            }
            node.kids.add(k);
        }
        st.nodes += ch.size();
        st.maxDepth = Math.max(st.maxDepth, node.depth + 1);
        if (node.policy != null || node.gpolicy != null) { // set by evaluate() only where priors apply (opponentPriors)
            double[] pr = node.gpolicy != null ? graphPriors(node.gpolicy, singletons(ch), ch, cfg)
                    : priors(node.policy, ch, world.live, cfg);
            if (pr != null) {
                for (int i = 0; i < ch.size(); i++) node.kids.get(i).prior = pr[i];
                st.netPriors++;
                if (!world.player.getId().equals(node.actor)) st.oppNetPriors++;
            }
        }
    }

    /** A graph network's priors: its option logits (copies of an option pooled, groups[i] holding opts[i]'s
     *  copies) through MageZero's setPriors; null when an option has no node in the policy. */
    static double[] graphPriors(GraphNet.Policy pol, List<List<MCTSNode>> groups, List<MCTSNode> opts, Config cfg) {
        double[] lg = pol.logits(groups);
        return lg == null ? null : priorsFromLogits(lg, opts, cfg);
    }

    static List<List<MCTSNode>> singletons(List<MCTSNode> opts) {
        List<List<MCTSNode>> out = new ArrayList<>(opts.size());
        for (MCTSNode c : opts) out.add(List.of(c));
        return out;
    }

    /** MageZero's setPriors from option logits: softmax(logit / T) plus a bonus for anything but Pass and mana abilities. */
    static double[] priorsFromLogits(double[] logits, List<MCTSNode> opts, Config cfg) {
        int n = opts.size();
        double[] out = new double[n];
        double mx = Double.NEGATIVE_INFINITY;
        for (double x : logits) mx = Math.max(mx, x);
        double sum = 0;
        for (int i = 0; i < n; i++) {
            out[i] = Math.exp((logits[i] - mx) / cfg.priorTemp);
            sum += out[i];
        }
        for (int i = 0; i < n; i++) {
            out[i] /= sum;
            Ability pa = opts.get(i).getPriorityAction();
            if (pa == null || (!pa.isManaAbility() && !(pa instanceof mage.abilities.common.PassAbility))) out[i] += cfg.priorBonus;
        }
        return out;
    }

    /** MageZero's setPriors on a list of options: softmax(logit / T) plus a bonus for anything but Pass. */
    static double[] priors(float[] policy, List<MCTSNode> opts, Game g, Config cfg) {
        int n = opts.size();
        double[] out = new double[n];
        double mx = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            int a;
            try {
                a = opts.get(i).getActionIndex(g);
            } catch (RuntimeException e) {
                return null;
            }
            if (a < 0 || a >= policy.length) return null;
            out[i] = policy[a];
            mx = Math.max(mx, out[i]);
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            out[i] = Math.exp((out[i] - mx) / cfg.priorTemp);
            sum += out[i];
        }
        for (int i = 0; i < n; i++) {
            out[i] /= sum;
            Ability pa = opts.get(i).getPriorityAction();
            if (pa == null || (!pa.isManaAbility() && !(pa instanceof mage.abilities.common.PassAbility))) out[i] += cfg.priorBonus;
        }
        return out;
    }

    /** rootPolicy for a graph network: a softmax over the options' logits (no prior temperature or bonus). */
    private static Result rootPolicyGraph(World world, Config cfg, Result res, long t0) {
        MCTSNode2 r = world.root;
        UUID me = world.player.getId();
        GraphNet.Ask q = GraphNet.ask(r);
        GraphNet.Out out = cfg.gnn.infer(r, q);
        res.stats.netEvals = 1;
        GraphNet.Policy pol = policyAllowed(r.actionType, me.equals(r.playerId), cfg) ? out.policy(q) : null;
        List<MCTSNode> ch = r.getChildren();
        double[] lg = pol == null ? null : pol.logits(singletons(ch));
        double[] logit = new double[ch.size()];
        double mx = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < ch.size(); k++) {
            logit[k] = lg == null ? 0.0 : lg[k];
            mx = Math.max(mx, logit[k]);
        }
        double sum = 0;
        for (int k = 0; k < ch.size(); k++) {
            logit[k] = Math.exp(logit[k] - mx);
            sum += logit[k];
        }
        for (int k = 0; k < ch.size(); k++) {
            RootChild c = new RootChild();
            c.label = label(ch.get(k), r.actionType, world.live, me);
            c.prior = logit[k] / sum;
            if (r.actionType == ActionEncoder.ActionType.CHOOSE_NUM) c.amount = ch.get(k).getAmountAction();
            res.children.add(c);
        }
        res.rootValue = (double) out.value;
        res.rootNet = (double) out.value;
        res.stats.evals = 1;
        res.stats.searchNanos = System.nanoTime() - t0;
        return res;
    }

    private static Node selectTree(Node node, UUID me, Config cfg) {
        if (node.kids.size() == 1) return node.kids.get(0);
        double sign = me.equals(node.actor) ? 1.0 : -1.0;
        double sqrtN = Math.sqrt(node.n);
        Node best = null;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (Node k : node.kids) {
            double q = k.n > 0 ? k.q() : 0.0;
            double val = sign * q + cfg.cPuct * k.prior * sqrtN / (1 + k.n);
            if (val > bestVal) {
                bestVal = val;
                best = k;
            }
        }
        return best;
    }

    private static void removeDead(Node node) {
        Node p = node.parent;
        if (p == null || p.kids == null) return;
        p.kids.remove(node);
        if (p.kids.isEmpty() && p.parent != null) removeDead(p);
    }

    // ============================================================================ policy only

    /** The network's policy at the root, no search (E0's reference): a softmax over the legal options. */
    public static Result rootPolicy(World world, Config cfg) {
        Result res = new Result();
        long t0 = System.nanoTime();
        MCTSNode2 r = world.root;
        UUID me = world.player.getId();
        r.expand();
        if (cfg.gnn != null) return rootPolicyGraph(world, cfg, res, t0);
        Set<Integer> sv = r.stateVector;
        long[] idx = new long[sv == null ? 0 : sv.size()];
        int i = 0;
        if (sv != null) for (int f : sv) idx[i++] = f;
        RemoteModelEvaluator.InferenceResult out = cfg.nn.infer(idx);
        res.stats.netEvals = 1;
        // the root is the searcher's decision, so opponentPriors does not matter here; applied for consistency
        float[] pol = policyAllowed(r.actionType, me.equals(r.playerId), cfg) ? head(out, r.actionType, me.equals(r.playerId)) : null;
        List<MCTSNode> ch = r.getChildren();
        double[] logit = new double[ch.size()];
        double mx = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < ch.size(); k++) {
            int a = -1;
            try {
                a = ch.get(k).getActionIndex(world.live);
            } catch (RuntimeException ignored) {
            }
            logit[k] = pol == null || a < 0 || a >= pol.length ? 0.0 : pol[a];
            mx = Math.max(mx, logit[k]);
        }
        double sum = 0;
        for (int k = 0; k < ch.size(); k++) {
            logit[k] = Math.exp(logit[k] - mx);
            sum += logit[k];
        }
        for (int k = 0; k < ch.size(); k++) {
            RootChild c = new RootChild();
            c.label = label(ch.get(k), r.actionType, world.live, me);
            c.prior = logit[k] / sum;
            if (r.actionType == ActionEncoder.ActionType.CHOOSE_NUM) c.amount = ch.get(k).getAmountAction();
            res.children.add(c);
        }
        res.rootValue = (double) out.value;
        res.rootNet = (double) out.value;
        res.stats.evals = 1;
        res.stats.searchNanos = System.nanoTime() - t0;
        return res;
    }

    // ============================================================================ IS-MCTS

    public static Result searchIS(List<World> worlds, Config cfg) {
        cfg.check();
        Result res = new Result();
        Stats st = res.stats;
        long t0 = System.nanoTime();
        long deadline = t0 + (long) (cfg.timeoutSec * 1e9);
        Random rng = new Random(cfg.seed ^ 0x5DEECE66DL);
        Node root = new Node(null, null);
        root.byKey = new LinkedHashMap<>();
        int rootOptions = 0;
        int maxIt = cfg.maxIterations > 0 ? cfg.maxIterations : 4 * cfg.budget + 200;
        while (st.sims < cfg.budget && st.iterations < maxIt) {
            if (System.nanoTime() > deadline) {
                st.timedOut = true;
                break;
            }
            st.iterations++;
            World w = worlds.get(rng.nextInt(worlds.size()));
            UUID me = w.player.getId();
            MCTSNode2 sh = shadowRoot(w, cfg, rng, st);
            if (sh == null) continue;
            Node cur = root;
            describe(cur, sh);
            if (!root.hasValue) { // as searchTree: not a simulation
                evaluate(root, sh, cfg, st);
                rootNet(root, sh, cfg, st);
            }
            double v;
            while (true) {
                if (cur.terminal) {
                    v = cur.win ? 1.0 : -1.0;
                    break;
                }
                if (cur != root && cur.n == 0) {
                    v = evaluate(cur, sh, cfg, st);
                    break;
                }
                long te = System.nanoTime();
                sh.expand();
                st.engineNanos += System.nanoTime() - te;
                LinkedHashMap<String, MCTSNode> opts = new LinkedHashMap<>();
                for (MCTSNode c : sh.getChildren()) opts.putIfAbsent(key(c, sh.actionType, sh.getGame(), me), c);
                if (opts.isEmpty()) {
                    v = cur.hasValue ? cur.value : 0.0;
                    break;
                }
                if (cur.byKey == null) cur.byKey = new LinkedHashMap<>();
                for (Map.Entry<String, MCTSNode> e : opts.entrySet()) {
                    Node ch = cur.byKey.get(e.getKey());
                    if (ch == null) {
                        ch = new Node(cur, e.getKey());
                        if (cur == root) {
                            ch.label = label(e.getValue(), sh.actionType, w.live, me);
                            if (sh.actionType == ActionEncoder.ActionType.CHOOSE_NUM) ch.amount = e.getValue().getAmountAction();
                        }
                        cur.byKey.put(e.getKey(), ch);
                        st.nodes++;
                        st.maxDepth = Math.max(st.maxDepth, ch.depth);
                    }
                    ch.avail++;
                }
                if (cur == root) rootOptions = Math.max(rootOptions, opts.size());
                Node next = null;
                MCTSNode2 nsh = null;
                // a shared node's actor can differ between worlds (describe() sets it for this
                // iteration's), and its policy was read once, in the world that created it: the
                // opponent-prior rule is applied to this iteration's actor
                boolean mine = me.equals(cur.actor);
                // isPolicyPerWorld: the policy for this world's actor and decision type at this node
                float[] curPolicy = cfg.gnn != null ? null : cfg.isPolicyPerWorld ? policyFor(cur, sh, mine, cfg, st)
                        : (mine || cfg.opponentPriors.equals("net")) ? cur.policy : null;
                GraphNet.Policy curG = cfg.gnn == null ? null : cfg.isPolicyPerWorld ? graphPolicyFor(cur, sh, mine, cfg, st, false)
                        : (mine || cfg.opponentPriors.equals("net")) ? cur.gpolicy : null;
                // a graph policy's option groups: every copy of an option (MageZero lists each card's)
                Map<String, List<MCTSNode>> copies = null;
                if (curG != null) {
                    copies = new HashMap<>();
                    for (MCTSNode c : sh.getChildren()) copies.computeIfAbsent(key(c, sh.actionType, sh.getGame(), me), x -> new ArrayList<>()).add(c);
                }
                while (!opts.isEmpty()) {
                    Map<String, Double> pri = null;
                    if (curG != null) {
                        List<MCTSNode> ol = new ArrayList<>(opts.values());
                        List<List<MCTSNode>> groups = new ArrayList<>();
                        for (String kk : opts.keySet()) groups.add(copies.get(kk));
                        double[] pr = graphPriors(curG, groups, ol, cfg);
                        if (pr == null && cfg.isPolicyPerWorld) { // this world's objects: read it here once
                            st.graphPolicyMisses++;
                            curG = graphPolicyFor(cur, sh, mine, cfg, st, true);
                            pr = curG == null ? null : graphPriors(curG, groups, ol, cfg);
                        }
                        if (pr != null) {
                            pri = new HashMap<>();
                            int i = 0;
                            for (String kk : opts.keySet()) pri.put(kk, pr[i++]);
                            st.netPriors++;
                            if (!mine) st.oppNetPriors++;
                        }
                    } else if (curPolicy != null) {
                        List<MCTSNode> ol = new ArrayList<>(opts.values());
                        double[] pr = priors(curPolicy, ol, sh.getGame(), cfg);
                        if (pr != null) {
                            pri = new HashMap<>();
                            int i = 0;
                            for (String kk : opts.keySet()) pri.put(kk, pr[i++]);
                            st.netPriors++;
                            if (!mine) st.oppNetPriors++;
                            if (!cfg.isPolicyPerWorld && !policyKey(cur.type, mine).equals(cur.policyKey)) st.policyMismatches++;
                        }
                    }
                    String k = selectIS(cur, opts.keySet(), me, cfg, pri);
                    Node ch = cur.byKey.get(k);
                    MCTSNode2 cand = (MCTSNode2) opts.get(k);
                    long tv = System.nanoTime();
                    cand.validateState();
                    st.engineSteps++;
                    st.engineNanos += System.nanoTime() - tv;
                    if (!cand.isTerminal() && cand.getPlayer().scriptFailed) {
                        st.scriptFailures++;
                        opts.remove(k);
                        ch.avail--;
                        continue;
                    }
                    next = ch;
                    nsh = cand;
                    break;
                }
                if (next == null) {
                    v = cur.hasValue ? cur.value : 0.0;
                    break;
                }
                cur = next;
                sh = nsh;
                describe(cur, sh);
            }
            backprop(cur, v, cfg, st);
            st.sims++;
        }
        st.searchNanos = System.nanoTime() - t0;
        if (root.byKey != null) {
            for (Node k : root.byKey.values()) {
                RootChild c = new RootChild();
                c.label = k.label;
                c.key = k.key;
                c.visits = k.n;
                c.q = k.n > 0 ? k.q() : null;
                c.prior = rootOptions > 0 ? 1.0 / rootOptions : 0.0;
                c.avail = k.avail;
                c.amount = k.amount;
                c.subDecisions = subDecisions(k);
                res.children.add(c);
                res.rootVisits += k.n;
            }
        }
        res.rootQ = root.n > 0 ? root.q() : null;
        res.rootW = root.w;
        res.rootN = root.n;
        res.rootValue = root.hasValue ? root.value : null;
        res.rootNet = root.hasNet ? root.net : null;
        return res;
    }

    /** A fresh determinization of world w at the decision: re-deal, then replay the prefix. */
    private static MCTSNode2 shadowRoot(World w, Config cfg, Random rng, Stats st) {
        for (int attempt = 0; attempt < 2; attempt++) {
            boolean redeal = cfg.redeal && attempt == 0;
            long te = System.nanoTime();
            Game g = w.root.getGame();
            w.root.resetRootGame(w.anchor.copy());
            if (redeal) {
                redeal(g, w.player.getId(), rng);
                st.redeals++;
            }
            MCTSNode2 sh = new MCTSNode2(w.player, g, w.type, new PlayerScript(w.prefixA), new PlayerScript(w.prefixB));
            sh.validateState();
            st.engineSteps++;
            st.engineNanos += System.nanoTime() - te;
            if (sh.isTerminal() || !sh.getPlayer().scriptFailed) return sh;
            if (redeal) st.redealFailures++;
        }
        return null;
    }

    /**
     * Re-deal the hidden cards: the opponent's hand is redrawn from its hand and library (the
     * world's unseen cards, which came from the belief, never from the real game), and both
     * libraries are shuffled. Uses the search's own random stream.
     */
    static void redeal(Game g, UUID me, Random rng) {
        for (Player p : g.getState().getPlayers().values()) {
            if (p.getId().equals(me)) {
                p.getLibrary().shuffle(rng);
                continue;
            }
            int h = p.getHand().size();
            List<Card> hand = new ArrayList<>(p.getHand().getCards(g));
            hand.sort(Comparator.comparing(c -> c.getId().toString())); // iteration order of a set is not seeded
            p.getHand().clear();
            for (Card c : hand) p.getLibrary().putOnBottom(c, g);
            p.getLibrary().shuffle(rng);
            for (int i = 0; i < h; i++) {
                Card c = p.getLibrary().drawFromTop(g);
                if (c == null) break;
                c.setZone(Zone.HAND, g);
                p.getHand().add(c);
            }
        }
    }

    private static String selectIS(Node node, Set<String> keys, UUID me, Config cfg, Map<String, Double> pri) {
        Iterator<String> it = keys.iterator();
        if (keys.size() == 1) return it.next();
        double sign = me.equals(node.actor) ? 1.0 : -1.0;
        double prior = 1.0 / keys.size();
        String best = null;
        double bestVal = Double.NEGATIVE_INFINITY;
        while (it.hasNext()) {
            String k = it.next();
            Node ch = node.byKey.get(k);
            double q = ch.n > 0 ? ch.q() : 0.0;
            double p = pri == null ? prior : pri.get(k);
            double val = sign * q + cfg.cPuct * p * Math.sqrt(ch.avail) / (1 + ch.n);
            if (val > bestVal) {
                bestVal = val;
                best = k;
            }
        }
        return best;
    }

    /** A world-independent name for an option: the same choice in any world gets the same key. */
    static String key(MCTSNode c, ActionEncoder.ActionType type, Game g, UUID me) {
        switch (type) {
            case PRIORITY: {
                Ability a = c.getPriorityAction();
                if (a == null) return "P:?";
                MageObject src = a.getSourceId() == null ? null : g.getObject(a.getSourceId());
                return "P:" + a + "@" + (src == null ? "" : src.getName());
            }
            case CHOOSE_TARGET:
                return "T:" + targetKey(g, c.getTargetAction(), me);
            case CHOOSE_USE:
                return "U:" + c.getUseAction();
            case CHOOSE_NUM:
                return "N:" + c.getAmountAction();
            default:
                return "C:" + c.getChoiceAction();
        }
    }

    static String targetKey(Game g, UUID id, UUID me) {
        if (id == null) return "null";
        if (TargetImpl.STOP_CHOOSING.equals(id)) return "Stop Choosing";
        Player pl = g.getPlayer(id);
        if (pl != null) return id.equals(me) ? "me" : "opponent";
        String who = "";
        Permanent perm = g.getPermanent(id);
        if (perm != null) {
            who = perm.isControlledBy(me) ? "|mine" : "|theirs";
        } else {
            Card card = g.getCard(id);
            if (card != null) who = card.isOwnedBy(me) ? "|mine" : "|theirs";
        }
        Zone z = g.getState().getZone(id);
        return g.getEntityValue(id, me) + "|" + z + who;
    }

    /** The label BridgePlayer gives the same option (what 17lands labels are matched against). */
    static String label(MCTSNode c, ActionEncoder.ActionType type, Game live, UUID me) {
        switch (type) {
            case PRIORITY:
                return c.getPriorityAction() == null ? "?" : c.getPriorityAction().toString();
            case CHOOSE_TARGET:
                return c.getTargetAction() == null ? "?" : live.getEntityName(c.getTargetAction(), me);
            case CHOOSE_USE:
                return c.getUseAction() ? "yes" : "no";
            case CHOOSE_NUM:
                return String.valueOf(c.getAmountAction());
            default:
                return String.valueOf(c.getChoiceAction());
        }
    }

    // ============================================================================ shared

    private static void describe(Node node, MCTSNode eng) {
        node.validated = true;
        node.terminal = eng.isTerminal();
        node.win = eng.isWinner();
        node.type = eng.actionType;
        node.actor = eng.playerId;
        node.turn = eng.getGame().getTurnNum();
    }

    /**
     * Score a leaf with cfg's leaf evaluator (Config.leaf), from the searcher's perspective, and
     * with priors on read the network's policy for the node's options:
     *  - the heuristic part (leaf heuristic or mix) is offline MageZero's: GameStateEvaluator3 at
     *    priority decisions; micro decisions inherit their parent's heuristic score (0 at a
     *    micro-decision root);
     *  - the network is called when the leaf needs its value (net, mix) or the node needs its
     *    policy (priors on, a decision with a policy head, the searcher's unless opponentPriors is
     *    net). With leaf heuristic its value is ignored.
     */
    private static double evaluate(Node node, MCTSNode eng, Config cfg, Stats st) {
        long te = System.nanoTime();
        String leaf = cfg.leafMode();
        double h = 0.0;
        if (!leaf.equals("net")) {
            if (eng.actionType == ActionEncoder.ActionType.PRIORITY) {
                h = GameStateEvaluator3.evaluateNormalized(eng.targetPlayer, eng.getGame());
            } else {
                h = node.parent != null && node.parent.hasHeur ? node.parent.heur : 0.0;
            }
            node.heur = h;
            node.hasHeur = true;
        }
        boolean mine = eng.targetPlayer.equals(eng.playerId);
        boolean wantPolicy = cfg.priors && policyAllowed(eng.actionType, mine, cfg);
        if (cfg.gnn != null && (!leaf.equals("heuristic") || wantPolicy)) {
            GraphNet.Ask q = GraphNet.ask(eng);
            GraphNet.Out out = inferGraph(eng, q, cfg, st);
            node.net = out.value;
            node.hasNet = true;
            if (wantPolicy) {
                node.gpolicy = out.policy(q);
                node.policyKey = policyKey(eng.actionType, mine);
            }
        } else if (!leaf.equals("heuristic") || wantPolicy) {
            RemoteModelEvaluator.InferenceResult out = infer(eng, cfg, st);
            node.net = out.value;
            node.hasNet = true;
            if (wantPolicy) {
                node.policy = head(out, eng.actionType, mine);
                node.policyKey = policyKey(eng.actionType, mine);
            }
        }
        double v;
        switch (leaf) {
            case "net":
                v = node.net;
                break;
            case "mix":
                v = cfg.leafMix * node.net + (1.0 - cfg.leafMix) * h;
                break;
            default:
                v = h;
        }
        st.evals++;
        st.evalNanos += System.nanoTime() - te;
        node.value = v;
        node.hasValue = true;
        return v;
    }

    /** The network's value at the root, for the output only (the search never reads it): one extra call when evaluate() made none. */
    private static void rootNet(Node root, MCTSNode eng, Config cfg, Stats st) {
        if (!cfg.hasNet() || root.hasNet || root.terminal) return;
        long te = System.nanoTime();
        root.net = cfg.gnn != null ? inferGraph(eng, GraphNet.ask(eng), cfg, st).value : infer(eng, cfg, st).value;
        root.hasNet = true;
        st.evalNanos += System.nanoTime() - te;
    }

    private static RemoteModelEvaluator.InferenceResult infer(MCTSNode eng, Config cfg, Stats st) {
        Set<Integer> sv = eng.stateVector;
        long[] idx = new long[sv == null ? 0 : sv.size()];
        int i = 0;
        if (sv != null) for (int f : sv) idx[i++] = f;
        st.netEvals++;
        return cfg.nn.infer(idx);
    }

    static GraphNet.Out inferGraph(MCTSNode eng, GraphNet.Ask q, Config cfg, Stats st) {
        st.netEvals++;
        return cfg.gnn.infer(eng, q);
    }

    /**
     * policyFor for a graph network: the policy for this iteration's actor and decision type at a shared
     * node, read in this world (sh) the first time the pair is met there, or again when `refresh` (an
     * option of this world had no node in it); a refresh adds this world's nodes to the policy.
     */
    private static GraphNet.Policy graphPolicyFor(Node cur, MCTSNode2 sh, boolean mine, Config cfg, Stats st, boolean refresh) {
        if (!cfg.priors || !policyAllowed(cur.type, mine, cfg)) return null;
        String k = policyKey(cur.type, mine);
        if (cur.gpolicyByKey == null) cur.gpolicyByKey = new HashMap<>();
        GraphNet.Policy pol = cur.gpolicyByKey.get(k);
        if (pol == null && !refresh && k.equals(cur.policyKey) && cur.gpolicy != null) {
            pol = cur.gpolicy;
            cur.gpolicyByKey.put(k, pol);
        }
        if (pol == null || refresh) {
            long te = System.nanoTime();
            GraphNet.Ask q = GraphNet.ask(sh);
            GraphNet.Policy fresh = inferGraph(sh, q, cfg, st).policy(q);
            st.evalNanos += System.nanoTime() - te;
            st.policyRefreshes++;
            if (pol != null && fresh != null && pol.byId != null && fresh.byId != null) {
                Map<UUID, Float> merged = new HashMap<>(pol.byId);
                merged.putAll(fresh.byId);
                fresh = new GraphNet.Policy(fresh.q, merged, fresh.use);
            }
            pol = fresh;
            cur.gpolicyByKey.put(k, pol);
        }
        return pol;
    }

    static String policyKey(ActionEncoder.ActionType type, boolean mine) {
        return (mine ? "me:" : "opp:") + type;
    }

    /**
     * IS-MCTS with isPolicyPerWorld: the policy for this iteration's actor and decision type at a
     * shared node, read from the network in this world (sh) the first time the pair is met there.
     */
    private static float[] policyFor(Node cur, MCTSNode2 sh, boolean mine, Config cfg, Stats st) {
        if (!cfg.priors || !policyAllowed(cur.type, mine, cfg)) return null;
        String k = policyKey(cur.type, mine);
        if (k.equals(cur.policyKey)) return cur.policy;
        if (cur.policyByKey == null) cur.policyByKey = new HashMap<>();
        float[] pol = cur.policyByKey.get(k);
        if (pol == null && !cur.policyByKey.containsKey(k)) {
            long te = System.nanoTime();
            pol = head(infer(sh, cfg, st), cur.type, mine);
            st.evalNanos += System.nanoTime() - te;
            st.policyRefreshes++;
            cur.policyByKey.put(k, pol);
        }
        return pol;
    }

    /** Whether a decision gets the network's prior: it has a policy head, and it is the searcher's or opponentPriors is net. */
    static boolean policyAllowed(ActionEncoder.ActionType type, boolean mine, Config cfg) {
        switch (type) {
            case PRIORITY:
            case CHOOSE_TARGET:
            case CHOOSE_USE:
                return mine || cfg.opponentPriors.equals("net");
            default:
                return false;
        }
    }

    /** The policy head for a decision (MageZero's): priority by who acts; target and binary are shared. */
    static float[] head(RemoteModelEvaluator.InferenceResult out, ActionEncoder.ActionType type, boolean mine) {
        switch (type) {
            case PRIORITY:
                return mine ? out.policy_player : out.policy_opponent;
            case CHOOSE_TARGET:
                return out.policy_target;
            case CHOOSE_USE:
                return out.policy_binary;
            default:
                return null;
        }
    }

    private static void backprop(Node leaf, double v, Config cfg, Stats st) {
        Node cur = leaf;
        double val = v;
        while (cur != null) {
            cur.n++;
            cur.w += val;
            Node p = cur.parent;
            if (p == null) break;
            int turns = Math.max(0, cur.turn - p.turn);
            st.edgeVisits++;
            if (p.type == ActionEncoder.ActionType.PRIORITY) st.priorityEdgeVisits++;
            st.turnEdgeSum += turns;
            switch (cfg.unit) {
                case "action":
                    if (p.type == ActionEncoder.ActionType.PRIORITY) val *= cfg.discount;
                    break;
                case "turn":
                    if (turns > 0) val *= Math.pow(cfg.discount, turns);
                    break;
                default:
                    val *= cfg.discount;
            }
            cur = p;
        }
    }

    /** Micro decisions after this option before the next priority decision, along the most visited path. */
    private static int subDecisions(Node node) {
        if (!node.validated) return -1;
        int c = 0;
        Node cur = node;
        while (cur != null && cur.validated && !cur.terminal && cur.type != ActionEncoder.ActionType.PRIORITY) {
            c++;
            Collection<Node> ks = cur.kids != null ? cur.kids : cur.byKey != null ? cur.byKey.values() : Collections.emptyList();
            Node best = null;
            for (Node k : ks) if (best == null || k.n > best.n) best = k;
            if (best == null || best.n == 0) break;
            cur = best;
        }
        return c;
    }
}
