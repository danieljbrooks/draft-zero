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
        public long seed = 0;
        /** IS-MCTS: re-deal the hidden cards of the chosen world every iteration */
        public boolean redeal = true;
        /** use the network's policy heads as PUCT priors (MageZero's setPriors: softmax at priorTemp, plus priorBonus off Pass) */
        public boolean priors = false;
        public double priorTemp = 1.5;
        public double priorBonus = 0.1;
        public double timeoutSec = 900;
        /** 0: 4 x budget + 200 */
        public int maxIterations = 0;
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
        public Double rootQ;
        public Double rootValue;
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
        double value;
        double prior = 1.0;
        float[] policy;           // the network's policy head for this node's decision (priors on)
        boolean hasValue, validated, terminal, win;
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
        res.rootValue = root.value;
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
        if (node.policy != null) {
            double[] pr = priors(node.policy, ch, world.live, cfg);
            if (pr != null) for (int i = 0; i < ch.size(); i++) node.kids.get(i).prior = pr[i];
        }
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
        Set<Integer> sv = r.stateVector;
        long[] idx = new long[sv == null ? 0 : sv.size()];
        int i = 0;
        if (sv != null) for (int f : sv) idx[i++] = f;
        RemoteModelEvaluator.InferenceResult out = cfg.nn.infer(idx);
        float[] pol;
        switch (r.actionType) {
            case PRIORITY:
                pol = me.equals(r.playerId) ? out.policy_player : out.policy_opponent;
                break;
            case CHOOSE_TARGET:
                pol = out.policy_target;
                break;
            case CHOOSE_USE:
                pol = out.policy_binary;
                break;
            default:
                pol = null;
        }
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
        res.stats.evals = 1;
        res.stats.searchNanos = System.nanoTime() - t0;
        return res;
    }

    // ============================================================================ IS-MCTS

    public static Result searchIS(List<World> worlds, Config cfg) {
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
            if (!root.hasValue) evaluate(root, sh, cfg, st); // as searchTree: not a simulation
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
                while (!opts.isEmpty()) {
                    Map<String, Double> pri = null;
                    if (cur.policy != null) {
                        List<MCTSNode> ol = new ArrayList<>(opts.values());
                        double[] pr = priors(cur.policy, ol, sh.getGame(), cfg);
                        if (pr != null) {
                            pri = new HashMap<>();
                            int i = 0;
                            for (String kk : opts.keySet()) pri.put(kk, pr[i++]);
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
        res.rootValue = root.hasValue ? root.value : null;
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

    /** Score a leaf: the value network, or offline MageZero's heuristic (micro decisions inherit). */
    private static double evaluate(Node node, MCTSNode eng, Config cfg, Stats st) {
        long te = System.nanoTime();
        double v;
        if (cfg.nn == null) {
            if (eng.actionType == ActionEncoder.ActionType.PRIORITY) {
                v = GameStateEvaluator3.evaluateNormalized(eng.targetPlayer, eng.getGame());
            } else {
                v = node.parent != null && node.parent.hasValue ? node.parent.value : 0.0;
            }
        } else {
            Set<Integer> sv = eng.stateVector;
            long[] idx = new long[sv == null ? 0 : sv.size()];
            int i = 0;
            if (sv != null) for (int f : sv) idx[i++] = f;
            RemoteModelEvaluator.InferenceResult out = cfg.nn.infer(idx);
            v = out.value;
            if (cfg.priors) {
                switch (eng.actionType) {
                    case PRIORITY:
                        node.policy = eng.targetPlayer.equals(eng.playerId) ? out.policy_player : out.policy_opponent;
                        break;
                    case CHOOSE_TARGET:
                        node.policy = out.policy_target;
                        break;
                    case CHOOSE_USE:
                        node.policy = out.policy_binary;
                        break;
                    default:
                        node.policy = null;
                }
            }
        }
        st.evals++;
        st.evalNanos += System.nanoTime() - te;
        node.value = v;
        node.hasValue = true;
        return v;
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
