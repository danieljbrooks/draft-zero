package mage.player.ai;

import mage.cards.Card;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import mage.game.ExileZone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import mage.game.stack.Spell;
import mage.game.stack.StackObject;
import mage.player.ai.encoder.ActionEncoder;
import mage.player.ai.score.GameStateEvaluator3;
import mage.players.Player;
import mage.players.PlayerScript;

import java.util.*;
import java.util.function.Function;

/**
 * Experiment #4's game player (docs/017 §6.2): a MageZero player whose search is the benchmark's.
 *
 * At every decision MageZero builds, validates, expands and scores its search root exactly as in its
 * own games (ComputerPlayerMCTS2.getNextAction). Instead of MageZero's MCTS, BenchSearch then searches
 * from that root: IS-MCTS, one shared tree over worlds re-dealt every simulation from the cards the
 * player hasn't seen (the opponent's hand and library, and both libraries' order).
 *
 * Closed decklists (docs/017 §3.4): with a `belief`, the worlds are K samples of the opponent's hidden
 * cards from belief.py's OpponentModel (tools/imitation_scale/belief_server.py), given the cards in its
 * known zones; each world is the search's starting state (MageZero's last priority checkpoint) with
 * the opponent's hand and library replaced by a sample's cards. Without one, the single world is the
 * live game, so the opponent's real decklist is known (though not its hand or library order). If no
 * belief world replays to the decision, that decision falls back to the live game (openFallbacks).
 *
 * MageZero then plays the chosen option, and the decision
 * is recorded for training as in MageZero's self-play: the state, the visit counts by action index,
 * and the search's root value (StateEncoder.addLabeledState).
 *
 * No tree reuse: every decision is a fresh search with `cfg.budget` new simulations.
 *
 * Policy-only play (cfg.policyOnly, docs/018): a decision with a policy head (priority, target, binary)
 * plays the network's own policy over MageZero's options, read once on the live game: no search, no
 * belief worlds, no training record. Options sharing an action index split its probability. Decisions
 * without a head, or with an option that has no action index, are searched as configured.
 */
public class BenchPlayer extends ComputerPlayerMCTS2 {

    /** the search's settings (BenchSearch.Config, as the bench op parses them) */
    public transient BenchSearch.Config cfg;
    /** per-search statistics, summed over the game */
    public transient BenchSearch.Stats stats = new BenchSearch.Stats();
    public transient int decisions, singleOption, fallbacks;
    /** policy-only play: decisions the policy played, and those it handed to the search */
    public transient int policyDecisions, policySearched;
    /** every card seen in the player's hand at one of its decisions (id -> name): 17lands' "in hand"
     *  (the opening hand and the cards drawn), for the games-in-hand win rate (docs/018) */
    public final transient Map<UUID, String> seenInHand = new LinkedHashMap<>();
    /** priority choices the engine could not carry out (MageZero's "failed to activate chosen
     *  ability"): the player passed instead */
    public transient int activationFailures;
    public transient long searchNanos;
    private transient Random rng;

    /** Samples of the opponent's hidden cards (closed decklists). */
    public interface Belief {
        /** k samples given the live game as `me` sees it; empty if the belief can't answer */
        List<Sample> sample(Game game, UUID me, UUID opponent, int k, long seed);
    }

    /** One sample: the opponent's hand, and every hidden card (its hand and library) in that world. */
    public static final class Sample {
        public final List<String> hand;
        public final List<String> hidden;

        public Sample(List<String> hand, List<String> hidden) {
            this.hand = hand;
            this.hidden = hidden;
        }
    }

    /** One searched decision, for training (docs/017 §6.6): the state, every legal option's action
     *  index with its visit count, the search's root value, the decision type, the turn, and the
     *  heuristic's score of the live game from this player's seat (GameStateEvaluator3, the
     *  baseline the value head is compared with). */
    public static final class Rec {
        public final int[] features;
        public final String type;
        public final int[] legal;
        public final int[] visits;
        public final double q;
        public final int turn;
        public final double heuristic;

        Rec(int[] features, String type, int[] legal, int[] visits, double q, int turn, double heuristic) {
            this.features = features;
            this.type = type;
            this.legal = legal;
            this.visits = visits;
            this.q = q;
            this.turn = turn;
            this.heuristic = heuristic;
        }
    }

    /** the searched decisions of the game, in order (filled when `record` is set) */
    public transient List<Rec> records = new ArrayList<>();
    public boolean record;

    public transient Belief belief;
    /** creates a card by name (the bridge's StateInjector.newCard) */
    public transient Function<String, Card> cardFactory;
    public int beliefWorlds = 8;
    public transient int beliefCalls, worldsBuilt, worldsFailed, openFallbacks;

    public BenchPlayer(String name, RangeOfInfluence range, int skill) {
        super(name, range, skill);
    }

    public BenchPlayer(final BenchPlayer p) {
        super(p);
        // cfg, stats and counters belong to the live player; copies inside simulations never search
    }

    @Override
    public BenchPlayer copy() {
        return new BenchPlayer(this);
    }

    /**
     * MageZero's priority, except when the chosen ability fails to activate (an "INVALID SCRIPT": the
     * engine can't carry out the searched option). MageZero throws then, and a test-mode player's game
     * ends ("Error in unit tests"; 2.6% of experiment #2's self-play games, docs/013-014; 2 of 12 in a
     * smoke test here). This
     * player restores the state from before its priority, takes a fresh last-priority snapshot (which
     * clears the action histories MageZero rebuilds its roots from) and passes.
     */
    @Override
    public boolean priority(Game game) {
        int bookmark = game.bookmarkState();
        try {
            return super.priority(game);
        } catch (IllegalStateException e) {
            if (e.getMessage() == null || !e.getMessage().startsWith("failed to activate chosen ability")) throw e;
            activationFailures++;
            game.restoreState(bookmark, "BenchPlayer: the chosen ability failed to activate");
            game.getState().setPriorityPlayerId(playerId);
            game.setLastPriority(playerId);
            pass(game);
            return false;
        }
    }

    /**
     * ComputerPlayerMCTS.createMCTSGame, with GraphMCTSPlayer as the simulation players when the seat
     * searches with a graph network: they keep the decision context the graph encoder needs (docs/022).
     */
    @Override
    protected Game createMCTSGame(Game game) {
        if (cfg == null || cfg.gnn == null) return super.createMCTSGame(game);
        Game mcts = game.createSimulationForAI();
        for (Player copyPlayer : mcts.getState().getPlayers().values()) {
            Player origPlayer = game.getState().getPlayers().get(copyPlayer.getId());
            GraphMCTSPlayer newPlayer = new GraphMCTSPlayer(copyPlayer.getId(), getId(), stateEncoder);
            newPlayer.restore(origPlayer);
            newPlayer.setMatchPlayer(origPlayer.getMatchPlayer());
            mcts.getState().getPlayers().put(copyPlayer.getId(), newPlayer);
        }
        mcts.pause();
        mcts.setMCTSSimulation(true);
        return mcts;
    }

    @Override
    protected MCTSNode2 getNextAction(Game game, ActionEncoder.ActionType actionType) {
        root = null;       // a fresh tree every decision: `budget` means new simulations
        return super.getNextAction(game, actionType);
    }

    /** MageZero's MCTS is replaced by BenchSearch (calculateActions). */
    @Override
    protected void applyMCTS(final Game game, final ActionEncoder.ActionType action) {
    }

    @Override
    protected MCTSNode calculateActions(Game game, ActionEncoder.ActionType action) {
        for (Card c : getHand().getCards(game)) seenInHand.putIfAbsent(c.getId(), c.getName());
        MCTSNode2 r = root;
        List<MCTSNode> kids = r.getChildren();
        if (kids.isEmpty()) return null;
        if (kids.size() == 1) {
            singleOption++;
            return kids.get(0);
        }
        if (rng == null) rng = new Random(cfg.seed ^ getId().getMostSignificantBits());
        if (cfg.policyOnly) {
            MCTSNode chosen = policyChoice(game, action, r, kids);
            if (chosen != null) return chosen;
            policySearched++;
        }
        UUID me = getId();
        // keys of MageZero's options, before the search re-deals the root's game
        Map<String, MCTSNode> byKey = new LinkedHashMap<>();
        for (MCTSNode c : kids) byKey.putIfAbsent(BenchSearch.key(c, action, r.getGame(), me), c);
        PlayerScript a = new PlayerScript(getPlayerHistory());
        PlayerScript b = new PlayerScript(game.getOpponent(playerId).getPlayerHistory());
        long t0 = System.nanoTime();
        List<BenchSearch.World> worlds = belief == null ? List.of() : beliefWorlds(game, action, a, b);
        if (worlds.isEmpty()) {
            if (belief != null) openFallbacks++;
            worlds = List.of(new BenchSearch.World(r, this, a, b, action, game));
        }
        BenchSearch.Config c = cfg.copyWithSeed(rng.nextLong());
        BenchSearch.Result res = BenchSearch.searchIS(worlds, c);
        searchNanos += System.nanoTime() - t0;
        stats.add(res.stats);
        decisions++;
        Map<String, Integer> visits = new HashMap<>();
        String bestKey = null;
        int bestN = -1;
        for (BenchSearch.RootChild k : res.children) {
            visits.merge(k.key, k.visits, Integer::sum);
        }
        for (Map.Entry<String, Integer> e : visits.entrySet()) {
            if (byKey.containsKey(e.getKey()) && e.getValue() > bestN) {
                bestN = e.getValue();
                bestKey = e.getKey();
            }
        }
        MCTSNode best;
        if (bestKey == null || bestN <= 0) {
            fallbacks++;   // no searched option matches one of MageZero's: play its first
            best = kids.get(0);
        } else {
            best = byKey.get(bestKey);
        }
        // the training record, as MageZero's calculateActions writes it (flat networks only: a graph
        // network's self-play records would need the graph and its option nodes, docs/022 §7)
        int[] vec = new int[ActionEncoder.ACTION_DIM];
        boolean ok = cfg.gnn == null;
        for (Map.Entry<String, MCTSNode> e : byKey.entrySet()) {
            int idx = e.getValue().getActionIndex(game);
            if (idx < 0) {
                ok = false;
                break;
            }
            vec[idx % ActionEncoder.ACTION_DIM] += visits.getOrDefault(e.getKey(), 0);
        }
        if (ok && stateEncoder != null) {
            stateEncoder.addLabeledState(r.stateVector, vec, res.rootQ == null ? 0.0 : res.rootQ, action, true);
        }
        if (ok && record && r.stateVector != null) {
            // one entry per distinct action index (options sharing an index are one to the network)
            Map<Integer, Integer> byIdx = new TreeMap<>();
            for (Map.Entry<String, MCTSNode> e : byKey.entrySet()) {
                byIdx.merge(e.getValue().getActionIndex(game) % ActionEncoder.ACTION_DIM, visits.getOrDefault(e.getKey(), 0), Integer::sum);
            }
            int[] legal = new int[byIdx.size()], vis = new int[byIdx.size()];
            int i = 0;
            for (Map.Entry<Integer, Integer> e : byIdx.entrySet()) {
                legal[i] = e.getKey();
                vis[i++] = e.getValue();
            }
            records.add(new Rec(r.stateVector.stream().mapToInt(Integer::intValue).sorted().toArray(), String.valueOf(action),
                    legal, vis, res.rootQ == null ? 0.0 : res.rootQ, game.getTurnNum(),
                    GameStateEvaluator3.evaluateNormalized(getId(), game)));
        }
        return best;
    }

    // ------------------------------------------------------------------ policy-only play

    /** The policy's option (cfg.policyOnly), or null when the decision has no head or an option no index. */
    private MCTSNode policyChoice(Game game, ActionEncoder.ActionType action, MCTSNode2 r, List<MCTSNode> kids) {
        boolean mine = getId().equals(r.playerId);
        if (cfg.gnn != null) return graphPolicyChoice(action, r, kids, mine);
        if (!BenchSearch.policyAllowed(action, mine, cfg) || r.stateVector == null) return null;
        int n = kids.size();
        int[] idx = new int[n];
        for (int k = 0; k < n; k++) {
            try {
                idx[k] = kids.get(k).getActionIndex(game);
            } catch (RuntimeException e) {
                return null;
            }
            if (idx[k] < 0) return null;
        }
        long t0 = System.nanoTime();
        long[] feats = new long[r.stateVector.size()];
        int i = 0;
        for (int f : r.stateVector) feats[i++] = f;
        float[] pol = BenchSearch.head(cfg.nn.infer(feats), action, mine);
        stats.netEvals++;
        searchNanos += System.nanoTime() - t0;
        if (pol == null) return null;
        // softmax over the distinct action indices (the network's view), shared equally by their options
        Map<Integer, Integer> share = new HashMap<>();
        for (int a : idx) share.merge(a % pol.length, 1, Integer::sum);
        double mx = Double.NEGATIVE_INFINITY;
        for (int a : share.keySet()) mx = Math.max(mx, pol[a]);
        int best = 0;
        double[] p = new double[n];
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            int a = idx[k] % pol.length;
            if (pol[a] > pol[idx[best] % pol.length]) best = k;
            double t = cfg.policyTemp > 0 ? Math.exp((pol[a] - mx) / cfg.policyTemp) : 0.0;
            p[k] = t / share.get(a);
            sum += p[k];
        }
        int pick = best;
        if (cfg.policyTemp > 0 && sum > 0) {
            double u = rng.nextDouble() * sum;
            for (int k = 0; k < n; k++) {
                u -= p[k];
                if (u <= 0) {
                    pick = k;
                    break;
                }
            }
        }
        decisions++;
        policyDecisions++;
        return kids.get(pick);
    }

    /** policyChoice for a graph network: a softmax over each option's own node score (each copy of a card is
     *  its own option here, so copies share the option's probability as the trainer's log-sum-exp does). */
    private MCTSNode graphPolicyChoice(ActionEncoder.ActionType action, MCTSNode2 r, List<MCTSNode> kids, boolean mine) {
        if (!BenchSearch.policyAllowed(action, mine, cfg)) return null;
        long t0 = System.nanoTime();
        GraphNet.Ask q = GraphNet.ask(r);
        GraphNet.Policy pol = BenchSearch.inferGraph(r, q, cfg, stats).policy(q);
        searchNanos += System.nanoTime() - t0;
        double[] lg = pol == null ? null : pol.logits(BenchSearch.singletons(kids));
        if (lg == null) return null;
        int n = kids.size();
        int best = 0;
        double mx = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < n; k++) {
            if (lg[k] > lg[best]) best = k;
            mx = Math.max(mx, lg[k]);
        }
        int pick = best;
        if (cfg.policyTemp > 0) {
            double[] p = new double[n];
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                p[k] = Math.exp((lg[k] - mx) / cfg.policyTemp);
                sum += p[k];
            }
            double u = rng.nextDouble() * sum;
            for (int k = 0; k < n; k++) {
                u -= p[k];
                if (u <= 0) {
                    pick = k;
                    break;
                }
            }
        }
        decisions++;
        policyDecisions++;
        return kids.get(pick);
    }

    // ------------------------------------------------------------------ closed decklists

    /** K worlds from the belief: the starting state with the opponent's hidden cards replaced. */
    private List<BenchSearch.World> beliefWorlds(Game game, ActionEncoder.ActionType action, PlayerScript a, PlayerScript b) {
        UUID me = getId();
        UUID opp = game.getOpponent(me).getId();
        List<Sample> samples;
        try {
            samples = belief.sample(game, me, opp, beliefWorlds, rng.nextLong());
        } catch (RuntimeException e) {
            samples = List.of();
        }
        beliefCalls++;
        Map<String, Integer> zonesNow = knownZones(game, opp);
        List<BenchSearch.World> out = new ArrayList<>();
        for (Sample s : samples) {
            try {
                Game sim = createMCTSGame(game.getLastPriority());
                // cards the opponent has put into a known zone since the starting state were hidden
                // then (in its hand, or drawn since): they go back in its hand, so the replay of
                // its actions since then can play them again
                Map<String, Integer> since = minus(zonesNow, knownZones(sim, opp));
                if (!replaceHidden(sim, opp, s, since)) {
                    worldsFailed++;
                    continue;
                }
                MCTSNode2 rk = new MCTSNode2(this, sim, action, new PlayerScript(a), new PlayerScript(b));
                rk.validateState();
                if (!rk.isTerminal() && rk.getPlayer().scriptFailed) {
                    worldsFailed++;
                    continue;
                }
                out.add(new BenchSearch.World(rk, this, a, b, action, game));
                worldsBuilt++;
            } catch (RuntimeException e) {
                worldsFailed++;
            }
        }
        return out;
    }

    /** The opponent's hand and library replaced by a sample's cards (same sizes where it can). */
    private boolean replaceHidden(Game sim, UUID opp, Sample s, Map<String, Integer> since) {
        Player p = sim.getPlayer(opp);
        int handN = p.getHand().size();
        int libN = p.getLibrary().size();
        Map<String, Integer> pool = new HashMap<>();
        for (String n : s.hidden) pool.merge(n, 1, Integer::sum);
        for (Map.Entry<String, Integer> e : since.entrySet()) pool.merge(e.getKey(), e.getValue(), Integer::sum);
        List<String> hand = new ArrayList<>();
        for (Map.Entry<String, Integer> e : since.entrySet()) {
            for (int i = 0; i < e.getValue(); i++) take(pool, e.getKey(), hand);
        }
        for (String n : s.hand) if (hand.size() < handN) take(pool, n, hand);
        List<String> rest = new ArrayList<>();
        for (Map.Entry<String, Integer> e : new TreeMap<>(pool).entrySet()) {
            for (int i = 0; i < e.getValue(); i++) rest.add(e.getKey());
        }
        Collections.shuffle(rest, rng);
        while (hand.size() < handN && !rest.isEmpty()) hand.add(rest.remove(rest.size() - 1));
        while (rest.size() > libN) rest.remove(rest.size() - 1);
        List<Card> handCards = new ArrayList<>(), libCards = new ArrayList<>();
        for (String n : hand) handCards.add(cardFactory.apply(n));
        for (String n : rest) libCards.add(cardFactory.apply(n));
        for (Card c : new ArrayList<>(p.getHand().getCards(sim))) p.getHand().remove(c);
        for (Card c : new ArrayList<>(p.getLibrary().getCards(sim))) p.getLibrary().remove(c.getId(), sim);
        sim.loadCards(new HashSet<>(handCards), opp);
        sim.loadCards(new HashSet<>(libCards), opp);
        for (Card c : handCards) {
            c.setZone(Zone.HAND, sim);
            p.getHand().add(c);
        }
        for (Card c : libCards) p.getLibrary().putOnBottom(c, sim);
        return true;
    }

    private static void take(Map<String, Integer> pool, String name, List<String> into) {
        Integer k = pool.get(name);
        if (k == null || k <= 0) return;
        pool.put(name, k - 1);
        into.add(name);
    }

    private static Map<String, Integer> minus(Map<String, Integer> a, Map<String, Integer> b) {
        Map<String, Integer> out = new TreeMap<>();
        for (Map.Entry<String, Integer> e : a.entrySet()) {
            int k = e.getValue() - b.getOrDefault(e.getKey(), 0);
            if (k > 0) out.put(e.getKey(), k);
        }
        return out;
    }

    /**
     * The opponent's cards in zones the player can see, by name: permanents it owns (not tokens),
     * its graveyard, its face-up exiled cards and its spells on the stack. Belief samples are
     * conditioned on these (the belief service's `seen` and `zones`).
     */
    public static Map<String, Integer> knownZones(Game g, UUID opp) {
        Map<String, Integer> out = new TreeMap<>();
        for (Permanent pm : g.getBattlefield().getAllPermanents()) {
            if (pm instanceof PermanentToken || !opp.equals(pm.getOwnerId())) continue;
            Card card = g.getCard(pm.getId());
            out.merge(card != null ? card.getName() : pm.getName(), 1, Integer::sum);
        }
        Player p = g.getPlayer(opp);
        if (p != null) for (Card c : p.getGraveyard().getCards(g)) out.merge(c.getName(), 1, Integer::sum);
        for (ExileZone z : g.getExile().getExileZones()) {
            for (Card c : z.getCards(g)) {
                if (opp.equals(c.getOwnerId()) && !c.isFaceDown(g)) out.merge(c.getName(), 1, Integer::sum);
            }
        }
        for (StackObject so : g.getStack()) {
            if (so instanceof Spell && opp.equals(so.getControllerId())) {
                Card c = ((Spell) so).getCard();
                if (c != null) out.merge(c.getName(), 1, Integer::sum);
            }
        }
        return out;
    }
}
