package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.CardInfo;
import mage.constants.MultiplayerAttackOption;
import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerDuel;
import mage.game.TwoPlayerMatch;
import mage.game.match.Match;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.player.ai.BenchPlayer;
import mage.player.ai.BenchSearch;
import mage.player.ai.encoder.StateEncoder;
import mage.util.RandomUtil;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

/**
 * The play op (docs/017 §6.5): one full game between two BenchPlayers, IS-MCTS at every decision
 * for both seats, set up as MageZero's own harness sets up its games (ParallelDataGenerator): a
 * TwoPlayerDuel in test mode, decks from .dck files, each player's StateEncoder with the
 * opponent's hand hidden.
 *
 *   options  deckA, deckB   .dck paths
 *            seatA, seatB   each seat's search, the bench op's settings (Bench.config): budget,
 *                           priors, priorTemp, priorBonus, leaf, leafMix, opponentPriors,
 *                           isPolicyPerWorld, discount, discountUnit, cPuct, timeoutSec (default 300
 *                           here), policyOnly and policyTemp (policy-only play, BenchPlayer: the
 *                           search settings then apply only to decisions without a policy head),
 *                           evaluator {type offline | remote, host, port}, and belief
 *                           {port, exclude, worlds}: closed decklists, the opponent's hidden cards
 *                           sampled by tools/imitation_scale/belief_server.py (`exclude`: the deck
 *                           file stem of the deck played against, whose draft leaves the pool;
 *                           `worlds`: samples per decision, default 8). Without it the opponent's
 *                           real decklist is known (docs/017 §3.4)
 *            seed           the game's random seed (shuffles, die rolls) and the searches'
 *            starting       "A" (default) or "B": who plays first
 *            maxTurns       the game stops after this turn (default 50; a stopped game has no winner)
 *            record         return each seat's training records (state features, the decision type,
 *                           the turn, each legal option's action index and visit count, the
 *                           search's root value)
 *   response winner ("A", "B", or null), turns, seats {A, B: decisions, singleOption, fallbacks,
 *            policyDecisions, policySearched, inHand {card name: copies seen in hand at the seat's
 *            decisions}, sims, evals, netEvals, engineSteps, searchSeconds, timedOut}, records when asked,
 *            timing_ms
 */
final class Play {
    private Play() {
    }

    private static final Map<String, CardInfo> CARD_INFO = new HashMap<>();
    private static final Map<String, DeckCardLists> DECKS = new HashMap<>();

    static JsonObject run(JsonObject opt) throws Exception {
        long t0 = System.nanoTime();
        String deckA = Worker.optString(opt, "deckA", null);
        String deckB = Worker.optString(opt, "deckB", null);
        if (deckA == null || deckB == null) throw new IllegalArgumentException("play needs options.deckA and options.deckB (.dck paths)");
        long seed = Worker.optLong(opt, "seed", 0L);
        int maxTurns = Worker.optInt(opt, "maxTurns", 50);
        boolean record = Worker.optBool(opt, "record", false);
        String starting = Worker.optString(opt, "starting", "A");
        if (!starting.equals("A") && !starting.equals("B")) throw new IllegalArgumentException("starting must be A or B");

        // a seeded game replays the same way (as the bridge's builds: docs/008 §6): reproducible ids,
        // the global and the game's own random streams, and libraries shuffled from a sorted order
        DeterministicIds.reset(seed);
        RandomUtil.setSeed(StateInjector.mix(seed, 3));
        Match match = new TwoPlayerMatch(new MatchOptions("draftzero play", "draftzero play", false));
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
                MulliganType.GAME_DEFAULT.getMulligan(0), 60, 20, 7);
        BenchPlayer a = player(game, match, "PlayerA", deckA, Worker.optObject(opt, "seatA"), seed * 2 + 1);
        BenchPlayer b = player(game, match, "PlayerB", deckB, Worker.optObject(opt, "seatB"), seed * 2 + 2);
        a.record = record;
        b.record = record;
        encoder(a, b);
        encoder(b, a);
        game.setLocalRandom(new Random(StateInjector.mix(seed, 2)));
        shuffleLibrary(game, a, StateInjector.mix(seed, 10));
        shuffleLibrary(game, b, StateInjector.mix(seed, 11));
        GameOptions options = new GameOptions();
        options.testMode = true;
        options.skipInitShuffling = true;   // shuffled above, from a sorted order
        options.stopOnTurn = maxTurns;
        options.stopAtStep = PhaseStep.END_TURN;
        game.setGameOptions(options);
        game.setStartingPlayerId(starting.equals("B") ? b.getId() : a.getId());
        long t1 = System.nanoTime();
        game.start(null);
        long t2 = System.nanoTime();

        JsonObject r = new JsonObject();
        r.addProperty("winner", a.hasWon() ? "A" : b.hasWon() ? "B" : null);
        r.addProperty("turns", game.getTurnNum());
        r.addProperty("starting", starting);
        r.addProperty("seed", seed);
        JsonObject seats = new JsonObject();
        seats.add("A", seatStats(a));
        seats.add("B", seatStats(b));
        r.add("seats", seats);
        if (record) {
            JsonObject recs = new JsonObject();
            recs.add("A", records(a));
            recs.add("B", records(b));
            r.add("records", recs);
        }
        JsonObject t = new JsonObject();
        t.addProperty("setup", Math.round((t1 - t0) / 1e6));
        t.addProperty("game", Math.round((t2 - t1) / 1e6));
        r.add("timing_ms", t);
        return r;
    }

    private static BenchPlayer player(Game game, Match match, String name, String deckPath, JsonObject seat, long seed)
            throws Exception {
        BenchPlayer p = new BenchPlayer(name, RangeOfInfluence.ONE, 6);
        p.setTestMode(true);
        JsonObject s = seat == null ? new JsonObject() : seat.deepCopy();
        if (!s.has("timeoutSec")) s.addProperty("timeoutSec", 300.0);
        BenchSearch.Config cfg = Bench.config(s);
        cfg.seed = seed;
        p.cfg = cfg;
        // MageZero scores its root before the search (getNextAction): with the network when one is
        // configured, offline otherwise. Policy-only play never reads that score: offline, one network
        // call a decision fewer
        p.nn = cfg.policyOnly ? null : cfg.nn;
        p.offlineMode = p.nn == null;
        p.allowMulligans = false;
        p.noNoise = true;
        JsonObject bel = seat == null ? null : Worker.optObject(seat, "belief");
        if (bel != null) {
            p.belief = new BeliefClient(Worker.optString(bel, "host", "127.0.0.1"), Worker.optInt(bel, "port", 50070),
                    Worker.optString(bel, "exclude", null));
            p.beliefWorlds = Worker.optInt(bel, "worlds", 8);
            p.cardFactory = n -> StateInjector.newCard(n, null, null);
        }
        DeckCardLists list;
        synchronized (DECKS) {
            list = DECKS.get(deckPath);
            if (list == null) {
                list = DeckImporter.importDeckFromFile(deckPath, true);
                DECKS.put(deckPath, list);
            }
        }
        Deck deck;
        synchronized (CARD_INFO) {
            deck = Deck.load(list, false, false, CARD_INFO);
        }
        if (deck.getMaindeckCards().size() < 40) {
            throw new IllegalArgumentException("deck " + deckPath + " has " + deck.getMaindeckCards().size() + " cards");
        }
        game.loadCards(deck.getCards(), p.getId());
        game.loadCards(deck.getSideboard(), p.getId());
        game.addPlayer(p, deck);
        match.addPlayer(p, deck);
        return p;
    }

    /** The library in a fixed order (by name, then id), then a seeded shuffle: Deck.load keeps the
     *  cards in a hash set, so the engine's own first shuffle would start from a different order
     *  every run. */
    private static void shuffleLibrary(Game game, BenchPlayer p, long seed) {
        List<mage.cards.Card> cards = new ArrayList<>(p.getLibrary().getCards(game));
        cards.sort(Comparator.comparing((mage.cards.Card c) -> c.getName()).thenComparing(c -> c.getId().toString()));
        p.getLibrary().clear();
        for (mage.cards.Card c : cards) p.getLibrary().putOnBottom(c, game);
        p.getLibrary().shuffle(new Random(seed));
    }

    private static void encoder(BenchPlayer p, BenchPlayer opponent) {
        StateEncoder e = new StateEncoder();
        e.setAgent(p.getId());
        e.setOpponent(opponent.getId());
        e.perfectInfo = false;      // the opponent's hand hidden, as in experiment #2's training
        p.stateEncoder = e;
    }

    private static JsonObject seatStats(BenchPlayer p) {
        JsonObject s = new JsonObject();
        s.addProperty("decisions", p.decisions);
        s.addProperty("singleOption", p.singleOption);
        s.addProperty("fallbacks", p.fallbacks);
        s.addProperty("policyDecisions", p.policyDecisions);
        s.addProperty("policySearched", p.policySearched);
        JsonObject seen = new JsonObject();
        Map<String, Integer> byName = new TreeMap<>();
        for (String n : p.seenInHand.values()) byName.merge(n, 1, Integer::sum);
        for (Map.Entry<String, Integer> e : byName.entrySet()) seen.addProperty(e.getKey(), e.getValue());
        s.add("inHand", seen);
        s.addProperty("activationFailures", p.activationFailures);
        s.addProperty("sims", p.stats.sims);
        s.addProperty("evals", p.stats.evals);
        s.addProperty("netEvals", p.stats.netEvals);
        s.addProperty("engineSteps", p.stats.engineSteps);
        s.addProperty("policyRefreshes", p.stats.policyRefreshes);
        s.addProperty("netPriors", p.stats.netPriors);
        if (p.cfg != null && p.cfg.gnn != null) s.addProperty("graphPolicyMisses", p.stats.graphPolicyMisses);
        s.addProperty("timedOut", p.stats.timedOut);
        s.addProperty("searchSeconds", Math.round(p.searchNanos / 1e6) / 1000.0);
        s.addProperty("redeals", p.stats.redeals);
        s.addProperty("redealFailures", p.stats.redealFailures);
        s.addProperty("closedDecklists", p.belief != null);
        if (p.belief != null) {
            s.addProperty("beliefCalls", p.beliefCalls);
            s.addProperty("worldsBuilt", p.worldsBuilt);
            s.addProperty("worldsFailed", p.worldsFailed);
            s.addProperty("openFallbacks", p.openFallbacks);
        }
        return s;
    }

    /** The belief service's client (tools/imitation_scale/belief_server.py). */
    static final class BeliefClient implements BenchPlayer.Belief {
        private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        private final URI uri;
        private final String exclude;

        BeliefClient(String host, int port, String exclude) {
            this.uri = URI.create("http://" + host + ":" + port + "/sample");
            this.exclude = exclude;
        }

        @Override
        public List<BenchPlayer.Sample> sample(Game game, UUID me, UUID opponent, int k, long seed) {
            Map<String, Integer> zones = BenchPlayer.knownZones(game, opponent);
            JsonObject req = new JsonObject();
            if (exclude != null) req.addProperty("exclude", exclude);
            JsonObject z = new JsonObject();
            for (Map.Entry<String, Integer> e : zones.entrySet()) z.addProperty(e.getKey(), e.getValue());
            req.add("seen", z);
            req.add("zones", z);
            req.addProperty("hand", game.getPlayer(opponent).getHand().size());
            req.addProperty("k", k);
            req.addProperty("seed", seed);
            try {
                HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(30))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(req.toString())).build(), HttpResponse.BodyHandlers.ofString());
                JsonObject body = JsonParser.parseString(r.body()).getAsJsonObject();
                if (r.statusCode() != 200 || !body.has("samples")) {
                    throw new IllegalStateException("belief service: " + (body.has("error") ? body.get("error").getAsString() : r.statusCode()));
                }
                List<BenchPlayer.Sample> out = new ArrayList<>();
                for (JsonElement e : body.getAsJsonArray("samples")) {
                    JsonObject o = e.getAsJsonObject();
                    out.add(new BenchPlayer.Sample(strings(o.getAsJsonArray("hand")), strings(o.getAsJsonArray("hidden"))));
                }
                return out;
            } catch (java.io.IOException | InterruptedException e) {
                throw new IllegalStateException("belief service unreachable at " + uri + ": " + e.getMessage());
            }
        }

        private static List<String> strings(JsonArray a) {
            List<String> out = new ArrayList<>();
            for (JsonElement e : a) out.add(e.getAsString());
            return out;
        }
    }

    /** A seat's training records (BenchPlayer.Rec): features, the decision type, the turn, every
     *  legal option's action index with its visit count, and the search's root value. */
    private static JsonArray records(BenchPlayer p) {
        JsonArray out = new JsonArray();
        for (BenchPlayer.Rec r : p.records) {
            JsonObject o = new JsonObject();
            JsonArray f = new JsonArray();
            for (int x : r.features) f.add(x);
            o.add("features", f);
            o.addProperty("type", r.type);
            o.addProperty("turn", r.turn);
            JsonArray legal = new JsonArray(), visits = new JsonArray();
            for (int x : r.legal) legal.add(x);
            for (int x : r.visits) visits.add(x);
            o.add("legal", legal);
            o.add("visits", visits);
            o.addProperty("q", r.q);
            o.addProperty("heuristic", r.heuristic);
            out.add(o);
        }
        return out;
    }
}
