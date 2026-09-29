package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import mage.cards.Card;
import mage.player.ai.BenchSearch;
import mage.player.ai.RemoteModelEvaluator;
import mage.players.Player;

import java.util.*;

/**
 * The bench op: one search of docs/012's first experiment on one decision.
 *
 * options.specs are the worlds, already determinized by the caller: the real world alone for
 * clairvoyant MCTS; K belief worlds for PIMC; W belief worlds for IS-MCTS. Each is built
 * (world k with worldSeeds[k], default mix(seed, 100 + k); every world shares idSeed so all ask
 * the same question), advanced to the decision player's first real decision and captured as
 * MageZero's search root. Then:
 *
 *   method "tree"    one MageZero-style tree per world, the budget split evenly (clairvoyant with
 *                    one world, PIMC with K); root options are merged by label and the choice is
 *                    the most visits summed over worlds
 *   method "ismcts"  one information-set tree over all the worlds (BenchSearch.searchIS)
 *
 * Other options: budget (total simulations), discount (0.99), discountUnit (ply | action | turn),
 * cPuct (1), evaluator {type offline|remote, host, port}, perfectInfo (false: the network does
 * not see the opponent's hand, as in experiment #2), preLand (a land option to play first),
 * redeal (IS-MCTS, true), timeoutSec, decisionPlayer, decideFrom, lenient.
 */
final class Bench {
    private Bench() {
    }

    static JsonObject run(JsonObject opt) throws Exception {
        long t0 = System.nanoTime();
        if (!opt.has("specs") || !opt.get("specs").isJsonArray()) throw new IllegalArgumentException("bench needs options.specs");
        List<Spec> specs = new ArrayList<>();
        for (JsonElement e : opt.getAsJsonArray("specs")) specs.add(Worker.specFrom(e));
        if (specs.isEmpty()) throw new IllegalArgumentException("options.specs is empty");
        String method = Worker.optString(opt, "method", "tree");
        if (!List.of("tree", "ismcts", "policy").contains(method)) throw new IllegalArgumentException("method must be tree, ismcts or policy");
        long seed = Worker.optLong(opt, "seed", 0L);
        long idSeed = Worker.optLong(opt, "idSeed", seed);
        boolean lenient = Worker.optBool(opt, "lenient", false);
        boolean perfectInfo = Worker.optBool(opt, "perfectInfo", false);
        String preLand = Worker.optString(opt, "preLand", null);
        BenchSearch.Config base = new BenchSearch.Config();
        base.budget = Worker.optInt(opt, "budget", 1000);
        base.discount = Worker.optDouble(opt, "discount", 0.99);
        base.unit = Worker.optString(opt, "discountUnit", "ply");
        if (!List.of("ply", "action", "turn").contains(base.unit)) throw new IllegalArgumentException("discountUnit must be ply, action or turn");
        base.cPuct = Worker.optDouble(opt, "cPuct", 1.0);
        base.redeal = Worker.optBool(opt, "redeal", true);
        base.timeoutSec = Worker.optDouble(opt, "timeoutSec", 900.0);
        base.priors = Worker.optBool(opt, "priors", false);
        base.priorTemp = Worker.optDouble(opt, "priorTemp", 1.5);
        base.priorBonus = Worker.optDouble(opt, "priorBonus", 0.1);
        if (base.budget < 1) throw new IllegalArgumentException("budget must be >= 1");
        JsonObject evalOpt = opt.has("evaluator") && opt.get("evaluator").isJsonObject() ? opt.getAsJsonObject("evaluator") : new JsonObject();
        String evalType = Worker.optString(evalOpt, "type", "offline");
        if ("remote".equals(evalType)) {
            base.nn = Coach.evaluator(Worker.optString(evalOpt, "host", "127.0.0.1"), Worker.optInt(evalOpt, "port", 50052));
        } else if (!"offline".equals(evalType)) {
            throw new IllegalArgumentException("evaluator.type must be offline or remote, got '" + evalType + "'");
        }
        if (base.priors && base.nn == null) throw new IllegalArgumentException("priors need evaluator.type remote");
        List<Long> worldSeeds = new ArrayList<>();
        if (opt.has("worldSeeds") && opt.get("worldSeeds").isJsonArray()) {
            for (JsonElement e : opt.getAsJsonArray("worldSeeds")) worldSeeds.add(e.getAsLong());
        }
        String dseat = Worker.decisionSeat(specs.get(0), opt);

        // ---- build every world and capture its search root
        List<BenchSearch.World> worlds = new ArrayList<>();
        Decision first = null;
        boolean consistent = true;
        double buildMs = 0, advanceMs = 0;
        JsonArray wj = new JsonArray();
        Set<String> hiddenNames = new TreeSet<>();
        List<String> warnings = new ArrayList<>();
        for (int i = 0; i < specs.size(); i++) {
            long sk = i < worldSeeds.size() ? worldSeeds.get(i) : StateInjector.mix(seed, 100 + i);
            long tb = System.nanoTime();
            StateInjector.Built b = StateInjector.build(specs.get(i), idSeed, sk, lenient);
            long ta = System.nanoTime();
            buildMs += (ta - tb) / 1e6;
            BridgePlayer d = b.players.get(dseat);
            d.stateEncoder.perfectInfo = perfectInfo;
            String[] reason = new String[1];
            Decision dec = Worker.advance(b, dseat, opt, BridgePlayer.Mode.ROOT, null, reason);
            advanceMs += (System.nanoTime() - ta) / 1e6;
            if (dec == null) throw new IllegalStateException("world " + i + ": " + reason[0]);
            if (d.capturedRoot == null) throw new IllegalStateException("world " + i + ": no search root captured");
            if (first == null) first = dec;
            else if (!first.type.equals(dec.type) || !Objects.equals(first.text, dec.text)) consistent = false;
            worlds.add(new BenchSearch.World(d.capturedRoot, d, d.capturedPrefixA, d.capturedPrefixB, d.capturedType, b.game));
            Player opp = b.players.get(Spec.other(dseat));
            JsonObject w = new JsonObject();
            w.addProperty("k", i);
            w.addProperty("seed", sk);
            w.add("oppHand", Dumper.names(opp.getHand().getCards(b.game)));
            wj.add(w);
            for (Card c : opp.getHand().getCards(b.game)) hiddenNames.add(c.getName());
            for (Card c : opp.getLibrary().getCards(b.game)) hiddenNames.add(c.getName());
            if (!b.warnings.isEmpty()) warnings.addAll(b.warnings);
        }

        // ---- search
        BenchSearch.Stats stats = new BenchSearch.Stats();
        Map<String, Agg> by = new LinkedHashMap<>();
        Double rootValue = null;
        if (method.equals("policy")) {
            if (base.nn == null) throw new IllegalArgumentException("method policy needs evaluator.type remote");
            BenchSearch.Result r = BenchSearch.rootPolicy(worlds.get(0), base);
            stats.add(r.stats);
            rootValue = r.rootValue;
            merge(by, r, first);
        } else if (method.equals("ismcts")) {
            BenchSearch.Config cfg = copy(base);
            cfg.seed = StateInjector.mix(seed, 200);
            BenchSearch.Result r = BenchSearch.searchIS(worlds, cfg);
            stats.add(r.stats);
            rootValue = r.rootValue;
            merge(by, r, first);
        } else {
            int k = worlds.size();
            for (int i = 0; i < k; i++) {
                BenchSearch.Config cfg = copy(base);
                cfg.budget = base.budget / k + (i < base.budget % k ? 1 : 0);
                cfg.seed = StateInjector.mix(seed, 200 + i);
                if (cfg.budget < 1) continue;
                BenchSearch.Result r = BenchSearch.searchTree(worlds.get(i), cfg);
                stats.add(r.stats);
                if (i == 0) rootValue = r.rootValue;
                merge(by, r, first);
            }
        }
        List<Agg> agg = new ArrayList<>(by.values());
        agg.sort(Comparator.comparingInt((Agg a) -> a.visits).reversed()
                .thenComparing(Comparator.comparingDouble((Agg a) -> a.prior).reversed())
                .thenComparing(Comparator.comparingDouble((Agg a) -> a.q == null ? -9 : a.q).reversed())
                .thenComparing(a -> a.label));

        JsonObject out = new JsonObject();
        out.add("decision", first.toJson(false));
        out.addProperty("consistent", consistent);
        JsonObject set = new JsonObject();
        set.addProperty("method", method);
        set.addProperty("worlds", worlds.size());
        set.addProperty("budget", base.budget);
        set.addProperty("discount", base.discount);
        set.addProperty("discountUnit", base.unit);
        set.addProperty("cPuct", base.cPuct);
        set.addProperty("evaluator", evalType);
        set.addProperty("perfectInfo", perfectInfo);
        set.addProperty("redeal", base.redeal);
        set.addProperty("priors", base.priors);
        set.addProperty("seed", seed);
        set.addProperty("idSeed", idSeed);
        set.addProperty("preLand", preLand);
        out.add("settings", set);
        JsonArray ch = new JsonArray();
        int total = 0;
        for (Agg a : agg) total += a.visits;
        for (Agg a : agg) {
            JsonObject j = new JsonObject();
            j.addProperty("label", a.label);
            j.addProperty("N", a.visits);
            j.addProperty("share", total > 0 ? (double) a.visits / total : 0.0);
            j.addProperty("Q", a.q);
            j.addProperty("sub", a.sub);
            j.addProperty("nKeys", a.keys.size());
            if (a.avail > 0) j.addProperty("avail", a.avail);
            if (method.equals("policy")) j.addProperty("prior", a.prior);
            ch.add(j);
        }
        out.add("children", ch);
        out.addProperty("best", agg.isEmpty() || (agg.get(0).visits == 0 && !method.equals("policy")) ? null : agg.get(0).label);
        out.addProperty("rootVisits", total);
        out.addProperty("rootValue", rootValue);
        out.add("worldInfo", wj);
        out.add("hiddenNames", Dumper.strings(new ArrayList<>(hiddenNames)));
        JsonObject s = new JsonObject();
        s.addProperty("sims", stats.sims);
        s.addProperty("iterations", stats.iterations);
        s.addProperty("scriptFailures", stats.scriptFailures);
        s.addProperty("engineSteps", stats.engineSteps);
        s.addProperty("evals", stats.evals);
        s.addProperty("nodes", stats.nodes);
        s.addProperty("maxDepth", stats.maxDepth);
        s.addProperty("redeals", stats.redeals);
        s.addProperty("redealFailures", stats.redealFailures);
        s.addProperty("edgeVisits", stats.edgeVisits);
        s.addProperty("priorityEdgeVisits", stats.priorityEdgeVisits);
        s.addProperty("turnEdgeSum", stats.turnEdgeSum);
        s.addProperty("timedOut", stats.timedOut);
        out.add("stats", s);
        out.add("warnings", Dumper.strings(warnings));
        JsonObject t = new JsonObject();
        t.addProperty("build", Math.round(buildMs));
        t.addProperty("advance", Math.round(advanceMs));
        t.addProperty("search", Math.round(stats.searchNanos / 1e6));
        t.addProperty("engine", Math.round(stats.engineNanos / 1e6));
        t.addProperty("eval", Math.round(stats.evalNanos / 1e6));
        t.addProperty("total", Math.round((System.nanoTime() - t0) / 1e6));
        out.add("timing_ms", t);
        return out;
    }

    private static BenchSearch.Config copy(BenchSearch.Config b) {
        BenchSearch.Config c = new BenchSearch.Config();
        c.budget = b.budget;
        c.discount = b.discount;
        c.unit = b.unit;
        c.cPuct = b.cPuct;
        c.nn = b.nn;
        c.seed = b.seed;
        c.redeal = b.redeal;
        c.timeoutSec = b.timeoutSec;
        c.maxIterations = b.maxIterations;
        c.priors = b.priors;
        c.priorTemp = b.priorTemp;
        c.priorBonus = b.priorBonus;
        return c;
    }

    static final class Agg {
        String label;
        int visits;
        double qSum;
        Double q;
        int sub = -1;
        int subVisits = -1;
        int avail;
        double prior;
        Set<String> keys = new LinkedHashSet<>();
    }

    /** Merge a search's root options into by-label totals (copies of a card share a label). */
    private static void merge(Map<String, Agg> by, BenchSearch.Result r, Decision first) {
        for (BenchSearch.RootChild c : r.children) {
            String label = c.label;
            if (c.amount != null) label = String.valueOf(first.numMin + c.amount);
            Agg a = by.computeIfAbsent(label, l -> {
                Agg x = new Agg();
                x.label = l;
                return x;
            });
            a.prior += c.prior;
            if (c.q != null) a.qSum += c.q * c.visits;
            a.visits += c.visits;
            a.q = a.visits > 0 ? a.qSum / a.visits : null;
            a.avail += c.avail;
            if (c.key != null) a.keys.add(c.key);
            if (c.subDecisions >= 0 && c.visits > a.subVisits) {
                a.sub = c.subDecisions;
                a.subVisits = c.visits;
            }
        }
    }
}
