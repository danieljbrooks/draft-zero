package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import mage.player.ai.RemoteModelEvaluator;

import java.util.*;

/**
 * The coach op: rate every legal option of the decision player's first decision with MageZero's
 * search, over K determinizations of the hidden information (PIMC at the root, xmage_state.md
 * §7.4 / magezero_mcts.md §3.3).
 *
 * Per determinization k (seed_k derived from the request seed): rebuild the state with seed_k
 * (library order below the known top cards and any handUnknown cards are drawn with it), re-draw
 * the hands of the `resample` seats from their libraries, then search once with a fresh tree.
 * Children with the same label (two copies of a card) are merged. Across determinizations: mean
 * and sd of Q, mean visit share, rank by mean Q; and for a given human action its rank and regret
 * (best mean Q minus its mean Q). Q is the searching player's mean backed-up value in [-1, 1].
 */
final class Coach {
    private static final Map<String, RemoteModelEvaluator> EVALUATORS = new HashMap<>();

    private Coach() {
    }

    static JsonObject run(Spec spec, JsonObject opt) throws Exception {
        long t0 = System.nanoTime();
        int k = Worker.optInt(opt, "determinizations", Worker.optInt(opt, "K", 4));
        int budget = Worker.optInt(opt, "budget", 300);
        long seed = Worker.optLong(opt, "seed", 0L);
        double timeout = Worker.optDouble(opt, "timeoutSec", 120.0);
        boolean lenient = Worker.optBool(opt, "lenient", false);
        boolean perfectInfo = Worker.optBool(opt, "perfectInfo", true);
        String dseat = Worker.decisionSeat(spec, opt);
        List<String> resample = new ArrayList<>();
        if (opt.has("resample") && !opt.get("resample").isJsonNull()) {
            for (JsonElement e : opt.getAsJsonArray("resample")) resample.add(e.getAsString());
        } else {
            resample.add(Spec.other(dseat));
        }
        for (String s : resample) if (!Spec.SEATS.contains(s)) throw new IllegalArgumentException("resample seat '" + s + "'");
        JsonObject evalOpt = opt.has("evaluator") && opt.get("evaluator").isJsonObject() ? opt.getAsJsonObject("evaluator") : new JsonObject();
        String evalType = Worker.optString(evalOpt, "type", "offline");
        JsonObject priors = opt.has("priors") && opt.get("priors").isJsonObject() ? opt.getAsJsonObject("priors") : new JsonObject();
        RemoteModelEvaluator nn = null;
        if ("remote".equals(evalType)) {
            nn = evaluator(Worker.optString(evalOpt, "host", "127.0.0.1"), Worker.optInt(evalOpt, "port", 50052));
        } else if (!"offline".equals(evalType)) {
            throw new IllegalArgumentException("evaluator.type must be offline or remote, got '" + evalType + "'");
        }
        if (k < 1 || budget < 1) throw new IllegalArgumentException("determinizations and budget must be >= 1");

        JsonArray dets = new JsonArray();
        List<Decision> decisions = new ArrayList<>();
        Decision first = null;
        boolean consistent = true;
        double buildMs = 0, searchS = 0;
        long sims = 0;
        for (int i = 0; i < k; i++) {
            long sk = StateInjector.mix(seed, 100 + i);
            long tb = System.nanoTime();
            StateInjector.Built b = StateInjector.build(spec, seed, sk, lenient);
            Random rr = new Random(StateInjector.mix(sk, 7));
            JsonObject sampled = new JsonObject();
            for (String s : resample) sampled.add(s, Dumper.strings(StateInjector.resampleHand(b, s, rr)));
            buildMs += (System.nanoTime() - tb) / 1e6;
            BridgePlayer d = b.players.get(dseat);
            d.searchBudget = budget;
            d.searchTimeout = timeout;
            d.noNoise = true;
            d.stateEncoder.perfectInfo = perfectInfo;
            if (nn != null) {
                d.nn = nn;
                d.offlineMode = false;
                d.noPolicyPriority = !Worker.optBool(priors, "priority", false);
                d.noPolicyTarget = !Worker.optBool(priors, "target", false);
                d.noPolicyUse = !Worker.optBool(priors, "binary", false);
                d.noPolicyOpponent = !Worker.optBool(priors, "opponent", false);
                if (priors.has("temperature")) d.priorTemp = priors.get("temperature").getAsDouble();
            }
            String[] reason = new String[1];
            Decision dec = Worker.advance(b, dseat, opt, BridgePlayer.Mode.SEARCH, null, reason);
            if (dec == null) throw new IllegalStateException("determinization " + i + ": " + reason[0]);
            if (first == null) first = dec;
            else if (!first.type.equals(dec.type) || !Objects.equals(first.text, dec.text)) consistent = false;
            decisions.add(dec);
            searchS += dec.searchSeconds;
            sims += dec.rootVisits;
            JsonObject dj = new JsonObject();
            dj.addProperty("k", i);
            dj.addProperty("seed", sk);
            dj.add("sampledHands", sampled);
            dj.addProperty("type", dec.type);
            dj.addProperty("text", dec.text);
            dj.addProperty("best", dec.best);
            dj.addProperty("rootVisits", dec.rootVisits);
            dj.addProperty("rootQ", dec.rootQ);
            dj.addProperty("value", dec.rootValue);
            dj.addProperty("seconds", Math.round(dec.searchSeconds * 1000) / 1000.0);
            JsonArray ch = new JsonArray();
            for (Decision.Child c : dec.children) ch.add(Decision.childJson(c));
            dj.add("children", ch);
            if (Worker.optBool(opt, "rootFeatures", false) && dec.rootFeatures != null) {
                // what `encode` must reproduce: MageZero's own encoding of the root (sorted ids)
                JsonArray fa = new JsonArray();
                dec.rootFeatures.stream().sorted().forEach(fa::add);
                dj.add("rootFeatures", fa);
            }
            if (!b.warnings.isEmpty()) dj.add("warnings", Dumper.strings(b.warnings));
            dets.add(dj);
        }

        List<Agg> agg = aggregate(decisions);
        JsonObject out = new JsonObject();
        out.add("decision", first.toJson(false));
        out.addProperty("consistent", consistent);
        JsonObject ev = new JsonObject();
        ev.addProperty("type", evalType);
        if (nn != null) ev.addProperty("url", "http://" + Worker.optString(evalOpt, "host", "127.0.0.1") + ":" + Worker.optInt(evalOpt, "port", 50052));
        ev.addProperty("budget", budget);
        ev.addProperty("determinizations", k);
        ev.addProperty("perfectInfo", perfectInfo);
        ev.add("resample", Dumper.strings(resample));
        ev.addProperty("deterministic", nn == null); // network search is timing dependent (MAX_PENDING async evals)
        out.add("settings", ev);
        out.add("determinizations", dets);
        JsonArray aj = new JsonArray();
        for (Agg a : agg) aj.add(a.toJson());
        out.add("aggregate", aj);
        out.addProperty("best", agg.isEmpty() ? null : agg.get(0).label);
        if (opt.has("humanAction") && !opt.get("humanAction").isJsonNull()) {
            out.add("human", human(opt.get("humanAction").getAsString(), first.type, agg));
        }
        JsonObject timing = new JsonObject();
        timing.addProperty("build", Math.round(buildMs));
        timing.addProperty("search", Math.round(searchS * 1000));
        timing.addProperty("total", Math.round((System.nanoTime() - t0) / 1e6));
        timing.addProperty("simsPerSec", searchS > 0 ? Math.round(sims / searchS) : 0);
        out.add("timing_ms", timing);
        return out;
    }

    static synchronized RemoteModelEvaluator evaluator(String host, int port) {
        String url = "http://" + host + ":" + port;
        RemoteModelEvaluator ev = EVALUATORS.get(url);
        if (ev != null) return ev;
        try {
            ev = new RemoteModelEvaluator(url); // health check on /healthz
        } catch (Exception e) {
            throw new IllegalStateException("cannot reach the MageZero inference server at " + url + " (" + e.getMessage()
                    + "); start it first, see java/mzbridge/README.md");
        }
        EVALUATORS.put(url, ev);
        return ev;
    }

    static void closeEvaluators() {
        for (RemoteModelEvaluator ev : EVALUATORS.values()) ev.close();
        EVALUATORS.clear();
    }

    static final class Agg {
        String label;
        int idx;
        List<Double> qs = new ArrayList<>();
        double shareSum;
        int visits;
        int nDet;
        Double meanQ;
        double sdQ;
        double share;
        int rank;

        JsonObject toJson() {
            JsonObject j = new JsonObject();
            j.addProperty("label", label);
            j.addProperty("idx", idx);
            j.addProperty("meanQ", meanQ);
            j.addProperty("sdQ", sdQ);
            j.addProperty("visitShare", share);
            j.addProperty("N", visits);
            j.addProperty("nDet", nDet);
            j.addProperty("rank", rank);
            return j;
        }
    }

    static List<Agg> aggregate(List<Decision> decisions) {
        Map<String, Agg> by = new LinkedHashMap<>();
        for (Decision d : decisions) {
            Map<String, Decision.Child> merged = d.mergedChildren();
            int total = merged.values().stream().mapToInt(c -> c.visits).sum();
            for (Decision.Child c : merged.values()) {
                Agg a = by.computeIfAbsent(c.label, l -> {
                    Agg x = new Agg();
                    x.label = l;
                    x.idx = c.idx;
                    return x;
                });
                a.nDet++;
                a.visits += c.visits;
                a.shareSum += total > 0 ? (double) c.visits / total : 0;
                if (c.q != null) a.qs.add(c.q);
            }
        }
        int k = decisions.size();
        for (Agg a : by.values()) {
            a.share = a.shareSum / k;
            if (!a.qs.isEmpty()) {
                double m = a.qs.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double v = 0;
                for (double q : a.qs) v += (q - m) * (q - m);
                a.meanQ = m;
                a.sdQ = a.qs.size() > 1 ? Math.sqrt(v / (a.qs.size() - 1)) : 0.0;
            }
        }
        List<Agg> out = new ArrayList<>(by.values());
        out.sort(Comparator.comparing((Agg a) -> a.meanQ == null ? Double.NEGATIVE_INFINITY : a.meanQ).reversed()
                .thenComparing(Comparator.comparingDouble((Agg a) -> a.share).reversed())
                .thenComparing(a -> a.label));
        for (int i = 0; i < out.size(); i++) out.get(i).rank = i + 1;
        return out;
    }

    /** Match the human's action to a label and report its rank and regret. */
    static JsonObject human(String action, String type, List<Agg> agg) {
        String want = action.trim();
        if ("CHOOSE_USE".equals(type)) {
            String w = want.toLowerCase(Locale.ROOT);
            if (w.equals("true") || w.equals("1") || w.equals("yes") || w.equals("attack")) want = "yes";
            if (w.equals("false") || w.equals("0") || w.equals("no")) want = "no";
        }
        Agg hit = null;
        for (Agg a : agg) if (a.label.equals(want)) hit = a;
        if (hit == null) {
            // 17lands ability text has no cost: "Create a 2/1 ..." matches "-2: Create a 2/1 ..."
            for (Agg a : agg) {
                int i = a.label.indexOf(": ");
                if (i > 0 && a.label.substring(i + 2).equals(want)) hit = a;
            }
        }
        if (hit == null) for (Agg a : agg) if (a.label.equalsIgnoreCase(want)) hit = a;
        JsonObject h = new JsonObject();
        h.addProperty("action", action);
        h.addProperty("found", hit != null);
        if (hit != null) {
            h.addProperty("label", hit.label);
            h.addProperty("rank", hit.rank);
            h.addProperty("meanQ", hit.meanQ);
            h.addProperty("sdQ", hit.sdQ);
            h.addProperty("visitShare", hit.share);
            Double best = agg.get(0).meanQ;
            h.addProperty("regret", best == null || hit.meanQ == null ? null : best - hit.meanQ);
        }
        return h;
    }
}
