package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import mage.cards.Card;
import mage.constants.PhaseStep;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.players.Player;

import java.util.*;

/**
 * The replay_turn op: replay one recorded turn through the engine with both seats as puppets and
 * check that it ends in the recorded end-of-turn state.
 *
 *   request  spec     a turn-start state (17lands: eot_rollover of user turn n, with the turn's later
 *                     draws on top of A's library in order)
 *            options  script    {seat, turn, lands[], casts[], activations[], attacks{}, attackGuess[],
 *                                blocks[[[blocker, attacker|null], ...], ...], blockPairing, opp[]}
 *                     expected  {life{A,B}, hand{A[], unknownA}, battlefield{A[], B[]}, deaths{...}}
 *                     seed, idSeed, lenient, substitute, encode, perfectInfo,
 *                     maxAttempts (default 12), policies[] (explicit attempts, else the built-in list)
 *   response reproduced (the verdict; the envelope's "ok" only says the request ran), attempt (1-based
 *            index of the attempt reported), policy, diff, diffKeys, items,
 *            targets, flags, notes, decisions, end, attemptLog, warnings, timing_ms
 *
 * Attempts differ only in the choices 17lands does not record: the order of A's spells (mana value
 * up or down), whether they are cast before or after combat, A's instants in combat, the windows of
 * B's instants, and the blocker pairing when several fit the deaths. The first attempt whose end
 * state matches wins; otherwise the attempt with the fewest differences is reported. Every attempt
 * rebuilds the spec with the same seeds, so the op is deterministic.
 *
 * Compared after the turn's cleanup step ("end the turn" effects such as Time Stop honoured), with
 * A's hand as the cleanup began (17lands' snapshot precedes the discard to hand size): life totals, A's hand (names;
 * unknown ids only by count) and both battlefields (name multisets per controller; tokens by name
 * without " Token"). A matching end state is not enough: every recorded play of A's must have
 * happened (no "undone:A"), and every recorded attack and block between spec permanents must have
 * been declared (ReplayRun.divergences), or the decisions would label plays the human did not
 * make. Deaths (non-token creatures, combat or not) and B's undone plays are reported ("deaths:A",
 * "undone:B") but not compared.
 */
final class TurnReplay {
    private TurnReplay() {
    }

    static final class Attempt {
        TurnScript.Policy policy;
        ReplayRun run;
        boolean ok;
        int nDiff;
        JsonObject diff;
        JsonArray diffKeys;
        JsonObject end;
        long buildMs, replayMs;
        List<String> warnings = List.of();
        String error;

        JsonObject summary() {
            JsonObject o = new JsonObject();
            o.add("policy", policy.toJson());
            o.addProperty("ok", ok);
            o.addProperty("nDiff", nDiff);
            o.add("diffKeys", diffKeys);
            if (error != null) o.addProperty("error", error);
            o.addProperty("ms", buildMs + replayMs);
            return o;
        }
    }

    static JsonObject run(Spec spec, JsonObject opt) {
        long t0 = System.nanoTime();
        long seed = Worker.optLong(opt, "seed", 0L);
        long idSeed = Worker.optLong(opt, "idSeed", seed);
        boolean lenient = Worker.optBool(opt, "lenient", false);
        boolean encode = Worker.optBool(opt, "encode", false);
        boolean perfectInfo = Worker.optBool(opt, "perfectInfo", false);
        Substitutions.Result subs = Substitutions.apply(spec, Worker.optObject(opt, "substitute"));
        int defaultTurn = spec.turn + (spec.activePlayer.equals(Worker.optString(Worker.optObject(opt, "script"), "seat", "A")) ? 0 : 1);
        TurnScript script = TurnScript.parse(Worker.optObject(opt, "script"), Worker.optObject(opt, "expected"), defaultTurn);
        if (script.turn < spec.turn || script.turn > spec.turn + 1) {
            throw new IllegalArgumentException("script.turn " + script.turn + " must be the spec's turn " + spec.turn + " or the next one");
        }
        List<TurnScript.Policy> policies = new ArrayList<>();
        boolean explicit = opt.has("policies") && opt.get("policies").isJsonArray();
        int max = Worker.optInt(opt, "maxAttempts", 12);
        if (explicit) {
            for (JsonElement e : opt.getAsJsonArray("policies")) policies.add(TurnScript.Policy.from(e.getAsJsonObject()));
        } else {
            policies = policies(script, spec, max);
        }
        Set<String> tried = new HashSet<>();
        Set<String> expanded = new HashSet<>();
        for (TurnScript.Policy p : policies) tried.add(signature(script, p));
        if (policies.isEmpty()) throw new IllegalArgumentException("no attempt to run (maxAttempts < 1 or policies empty)");

        JsonArray log = new JsonArray();
        Attempt best = null;
        int bestIdx = 0;
        long buildMs = 0, replayMs = 0;
        for (int i = 0; i < policies.size(); i++) {
            Attempt a = attempt(spec, script, policies.get(i), seed, idSeed, lenient, encode, perfectInfo);
            buildMs += a.buildMs;
            replayMs += a.replayMs;
            log.add(a.summary());
            if (best == null || a.nDiff < best.nDiff || a.ok) {
                best = a;
                bestIdx = i;
            }
            if (a.ok) break;
            if (!explicit && a.run != null) {
                // the attempt guessed a mode, a "may" answer, a target ... : try the other answers
                // next (same ordering), ahead of the remaining generic attempts; each kind of
                // guess is expanded once per request, from the first attempt that made it
                List<TurnScript.Policy> more = new ArrayList<>();
                TurnScript.Policy base = policies.get(i);
                Set<String> fresh = new HashSet<>(a.run.flags);
                fresh.removeAll(expanded);
                expanded.addAll(a.run.flags);
                if (fresh.contains("guessed_mode")) {
                    for (int m = 1; m <= 2; m++) {
                        int k = base.mode + m;
                        more.add(base.with(p -> p.mode = k));
                    }
                }
                if (fresh.contains("guessed_use")) {
                    more.add(base.with(p -> p.may = "no"));
                    more.add(base.with(p -> p.may = "yes"));
                }
                if (fresh.contains("guessed_target")) more.add(base.with(p -> p.targetOrder = "low"));
                if (fresh.contains("sac_mana_kept")) more.add(base.with(p -> p.sacMana = "use"));
                if (fresh.contains("attack_defender_player_assumed")) more.add(base.with(p -> p.defender = "planeswalker"));
                int at = i + 1;
                for (TurnScript.Policy p : more) {
                    if (tried.add(signature(script, p))) policies.add(at++, p);
                }
                while (policies.size() > max) policies.remove(policies.size() - 1);
            }
        }
        ReplayRun r = best.run;
        JsonObject out = new JsonObject();
        out.addProperty("reproduced", best.ok);
        out.addProperty("attempt", bestIdx + 1);
        out.addProperty("attempts", log.size());
        out.add("policy", best.policy.toJson());
        out.add("diff", best.diff);
        out.add("diffKeys", best.diffKeys);
        if (best.error != null) out.addProperty("error", best.error);
        JsonArray items = new JsonArray();
        if (r != null) {
            for (TurnScript.Item it : r.items) {
                JsonObject o = it.toJson();
                o.addProperty("due", r.due.get(it));
                items.add(o);
            }
        }
        out.add("items", items);
        out.add("targets", r == null ? new JsonArray() : r.targets);
        out.add("flags", Dumper.strings(r == null ? List.of() : r.flags));
        out.add("notes", Dumper.strings(r == null ? List.of() : r.notes));
        out.add("decisions", r == null ? new JsonArray() : r.decisions);
        out.add("end", best.end);
        out.add("attemptLog", log);
        List<String> warnings = new ArrayList<>(subs.warnings);
        warnings.addAll(best.warnings);
        out.add("warnings", Dumper.strings(warnings));
        JsonObject t = new JsonObject();
        t.addProperty("build", buildMs);
        t.addProperty("replay", replayMs);
        t.addProperty("total", Worker.ms(t0, System.nanoTime()));
        out.add("timing_ms", t);
        return out;
    }

    /** The built-in attempts, without the ones that would replay exactly like an earlier one. */
    static List<TurnScript.Policy> policies(TurnScript script, Spec spec, int max) {
        List<TurnScript.Policy> cands = new ArrayList<>();
        cands.add(pol("mv_asc", "main1", "main", "auto", 0));
        cands.add(pol("mv_desc", "main1", "main", "auto", 0));
        TurnScript.Policy landLast = pol("mv_asc", "main1", "main", "auto", 0);
        landLast.land = "last";
        cands.add(landLast);
        cands.add(pol("mv_asc", "creatures_main2", "combat", "auto", 0));
        cands.add(pol("mv_asc", "main2", "main", "auto", 0));
        cands.add(pol("mv_asc", "main1", "combat", "auto", 0));
        for (String w : List.of("after_attackers", "end_step", "respond", "main1", "after_blockers", "upkeep")) {
            cands.add(pol("mv_asc", "main1", "main", w, 0));
        }
        for (int k = 1; k < Math.min(script.blocks.size(), 5); k++) cands.add(pol("mv_asc", "main1", "main", "auto", k));
        cands.add(pol("listed", "main1", "main", "auto", 0));
        cands.add(pol("mv_desc", "creatures_main2", "main", "auto", 0));
        if (modal(script)) {
            for (int m = 1; m <= 2; m++) {
                TurnScript.Policy p = pol("mv_asc", "main1", "main", "auto", 0);
                p.mode = m;
                cands.add(2 + m - 1, p);        // right after the two orderings: a wrong mode is common
            }
        }
        List<TurnScript.Policy> out = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (TurnScript.Policy p : cands) {
            if (out.size() >= max) break;
            if (seen.add(signature(script, p))) out.add(p);
        }
        return out;
    }

    /** Does the turn cast a modal spell ("Choose one --")? */
    static boolean modal(TurnScript script) {
        for (TurnScript.Item it : script.items) {
            ReplayRun.Facts f = ReplayRun.Facts.of(it.name);
            if (f != null && (f.rules.contains("choose one") || f.rules.contains("choose two") || f.rules.contains("choose up to"))) return true;
        }
        return false;
    }

    static TurnScript.Policy pol(String order, String aMain, String aInstants, String bWindow, int pairing) {
        TurnScript.Policy p = new TurnScript.Policy();
        p.order = order;
        p.aMain = aMain;
        p.aInstants = aInstants;
        p.bWindow = bWindow;
        p.pairing = pairing;
        return p;
    }

    /** What an attempt actually does differently: each item's window and A's order, and the pairing. */
    static String signature(TurnScript script, TurnScript.Policy p) {
        ReplayRun probe = new ReplayRun(null, script, p, false, false);
        StringBuilder sb = new StringBuilder();
        for (TurnScript.Item it : probe.mine) sb.append(it.key).append('@').append(probe.due.get(it)).append(';');
        sb.append('|');
        for (TurnScript.Item it : probe.theirs) sb.append(it.key).append('@').append(probe.due.get(it)).append(';');
        sb.append("|pairing=").append(script.blocks.isEmpty() ? 0 : Math.min(p.pairing, script.blocks.size() - 1));
        sb.append("|mode=").append(p.mode).append("|may=").append(p.may).append("|to=").append(p.targetOrder)
                .append("|sac=").append(p.sacMana).append("|def=").append(p.defender);
        return sb.toString();
    }

    static Attempt attempt(Spec spec, TurnScript script, TurnScript.Policy pol, long seed, long idSeed, boolean lenient,
                           boolean encode, boolean perfectInfo) {
        Attempt a = new Attempt();
        a.policy = pol;
        long t0 = System.nanoTime();
        StateInjector.Built b = StateInjector.build(spec, idSeed, seed, lenient);
        a.warnings = b.warnings;
        Game game = b.game;
        ReplayRun run = new ReplayRun(b, script, pol, encode, perfectInfo);
        a.run = run;
        // both seats become script-following puppets: replace the player objects in the game state
        for (String s : Spec.SEATS) {
            BridgePlayer old = b.players.get(s);
            ReplayPlayer rp = new ReplayPlayer(old, run);
            game.getState().getPlayers().put(old.getId(), rp);
            b.players.put(s, rp);
        }
        game.getState().addWatcher(new ReplayWatcher());
        // stop after the replayed turn's cleanup step (Phase.checkStopOnStepOption): an "end the
        // turn" effect skips to cleanup; the hand is read as cleanup began (ReplayWatcher)
        game.getOptions().stopOnTurn = script.turn;
        game.getOptions().stopAtStep = PhaseStep.CLEANUP;
        StateInjector.anchor(b, false);
        DeterministicIds.reset(StateInjector.mix(b.idSeed, 5));
        long t1 = System.nanoTime();
        try {
            game.resume();
        } catch (RuntimeException e) {
            run.fail(e);
        }
        long t2 = System.nanoTime();
        a.buildMs = Worker.ms(t0, t1);
        a.replayMs = Worker.ms(t1, t2);
        compare(a, run, game);
        return a;
    }

    // =================================================================================== comparison

    static void compare(Attempt a, ReplayRun run, Game game) {
        TurnScript.Expected ex = run.script.expected;
        JsonObject diff = new JsonObject();
        JsonArray keys = new JsonArray();
        int n = 0;
        if (run.failure != null) {
            a.error = run.failure.getClass().getSimpleName() + ": " + run.failure.getMessage();
            keys.add("engine_error");
            n += 100;
        }
        boolean reached = game.getTurnNum() == run.turn && game.getTurnStepType() == PhaseStep.CLEANUP;
        if (game.checkIfGameIsOver()) {
            keys.add("game_over");
            n += 1;
        } else if (!reached && run.failure == null) {
            keys.add("end_not_reached");
            diff.addProperty("stoppedAt", game.getTurnNum() + ":" + game.getTurnStepType());
            n += 100;
        }
        JsonObject end = new JsonObject();
        JsonObject life = new JsonObject();
        JsonObject lifeDiff = new JsonObject();
        for (String s : Spec.SEATS) {
            Player p = game.getPlayer(run.b.players.get(s).getId());
            life.addProperty(s, p.getLife());
            Integer want = ex.life.get(s);
            if (want != null && want != p.getLife()) {
                JsonArray d = new JsonArray();
                d.add(want);
                d.add(p.getLife());
                lifeDiff.add(s, d);
                keys.add("life:" + s);
                n += 1;
            }
        }
        end.add("life", life);
        if (lifeDiff.size() > 0) diff.add("life", lifeDiff);

        // A's hand as the cleanup step began (17lands' snapshot comes before the discard to hand size)
        Player pa = game.getPlayer(run.b.players.get(run.seat).getId());
        Map<String, Integer> hand = new TreeMap<>();
        ReplayWatcher hw = game.getState().getWatcher(ReplayWatcher.class);
        List<String> atCleanup = hw == null ? null : hw.handAtCleanup(run.turn, run.seat);
        if (atCleanup != null) {
            for (String nm : atCleanup) hand.merge(nm, 1, Integer::sum);
        } else {
            for (Card c : pa.getHand().getCards(game)) hand.merge(c.getName(), 1, Integer::sum);
        }
        JsonObject handEnd = new JsonObject();
        handEnd.add("A", expand(hand));
        end.add("hand", handEnd);
        if (ex.hasHand) {
            Map<String, Integer> missing = minus(ex.handA, hand);
            Map<String, Integer> extra = minus(hand, ex.handA);
            int extraN = extra.values().stream().mapToInt(Integer::intValue).sum();
            // unknown ids in the recorded hand absorb that many extra cards
            if (!missing.isEmpty() || extraN != ex.handUnknownA) {
                JsonObject h = new JsonObject();
                h.add("missing", expand(missing));
                h.add("extra", expand(extra));
                if (ex.handUnknownA > 0) h.addProperty("unknown", ex.handUnknownA);
                diff.add("hand", h);
                if (!missing.isEmpty()) keys.add("hand:A:missing");
                if (extraN > ex.handUnknownA) keys.add("hand:A:extra");
                if (missing.isEmpty() && extraN < ex.handUnknownA) keys.add("hand:A:short");
                n += sum(missing) + Math.abs(extraN - ex.handUnknownA);
            }
        }

        // battlefields
        Map<String, Map<String, Integer>> bf = run.battlefieldCounts(game);
        JsonObject bfEnd = new JsonObject();
        JsonObject bfDiff = new JsonObject();
        for (String s : Spec.SEATS) {
            bfEnd.add(s, expand(bf.get(s)));
            if (!ex.hasBattlefield) continue;
            Map<String, Integer> want = ex.battlefield.getOrDefault(s, Map.of());
            Map<String, Integer> missing = minus(want, bf.get(s));
            Map<String, Integer> extra = minus(bf.get(s), want);
            if (!missing.isEmpty() || !extra.isEmpty()) {
                JsonObject d = new JsonObject();
                d.add("missing", expand(missing));
                d.add("extra", expand(extra));
                bfDiff.add(s, d);
                if (!missing.isEmpty()) keys.add("bf:" + s + ":missing");
                if (!extra.isEmpty()) keys.add("bf:" + s + ":extra");
                n += sum(missing) + sum(extra);
            }
        }
        end.add("battlefield", bfEnd);
        if (bfDiff.size() > 0) diff.add("battlefield", bfDiff);

        // script items that never happened: the scripted seat's fail the verdict (its decisions
        // would miss a recorded play); the other seat's are reported only (its hand is hidden)
        JsonArray undone = new JsonArray();
        int undoneMine = 0;
        for (TurnScript.Item it : run.items) {
            if (!it.done) {
                undone.add(it.describe());
                if (it.seat.equals(run.seat)) undoneMine++;
                if (!keys.contains(new com.google.gson.JsonPrimitive("undone:" + it.seat))) keys.add("undone:" + it.seat);
            }
        }
        if (undone.size() > 0) diff.add("undone", undone);

        // recorded attacks and blocks the engine did not declare (ReplayRun.divergences)
        Map<String, List<String>> div = run.divergences(game);
        if (!div.isEmpty()) {
            JsonObject dv = new JsonObject();
            for (Map.Entry<String, List<String>> e : div.entrySet()) {
                dv.add(e.getKey(), Dumper.strings(e.getValue()));
                keys.add(e.getKey());
                n += e.getValue().size();
            }
            diff.add("divergence", dv);
        }

        // deaths (info)
        ReplayWatcher w = game.getState().getWatcher(ReplayWatcher.class);
        JsonObject deaths = new JsonObject();
        Map<String, Map<String, Map<String, Integer>>> got = new LinkedHashMap<>();
        if (w != null) {
            for (String e : w.deaths()) {
                String[] f = e.split("\\|", 4);
                if (Integer.parseInt(f[0]) != run.turn) continue;
                got.computeIfAbsent(f[1], k -> new LinkedHashMap<>()).computeIfAbsent(f[2], k -> new TreeMap<>()).merge(f[3], 1, Integer::sum);
            }
        }
        for (String s : Spec.SEATS) {
            JsonObject ds = new JsonObject();
            for (String kind : List.of("combat", "noncombat")) ds.add(kind, expand(got.getOrDefault(s, Map.of()).getOrDefault(kind, Map.of())));
            deaths.add(s, ds);
        }
        end.add("deaths", deaths);
        if (!ex.deaths.isEmpty()) {
            JsonObject dd = new JsonObject();
            for (String s : Spec.SEATS) {
                Map<String, Integer> all = new TreeMap<>();
                Map<String, Integer> wantAll = new TreeMap<>();
                got.getOrDefault(s, Map.of()).values().forEach(m -> m.forEach((k, v) -> all.merge(k, v, Integer::sum)));
                ex.deaths.getOrDefault(s, Map.of()).values().forEach(m -> m.forEach((k, v) -> wantAll.merge(k, v, Integer::sum)));
                Map<String, Integer> missing = minus(wantAll, all);
                Map<String, Integer> extra = minus(all, wantAll);
                if (!missing.isEmpty() || !extra.isEmpty()) {
                    JsonObject d = new JsonObject();
                    d.add("missing", expand(missing));
                    d.add("extra", expand(extra));
                    dd.add(s, d);
                }
            }
            if (dd.size() > 0) {
                diff.add("deathsInfo", dd);
                for (String s : dd.keySet()) keys.add("deaths:" + s);       // reported, not compared
            }
        }
        a.end = end;
        a.diff = diff;
        a.diffKeys = keys;
        a.nDiff = n + undone.size();
        boolean compared = !ex.life.isEmpty() || ex.hasHand || ex.hasBattlefield;
        a.ok = compared && run.failure == null && reached && !game.checkIfGameIsOver()
                && lifeDiff.size() == 0 && !diff.has("hand") && bfDiff.size() == 0
                && undoneMine == 0 && div.isEmpty();
    }

    static Map<String, Integer> minus(Map<String, Integer> a, Map<String, Integer> b) {
        Map<String, Integer> out = new TreeMap<>();
        for (Map.Entry<String, Integer> e : a.entrySet()) {
            int d = e.getValue() - b.getOrDefault(e.getKey(), 0);
            if (d > 0) out.put(e.getKey(), d);
        }
        return out;
    }

    static int sum(Map<String, Integer> m) {
        return m.values().stream().mapToInt(Integer::intValue).sum();
    }

    static JsonArray expand(Map<String, Integer> m) {
        JsonArray a = new JsonArray();
        for (Map.Entry<String, Integer> e : m.entrySet()) for (int i = 0; i < e.getValue(); i++) a.add(e.getKey());
        return a;
    }
}
