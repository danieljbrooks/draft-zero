package org.draftzero.mzbridge;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.cards.repository.CardRepository;
import mage.cards.repository.RepositoryUtil;
import mage.game.Game;
import mage.player.ai.ComputerPlayerMCTS2;
import mage.player.ai.encoder.ActionEncoder;
import mage.player.ai.encoder.StateEncoder;
import mage.player.ai.score.GameStateEvaluator3;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Long-lived bridge worker: JSON lines in on stdin, one JSON line out per request on stdout,
 * logs on stderr. Run it from a directory holding a private copy of xmage/db (H2 is opened by a
 * relative path and cannot be shared between JVMs; critique C3). See README.md for the protocol.
 *
 *   request  {"id": 1, "op": "build", "spec": {...StateSpec v1...}, "options": {...}}
 *   response {"id": 1, "op": "build", "ok": true, ...}   or   {"id": 1, "ok": false, "error": {...}}
 *
 * On start it prints one {"event": "ready", ...} line once the card database is open.
 */
public final class Worker {
    static final Gson GSON = new GsonBuilder().serializeNulls().disableHtmlEscaping().create();
    static final Gson SPEC_GSON = new Gson();
    public static final String VERSION = "mzbridge 1 (StateSpec v1)";

    private Worker() {
    }

    public static void main(String[] args) throws Exception {
        long jvmStart = ManagementFactory.getRuntimeMXBean().getStartTime();
        long mainAt = System.currentTimeMillis();
        long t0 = System.nanoTime();
        // before anything creates a UUID (card database, decks) or the ids cannot be made reproducible
        boolean ids = DeterministicIds.install();
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8);
        System.setOut(System.err); // stray prints from XMage must not corrupt the protocol
        if (!Files.isRegularFile(Paths.get("db", "cards.h2.mv.db"))) {
            JsonObject r = new JsonObject();
            r.addProperty("event", "ready");
            r.addProperty("ok", false);
            r.addProperty("error", "no db/cards.h2.mv.db in the working directory " + Paths.get("").toAbsolutePath()
                    + " (run the worker through java/mzbridge/run.sh, which prepares a private copy of xmage/db)");
            out.println(GSON.toJson(r));
            System.exit(2);
        }
        long tdb = System.nanoTime();
        RepositoryUtil.bootstrapLocalDb();
        long dbMs = (System.nanoTime() - tdb) / 1_000_000;
        ComputerPlayerMCTS2.SHOW_THREAD_INFO = false;

        JsonObject ready = new JsonObject();
        ready.addProperty("event", "ready");
        ready.addProperty("ok", true);
        ready.addProperty("version", VERSION);
        ready.addProperty("jvm_to_main_ms", mainAt - jvmStart);
        ready.addProperty("db_ms", dbMs);
        ready.addProperty("startup_ms", (System.nanoTime() - t0) / 1_000_000 + (mainAt - jvmStart));
        ready.addProperty("deterministicIds", ids);
        ready.addProperty("actionVocab", ActionEncoder.VOCAB_PATH);
        ready.addProperty("actionDim", ActionEncoder.ACTION_DIM);
        out.println(GSON.toJson(ready));

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        boolean quit = false;
        while (!quit && (line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            JsonObject resp;
            JsonElement id = null;
            String op = null;
            long tr = System.nanoTime();
            try {
                JsonObject req = JsonParser.parseString(line).getAsJsonObject();
                id = req.get("id");
                op = optString(req, "op", null);
                JsonObject opt = req.has("options") && req.get("options").isJsonObject() ? req.getAsJsonObject("options") : new JsonObject();
                if (op == null) throw new IllegalArgumentException("request has no op");
                switch (op) {
                    case "ping":
                        resp = new JsonObject();
                        resp.addProperty("version", VERSION);
                        break;
                    case "build":
                        resp = build(spec(req), opt);
                        break;
                    case "encode":
                        resp = encode(spec(req), opt);
                        break;
                    case "coach":
                        // with options.specs (pre-determinized specs) the request spec is optional
                        resp = Coach.run(req.has("spec") && !req.get("spec").isJsonNull() ? spec(req) : null, opt);
                        break;
                    case "bench":
                        resp = Bench.run(opt); // docs/012's search benchmark (Bench.java)
                        break;
                    case "play":
                        resp = Play.run(opt); // one full game, IS-MCTS for both seats (docs/017)
                        break;
                    case "replay_turn":
                        resp = TurnReplay.run(spec(req), opt); // one recorded turn, both seats scripted (README)
                        break;
                    case "quit":
                        resp = new JsonObject();
                        quit = true;
                        break;
                    default:
                        throw new IllegalArgumentException("unknown op '" + op + "' (ping, build, encode, coach, bench, play, replay_turn, quit)");
                }
                resp.addProperty("ok", true);
            } catch (Throwable e) {
                resp = error(e);
            } finally {
                DeterministicIds.off();
            }
            JsonObject full = new JsonObject();
            full.add("id", id);
            full.addProperty("op", op);
            for (Map.Entry<String, JsonElement> e : resp.entrySet()) full.add(e.getKey(), e.getValue());
            if (!full.has("timing_ms")) {
                JsonObject t = new JsonObject();
                t.addProperty("total", Math.round((System.nanoTime() - tr) / 1e6));
                full.add("timing_ms", t);
            }
            out.println(GSON.toJson(full));
        }
        Coach.closeEvaluators();
        try {
            // SHUTDOWN IMMEDIATELY: no writes. closeDB(true) would run SHUTDOWN COMPACT, which rewrites
            // the whole 290 MB file (breaking the clone's block sharing) and leaves a truncated
            // database behind if the process is killed while it runs.
            CardRepository.instance.closeDB(false);
        } catch (Exception ignored) {
        }
        System.exit(0);
    }

    // ------------------------------------------------------------------ ops

    static JsonObject build(Spec spec, JsonObject opt) {
        long t0 = System.nanoTime();
        long seed = optLong(opt, "seed", 0L);
        long idSeed = optLong(opt, "idSeed", seed);
        Substitutions.Result subs = Substitutions.apply(spec, optObject(opt, "substitute"));
        StateInjector.Built b = StateInjector.build(spec, idSeed, seed, optBool(opt, "lenient", false));
        b.warnings.addAll(0, subs.warnings);
        long t1 = System.nanoTime();
        JsonObject dump = Dumper.dump(b, optBool(opt, "dumpLibrary", false));
        String seat = decisionSeat(spec, opt);
        JsonObject[] atDecision = new JsonObject[1];
        boolean dumpAt = optBool(opt, "dumpDecisionState", false);
        BridgePlayer.Listener l = dumpAt ? (game, p, d) -> atDecision[0] = Dumper.dump(b, false) : null;
        String[] reason = new String[1];
        Decision d = optBool(opt, "advance", true) ? advance(b, seat, opt, BridgePlayer.Mode.CAPTURE, l, reason) : null;
        long t2 = System.nanoTime();
        JsonObject r = new JsonObject();
        r.addProperty("seed", seed);
        r.add("dump", dump);
        r.add("decision", d == null ? null : d.toJson(false));
        if (d == null && reason[0] != null) r.addProperty("noDecision", reason[0]);
        if (atDecision[0] != null) r.add("decisionState", atDecision[0]);
        r.add("warnings", Dumper.strings(b.warnings));
        if (!subs.applied.isEmpty()) r.add("substitutions", subs.toJson());
        JsonObject t = new JsonObject();
        t.addProperty("build", ms(t0, t1));
        t.addProperty("advance", ms(t1, t2));
        t.addProperty("total", ms(t0, t2));
        r.add("timing_ms", t);
        return r;
    }

    static JsonObject encode(Spec spec, JsonObject opt) {
        long t0 = System.nanoTime();
        long seed = optLong(opt, "seed", 0L);
        boolean perfectInfo = optBool(opt, "perfectInfo", false);
        Substitutions.Result subs = Substitutions.apply(spec, optObject(opt, "substitute"));
        StateInjector.Built b = StateInjector.build(spec, optLong(opt, "idSeed", seed), seed, optBool(opt, "lenient", false));
        b.warnings.addAll(0, subs.warnings);
        long t1 = System.nanoTime();
        // the built state before advancing, as the build op dumps it (labels map spec aliases to names)
        JsonObject dump = optBool(opt, "dump", false) ? Dumper.dump(b, false) : null;
        String seat = decisionSeat(spec, opt);
        boolean heuristic = optBool(opt, "heuristic", false);
        int[][] features = new int[1][];
        double[] encMs = new double[1];
        Double[] heur = new Double[1];
        BridgePlayer.Listener l = (game, p, d) -> {
            long te = System.nanoTime();
            StateEncoder enc = new StateEncoder();
            UUID me = p.getId(); // the same object as the decision player id: StateEncoder compares with ==
            enc.setAgent(me);
            enc.setOpponent(game.getOpponent(me).getId());
            enc.perfectInfo = perfectInfo;
            Set<Integer> fv = enc.processState(game, me, ActionEncoder.ActionType.valueOf(d.type), d.text);
            features[0] = fv.stream().mapToInt(Integer::intValue).sorted().toArray();
            encMs[0] = (System.nanoTime() - te) / 1e6;
            // offline MageZero's leaf score (docs/017's value pilot), from the decision player's seat
            if (heuristic) heur[0] = GameStateEvaluator3.evaluateNormalized(me, game);
        };
        String[] reason = new String[1];
        Decision d = advance(b, seat, opt, BridgePlayer.Mode.CAPTURE, l, reason);
        long t2 = System.nanoTime();
        JsonObject r = new JsonObject();
        r.addProperty("seed", seed);
        r.addProperty("perfectInfo", perfectInfo);
        r.add("decision", d == null ? null : d.toJson(false));
        if (d == null) r.addProperty("noDecision", reason[0]);
        JsonArray fa = new JsonArray();
        if (features[0] != null) for (int f : features[0]) fa.add(f);
        r.add("features", fa);
        r.addProperty("nFeatures", fa.size());
        if (heur[0] != null) r.addProperty("heuristic", heur[0]);
        if (dump != null) r.add("dump", dump);
        r.add("warnings", Dumper.strings(b.warnings));
        if (!subs.applied.isEmpty()) r.add("substitutions", subs.toJson());
        JsonObject t = new JsonObject();
        t.addProperty("build", ms(t0, t1));
        t.addProperty("advance", ms(t1, t2));
        t.addProperty("encode", Math.round(encMs[0] * 10) / 10.0);
        t.addProperty("total", ms(t0, t2));
        r.add("timing_ms", t);
        return r;
    }

    /**
     * Resume the built game until the decision player's first non-trivial decision. The other seat
     * is a puppet. Returns null (with a reason) if the game ends or the safety stop is reached first.
     */
    static Decision advance(StateInjector.Built b, String seat, JsonObject opt, BridgePlayer.Mode mode,
                            BridgePlayer.Listener l, String[] reason) {
        BridgePlayer d = b.players.get(seat);
        BridgePlayer o = b.players.get(Spec.other(seat));
        d.role = BridgePlayer.Role.DECIDER;
        d.mode = mode;
        d.preLand = opt == null ? null : optString(opt, "preLand", null);
        d.listener = l;
        if (opt != null && opt.has("decideFrom") && opt.get("decideFrom").isJsonObject()) {
            JsonObject from = opt.getAsJsonObject("decideFrom");
            d.fromTurn = optInt(from, "turn", b.spec.turn);
            String step = optString(from, "step", null);
            d.fromStep = step == null ? -1 : Spec.turnSteps().indexOf(step);
            if (step != null && d.fromStep < 0) throw new IllegalArgumentException("decideFrom.step '" + step + "' is not a PhaseStep");
            if (d.fromTurn > b.spec.turn + 1) {
                throw new IllegalArgumentException("decideFrom.turn " + d.fromTurn + " is past the safety stop (end of turn " + (b.spec.turn + 1) + ")");
            }
        }
        o.role = BridgePlayer.Role.PUPPET;
        o.mode = mode; // the puppet re-anchors at MageZero's checkpoints too (BridgePlayer.checkpoint)
        StateInjector.anchor(b, mode != BridgePlayer.Mode.CAPTURE);
        DeterministicIds.reset(StateInjector.mix(b.idSeed, 5));
        Game game = b.game;
        try {
            game.resume();
        } catch (RuntimeException e) {
            if (d.failure != null) throw d.failure;
            throw e;
        }
        if (d.failure != null) throw d.failure;
        if (d.decision == null) {
            if (game.checkIfGameIsOver()) {
                reason[0] = "the game ended before " + seat + " had a decision (winner: " + game.getWinner() + ")";
            } else {
                reason[0] = "no decision for " + seat + " with 2+ options before the end of turn " + (b.spec.turn + 1);
                if (opt == null || !opt.has("decisionPlayer")) {
                    // the usual cause: an end-of-turn snapshot of the other seat's turn (17lands eot_rollover)
                    reason[0] += " (decisionPlayer defaulted to the spec's " + (b.spec.priorityPlayer != null ? "priorityPlayer" : "activePlayer")
                            + " " + seat + "; pass options.decisionPlayer, and decideFrom for an end-of-turn entry)";
                }
            }
        }
        return d.decision;
    }

    // ------------------------------------------------------------------ helpers

    static Spec spec(JsonObject req) {
        if (!req.has("spec") || !req.get("spec").isJsonObject()) throw new IllegalArgumentException("request has no spec object");
        return specFrom(req.get("spec"));
    }

    /** A StateSpec JSON object -> Spec, with every absent list defaulted to empty. */
    static Spec specFrom(JsonElement json) {
        if (json == null || !json.isJsonObject()) throw new IllegalArgumentException("a spec must be a JSON object");
        Spec s = SPEC_GSON.fromJson(json, Spec.class);
        if (s.players == null) s.players = new LinkedHashMap<>();
        if (s.stack == null) s.stack = new ArrayList<>();
        if (s.attackers == null) s.attackers = new ArrayList<>();
        if (s.blockers == null) s.blockers = new ArrayList<>();
        if (s.passedPlayers == null) s.passedPlayers = new ArrayList<>();
        for (Spec.PlayerState p : s.players.values()) {
            if (p == null) continue;
            if (p.decklist == null) p.decklist = new ArrayList<>();
            if (p.hand == null) p.hand = new ArrayList<>();
            if (p.graveyard == null) p.graveyard = new ArrayList<>();
            if (p.exile == null) p.exile = new ArrayList<>();
            if (p.libraryTop == null) p.libraryTop = new ArrayList<>();
            if (p.battlefield == null) p.battlefield = new ArrayList<>();
            if (p.decklistSource == null) p.decklistSource = "exact";
            for (Spec.Perm perm : p.battlefield) if (perm.counters == null) perm.counters = new LinkedHashMap<>();
        }
        for (Spec.StackItem si : s.stack) if (si.targets == null) si.targets = new ArrayList<>();
        return s;
    }

    /** options.decisionPlayer, else the spec's priorityPlayer, else its active player. */
    static String decisionSeat(Spec spec, JsonObject opt) {
        String s = optString(opt, "decisionPlayer", null);
        if (s == null) s = spec.priorityPlayer != null ? spec.priorityPlayer : spec.activePlayer;
        if (!Spec.SEATS.contains(s)) throw new IllegalArgumentException("decisionPlayer must be A or B, got '" + s + "'");
        return s;
    }

    static JsonObject error(Throwable e) {
        JsonObject r = new JsonObject();
        r.addProperty("ok", false);
        JsonObject err = new JsonObject();
        err.addProperty("type", e.getClass().getSimpleName());
        err.addProperty("message", String.valueOf(e.getMessage()));
        if (e instanceof StateInjector.SpecException) {
            err.add("problems", Dumper.strings(((StateInjector.SpecException) e).problems));
        } else {
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String[] lines = sw.toString().split("\n");
            err.addProperty("trace", String.join("\n", Arrays.asList(lines).subList(0, Math.min(12, lines.length))));
            System.err.println("[mzbridge] request failed: " + sw);
        }
        r.add("error", err);
        return r;
    }

    static long ms(long a, long b) {
        return Math.round((b - a) / 1e6);
    }

    static JsonObject optObject(JsonObject o, String k) {
        return o != null && o.has(k) && o.get(k).isJsonObject() ? o.getAsJsonObject(k) : null;
    }

    static String optString(JsonObject o, String k, String def) {
        return o != null && o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsString() : def;
    }

    static int optInt(JsonObject o, String k, int def) {
        return o != null && o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsInt() : def;
    }

    static long optLong(JsonObject o, String k, long def) {
        return o != null && o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsLong() : def;
    }

    static double optDouble(JsonObject o, String k, double def) {
        return o != null && o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsDouble() : def;
    }

    static boolean optBool(JsonObject o, String k, boolean def) {
        return o != null && o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsBoolean() : def;
    }
}
