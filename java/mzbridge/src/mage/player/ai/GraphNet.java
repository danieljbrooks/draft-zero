package mage.player.ai;

import mage.game.Game;
import mage.player.ai.encoder.ActionEncoder;
import mage.players.Player;
import org.draftzero.mzbridge.GraphRecord;
import org.draftzero.mzbridge.graph.FeatureGraph;
import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessagePack;
import org.msgpack.core.MessageUnpacker;

import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

import static mage.target.TargetImpl.STOP_CHOOSING;

/**
 * A graph network for the search (docs/022 §3.3): the client of tools/imitation_scale/graph_server.py,
 * and the mapping from a search node to the graph encoder's question and back from per-node scores to
 * the node's options.
 *
 * The protocol is MageZero's graph server's (WillWroble/MageZero graph-encoder, server.py): a msgpack
 * POST /evaluate with the state's nodes and edges (indices, values, offsets, edge_child, edge_parent,
 * edge_label, edge_offsets), answered with one map per state: policy_priority and policy_target (one
 * score per node, in the order sent), policy_binary [no, yes] and value. Calls are synchronous: the
 * search is single-threaded per game, and the server batches across games.
 */
public final class GraphNet {

    /** One state's network output, its per-node scores keyed by the nodes' engine ids. */
    public static final class Out {
        public final float value;
        final float[] priority;
        final float[] target;
        final float[] use;
        final Map<UUID, Integer> index;

        Out(float value, float[] priority, float[] target, float[] use, Map<UUID, Integer> index) {
            this.value = value;
            this.priority = priority;
            this.target = target;
            this.use = use;
            this.index = index;
        }

        /** The policy of the decision the state was encoded at (null: no head reads it). */
        Policy policy(Ask q) {
            switch (q.type) {
                case PRIORITY:
                    return new Policy(q, scores(priority), null);
                case CHOOSE_TARGET:
                    return new Policy(q, scores(target), null);
                case CHOOSE_USE:
                    return q.attack ? new Policy(q, scores(target), null) : new Policy(q, null, use);
                default:
                    return null;
            }
        }

        private Map<UUID, Float> scores(float[] s) {
            Map<UUID, Float> out = new HashMap<>();
            for (Map.Entry<UUID, Integer> e : index.entrySet()) {
                float v = s[e.getValue()];
                if (!Float.isInfinite(v) && !Float.isNaN(v)) out.put(e.getKey(), v);
            }
            return out;
        }
    }

    /**
     * A node's policy over its options, kept world-independent: logits by the options' engine ids
     * (abilities, targets, players, Stop Choosing; they are the same objects in every belief world
     * for the searcher's own options), or the yes/no logits.
     */
    public static final class Policy {
        final Ask q;
        final Map<UUID, Float> byId;
        final float[] use;

        Policy(Ask q, Map<UUID, Float> byId, float[] use) {
            this.q = q;
            this.byId = byId;
            this.use = use;
        }

        /** The logit of one option, or null when its node isn't in this policy. */
        Double logit(MCTSNode c) {
            if (use != null) return (double) use[c.getUseAction() ? 1 : 0];
            UUID id;
            switch (q.type) {
                case PRIORITY:
                    id = c.getPriorityAction() == null ? null : GraphRecord.actionId(c.getPriorityAction());
                    break;
                case CHOOSE_TARGET:
                    id = c.getTargetAction();
                    break;
                case CHOOSE_USE: // an attack: no is Stop Choosing, yes the defending player
                    id = c.getUseAction() ? q.defender : STOP_CHOOSING;
                    break;
                default:
                    id = null;
            }
            Float v = id == null ? null : byId.get(id);
            return v == null ? null : (double) v;
        }

        /** Per group of options (copies of one option, as the trainer's option sets), the log-sum-exp of
         *  their logits; null if any option has no logit. */
        double[] logits(List<List<MCTSNode>> groups) {
            double[] out = new double[groups.size()];
            for (int i = 0; i < groups.size(); i++) {
                double mx = Double.NEGATIVE_INFINITY;
                double[] xs = new double[groups.get(i).size()];
                for (int j = 0; j < xs.length; j++) {
                    Double v = logit(groups.get(i).get(j));
                    if (v == null) return null;
                    xs[j] = v;
                    mx = Math.max(mx, v);
                }
                double s = 0;
                for (double x : xs) s += Math.exp(x - mx);
                out[i] = mx + Math.log(s);
            }
            return out;
        }
    }

    /**
     * The engine id of the graph node that stands for option `c` at a decision asked as `q`, as
     * Policy.logit reads it: a priority option's ability (Pass's fixed node), a target, Stop Choosing or
     * the defending player for an attack; null for a yes/no question (the use head reads it, no node).
     */
    static UUID optionNodeId(Ask q, MCTSNode c) {
        switch (q.type) {
            case PRIORITY:
                return c.getPriorityAction() == null ? null : GraphRecord.actionId(c.getPriorityAction());
            case CHOOSE_TARGET:
                return c.getTargetAction();
            case CHOOSE_USE:
                return q.attack ? (c.getUseAction() ? q.defender : STOP_CHOOSING) : null;
            default:
                return null;
        }
    }

    /** What the graph encoder is asked at a search node, from MageZero's paused simulation player. */
    static final class Ask {
        final ActionEncoder.ActionType type;
        final boolean attack;
        final UUID defender;
        final GraphRecord.Ask ask;

        Ask(ActionEncoder.ActionType type, boolean attack, UUID defender, GraphRecord.Ask ask) {
            this.type = type;
            this.attack = attack;
            this.defender = defender;
            this.ask = ask;
        }
    }

    private static final Field DECISION_TEXT;

    static {
        try {
            DECISION_TEXT = MCTSPlayer.class.getDeclaredField("decisionText");
            DECISION_TEXT.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException("MCTSPlayer.decisionText not found: the MageZero jars changed", e);
        }
    }

    /**
     * The graph encoder's question at a node: MageZero's decision type and text, with the graph-encoder
     * branch's form of attacks (a target choice, the attacker as DecisionSource) and blocks (the blocker
     * as DecisionSource), and the source of targets and "may" questions (GraphMCTSPlayer).
     */
    static Ask ask(MCTSNode eng) {
        Game g = eng.getGame();
        Player pl = g.getPlayer(eng.playerId);
        GraphMCTSPlayer gp = pl instanceof GraphMCTSPlayer ? (GraphMCTSPlayer) pl : null;
        String text;
        try {
            text = pl instanceof MCTSPlayer ? (String) DECISION_TEXT.get(pl) : null;
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        if (text == null) text = "priority";
        ActionEncoder.ActionType type = eng.actionType;
        List<List<UUID>> none = List.of();
        switch (type) {
            case PRIORITY:
                return new Ask(type, false, null, new GraphRecord.Ask("PRIORITY", "priority", null, null, none));
            case CHOOSE_USE:
                if (text.startsWith("attack with: ") && gp != null && gp.attacker != null) {
                    return new Ask(type, true, gp.defender, GraphRecord.Ask.attack(g, gp.attacker, gp.defender));
                }
                return new Ask(type, false, null, new GraphRecord.Ask("CHOOSE_USE", text, gp == null ? null : gp.decisionSource, null, none));
            case CHOOSE_TARGET:
                if (text.startsWith("choose which creature to block for ") && gp != null && gp.blocker != null) {
                    return new Ask(type, false, null, GraphRecord.Ask.block(gp.blocker, none));
                }
                return new Ask(type, false, null, new GraphRecord.Ask("CHOOSE_TARGET", text,
                        gp == null ? null : gp.decisionSource, gp == null ? null : gp.decisionCards, none));
            default:
                return new Ask(type, false, null, new GraphRecord.Ask(type.name(), text, null, null, none));
        }
    }

    private static final Map<String, GraphNet> CACHE = new HashMap<>();

    /** One client per server per JVM, health-checked once. */
    public static synchronized GraphNet of(String host, int port) {
        String k = host + ":" + port;
        GraphNet n = CACHE.get(k);
        if (n == null) {
            n = new GraphNet("http://" + host + ":" + port);
            CACHE.put(k, n);
        }
        return n;
    }

    private final HttpClient http;
    private final URI evalUri;
    public long calls;

    private GraphNet(String base) {
        this.http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).connectTimeout(Duration.ofSeconds(10)).build();
        this.evalUri = URI.create(base + "/evaluate");
        try {
            HttpResponse<String> r = http.send(HttpRequest.newBuilder(URI.create(base + "/healthz")).GET().build(),
                    HttpResponse.BodyHandlers.ofString());
            if (r.statusCode() != 200) throw new IllegalStateException("graph server health check: HTTP " + r.statusCode());
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("graph server at " + base + " is not answering: " + e, e);
        }
    }

    /** The network's output for the state at search node `eng`, from the searcher's seat. */
    Out infer(MCTSNode eng, Ask q) {
        FeatureGraph.GraphArrays a = GraphRecord.arrays(eng.getGame(), eng.targetPlayer, eng.playerId, q.ask, false);
        return infer(a);
    }

    public Out infer(FeatureGraph.GraphArrays a) {
        try {
            MessageBufferPacker pk = MessagePack.newDefaultBufferPacker();
            pk.packMapHeader(7);
            ints(pk, "indices", a.ids);
            ints(pk, "values", a.values);
            pk.packString("offsets").packArrayHeader(1).packLong(0);
            ints(pk, "edge_child", a.edgeChild);
            ints(pk, "edge_parent", a.edgeParent);
            ints(pk, "edge_label", a.edgeLabel);
            pk.packString("edge_offsets").packArrayHeader(1).packLong(0);
            pk.close();
            HttpRequest req = HttpRequest.newBuilder(evalUri).header("Content-Type", "application/x-msgpack")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(pk.toByteArray())).build();
            HttpResponse<byte[]> resp = http.send(req, HttpResponse.BodyHandlers.ofByteArray());
            if (resp.statusCode() != 200) throw new IllegalStateException("graph server: HTTP " + resp.statusCode() + " " + new String(resp.body()));
            calls++;
            MessageUnpacker up = MessagePack.newDefaultUnpacker(resp.body());
            int n = up.unpackArrayHeader();
            if (n != 1) throw new IllegalStateException("graph server answered " + n + " states for 1");
            float[] pri = null, tgt = null, use = null;
            float value = 0f;
            int fields = up.unpackMapHeader();
            for (int f = 0; f < fields; f++) {
                String k = up.unpackString();
                switch (k) {
                    case "policy_priority":
                        pri = floats(up);
                        break;
                    case "policy_target":
                        tgt = floats(up);
                        break;
                    case "policy_binary":
                        use = floats(up);
                        break;
                    case "value":
                        value = (float) up.unpackDouble();
                        break;
                    default:
                        up.skipValue();
                }
            }
            if (pri == null || tgt == null || use == null || pri.length != a.ids.length) {
                throw new IllegalStateException("graph server: missing or misaligned fields");
            }
            return new Out(value, pri, tgt, use, a.localIndex);
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("graph server call failed: " + e, e);
        }
    }

    private static void ints(MessageBufferPacker pk, String key, int[] xs) throws IOException {
        pk.packString(key);
        pk.packArrayHeader(xs.length);
        for (int x : xs) pk.packInt(x);
    }

    private static float[] floats(MessageUnpacker up) throws IOException {
        int n = up.unpackArrayHeader();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            switch (up.getNextFormat().getValueType()) {
                case FLOAT:
                    out[i] = (float) up.unpackDouble();
                    break;
                case NIL:
                    up.unpackNil();
                    out[i] = Float.NEGATIVE_INFINITY;
                    break;
                default:
                    out[i] = (float) up.unpackLong();
            }
        }
        return out;
    }
}
