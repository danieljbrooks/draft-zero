package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import mage.abilities.Ability;
import mage.abilities.common.PassAbility;
import mage.cards.Cards;
import mage.game.Game;
import mage.player.ai.encoder.ActionEncoder;
import mage.target.common.TargetAttackingCreature;
import mage.target.common.TargetDefender;
import org.draftzero.mzbridge.graph.FeatureGraph;
import org.draftzero.mzbridge.graph.StateEncoder;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

import static mage.target.TargetImpl.STOP_CHOOSING;

/**
 * A decision state as MageZero's graph encoder sees it (the vendored copy in graph/, docs/022 §3.1):
 * the state graph in the per-state arrays of MageZero's LabeledStateWriter, and each legal option as
 * the graph nodes that stand for it. Arrays travel as base64 of little-endian int32 (JSON int lists
 * would be ~3x larger).
 *
 * The graph encoder's own players (WillWroble/mage graph-encoder) ask two questions differently
 * from v0.2, and the graph keeps their form, so a network trained here reads states as theirs do:
 *   - an attack is a target choice for the attacker (ChooseToAttackAbility): the defending player
 *     or Stop Choosing, with the attacker as the DecisionSource (v0.2: "attack with: X?" yes/no);
 *   - a block names its blocker by DecisionSource, not inside the text.
 */
public final class GraphRecord {
    static final String ATTACK_TEXT = "attack with: {this} ?";
    static final String BLOCK_TEXT = "choose which creature to block for {this}";

    /** A decision as the graph encoder is asked it. */
    public static final class Ask {
        public final String type;
        public final String text;
        public final UUID source;
        public final Cards fromCards;
        /** Per legal option (in the decision's legal order), the engine ids of its nodes. */
        public final List<List<UUID>> options;

        public Ask(String type, String text, UUID source, Cards fromCards, List<List<UUID>> options) {
            this.type = type;
            this.text = text;
            this.source = source;
            this.fromCards = fromCards;
            this.options = options;
        }

        /** A priority, target or "may" decision, as the bridge's Decision records it. */
        static Ask of(Decision d) {
            return new Ask(d.type, d.text, d.source, d.fromCards, d.optionIds());
        }

        /** "Attack with this creature?" with options [no, yes]: a target choice between Stop
         *  Choosing and the defending player (MageZero's ComputerPlayer.selectAttackersOneAtATime
         *  on the graph-encoder branch). */
        public static Ask attack(Game game, UUID attacker, UUID defendingPlayer) {
            String text = ATTACK_TEXT + ":Choose a target:" + new TargetDefender(game.getCombat().getDefenders()).getTargetName();
            List<List<UUID>> options = new ArrayList<>();
            options.add(List.of(STOP_CHOOSING));
            options.add(List.of(defendingPlayer));
            return new Ask("CHOOSE_TARGET", text, attacker, null, options);
        }

        /** "Which attacker does this creature block?", options in the decision's legal order. */
        public static Ask block(UUID blocker, List<List<UUID>> options) {
            String text = BLOCK_TEXT + ":Choose a target:" + new TargetAttackingCreature(0, 1).getTargetName();
            return new Ask("CHOOSE_TARGET", text, blocker, null, options);
        }
    }

    private GraphRecord() {
    }

    /** The node id of a priority action: the ability's own id, except Pass. The graph encoder has one
     *  Pass node with a fixed id (FeatureGraph.PASS_ABILITY_ID), which the graph-encoder branch's
     *  PassAbility takes as its id; v0.2's PassAbility has a random one. */
    public static UUID actionId(Ability a) {
        return a instanceof PassAbility ? FeatureGraph.PASS_ABILITY_ID : a.getId();
    }

    /**
     * The graph of `game` from `me`'s seat (`me` is the encoder's agent and the decision player)
     * with the opponent's hand hidden unless `perfectInfo`. Fields: type, text; nodes as ids and
     * values; edges as child, parent and label (node indices local to this graph); options, one
     * array of node indices per legal option; missing, the options with no node (absent when 0).
     */
    static JsonObject encode(Game game, UUID me, Ask ask, boolean perfectInfo) {
        FeatureGraph.GraphArrays a = arrays(game, me, me, ask, perfectInfo);
        JsonObject r = new JsonObject();
        r.addProperty("type", ask.type);
        r.addProperty("text", ask.text);
        r.addProperty("ids", b64(a.ids));
        r.addProperty("values", b64(a.values));
        r.addProperty("edge_child", b64(a.edgeChild));
        r.addProperty("edge_parent", b64(a.edgeParent));
        r.addProperty("edge_label", b64(a.edgeLabel));
        JsonArray opts = new JsonArray();
        int missing = 0;
        for (List<UUID> ids : ask.options) {
            JsonArray o = new JsonArray();
            for (UUID id : ids) {
                Integer k = a.localIndex.get(id);
                if (k != null) o.add(k);
            }
            if (o.size() == 0) missing++;
            opts.add(o);
        }
        r.add("options", opts);
        if (missing > 0) r.addProperty("missing", missing);
        return r;
    }

    /**
     * The state graph of `game`: `agent` is the encoder's agent (PlayerA: the seat whose view this is,
     * whose value the network predicts) and `decisionPlayer` the player deciding (only its hand is
     * shown, unless perfectInfo). In the imitation data they are the same player; a search also
     * scores the opponent's decisions from the searcher's seat, as its flat encodings do.
     */
    public static FeatureGraph.GraphArrays arrays(Game game, UUID agent, UUID decisionPlayer, Ask ask, boolean perfectInfo) {
        StateEncoder enc = new StateEncoder();
        enc.setAgent(agent);
        enc.setOpponent(game.getOpponent(agent).getId());
        enc.perfectInfo = perfectInfo;
        FeatureGraph g = enc.processState(game, decisionPlayer, ActionEncoder.ActionType.valueOf(ask.type), ask.text,
                ask.fromCards, ask.source);
        return g.getGraphArrays();
    }

    static String b64(int[] xs) {
        ByteBuffer bb = ByteBuffer.allocate(4 * xs.length).order(ByteOrder.LITTLE_ENDIAN);
        for (int x : xs) bb.putInt(x);
        return Base64.getEncoder().encodeToString(bb.array());
    }
}
