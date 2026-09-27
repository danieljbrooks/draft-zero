package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One decision of the decision player: what MageZero is asked (type + text, exactly the strings
 * MCTSPlayer feeds to the StateEncoder) and the legal options with their MageZero labels and
 * policy-head indices. After a search it also holds the root's children.
 */
public class Decision {
    /** mage.player.ai.encoder.ActionEncoder.ActionType name: PRIORITY, CHOOSE_TARGET, CHOOSE_USE, MAKE_CHOICE, CHOOSE_NUM. */
    public String type;
    /** The decisionText MageZero encodes: "priority", "attack with: X?", "<rule>:Choose a target:<name>", ... */
    public String text;
    public String player;          // seat "A"/"B"
    public int turn;
    public String phase;
    public String step;
    public String activePlayer;
    public int stackSize;
    /**
     * Priority windows the decision player passed before this decision (only Pass legal, or before
     * options.decideFrom). Non-zero for a PRIORITY_* entry means the spec's own window was not the
     * decision: compare `where` with the spec before using the decision as a label.
     */
    public int passedBefore;
    /** For CHOOSE_NUM: labels are min..max, the MCTS amount action is an offset from min. */
    public int numMin;
    public final List<Option> legal = new ArrayList<>();

    // filled by a search
    public List<Child> children;
    public int rootVisits;
    public Double rootQ;
    public Double rootValue;
    public String best;
    public double searchSeconds;
    /** the root's StateEncoder features as MageZero's own search computed them (MCTSNode.stateVector) */
    public java.util.Set<Integer> rootFeatures;

    public static class Option {
        public String label;
        public int idx;
        public int n = 1;          // multiplicity (two copies of a card in hand give two "Cast X" options)
        public String source;      // source object name for abilities, when it is not already in the label

        Option(String label, int idx) {
            this.label = label;
            this.idx = idx;
        }
    }

    public static class Child {
        public String label;
        public int idx;
        public int visits;
        public Double q;           // mean value from the searching player's view; null when unvisited
        public double prior;

        Child(String label, int idx, int visits, Double q, double prior) {
            this.label = label;
            this.idx = idx;
            this.visits = visits;
            this.q = q;
            this.prior = prior;
        }
    }

    /** Add an option, merging duplicates of the same label. */
    void add(String label, int idx, String source) {
        for (Option o : legal) {
            if (o.label.equals(label)) {
                o.n++;
                return;
            }
        }
        Option o = new Option(label, idx);
        o.source = source;
        legal.add(o);
    }

    JsonObject toJson(boolean withChildren) {
        JsonObject d = new JsonObject();
        d.addProperty("player", player);
        d.addProperty("type", type);
        d.addProperty("text", text);
        JsonObject where = new JsonObject();
        where.addProperty("turn", turn);
        where.addProperty("phase", phase);
        where.addProperty("step", step);
        where.addProperty("activePlayer", activePlayer);
        where.addProperty("stack", stackSize);
        where.addProperty("passedBefore", passedBefore);
        d.add("where", where);
        JsonArray arr = new JsonArray();
        for (Option o : legal) {
            JsonObject j = new JsonObject();
            j.addProperty("label", o.label);
            j.addProperty("idx", o.idx);
            if (o.n > 1) j.addProperty("n", o.n);
            if (o.source != null) j.addProperty("source", o.source);
            arr.add(j);
        }
        d.add("legal", arr);
        if (withChildren && children != null) {
            JsonArray ch = new JsonArray();
            for (Child c : children) ch.add(childJson(c));
            d.add("children", ch);
            d.addProperty("rootVisits", rootVisits);
            d.addProperty("rootQ", rootQ);
            d.addProperty("value", rootValue);
            d.addProperty("best", best);
            d.addProperty("searchSeconds", Math.round(searchSeconds * 1000.0) / 1000.0);
        }
        return d;
    }

    static JsonObject childJson(Child c) {
        JsonObject j = new JsonObject();
        j.addProperty("label", c.label);
        j.addProperty("idx", c.idx);
        j.addProperty("N", c.visits);
        j.addProperty("Q", c.q);
        j.addProperty("prior", c.prior);
        return j;
    }

    /** Children merged by label: visits summed, Q visit-weighted (duplicate cards are one action). */
    Map<String, Child> mergedChildren() {
        Map<String, Child> out = new LinkedHashMap<>();
        Map<String, Double> qsum = new LinkedHashMap<>();
        for (Child c : children) {
            Child m = out.get(c.label);
            if (m == null) {
                m = new Child(c.label, c.idx, 0, null, 0);
                out.put(c.label, m);
                qsum.put(c.label, 0.0);
            }
            m.visits += c.visits;
            m.prior += c.prior;
            if (c.q != null && c.visits > 0) qsum.put(c.label, qsum.get(c.label) + c.q * c.visits);
        }
        for (Child m : out.values()) {
            m.q = m.visits > 0 ? qsum.get(m.label) / m.visits : null;
        }
        return out;
    }
}
