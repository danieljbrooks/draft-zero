package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.*;

/**
 * What the replay_turn op replays: one turn of the scripted seat (the 17lands user, A) and what
 * the other seat (B) was recorded doing during it, plus the end-of-turn snapshot the replay must
 * reach. Parsed from options.script and options.expected (see README.md, replay_turn).
 *
 * 17lands records per-turn multisets with no order, phase or targets, so the script is a
 * partially ordered set: lands first, then spells and activations in the order a {@link Policy}
 * imposes, each one tried at every later window until it is legal (a cantrip's card is cast after
 * the cantrip drew it). Attacks and blocks are exact per creature; B's instants get a window.
 */
final class TurnScript {

    /** Windows of the replayed turn, in turn order ("respond" is any time an A spell is on the stack). */
    static final List<String> WINDOWS = List.of("upkeep", "main1", "after_attackers", "after_blockers", "main2", "end_step");

    static final class Item {
        String seat;          // "A" or "B"
        String kind;          // land | cast | activation
        String key;           // MageZero label: "Play Plains", "Cast X", "Flashback {2}{U}", ability rule text
        String name;          // card name (casts, lands) or source name (activations)
        String family;        // creature | noncreature | instant_sorcery | flash (B's inferred flash permanents)
        String window;        // requested window; null = decided from the card (TurnReplay.defaultWindow)
        int mv = -1;          // mana value (-1: read from the card)
        int order;            // position in the policy order
        // replay state
        boolean done;
        String failedAt;      // "turn:stepNum" of a failed activation: not retried in that step
        String doneAt;        // "<step>" where it was taken

        String describe() {
            return seat + ": " + key + (kind.equals("activation") && name != null ? " [" + name + "]" : "");
        }

        JsonObject toJson() {
            JsonObject o = new JsonObject();
            o.addProperty("seat", seat);
            o.addProperty("kind", kind);
            o.addProperty("key", key);
            if (name != null) o.addProperty("name", name);
            if (window != null) o.addProperty("window", window);
            o.addProperty("done", done);
            if (doneAt != null) o.addProperty("at", doneAt);
            return o;
        }
    }

    /** One block alternative: blocker ref ("B:alias" / "new:Name") -> attacker ref ("A:alias" / "new:Name") or null. */
    static final class Block {
        String blocker;
        String attacker;
    }

    /** One attempt's choices (the bounded search over orderings and windows). */
    static final class Policy {
        String order = "mv_asc";       // mv_asc | mv_desc | listed
        String aMain = "main1";        // main1 | creatures_main2 | main2: where A's non-land spells go
        String aInstants = "main";     // main | combat: A's instants (combat = after blockers)
        String bWindow = "auto";       // auto | one window for all of B's items
        int pairing = 0;               // index into the block alternatives
        int mode = 0;                  // modal spells: the k-th available mode
        String may = "default";        // "may" questions in the turn: default (the puppet's) | yes | no
        String targetOrder = "high";   // heuristic target ties: high (biggest creature first) | low
        String sacMana = "keep";       // Treasure-style mana: keep (spend only when needed) | use (auto-tap's choice)
        String land = "first";         // the land drop: first | last (after the main phase's spells: landfall, lifegain lands)
        String defender = "auto";      // attacks: auto (a planeswalker the snapshot kills, else the player) | planeswalker

        Policy with(java.util.function.Consumer<Policy> change) {
            Policy p = new Policy();
            p.order = order;
            p.aMain = aMain;
            p.aInstants = aInstants;
            p.bWindow = bWindow;
            p.pairing = pairing;
            p.mode = mode;
            p.may = may;
            p.targetOrder = targetOrder;
            p.sacMana = sacMana;
            p.land = land;
            p.defender = defender;
            change.accept(p);
            return p;
        }

        String describe() {
            return "order=" + order + " aMain=" + aMain + " aInstants=" + aInstants + " bWindow=" + bWindow + " pairing=" + pairing
                    + " mode=" + mode + " may=" + may + " targetOrder=" + targetOrder + " sacMana=" + sacMana + " land=" + land
                    + " defender=" + defender;
        }

        JsonObject toJson() {
            JsonObject o = new JsonObject();
            o.addProperty("order", order);
            o.addProperty("aMain", aMain);
            o.addProperty("aInstants", aInstants);
            o.addProperty("bWindow", bWindow);
            o.addProperty("pairing", pairing);
            o.addProperty("mode", mode);
            o.addProperty("may", may);
            o.addProperty("targetOrder", targetOrder);
            o.addProperty("sacMana", sacMana);
            o.addProperty("land", land);
            o.addProperty("defender", defender);
            return o;
        }

        static Policy from(JsonObject o) {
            Policy p = new Policy();
            p.order = Worker.optString(o, "order", p.order);
            p.aMain = Worker.optString(o, "aMain", p.aMain);
            p.aInstants = Worker.optString(o, "aInstants", p.aInstants);
            p.bWindow = Worker.optString(o, "bWindow", p.bWindow);
            p.pairing = Worker.optInt(o, "pairing", p.pairing);
            p.mode = Worker.optInt(o, "mode", p.mode);
            p.may = Worker.optString(o, "may", p.may);
            p.targetOrder = Worker.optString(o, "targetOrder", p.targetOrder);
            p.sacMana = Worker.optString(o, "sacMana", p.sacMana);
            p.land = Worker.optString(o, "land", p.land);
            p.defender = Worker.optString(o, "defender", p.defender);
            return p;
        }
    }

    /** The end-of-turn snapshot: canonical permanent names per seat, A's hand, life totals, deaths. */
    static final class Expected {
        final Map<String, Map<String, Integer>> battlefield = new LinkedHashMap<>();
        final Map<String, Integer> handA = new TreeMap<>();
        int handUnknownA;
        final Map<String, Integer> life = new LinkedHashMap<>();
        /** seat -> "combat"/"noncombat" -> names of non-token creatures that died */
        final Map<String, Map<String, Map<String, Integer>>> deaths = new LinkedHashMap<>();
        boolean hasHand, hasBattlefield;
    }

    String seat = "A";
    int turn;                                   // global turn number of the replayed turn
    final List<Item> items = new ArrayList<>();
    /** attack plan: "A:alias" -> attacked; "new:Name" -> attacked (a creature that entered this turn) */
    final Map<String, Boolean> attacks = new LinkedHashMap<>();
    /** names whose attacking copy is a guess (labels.attack_notes) */
    final Set<String> attackGuess = new HashSet<>();
    final List<List<Block>> blocks = new ArrayList<>();
    String blockPairing = "none";
    /** recorded actions of the scripted seat the request could not script (an ability with no XMage
     *  key): the replay cannot do them, so it never labels a Pass "exact" and never reproduces */
    int unscripted;
    Expected expected;

    static TurnScript parse(JsonObject script, JsonObject expected, int defaultTurn) {
        if (script == null) throw new IllegalArgumentException("replay_turn needs options.script");
        TurnScript s = new TurnScript();
        s.seat = Worker.optString(script, "seat", "A");
        if (!Spec.SEATS.contains(s.seat)) throw new IllegalArgumentException("script.seat must be A or B");
        s.turn = Worker.optInt(script, "turn", defaultTurn);
        int k = 0;
        for (String list : List.of("lands", "casts", "activations", "opp")) {
            if (!script.has(list)) continue;
            for (JsonElement e : script.getAsJsonArray(list)) {
                JsonObject o = e.getAsJsonObject();
                Item it = new Item();
                it.seat = Worker.optString(o, "seat", list.equals("opp") ? Spec.other(s.seat) : s.seat);
                it.kind = Worker.optString(o, "kind", list.equals("lands") ? "land" : list.equals("activations") ? "activation" : "cast");
                it.key = Worker.optString(o, "key", null);
                it.name = Worker.optString(o, "name", Worker.optString(o, "source", null));
                it.family = Worker.optString(o, "family", null);
                it.window = Worker.optString(o, "window", null);
                it.mv = Worker.optInt(o, "mv", -1);
                it.order = k++;
                if (it.key == null) throw new IllegalArgumentException("script item without a key: " + o);
                if (it.window != null && !WINDOWS.contains(it.window) && !it.window.equals("respond")) {
                    throw new IllegalArgumentException("unknown window '" + it.window + "' (" + WINDOWS + ", respond)");
                }
                s.items.add(it);
            }
        }
        JsonObject att = Worker.optObject(script, "attacks");
        if (att != null) for (Map.Entry<String, JsonElement> e : att.entrySet()) s.attacks.put(e.getKey(), e.getValue().getAsBoolean());
        if (script.has("attackGuess")) for (JsonElement e : script.getAsJsonArray("attackGuess")) s.attackGuess.add(e.getAsString());
        if (script.has("blocks")) {
            // [[[blocker, attacker|null], ...], ...]: alternatives of complete pairings
            for (JsonElement alt : script.getAsJsonArray("blocks")) {
                List<Block> one = new ArrayList<>();
                for (JsonElement pair : alt.getAsJsonArray()) {
                    JsonArray p = pair.getAsJsonArray();
                    Block b = new Block();
                    b.blocker = p.get(0).getAsString();
                    b.attacker = p.size() > 1 && !p.get(1).isJsonNull() ? p.get(1).getAsString() : null;
                    one.add(b);
                }
                s.blocks.add(one);
            }
        }
        s.blockPairing = Worker.optString(script, "blockPairing", s.blockPairing);
        s.unscripted = Worker.optInt(script, "unscripted", 0);
        s.expected = parseExpected(expected);
        return s;
    }

    static Expected parseExpected(JsonObject o) {
        Expected x = new Expected();
        if (o == null) return x;
        JsonObject life = Worker.optObject(o, "life");
        if (life != null) for (Map.Entry<String, JsonElement> e : life.entrySet()) x.life.put(e.getKey(), e.getValue().getAsInt());
        JsonObject bf = Worker.optObject(o, "battlefield");
        if (bf != null) {
            x.hasBattlefield = true;
            for (Map.Entry<String, JsonElement> e : bf.entrySet()) x.battlefield.put(e.getKey(), counts(e.getValue().getAsJsonArray()));
        }
        JsonObject hand = Worker.optObject(o, "hand");
        if (hand != null && hand.has("A")) {
            x.hasHand = true;
            x.handA.putAll(counts(hand.getAsJsonArray("A")));
            x.handUnknownA = Worker.optInt(hand, "unknownA", 0);
        }
        JsonObject deaths = Worker.optObject(o, "deaths");
        if (deaths != null) {
            for (Map.Entry<String, JsonElement> e : deaths.entrySet()) {
                Map<String, Map<String, Integer>> m = new LinkedHashMap<>();
                for (Map.Entry<String, JsonElement> f : e.getValue().getAsJsonObject().entrySet()) {
                    m.put(f.getKey(), counts(f.getValue().getAsJsonArray()));
                }
                x.deaths.put(e.getKey(), m);
            }
        }
        return x;
    }

    static Map<String, Integer> counts(JsonArray a) {
        Map<String, Integer> m = new TreeMap<>();
        for (JsonElement e : a) m.merge(e.getAsString(), 1, Integer::sum);
        return m;
    }

    /** A fresh copy of the items for one attempt (replay state reset). */
    List<Item> freshItems() {
        List<Item> out = new ArrayList<>();
        for (Item i : items) {
            Item c = new Item();
            c.seat = i.seat;
            c.kind = i.kind;
            c.key = i.key;
            c.name = i.name;
            c.family = i.family;
            c.window = i.window;
            c.mv = i.mv;
            c.order = i.order;
            out.add(c);
        }
        return out;
    }
}
