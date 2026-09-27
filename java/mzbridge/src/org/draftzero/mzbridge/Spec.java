package org.draftzero.mzbridge;

import java.util.*;

/**
 * Gson mirror of StateSpec v1 (src/draftzero/gameplay/statespec.py). Keep the two in step: a field
 * added there is added here. Unknown JSON keys are ignored (forward compatibility), and
 * {@code provenance} / {@code labels} are not read at all: the bridge builds states, it does not
 * interpret where they came from.
 *
 * Conventions (statespec.py's docstring lists them too):
 *  - {@code stack} is listed bottom to top (cast order): the last item resolves first.
 *  - Attacking creatures get their tapped state from the engine (declaring an attack taps a
 *    creature without vigilance); a spec's {@code tapped} flag on an attacker is ignored.
 *  - A permanent is listed under its CONTROLLER's seat; {@code owner} names the other seat when
 *    control changed, and its card then comes out of the owner's decklist.
 *
 * {@link #validate()} and statespec.StateSpec.validate() apply the same rules: change both.
 */
public class Spec {
    public static final int SCHEMA_VERSION = 1;
    public static final List<String> SEATS = List.of("A", "B");
    public static final List<String> ENTER_MODES = List.of("PRIORITY_FRESH", "BEGIN_STEP", "PRIORITY_HELD");
    public static final List<String> DECKLIST_SOURCES = List.of("exact", "belief", "placeholder");
    /** mage.constants.TurnPhase -> the mage.constants.PhaseStep values inside it, in turn order. */
    public static final LinkedHashMap<String, List<String>> PHASE_STEPS = new LinkedHashMap<>();

    static {
        PHASE_STEPS.put("BEGINNING", List.of("UNTAP", "UPKEEP", "DRAW"));
        PHASE_STEPS.put("PRECOMBAT_MAIN", List.of("PRECOMBAT_MAIN"));
        PHASE_STEPS.put("COMBAT", List.of("BEGIN_COMBAT", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS",
                "FIRST_COMBAT_DAMAGE", "COMBAT_DAMAGE", "END_COMBAT"));
        PHASE_STEPS.put("POSTCOMBAT_MAIN", List.of("POSTCOMBAT_MAIN"));
        PHASE_STEPS.put("END", List.of("END_TURN", "CLEANUP"));
    }

    public int version = SCHEMA_VERSION;
    public String comment = "";
    public int turn = 1;
    public String activePlayer = "A";
    public String phase = "PRECOMBAT_MAIN";
    public String step = "PRECOMBAT_MAIN";
    public String enterMode = "PRIORITY_FRESH";
    public String priorityPlayer;
    public List<String> passedPlayers = new ArrayList<>();
    public String startingPlayer;
    public Map<String, PlayerState> players = new LinkedHashMap<>();
    public List<StackItem> stack = new ArrayList<>();
    public List<Attack> attackers = new ArrayList<>();
    public List<Block> blockers = new ArrayList<>();

    public static class Perm {
        public String name;
        public String set;
        public String number;
        public String token;
        public String tokenClass;
        public String id;
        public int count = 1;
        public boolean tapped = false;
        public boolean sick = false;
        public int damage = 0;
        public Map<String, Integer> counters = new LinkedHashMap<>();
        public String attachTo;
        public boolean faceDown = false;
        /** the other seat, when it owns this permanent (control changed); null = the listing seat */
        public String owner;

        boolean isToken() {
            return token != null || tokenClass != null;
        }

        String describe() {
            if (tokenClass != null) return "token " + tokenClass;
            if (token != null) return "token " + token + (set == null ? "" : "/" + set);
            return name;
        }
    }

    public static class PlayerState {
        public String name = "";
        public int life = 20;
        public List<String> decklist = new ArrayList<>();
        public String decklistSource = "exact";
        public int landsPlayed = 0;
        public List<String> hand = new ArrayList<>();
        public int handUnknown = 0;
        public List<String> graveyard = new ArrayList<>();
        public List<String> exile = new ArrayList<>();
        public List<String> libraryTop = new ArrayList<>();
        public Integer librarySize;
        public String manaPool;
        public List<Perm> battlefield = new ArrayList<>();
    }

    public static class StackItem {
        public String controller;
        public String card;
        public List<String> targets = new ArrayList<>();
    }

    public static class Attack {
        public String attacker;
        public String defender;
    }

    public static class Block {
        public String blocker;
        public String attacker;
    }

    // ------------------------------------------------------------------ derived values

    /** Who was on the play: explicit, or the active player on odd turns. */
    public String starting() {
        if (startingPlayer != null) return startingPlayer;
        return turn % 2 == 1 ? activePlayer : other(activePlayer);
    }

    public static String other(String seat) {
        return "A".equals(seat) ? "B" : "A";
    }

    public static String phaseOf(String step) {
        for (Map.Entry<String, List<String>> e : PHASE_STEPS.entrySet()) {
            if (e.getValue().contains(step)) return e.getKey();
        }
        return null;
    }

    /** All steps of a turn in order. */
    public static List<String> turnSteps() {
        List<String> out = new ArrayList<>();
        for (List<String> s : PHASE_STEPS.values()) out.addAll(s);
        return out;
    }

    /** Aliases "<seat>:<id>" (and "<seat>:<id>#k" for count > 1) defined by battlefield entries. */
    public Set<String> aliases() {
        Set<String> out = new LinkedHashSet<>();
        for (String seat : SEATS) {
            PlayerState p = players.get(seat);
            if (p == null) continue;
            for (Perm perm : p.battlefield) {
                if (perm.id == null) continue;
                out.add(seat + ":" + perm.id);
                for (int k = 1; perm.count > 1 && k <= perm.count; k++) out.add(seat + ":" + perm.id + "#" + k);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ validation

    /**
     * Problems that would make the bridge reject or misbuild this spec: the checks of
     * statespec.StateSpec.validate() plus the ones only the engine side cares about. Empty = fine.
     */
    public List<String> validate() {
        List<String> errs = new ArrayList<>();
        if (version != SCHEMA_VERSION) errs.add("version " + version + " != " + SCHEMA_VERSION);
        if (players == null || !new HashSet<>(players.keySet()).equals(new HashSet<>(SEATS))) {
            errs.add("players must be exactly [A, B], got " + (players == null ? "null" : new TreeSet<>(players.keySet())));
            return errs; // everything below needs both seats
        }
        if (!SEATS.contains(activePlayer)) errs.add("activePlayer " + q(activePlayer));
        if (startingPlayer != null && !SEATS.contains(startingPlayer)) errs.add("startingPlayer " + q(startingPlayer));
        if (!PHASE_STEPS.containsKey(phase)) {
            errs.add("phase " + q(phase));
        } else if (!PHASE_STEPS.get(phase).contains(step)) {
            errs.add("step " + q(step) + " is not in phase " + phase);
        }
        if (!ENTER_MODES.contains(enterMode)) errs.add("enterMode " + q(enterMode));
        if ("PRIORITY_HELD".equals(enterMode) && !SEATS.contains(priorityPlayer)) errs.add("PRIORITY_HELD needs priorityPlayer");
        if (priorityPlayer != null && !SEATS.contains(priorityPlayer)) errs.add("priorityPlayer " + q(priorityPlayer));
        for (String s : passedPlayers) if (!SEATS.contains(s)) errs.add("passedPlayers entry " + q(s));
        if (turn < 1) errs.add("turn " + turn);
        if (turn == 1 && startingPlayer != null && !startingPlayer.equals(activePlayer)) {
            errs.add("turn 1 belongs to the starting player: startingPlayer " + startingPlayer + " != activePlayer " + activePlayer);
        }
        if ("UNTAP".equals(step)) {
            errs.add("UNTAP cannot be entered: use the previous turn's END_TURN with PRIORITY_FRESH (the engine "
                    + "then plays cleanup and this turn's untap, upkeep and draw)");
        }
        if ("CLEANUP".equals(step) && !"BEGIN_STEP".equals(enterMode)) {
            errs.add("CLEANUP has no priority window: use BEGIN_STEP at CLEANUP (the engine runs the cleanup) or END_TURN");
        }
        if ("BEGIN_STEP".equals(enterMode) && !stack.isEmpty()) {
            errs.add("BEGIN_STEP with a non-empty stack: a step only begins once the stack is empty");
        }
        Set<String> aliases = aliases();
        Set<String> seen = new HashSet<>();
        Map<String, List<String>> owned = owned();
        for (String seat : SEATS) {
            PlayerState p = players.get(seat);
            if (p == null) {
                errs.add(seat + ": missing");
                continue;
            }
            if (!DECKLIST_SOURCES.contains(p.decklistSource)) errs.add(seat + ".decklistSource " + q(p.decklistSource));
            if (p.decklist.size() < 40) errs.add(seat + ".decklist has " + p.decklist.size() + " cards; XMage needs at least 40");
            if (p.handUnknown < 0) errs.add(seat + ".handUnknown < 0");
            if (p.landsPlayed < 0) errs.add(seat + ".landsPlayed < 0");
            if (p.librarySize != null && p.librarySize < p.libraryTop.size()) {
                errs.add(seat + ".librarySize " + p.librarySize + " < libraryTop count " + p.libraryTop.size());
            }
            if (p.manaPool != null && !p.manaPool.matches("[WUBRGC]*")) errs.add(seat + ".manaPool " + q(p.manaPool) + " (use W U B R G C)");
            Map<String, Integer> pool = multiset(p.decklist);
            List<String> used = owned.get(seat);
            Map<String, Integer> short_ = new TreeMap<>();
            for (Map.Entry<String, Integer> e : multiset(used).entrySet()) {
                int have = pool.getOrDefault(e.getKey(), 0);
                if (e.getValue() > have) short_.put(e.getKey(), e.getValue() - have);
            }
            if (!short_.isEmpty()) errs.add(seat + ": cards named in zones but missing from decklist: " + short_);
            int remaining = p.decklist.size() - used.size();
            if (p.handUnknown > Math.max(0, remaining)) {
                errs.add(seat + ".handUnknown " + p.handUnknown + " > " + remaining + " cards left in the library");
            }
            for (Perm perm : p.battlefield) {
                if (perm.name == null && perm.token == null && perm.tokenClass == null) errs.add(seat + ": battlefield entry with no name/token/tokenClass");
                if (perm.attachTo != null && !aliases.contains(perm.attachTo)) errs.add(seat + ": attachTo " + q(perm.attachTo) + " names no permanent alias");
                if (perm.count < 1) errs.add(seat + ": battlefield count " + perm.count);
                if (perm.damage < 0) errs.add(seat + ": negative damage on " + perm.describe());
                if (perm.id != null && !seen.add(seat + ":" + perm.id)) errs.add(seat + ": duplicate alias " + q(perm.id));
                if (perm.attachTo != null && perm.count > 1) errs.add(seat + ": attachTo with count > 1 on " + perm.describe());
                for (Map.Entry<String, Integer> c : perm.counters.entrySet()) {
                    if (c.getValue() == null || c.getValue() < 0) errs.add(seat + ": counter " + c.getKey() + " on " + perm.describe() + " must be >= 0");
                }
                if (perm.owner != null && !SEATS.contains(perm.owner)) errs.add(seat + ": owner " + q(perm.owner) + " on " + perm.describe());
                if (perm.owner != null && perm.isToken()) errs.add(seat + ": owner on a token (" + perm.describe() + ") is not supported");
            }
        }
        for (Attack a : attackers) {
            if (!aliases.contains(a.attacker)) errs.add("attacker " + q(a.attacker) + " names no permanent alias");
            if (a.attacker != null && !a.attacker.startsWith(activePlayer + ":")) errs.add("attacker " + q(a.attacker) + " is not controlled by the active player");
            if (a.defender == null || !(a.defender.equals("player:" + other(activePlayer)) || aliases.contains(a.defender))) {
                errs.add("defender " + q(a.defender) + " must be player:" + other(activePlayer) + " or a permanent alias");
            }
        }
        for (Block b : blockers) {
            if (!aliases.contains(b.blocker) || !aliases.contains(b.attacker)) errs.add("block " + q(b.blocker) + "->" + q(b.attacker) + " names no permanent alias");
        }
        List<String> steps = turnSteps();
        int at = steps.indexOf(step);
        if (!attackers.isEmpty()) {
            if (!"COMBAT".equals(phase) || at < steps.indexOf("DECLARE_ATTACKERS")
                    || ("DECLARE_ATTACKERS".equals(step) && "BEGIN_STEP".equals(enterMode))) {
                errs.add("attackers need a position after attackers were declared (COMBAT phase, DECLARE_ATTACKERS "
                        + "with PRIORITY_FRESH/PRIORITY_HELD or a later combat step); got " + step + "/" + enterMode);
            }
        }
        if (!blockers.isEmpty()) {
            if (attackers.isEmpty()) errs.add("blockers without attackers");
            if (!"COMBAT".equals(phase) || at < steps.indexOf("DECLARE_BLOCKERS")
                    || ("DECLARE_BLOCKERS".equals(step) && "BEGIN_STEP".equals(enterMode))) {
                errs.add("blockers need a position after blockers were declared (DECLARE_BLOCKERS with "
                        + "PRIORITY_FRESH/PRIORITY_HELD or a later combat step); got " + step + "/" + enterMode);
            }
        }
        for (StackItem si : stack) {
            if (!SEATS.contains(si.controller)) errs.add("stack controller " + q(si.controller));
            if (si.card == null) errs.add("stack item with no card");
            for (String t : si.targets) {
                if (!(t.equals("player:A") || t.equals("player:B") || aliases.contains(t))) errs.add("stack target " + q(t) + " names no permanent alias or player");
            }
        }
        return errs;
    }

    /**
     * Cards each seat OWNS outside its library: its hand, graveyard, exile and known library top,
     * the non-token permanents it owns on either battlefield (a permanent's owner is the listing
     * seat unless {@code owner} names the other one), and the stack spells it controls.
     */
    Map<String, List<String>> owned() {
        Map<String, List<String>> owned = new LinkedHashMap<>();
        for (String seat : SEATS) owned.put(seat, new ArrayList<>());
        for (String seat : SEATS) {
            PlayerState p = players.get(seat);
            if (p == null) continue;
            List<String> mine = owned.get(seat);
            mine.addAll(p.hand);
            mine.addAll(p.graveyard);
            mine.addAll(p.exile);
            mine.addAll(p.libraryTop);
        }
        for (String seat : SEATS) {
            PlayerState p = players.get(seat);
            if (p == null) continue;
            for (Perm perm : p.battlefield) {
                if (perm.name == null || perm.isToken()) continue;
                // an unknown owner is reported by validate(); count the card where it is listed
                String own = perm.owner != null && SEATS.contains(perm.owner) ? perm.owner : seat;
                for (int k = 0; k < perm.count; k++) owned.get(own).add(perm.name);
            }
        }
        for (StackItem si : stack) {
            if (si.controller != null && owned.containsKey(si.controller) && si.card != null) owned.get(si.controller).add(si.card);
        }
        return owned;
    }

    /** The seat that owns a battlefield entry listed under {@code seat}. */
    static String ownerSeat(String seat, Perm perm) {
        return perm.owner != null ? perm.owner : seat;
    }

    static Map<String, Integer> multiset(List<String> xs) {
        Map<String, Integer> out = new HashMap<>();
        for (String x : xs) out.merge(x, 1, Integer::sum);
        return out;
    }

    static String q(Object o) {
        return o == null ? "null" : "'" + o + "'";
    }
}
