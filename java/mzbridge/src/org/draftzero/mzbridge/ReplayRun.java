package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.cards.Card;
import mage.cards.Cards;
import mage.cards.repository.CardInfo;
import mage.constants.CardType;
import mage.constants.Outcome;
import mage.constants.PhaseStep;
import mage.game.Game;
import mage.game.GameImpl;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentCard;
import mage.game.permanent.PermanentToken;
import mage.game.stack.Spell;
import mage.game.stack.StackObject;
import mage.player.ai.encoder.ActionEncoder;
import mage.player.ai.encoder.StateEncoder;
import mage.player.ai.score.GameStateEvaluator3;
import mage.target.Target;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static mage.target.TargetImpl.STOP_CHOOSING;

/**
 * One attempt of a replay_turn request: the script state both {@link ReplayPlayer}s share, the
 * decisions the scripted seat reaches, the targets chosen, and the comparison with the snapshot.
 *
 * Windows of the replayed turn (index): upkeep/draw 0, main1 and beginning of combat 1, after
 * attackers 2, after blockers and combat damage 3, main2 4, end step 5. An item is due from its
 * window on and is taken at the first priority where it is due and legal, so an item that is
 * not legal yet (its card is still to be drawn or bounced back, the mana is not there) carries
 * over to later windows. Sorcery-speed legality is XMage's own (getPlayable).
 */
final class ReplayRun {
    static final int LOOP_GUARD = 3000;

    final StateInjector.Built b;
    final TurnScript script;
    final TurnScript.Policy policy;
    final List<TurnScript.Item> items;
    final String seat;            // the scripted (active) seat
    final String opp;
    final int turn;
    final boolean encode;
    final boolean perfectInfo;
    /** record the scripted seat's priority decisions at every stop with a real choice (end step,
     *  combat, a spell on the stack), not only in its main phases or where it plays */
    boolean allStops;
    /** each recorded decision also carries GameStateEvaluator3's score from the deciding seat */
    boolean heuristic;
    /** each recorded decision also carries its state as MageZero's graph encoder sees it, with the
     *  legal options as graph nodes (GraphRecord; docs/022 §3.1) */
    boolean graph;
    /** whose decisions are recorded: the scripted seat (default), or the other one (the 17lands
     *  user during the opponent's turn: its instants, flash and blocks; docs/017 §2.2) */
    String recordSeat;
    final TargetResolver resolver;
    final Map<TurnScript.Item, String> due = new IdentityHashMap<>();
    final List<TurnScript.Item> mine = new ArrayList<>();
    final List<TurnScript.Item> theirs = new ArrayList<>();
    final List<TurnScript.Block> blockPlan;
    final JsonArray decisions = new JsonArray();
    final JsonArray targets = new JsonArray();
    final Set<String> flags = new TreeSet<>();
    final List<String> notes = new ArrayList<>();
    /** zone-change counter of every injected permanent: a different one means it left and came back */
    final Map<UUID, Integer> zcc0 = new HashMap<>();
    /** spec alias -> the compared name of its permanent at the start (cardKey) */
    final Map<String, String> aliasName = new HashMap<>();
    RuntimeException failure;
    int priorityCalls, attackCalls, blockCalls;
    /** mana producers the paying seat must not tap for the item being paid (reserved for later items) */
    Set<UUID> hidden;
    String hiddenFor;

    ReplayRun(StateInjector.Built b, TurnScript script, TurnScript.Policy policy, boolean encode, boolean perfectInfo) {
        this.b = b;
        this.script = script;
        this.policy = policy;
        this.items = script.freshItems();
        this.seat = script.seat;
        this.opp = Spec.other(seat);
        this.recordSeat = seat;
        this.turn = script.turn;
        this.encode = encode;
        this.perfectInfo = perfectInfo;
        this.resolver = new TargetResolver(this);
        this.blockPlan = script.blocks.isEmpty() ? List.of() : script.blocks.get(Math.min(policy.pairing, script.blocks.size() - 1));
        for (TurnScript.Item it : items) {
            due.put(it, dueWindow(it));
            (it.seat.equals(seat) ? mine : theirs).add(it);
        }
        boolean landLast = policy.land.equals("last");
        Comparator<TurnScript.Item> byKind = Comparator.comparingInt(i -> i.kind.equals("land") ? (landLast ? 3 : 0) : i.kind.equals("cast") ? 1 : 2);
        Comparator<TurnScript.Item> byPolicy;
        switch (policy.order) {
            case "mv_desc":
                byPolicy = Comparator.comparingInt((TurnScript.Item i) -> -mv(i));
                break;
            case "listed":
                byPolicy = Comparator.comparingInt(i -> 0);
                break;
            default:
                byPolicy = Comparator.comparingInt(ReplayRun::mv);
        }
        mine.sort(byKind.thenComparing(byPolicy).thenComparingInt(i -> i.order));
        theirs.sort(Comparator.comparingInt((TurnScript.Item i) -> i.order));
        if (b != null) {   // null: TurnReplay.signature only reads the windows and the order
            for (Permanent pm : b.game.getBattlefield().getAllPermanents()) zcc0.put(pm.getId(), pm.getZoneChangeCounter(b.game));
            for (Map.Entry<String, UUID> e : b.aliases.entrySet()) {
                Permanent pm = b.game.getPermanent(e.getValue());
                if (pm != null) aliasName.put(e.getKey(), cardKey(pm));
            }
        }
    }

    // =================================================================================== windows

    static int windowIndex(PhaseStep st) {
        if (st == null) return -1;
        switch (st) {
            case UNTAP:
            case UPKEEP:
            case DRAW:
                return 0;
            case PRECOMBAT_MAIN:
            case BEGIN_COMBAT:
                return 1;
            case DECLARE_ATTACKERS:
                return 2;
            case DECLARE_BLOCKERS:
            case FIRST_COMBAT_DAMAGE:
            case COMBAT_DAMAGE:
            case END_COMBAT:
                return 3;
            case POSTCOMBAT_MAIN:
                return 4;
            case END_TURN:
                return 5;
            default:
                return 6;
        }
    }

    static int windowIndex(String w) {
        return TurnScript.WINDOWS.indexOf(w);
    }

    /** The window an item is due from, under this attempt's policy. */
    String dueWindow(TurnScript.Item it) {
        if (it.seat.equals(seat)) {
            if (it.window != null) return it.window;
            if (it.kind.equals("land")) return "main1";
            if (it.kind.equals("activation")) return policy.aMain.equals("main2") ? "main2" : "main1";
            Facts f = Facts.of(it.name);
            // a counterspell of A's answers one of B's spells in the turn
            if (f != null && f.rules.contains("counter target")) return "respond";
            // a creature that entered this turn and attacked was cast before combat, whatever the policy
            boolean attackedNew = script.attacks.keySet().stream().anyMatch(k -> k.equals("new:" + it.name) || k.startsWith("new:" + it.name + "#"));
            if (attackedNew) return "main1";
            if (f != null && f.instant && policy.aInstants.equals("combat")) return "after_blockers";
            switch (policy.aMain) {
                case "main2":
                    return "main2";
                case "creatures_main2":
                    return f != null && f.creature ? "main2" : "main1";
                default:
                    return "main1";
            }
        }
        if (!policy.bWindow.equals("auto")) return policy.bWindow;
        if (it.window != null) return it.window;
        return defaultTheirWindow(it);
    }

    /** Where the opponent's recorded instant-speed play most plausibly happened, from its card text. */
    String defaultTheirWindow(TurnScript.Item it) {
        if (it.kind.equals("activation")) {
            String t = it.key.toLowerCase(Locale.ROOT);
            if (t.contains("blocking") || t.contains("blocked")) return "after_blockers";
            if (t.contains("attacking")) return "after_attackers";
            return "end_step";
        }
        Facts f = Facts.of(it.name);
        if (f == null) return "end_step";
        String r = f.rules;
        if (r.contains("counter target")) return "respond";
        if (f.creature) {
            // a flash creature that blocked came in after the attack was declared
            for (TurnScript.Block bl : blockPlan) if (bl.blocker.equals("new:" + it.name) || bl.blocker.startsWith("new:" + it.name + "#")) return "after_attackers";
            return "end_step";
        }
        if (r.contains("blocking") || r.contains("blocked")) return "after_blockers";
        if (r.contains("attacking")) return "after_attackers";
        if (r.contains("gets +") || r.contains("hexproof") || r.contains("indestructible") || r.contains("protection from")) {
            boolean blocks = blockPlan.stream().anyMatch(x -> x.attacker != null);
            return blocks ? "after_blockers" : "respond";
        }
        if (r.contains("destroy target") || r.contains("damage to target") || r.contains("exile target")
                || r.contains("return target") || r.contains("gets -") || r.contains("tap target")) {
            return "after_attackers";
        }
        return "end_step";
    }

    boolean inTurn(Game game) {
        return game.getTurnNum() == turn && seat.equals(b.seatOf(game.getActivePlayerId()));
    }

    /** Whether a script item's window has come: the scripted seat's plays at their window once the
     *  stack is empty, a "respond" item when it can answer the top of the stack; the other seat's
     *  "respond" items fall back to the end step when nothing ever triggered them. */
    boolean dueNow(TurnScript.Item it, Game game, int w, boolean stackEmpty, boolean scripted) {
        String d = due.get(it);
        if (d.equals("respond")) {
            if (!stackEmpty) return respondNow(it, game);
            return !scripted && w >= windowIndex("end_step");
        }
        return stackEmpty && windowIndex(d) <= w;
    }

    static boolean isMain(Game game) {
        PhaseStep st = game.getTurnStepType();
        return st == PhaseStep.PRECOMBAT_MAIN || st == PhaseStep.POSTCOMBAT_MAIN;
    }

    static int mv(TurnScript.Item i) {
        if (i.mv >= 0) return i.mv;
        Facts f = Facts.of(i.name);
        return f == null ? 0 : f.mv;
    }

    // =================================================================================== priority

    boolean priority(ReplayPlayer p, Game game) {
        if (++priorityCalls > LOOP_GUARD) {
            fail(new IllegalStateException("replay loop guard: more than " + LOOP_GUARD + " priority calls"));
            game.pause();
            return false;
        }
        game.getState().setPriorityPlayerId(p.getId());
        List<ActivatedAbility> playable = p.playable(game);
        boolean nonTrivial = playable.size() >= 2;
        // MageZero re-anchors (and so clears both players' micro-decision histories) at every
        // priority with a real choice and at its combat checkpoints, for either seat
        if (nonTrivial || game.isCheckPoint(p.getId())) ((GameImpl) game).clearHistory();
        if (!inTurn(game)) {
            p.pass(game);
            return false;
        }
        int w = windowIndex(game.getTurnStepType());
        boolean stackEmpty = game.getStack().isEmpty();
        TurnScript.Item pick = null;
        ActivatedAbility ability = null;
        String here = game.getTurnNum() + ":" + game.getState().getStepNum();
        if (p.seat.equals(seat)) {
            for (TurnScript.Item it : mine) {
                if (it.done || here.equals(it.failedAt)) continue;
                boolean now = dueNow(it, game, w, stackEmpty, true);
                if (now) {
                    ActivatedAbility a = find(playable, it, game);
                    if (a != null) {
                        pick = it;
                        ability = a;
                        break;
                    }
                }
            }
        } else {
            for (TurnScript.Item it : theirs) {
                if (it.done || here.equals(it.failedAt)) continue;
                boolean now = dueNow(it, game, w, stackEmpty, false);
                if (!now) continue;
                ActivatedAbility a = find(playable, it, game);
                if (a != null) {
                    pick = it;
                    ability = a;
                    break;
                }
            }
        }
        JsonObject rec = null;
        boolean scripted = p.seat.equals(seat);
        if (p.seat.equals(recordSeat) && nonTrivial && (allStops || (scripted && isMain(game)) || pick != null)) {
            List<TurnScript.Item> own = scripted ? mine : theirs;
            // the label set: in the scripted seat's main phases, every play it has left (17lands has
            // no order inside a turn); elsewhere only the plays the replay has due at this stop, so
            // an instant it cast later is not labelled at an earlier stop (docs/017 §2.2)
            boolean dueOnly = !scripted || !isMain(game);
            JsonObject d = decision(game, p, "PRIORITY", "priority");
            rec = d;
            // sorted by label: the engine's playable order can follow hash order (a card that
            // returned to hand), which would make identical replays list options differently
            TreeMap<String, ActivatedAbility> byLabel = new TreeMap<>();
            TreeMap<String, List<UUID>> idsByLabel = new TreeMap<>();
            for (ActivatedAbility a : playable) {
                byLabel.putIfAbsent(a.toString(), a);
                idsByLabel.computeIfAbsent(a.toString(), k -> new ArrayList<>()).add(GraphRecord.actionId(a));
            }
            JsonArray legal = new JsonArray();
            JsonArray set = new JsonArray();
            for (Map.Entry<String, ActivatedAbility> e : byLabel.entrySet()) {
                String label = e.getKey();
                JsonObject o = new JsonObject();
                o.addProperty("label", label);
                o.addProperty("idx", p.actionEncoder.getActionIndex(e.getValue(), true));
                legal.add(o);
                for (TurnScript.Item it : own) if (!it.done && (it.key.equals(label) || flashbackOf(it, e.getValue(), game))
                        && (!dueOnly || dueNow(it, game, w, stackEmpty, scripted))) {
                    set.add(label);
                    break;
                }
            }
            d.add("legal", legal);
            d.add("set", set);
            boolean left = (scripted && script.unscripted > 0) || own.stream().anyMatch(i -> !i.done);
            if (pick != null) {
                d.addProperty("chosen", ability.toString());
                d.addProperty("label_kind", "imputed_order");
                d.addProperty("evidence", "recorded");
            } else {
                d.addProperty("chosen", "Pass");
                d.addProperty("label_kind", left ? "imputed_order" : "exact");
                d.addProperty("evidence", left ? "policy" : "recorded_none_left");
            }
            finishDecision(d, game, p, "PRIORITY", "priority",
                    new GraphRecord.Ask("PRIORITY", "priority", null, null, new ArrayList<>(idsByLabel.values())));
        }
        if (pick == null) {
            p.pass(game);
            return false;
        }
        reserveMana(p, game, pick, ability);
        boolean ok;
        try {
            ok = p.activateAbility((ActivatedAbility) ability.copy(), game);
        } finally {
            hidden = null;
            hiddenFor = null;
        }
        if (!ok) {
            // not payable or no legal target after all: skip it in this step and look again (the
            // engine gives the player priority again, since it has not passed)
            flags.add("activation_failed");
            notes.add("could not activate " + pick.describe() + " at " + game.getTurnStepType());
            pick.failedAt = here;
            if (rec != null) decisions.remove(rec);     // it did not happen: the retry records the real choice
            return true;
        }
        pick.done = true;
        pick.doneAt = String.valueOf(game.getTurnStepType());
        return true;
    }

    /** The playable ability for a script item: same label (ability.toString()), same source name for activations. */
    ActivatedAbility find(List<ActivatedAbility> playable, TurnScript.Item it, Game game) {
        ActivatedAbility loose = null;
        for (ActivatedAbility a : playable) {
            if (!a.toString().equals(it.key)) continue;
            if (!it.kind.equals("activation") || it.name == null) return a;
            MageObject src = a.getSourceId() == null ? null : game.getObject(a.getSourceId());
            String sn = src instanceof Permanent ? cardKey((Permanent) src) : src == null ? null : src.getName();
            if (it.name.equals(sn)) return a;
            if (loose == null) loose = a;
        }
        if (loose != null) {
            flags.add("activation_source_mismatch");
            return loose;
        }
        for (ActivatedAbility a : playable) {
            if (flashbackOf(it, a, game)) {
                flags.add("cast_as_flashback");
                return a;
            }
        }
        return null;
    }

    /**
     * A flashback cast of an item recorded as "Cast X": 17lands logs flashback as a cast, and
     * labels only catches it when no copy was in hand at the turn start, so a flashback granted by
     * another card (Sphinx of Forgotten Lore) or the copy cast from hand earlier in the same turn
     * (Think Twice) comes as "Cast X". Taken only when no "Cast X" is playable.
     */
    boolean flashbackOf(TurnScript.Item it, ActivatedAbility a, Game game) {
        if (!it.kind.equals("cast") || it.name == null || !it.key.startsWith("Cast ") || !a.toString().startsWith("Flashback")) return false;
        MageObject src = a.getSourceId() == null ? null : game.getObject(a.getSourceId());
        return src != null && it.name.equals(src.getName()) && game.getState().getZone(a.getSourceId()) == mage.constants.Zone.GRAVEYARD;
    }

    /** "respond": a counterspell waits for the spell the snapshot says did not resolve; other items for a spell or ability of the scripted seat that targets theirs. */
    boolean respondNow(TurnScript.Item it, Game game) {
        StackObject top = game.getStack().getFirst();
        if (top == null) return false;
        String by = b.seatOf(top.getControllerId());
        String other = Spec.other(it.seat);
        if (by == null || !by.equals(other)) return false;
        Facts f = Facts.of(it.name);
        if (f != null && f.rules.contains("counter target")) {
            if (resolver.shouldCounter(game, top, it.seat)) return true;
            // an instant or sorcery leaves no trace in the snapshot: counter it unless a permanent
            // spell the snapshot does not show is still to come from that seat
            return top instanceof Spell && !((Spell) top).getCard().isPermanent(game) && !permanentToCounterLater(other, game);
        }
        for (Target t : top.getStackAbility().getTargets()) {
            for (UUID id : t.getTargets()) {
                Permanent pm = game.getPermanent(id);
                if (pm != null && it.seat.equals(b.seatOf(pm.getControllerId()))) return true;
                if (it.seat.equals(b.seatOf(id))) return true;
            }
        }
        return false;
    }

    /** Does `seat` still have to cast a permanent spell that the snapshot does not show on its battlefield? */
    private boolean permanentToCounterLater(String seatOf, Game game) {
        Map<String, Integer> now = battlefieldCounts(game).getOrDefault(seatOf, Map.of());
        Map<String, Integer> want = script.expected.battlefield.getOrDefault(seatOf, Map.of());
        for (TurnScript.Item i : (seatOf.equals(seat) ? mine : theirs)) {
            if (i.done || !i.kind.equals("cast")) continue;
            Facts f = Facts.of(i.name);
            if (f != null && (f.creature || f.permanent) && want.getOrDefault(i.name, 0) <= now.getOrDefault(i.name, 0)) return true;
        }
        return false;
    }

    // =================================================================================== mana

    /**
     * XMage's auto-tap pays each cost with the first fitting producers, blind to what the rest of
     * the turn needs (a Forest spent on a generic {1} leaves {G}{G} unpayable). Before an item is
     * paid, the colored pips of the seat's later items are matched to producers (scarcest color
     * first, least flexible producer first) after this item's own pips, and those producers are
     * hidden from the payment (ReplayPlayer.getAvailableManaProducers). Sacrifice producers
     * (Treasure, Food-style) are hidden too while the rest can pay: humans keep them.
     */
    void reserveMana(ReplayPlayer p, Game game, TurnScript.Item current, ActivatedAbility ab) {
        hidden = null;
        hiddenFor = null;
        List<Permanent> producers = new ArrayList<>();
        for (MageObject o : p.allManaProducers(game)) if (o instanceof Permanent) producers.add((Permanent) o);
        if (producers.isEmpty()) return;
        Map<UUID, Set<Character>> colors = new HashMap<>();
        Set<UUID> sacrifice = new HashSet<>();
        for (Permanent pm : producers) {
            Set<Character> cs = new TreeSet<>();
            for (mage.abilities.mana.ActivatedManaAbilityImpl a : pm.getAbilities().getActivatedManaAbilities(mage.constants.Zone.BATTLEFIELD)) {
                for (mage.Mana m : a.getNetMana(game)) cs.addAll(colorsOf(m));
                for (mage.abilities.costs.Cost c : a.getCosts()) {
                    if (c instanceof mage.abilities.costs.common.SacrificeSourceCost) sacrifice.add(pm.getId());
                }
            }
            colors.put(pm.getId(), cs);
        }
        mage.Mana cur = ab.getManaCostsToPay().getMana();
        List<Character> curPips = pips(cur);
        int curTotal = cur.count();
        List<Character> later = new ArrayList<>();
        for (TurnScript.Item it : (current.seat.equals(seat) ? mine : theirs)) {
            if (it == current || it.done || !it.kind.equals("cast")) continue;
            Facts f = Facts.of(it.name);
            if (f != null && f.cost != null) later.addAll(pips(f.cost));
        }
        Set<UUID> used = new HashSet<>();
        assign(curPips, producers, colors, used);
        Set<UUID> forLater = assign(later, producers, colors, used);
        Set<UUID> hide = new HashSet<>(forLater);
        // never hide so much that this item cannot be paid
        List<UUID> order = new ArrayList<>(forLater);
        Collections.sort(order);
        while (producers.size() - hide.size() < curTotal && !order.isEmpty()) hide.remove(order.remove(order.size() - 1));
        // Treasure-style producers (policy sacMana=keep) and creatures that are about to attack
        // (a Llanowar Elves tapped for mana cannot attack) are spared while the rest can pay
        List<Permanent> spare = new ArrayList<>();
        for (Permanent pm : producers) {
            if (hide.contains(pm.getId())) continue;
            if ((sacrifice.contains(pm.getId()) && policy.sacMana.equals("keep")) || (pm.isCreature(game) && plannedAttacker(pm, game))) {
                spare.add(pm);
            }
        }
        List<Permanent> keep = new ArrayList<>();
        for (Permanent pm : producers) if (!hide.contains(pm.getId()) && !spare.contains(pm)) keep.add(pm);
        if (!spare.isEmpty() && keep.size() >= curTotal && assign(curPips, keep, colors, new HashSet<>()).size() == curPips.size()) {
            for (Permanent pm : spare) {
                hide.add(pm.getId());
                if (sacrifice.contains(pm.getId())) flags.add("sac_mana_kept");
            }
        }
        if (!hide.isEmpty()) {
            hidden = hide;
            hiddenFor = p.seat;
        }
    }

    /** Match pips to unused producers (scarcest color first, least flexible producer first); returns the producers taken. */
    private static Set<UUID> assign(List<Character> pips, List<Permanent> producers, Map<UUID, Set<Character>> colors, Set<UUID> used) {
        Set<UUID> taken = new HashSet<>();
        Map<Character, Long> supply = new HashMap<>();
        for (char c : new char[]{'W', 'U', 'B', 'R', 'G', 'C'}) {
            supply.put(c, producers.stream().filter(pm -> colors.get(pm.getId()).contains(c)).count());
        }
        List<Character> sorted = new ArrayList<>(pips);
        sorted.sort(Comparator.comparingLong(c -> supply.getOrDefault(c, 0L)));
        for (char c : sorted) {
            Permanent best = null;
            for (Permanent pm : producers) {
                if (used.contains(pm.getId()) || !colors.get(pm.getId()).contains(c)) continue;
                if (best == null || colors.get(pm.getId()).size() < colors.get(best.getId()).size()
                        || (colors.get(pm.getId()).size() == colors.get(best.getId()).size() && pm.getId().compareTo(best.getId()) < 0)) {
                    best = pm;
                }
            }
            if (best != null) {
                used.add(best.getId());
                taken.add(best.getId());
            }
        }
        return taken;
    }

    private static List<Character> pips(mage.Mana m) {
        List<Character> out = new ArrayList<>();
        for (int i = 0; i < m.getWhite(); i++) out.add('W');
        for (int i = 0; i < m.getBlue(); i++) out.add('U');
        for (int i = 0; i < m.getBlack(); i++) out.add('B');
        for (int i = 0; i < m.getRed(); i++) out.add('R');
        for (int i = 0; i < m.getGreen(); i++) out.add('G');
        for (int i = 0; i < m.getColorless(); i++) out.add('C');
        return out;
    }

    private static Set<Character> colorsOf(mage.Mana m) {
        Set<Character> cs = new TreeSet<>();
        if (m.getAny() > 0) cs.addAll(List.of('W', 'U', 'B', 'R', 'G'));
        if (m.getWhite() > 0) cs.add('W');
        if (m.getBlue() > 0) cs.add('U');
        if (m.getBlack() > 0) cs.add('B');
        if (m.getRed() > 0) cs.add('R');
        if (m.getGreen() > 0) cs.add('G');
        if (m.getColorless() > 0) cs.add('C');
        return cs;
    }

    // =================================================================================== combat

    void selectAttackers(ReplayPlayer p, Game game, UUID attackingPlayerId) {
        attackCalls++;
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) return;
        Set<UUID> defenders = game.getCombat().getDefenders();
        UUID oppId = b.players.get(opp).getId();
        UUID playerDef = defenders.contains(oppId) ? oppId : defenders.iterator().next();
        // 17lands does not record what was attacked: the player, unless a planeswalker the snapshot
        // no longer shows is there (then attackers go at it until their power covers its loyalty),
        // or the attempt's policy sends everything at a planeswalker
        List<Permanent> walkers = new ArrayList<>();
        for (UUID d : defenders) {
            Permanent pw = game.getPermanent(d);
            if (pw != null && pw.isPlaneswalker(game)) walkers.add(pw);
        }
        walkers.sort(Comparator.comparing(Permanent::getId));
        Permanent doomedWalker = null;
        Map<String, Map<String, Integer>> bfNow = battlefieldCounts(game);
        for (Permanent pw : walkers) {
            String key = cardKey(pw);
            if (script.expected.hasBattlefield && bfNow.get(opp).getOrDefault(key, 0)
                    > script.expected.battlefield.getOrDefault(opp, Map.of()).getOrDefault(key, 0)) {
                doomedWalker = pw;
                break;
            }
        }
        int walkerNeed = doomedWalker == null ? 0 : doomedWalker.getCounters(game).getCount(mage.counters.CounterType.LOYALTY);
        if (!walkers.isEmpty()) flags.add(doomedWalker != null ? "attack_defender_from_fate" : "attack_defender_player_assumed");
        List<Permanent> avail = new ArrayList<>(p.getAvailableAttackers(game));
        avail.sort(Comparator.comparing(Permanent::getId)); // MageZero's question order
        Map<String, Integer> newLeft = newRefs(script.attacks.entrySet().stream()
                .filter(e -> e.getKey().startsWith("new:") && e.getValue()).map(Map.Entry::getKey).toList());
        // copies of a card whose attacking copy labels could not tell (attackGuess): as many of them
        // attack as the plan says, the planned aliases first, then the other copies (the planned
        // one may not have untapped: Slumbering Cerberus)
        Map<String, Integer> guessLeft = new HashMap<>();
        for (Map.Entry<String, Boolean> e : script.attacks.entrySet()) {
            String nm = aliasName.get(e.getKey());
            if (e.getValue() && nm != null && script.attackGuess.contains(nm)) guessLeft.merge(nm, 1, Integer::sum);
        }
        // creatures the engine already declared ("attacks each combat if able" requirements) are not
        // asked about; one among them uses up its "new:" entry or its guessed copy's count
        for (UUID id : game.getCombat().getAttackers()) {
            Permanent pm = game.getPermanent(id);
            if (pm != null && aliasIfUnmoved(pm, game) == null) take(newLeft, cardKey(pm));
            else if (pm != null) take(guessLeft, cardKey(pm));
        }
        Set<UUID> guessedYes = new HashSet<>();
        for (Permanent pm : avail) {
            String a = aliasIfUnmoved(pm, game);
            if (a != null && Boolean.TRUE.equals(script.attacks.get(a)) && take(guessLeft, cardKey(pm))) guessedYes.add(pm.getId());
        }
        for (Permanent pm : avail) {
            if (aliasIfUnmoved(pm, game) != null && !guessedYes.contains(pm.getId()) && take(guessLeft, cardKey(pm))) guessedYes.add(pm.getId());
        }
        for (Permanent pm : avail) {
            String alias = aliasIfUnmoved(pm, game);
            boolean yes;
            String evidence;
            if (attackCalls > 1) {
                yes = false;           // the plan broke an attack restriction: declare nothing
                evidence = "plan_illegal";
                flags.add("attack_plan_illegal");
            } else if (alias != null && script.attackGuess.contains(cardKey(pm))
                    && (script.attacks.containsKey(alias) || guessedYes.contains(pm.getId()))) {
                yes = guessedYes.contains(pm.getId());
                evidence = "recorded";
            } else if (alias != null && script.attacks.containsKey(alias)) {
                yes = script.attacks.get(alias);
                evidence = "recorded";
            } else if (take(newLeft, cardKey(pm))) {
                yes = true;
                evidence = "recorded_new";
            } else {
                yes = false;
                evidence = alias == null ? "not_in_plan_new" : "not_in_plan";
            }
            String text = "attack with: " + pm.getName() + "?";
            JsonObject d = decision(game, p, "CHOOSE_USE", text);
            JsonArray legal = new JsonArray();
            legal.add(option("no", 0));
            legal.add(option("yes", 1));
            d.add("legal", legal);
            d.addProperty("chosen", yes ? "yes" : "no");
            boolean guess = script.attackGuess.contains(cardKey(pm)) && !evidence.equals("plan_illegal");
            // not_in_plan: labels left the creature out (it may not have untapped, it died outside
            // combat, a hostile Aura): it did not attack, but whether the question arose depends
            // on this attempt's order, so "no" is not a recorded answer
            boolean imputed = evidence.equals("plan_illegal") || evidence.equals("not_in_plan") || guess;
            d.addProperty("label_kind", imputed ? "imputed_order" : "exact");
            d.addProperty("evidence", guess ? "copy_guess" : evidence);
            if (alias != null) d.addProperty("alias", alias);
            finishDecision(d, game, p, "CHOOSE_USE", text, GraphRecord.Ask.attack(game, pm.getId(), playerDef));
            p.getPlayerHistory().useSequence.add(yes);
            if (yes) {
                UUID defender = playerDef;
                if (policy.defender.equals("planeswalker") && !walkers.isEmpty()) {
                    defender = walkers.get(0).getId();
                } else if (doomedWalker != null && walkerNeed > 0) {
                    defender = doomedWalker.getId();
                    walkerNeed -= Math.max(1, pm.getPower().getValue());
                }
                p.declareAttacker(pm.getId(), defender, game, false);
            }
        }
        game.getPlayers().resetPassed();
        for (Map.Entry<String, Boolean> e : script.attacks.entrySet()) {
            if (!e.getValue() || e.getKey().startsWith("new:") || script.attackGuess.contains(aliasName.get(e.getKey()))) continue;
            UUID id = b.aliases.get(e.getKey());
            Permanent pm = id == null ? null : game.getPermanent(id);
            if (pm == null || !pm.isAttacking()) {
                flags.add("attack_unrealised");
                notes.add("recorded attacker " + e.getKey() + " did not attack (not on the battlefield or not able to)");
            }
        }
        for (Map.Entry<String, Integer> e : newLeft.entrySet()) {
            if (e.getValue() > 0) {
                // only flagged (divergences): 17lands sometimes lists a turn's attackers twice,
                // and labels turns the second copy into a "new:" attacker
                flags.add("attack_new_unrealised");
                notes.add("recorded new attacker " + e.getKey() + " x" + e.getValue() + " was not there to attack");
            }
        }
    }

    void selectBlockers(ReplayPlayer p, Ability source, Game game, UUID defendingPlayerId) {
        blockCalls++;
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) return;
        List<Permanent> blockers = new ArrayList<>(p.getAvailableBlockers(game));
        blockers.sort(Comparator.comparing(Permanent::getId));
        Map<String, Integer> newBlockers = newRefs(blockPlan.stream().map(x -> x.blocker).filter(x -> x.startsWith("new:")).toList());
        Set<UUID> usedNewAttackers = new HashSet<>();
        Set<TurnScript.Block> realised = new HashSet<>();
        boolean recordBlocks = p.seat.equals(recordSeat) && !recordSeat.equals(seat) && inTurn(game);
        for (Permanent blk : blockers) {
            UUID target = null;
            if (blockCalls > 1) {
                flags.add("block_plan_illegal");
            } else {
                String alias = aliasIfUnmoved(blk, game);
                TurnScript.Block plan = null;
                for (TurnScript.Block x : blockPlan) if (alias != null && x.blocker.equals(alias)) plan = x;
                if (plan == null && take(newBlockers, cardKey(blk))) {
                    for (TurnScript.Block x : blockPlan) {
                        if (x.blocker.startsWith("new:") && baseName(x.blocker).equals(cardKey(blk)) && !realised.contains(x)) {
                            plan = x;
                            break;
                        }
                    }
                }
                if (plan != null) {
                    realised.add(plan);
                    if (plan.attacker != null) {
                        target = attackerId(plan.attacker, game, usedNewAttackers);
                        if (target == null) {
                            flags.add("block_attacker_missing");
                            notes.add("blocker " + plan.blocker + ": its attacker " + plan.attacker + " is not attacking");
                        }
                    }
                }
            }
            if (recordBlocks && blockCalls == 1) recordBlock(p, blk, target, game);
            if (target != null) {
                p.getPlayerHistory().targetSequence.add(target);
                p.declareBlocker(defendingPlayerId, blk.getId(), target, game);
                if (blk.getBlocking() == 0) {
                    flags.add("block_refused");
                    notes.add("XMage refused the block " + cardKey(blk) + " -> " + game.getEntityName(target, null));
                }
            } else {
                p.getPlayerHistory().targetSequence.add(STOP_CHOOSING);
            }
        }
        game.getPlayers().resetPassed();
        for (TurnScript.Block x : blockPlan) {
            if (x.attacker != null && !realised.contains(x)) {
                flags.add(x.blocker.startsWith("new:") ? "block_new_unrealised" : "block_unrealised");
                notes.add("recorded blocker " + x.blocker + " was not able to block");
            }
        }
    }

    /** One block question as MageZero asks it (ChooseCreatureToBlockAbility: a target per blocker),
     *  with the scripted answer; exact when the turn's pairing is (unique, or no block at all). */
    void recordBlock(ReplayPlayer p, Permanent blk, UUID target, Game game) {
        List<String[]> labelled = new ArrayList<>();
        for (UUID a : game.getCombat().getAttackers()) {
            if (blk.canBlock(a, game)) labelled.add(new String[]{BridgePlayer.targetLabel(game, a, p.getId()), a.toString()});
        }
        if (labelled.isEmpty()) return;
        labelled.add(new String[]{"Stop Choosing", STOP_CHOOSING.toString()});
        labelled.sort(Comparator.comparing((String[] x) -> x[0]).thenComparing(x -> x[1]));
        String text = "choose which creature to block for " + blk.getName() + ":Choose a target:attacking creature";
        JsonObject d = decision(game, p, "CHOOSE_TARGET", text);
        JsonArray legal = new JsonArray();
        Set<String> seen = new HashSet<>();
        for (String[] l : labelled) if (seen.add(l[0])) legal.add(option(l[0], p.actionEncoder.getTargetIndex(l[0])));
        if (legal.size() < 2) return;
        GraphRecord.Ask ask = GraphRecord.Ask.block(blk.getId(), optionIds(labelled));
        d.add("legal", legal);
        d.addProperty("chosen", target == null ? "Stop Choosing" : BridgePlayer.targetLabel(game, target, p.getId()));
        boolean exact = script.blockPairing == null || script.blockPairing.equals("unique") || script.blockPairing.equals("none");
        d.addProperty("label_kind", exact ? "exact" : "guessed_target");
        d.addProperty("evidence", "block_" + (script.blockPairing == null ? "none" : script.blockPairing));
        finishDecision(d, game, p, "CHOOSE_TARGET", text, ask);
    }

    /** The engine ids behind each distinct label of `labelled` ({label, uuid} rows sorted by label),
     *  in the order the labels were first added to the decision's legal list. */
    static List<List<UUID>> optionIds(List<String[]> labelled) {
        LinkedHashMap<String, List<UUID>> out = new LinkedHashMap<>();
        for (String[] l : labelled) out.computeIfAbsent(l[0], k -> new ArrayList<>()).add(UUID.fromString(l[1]));
        return new ArrayList<>(out.values());
    }

    private UUID attackerId(String ref, Game game, Set<UUID> usedNew) {
        Set<UUID> attacking = new HashSet<>(game.getCombat().getAttackers());
        String name;
        if (!ref.startsWith("new:")) {
            UUID id = b.aliases.get(ref);
            if (id != null && attacking.contains(id)) return id;
            // which copy attacked is a guess: the blocked one is the attacking copy of that name
            name = aliasName.get(ref);
            if (name == null || !script.attackGuess.contains(name)) return null;
        } else {
            name = baseName(ref);
        }
        List<UUID> ids = new ArrayList<>(attacking);
        Collections.sort(ids);
        for (UUID id : ids) {
            Permanent pm = game.getPermanent(id);
            if (pm != null && cardKey(pm).equals(name) && !usedNew.contains(id)) {
                usedNew.add(id);
                return id;
            }
        }
        return null;
    }

    /** Before combat: will this creature attack under the plan (its alias, or a "new:" entry of its name)? */
    boolean plannedAttacker(Permanent pm, Game game) {
        if (!inTurn(game) || windowIndex(game.getTurnStepType()) >= 2 || !seat.equals(b.seatOf(pm.getControllerId()))) return false;
        String alias = aliasIfUnmoved(pm, game);
        if (alias != null && script.attacks.containsKey(alias)) return script.attacks.get(alias);
        String name = cardKey(pm);
        for (Map.Entry<String, Boolean> e : script.attacks.entrySet()) {
            if (e.getValue() && e.getKey().startsWith("new:") && baseName(e.getKey()).equals(name)) return true;
        }
        return false;
    }

    /** The spec alias of an injected permanent that has not left the battlefield since. */
    String aliasIfUnmoved(Permanent pm, Game game) {
        Integer z = zcc0.get(pm.getId());
        if (z == null || z != pm.getZoneChangeCounter(game)) return null;
        return b.aliasOf(pm.getId());
    }

    static String baseName(String ref) {
        String n = ref.startsWith("new:") ? ref.substring(4) : ref;
        int h = n.lastIndexOf('#');
        return h > 0 ? n.substring(0, h) : n;
    }

    private static Map<String, Integer> newRefs(List<String> refs) {
        Map<String, Integer> m = new HashMap<>();
        for (String r : refs) m.merge(baseName(r), 1, Integer::sum);
        return m;
    }

    private static boolean take(Map<String, Integer> left, String name) {
        Integer k = left.get(name);
        if (k == null || k <= 0) return false;
        left.put(name, k - 1);
        return true;
    }

    /**
     * Recorded plays of the replayed turn that the engine did not carry out, whatever the end state.
     * An end state reached without them is a coincidence (a Stab on the user's own doomed creature
     * instead of the opponent's; a blocker killed before combat instead of in it), and the turn's
     * decisions would label the wrong play, so any of them means "not reproduced":
     *   unscripted:A          recorded actions of A the request could not script (script.unscripted)
     *   attack:A:unrealised   a recorded attacker (spec alias) the engine did not declare
     *   attack:A:extra        a creature recorded as not attacking that was declared anyway
     *   block:B:unrealised    a recorded block between two spec aliases the engine did not declare
     * (undone items are counted by TurnReplay.compare). Read from the engine's own declarations
     * (ReplayWatcher), so an attacker the puppet was never asked about counts too. Attackers and
     * blockers that entered during the turn ("new:" refs) are only flagged, as the soft flags
     * attack_new_unrealised / block_new_unrealised: 17lands sometimes lists a turn's attackers or
     * blockers twice, which labels turns into "new:" entries nothing can realise.
     * Returns key -> the refs concerned.
     */
    Map<String, List<String>> divergences(Game game) {
        Map<String, List<String>> out = new TreeMap<>();
        if (script.unscripted > 0) out.put("unscripted:" + seat, List.of(String.valueOf(script.unscripted)));
        ReplayWatcher w = game.getState().getWatcher(ReplayWatcher.class);
        if (w == null) return out;
        Set<String> att = w.attackers(turn);
        Map<String, Integer> guessPlanned = new TreeMap<>();
        Map<String, Integer> guessDid = new TreeMap<>();
        for (Map.Entry<String, Boolean> e : script.attacks.entrySet()) {
            if (e.getKey().startsWith("new:")) continue;
            UUID id = b.aliases.get(e.getKey());
            boolean did = id != null && att.contains(id.toString());
            String nm = aliasName.get(e.getKey());
            if (nm != null && script.attackGuess.contains(nm)) {
                // which copy attacked is a guess: only the number of attacking copies is recorded
                if (e.getValue()) guessPlanned.merge(nm, 1, Integer::sum);
                continue;
            }
            if (e.getValue() != did) {
                out.computeIfAbsent("attack:" + seat + (did ? ":extra" : ":unrealised"), k -> new ArrayList<>()).add(e.getKey());
            }
        }
        Set<UUID> counted = new HashSet<>();
        for (Map.Entry<String, String> e : aliasName.entrySet()) {
            UUID id = b.aliases.get(e.getKey());
            if (script.attackGuess.contains(e.getValue()) && att.contains(id.toString()) && counted.add(id)) guessDid.merge(e.getValue(), 1, Integer::sum);
        }
        Set<String> guessed = new TreeSet<>(guessPlanned.keySet());
        guessed.addAll(guessDid.keySet());
        for (String nm : guessed) {
            int d = guessDid.getOrDefault(nm, 0) - guessPlanned.getOrDefault(nm, 0);
            if (d != 0) out.computeIfAbsent("attack:" + seat + (d > 0 ? ":extra" : ":unrealised"), k -> new ArrayList<>()).add(nm + " x" + Math.abs(d));
        }
        Set<String> blk = w.blocks(turn);
        for (TurnScript.Block x : blockPlan) {
            if (x.attacker == null || x.blocker.startsWith("new:") || x.attacker.startsWith("new:")) continue;
            UUID bl = b.aliases.get(x.blocker);
            UUID at = b.aliases.get(x.attacker);
            boolean did = bl != null && at != null && blk.contains(bl + "|" + at);
            String an = aliasName.get(x.attacker);
            if (!did && bl != null && an != null && script.attackGuess.contains(an)) {
                // a guessed copy: any copy of that name blocked by this blocker
                for (Map.Entry<String, String> e : aliasName.entrySet()) {
                    if (an.equals(e.getValue()) && blk.contains(bl + "|" + b.aliases.get(e.getKey()))) did = true;
                }
            }
            if (!did) {
                out.computeIfAbsent("block:" + opp + ":unrealised", k -> new ArrayList<>()).add(x.blocker + " -> " + x.attacker);
            }
        }
        return out;
    }

    // =================================================================================== targets

    boolean chooseTarget(ReplayPlayer p, Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (fromCards != null && fromCards.isEmpty()) return false;
        UUID controller = target.getAffectedAbilityControllerId(p.getId());
        if (target.isChoiceCompleted(controller, source, game, fromCards)) return false;
        boolean inTurn = inTurn(game);
        for (int guard = 0; guard < 50; guard++) {
            Set<UUID> possible = new HashSet<>();
            for (UUID id : target.possibleTargets(controller, source, game, fromCards)) if (!target.contains(id)) possible.add(id);
            if (possible.isEmpty()) break;
            boolean canStop = target.isChosen(game);
            TargetResolver.Pick pk = resolver.pick(p, outcome, target, source, game, possible, canStop);
            boolean record = inTurn && p.seat.equals(recordSeat) && pk.options >= 2;
            List<String[]> labelled = new ArrayList<>();
            if (record) {
                for (UUID id : possible) labelled.add(new String[]{BridgePlayer.targetLabel(game, id, p.getId()), id.toString()});
                if (canStop) labelled.add(new String[]{"Stop Choosing", STOP_CHOOSING.toString()});
                // copies with one label (two Stabs in hand to discard) leave nothing to learn
                record = labelled.stream().map(x -> x[0]).distinct().count() >= 2;
            }
            if (record) {
                String text = (source == null ? "null" : source.getRule()) + ":Choose a target:" + target.getTargetName();
                JsonObject d = decision(game, p, "CHOOSE_TARGET", text);
                labelled.sort(Comparator.comparing((String[] x) -> x[0]).thenComparing(x -> x[1]));
                JsonArray legal = new JsonArray();
                Set<String> seen = new HashSet<>();
                for (String[] l : labelled) if (seen.add(l[0])) legal.add(option(l[0], p.actionEncoder.getTargetIndex(l[0])));
                d.add("legal", legal);
                d.addProperty("chosen", pk.label);
                d.addProperty("label_kind", pk.evidence.equals("fate") ? "exact" : "guessed_target");
                d.addProperty("evidence", pk.evidence);
                finishDecision(d, game, p, "CHOOSE_TARGET", text, new GraphRecord.Ask("CHOOSE_TARGET", text,
                        source == null ? null : source.getSourceId(), fromCards, optionIds(labelled)));
            }
            if (inTurn && pk.options >= 2) {
                JsonObject t = new JsonObject();
                t.addProperty("seat", p.seat);
                t.addProperty("source", source == null ? null : sourceName(source, game));
                t.addProperty("target", pk.label);
                t.addProperty("evidence", pk.evidence);
                t.addProperty("options", pk.options);
                targets.add(t);
                if (!pk.evidence.equals("fate")) flags.add("guessed_target");
            }
            if (STOP_CHOOSING.equals(pk.id)) {
                p.getPlayerHistory().targetSequence.add(STOP_CHOOSING);
                break;
            }
            int before = target.getTargets().size();
            target.addTarget(pk.id, source, game);
            if (target.getTargets().size() == before) target.add(pk.id, game);
            p.getPlayerHistory().targetSequence.add(pk.id);
            if (target.getTargets().size() == before) break; // refused: do not loop
            if (target.isChoiceCompleted(controller, source, game, fromCards)) break;
        }
        return target.isChosen(game) && !target.getTargets().isEmpty();
    }

    static String sourceName(Ability source, Game game) {
        MageObject o = source.getSourceId() == null ? null : game.getObject(source.getSourceId());
        return o == null ? source.getRule() : o.getName();
    }

    /** "May" questions (and kicker, "pay X?"): the puppet's default answer, recorded as a guess for the scripted seat. */
    JsonObject beforeUse(ReplayPlayer p, String message, Game game) {
        if (!inTurn(game)) return null;
        flags.add("guessed_use");
        if (!p.seat.equals(recordSeat)) return null;
        JsonObject d = decision(game, p, "CHOOSE_USE", message);
        JsonArray legal = new JsonArray();
        legal.add(option("no", 0));
        legal.add(option("yes", 1));
        d.add("legal", legal);
        if (encode) d.add("features", features(game, p, "CHOOSE_USE", message));
        return d;
    }

    void afterUse(JsonObject d, boolean out) {
        if (d == null) return;
        d.addProperty("chosen", out ? "yes" : "no");
        d.addProperty("label_kind", "guessed_target");
        d.addProperty("evidence", "default");
        decisions.add(d);
    }

    /**
     * A modal spell or ability: MageZero's own options are [no mode] + the available modes, chosen
     * by index (a CHOOSE_NUM decision). 17lands does not say which mode, so the attempt's policy
     * picks the k-th available mode (k = policy.mode, clamped), one mode per choice ("choose one").
     */
    mage.abilities.Mode chooseMode(ReplayPlayer p, mage.abilities.Modes modes, Ability source, Game game) {
        List<mage.abilities.Mode> options = new ArrayList<>();
        options.add(null);
        for (mage.abilities.Mode m : modes.getAvailableModes(source, game)) {
            if (!modes.getSelectedModes().contains(m.getId()) && m.getTargets().canChoose(source.getControllerId(), source, game)) options.add(m);
        }
        int idx;
        String evidence = "only_mode";
        if (options.size() == 1 || !modes.getSelectedModes().isEmpty()) {
            idx = 0;                                            // nothing (more) to choose: stop
        } else if (options.size() == 2) {
            idx = 1;
        } else {
            // a mode whose effect explains the snapshot: it creates a token the snapshot has more of,
            // or it removes / shrinks a permanent the snapshot no longer shows
            double[] sc = new double[options.size()];
            Map<String, Map<String, Integer>> bfNow = battlefieldCounts(game);
            String mine = p.seat;
            for (int i = 1; i < options.size(); i++) sc[i] = modeScore(options.get(i), source, game, bfNow, mine);
            int best = 1;
            for (int i = 2; i < options.size(); i++) if (sc[i] > sc[best]) best = i;
            int ties = 0;
            for (int i = 1; i < options.size(); i++) if (sc[i] == sc[best]) ties++;
            if (sc[best] >= 10 && ties == 1) {
                idx = best;
                evidence = "fate";
            } else {
                idx = 1 + Math.min(policy.mode, options.size() - 2);
                evidence = "policy";
                flags.add("guessed_mode");
            }
        }
        if (inTurn(game) && p.seat.equals(recordSeat) && options.size() > 1) {
            String text = "choose num for " + source;
            JsonObject d = decision(game, p, "CHOOSE_NUM", text);
            JsonArray legal = new JsonArray();
            for (int i = 0; i < options.size(); i++) legal.add(option(String.valueOf(i), -1));
            d.add("legal", legal);
            d.addProperty("chosen", String.valueOf(idx));
            d.addProperty("label_kind", evidence.equals("policy") ? "guessed_target" : "exact");
            d.addProperty("evidence", evidence);
            finishDecision(d, game, p, "CHOOSE_NUM", text, null);
        }
        // what ComputerPlayerMCTS.chooseMode records: nothing when there is no choice (makeChoiceAmount min >= max)
        if (options.size() > 1) p.getPlayerHistory().numSequence.add(idx);
        return options.get(idx);
    }

    /** The color most creatures of the affected player have (the chooser's for a benefit), then its deck's main color. */
    String color(ReplayPlayer p, Outcome outcome, Set<String> options, Game game) {
        UUID who = outcome != null && !outcome.isGood() ? game.getOpponent(p.getId()).getId() : p.getId();
        Map<String, Integer> n = new LinkedHashMap<>();
        for (String c : List.of("White", "Blue", "Black", "Red", "Green")) if (options.contains(c)) n.put(c, 0);
        for (Permanent pm : game.getBattlefield().getAllActivePermanents(who)) {
            if (!pm.isCreature(game)) continue;
            mage.ObjectColor col = pm.getColor(game);
            if (col.isWhite()) n.computeIfPresent("White", (k, v) -> v + 100);
            if (col.isBlue()) n.computeIfPresent("Blue", (k, v) -> v + 100);
            if (col.isBlack()) n.computeIfPresent("Black", (k, v) -> v + 100);
            if (col.isRed()) n.computeIfPresent("Red", (k, v) -> v + 100);
            if (col.isGreen()) n.computeIfPresent("Green", (k, v) -> v + 100);
        }
        for (Card c : game.getCards()) {
            if (!c.isOwnedBy(who)) continue;
            mage.ObjectColor col = c.getColor(game);
            if (col.isWhite()) n.computeIfPresent("White", (k, v) -> v + 1);
            if (col.isBlue()) n.computeIfPresent("Blue", (k, v) -> v + 1);
            if (col.isBlack()) n.computeIfPresent("Black", (k, v) -> v + 1);
            if (col.isRed()) n.computeIfPresent("Red", (k, v) -> v + 1);
            if (col.isGreen()) n.computeIfPresent("Green", (k, v) -> v + 1);
        }
        String best = null;
        for (Map.Entry<String, Integer> e : n.entrySet()) if (best == null || e.getValue() > n.get(best)) best = e.getKey();
        return best != null ? best : new TreeSet<>(options).first();
    }

    private double modeScore(mage.abilities.Mode m, Ability source, Game game, Map<String, Map<String, Integer>> bfNow, String mine) {
        String t = m.getEffects().getText(m).toLowerCase(Locale.ROOT);
        TurnScript.Expected ex = script.expected;
        if (!ex.hasBattlefield) return 0;
        double s = 0;
        if (t.contains("create")) {
            for (Map.Entry<String, Integer> e : ex.battlefield.getOrDefault(mine, Map.of()).entrySet()) {
                int more = e.getValue() - bfNow.getOrDefault(mine, Map.of()).getOrDefault(e.getKey(), 0);
                if (more > 0 && t.contains(e.getKey().toLowerCase(Locale.ROOT))) s += 10;
            }
        }
        boolean removes = t.contains("destroy") || t.contains("damage") || t.contains("exile") || t.contains("gets -")
                || t.contains("sacrifice") || t.contains("return target");
        if (removes) {
            String other = Spec.other(mine);
            for (Map.Entry<String, Integer> e : bfNow.getOrDefault(other, Map.of()).entrySet()) {
                if (e.getValue() > ex.battlefield.getOrDefault(other, Map.of()).getOrDefault(e.getKey(), 0)) {
                    s += 5;
                    break;
                }
            }
        }
        return s;
    }

    // =================================================================================== decisions

    JsonObject decision(Game game, ReplayPlayer p, String type, String text) {
        JsonObject d = new JsonObject();
        d.addProperty("type", type);
        d.addProperty("text", text);
        JsonObject where = new JsonObject();
        where.addProperty("turn", game.getTurnNum());
        where.addProperty("step", String.valueOf(game.getTurnStepType()));
        where.addProperty("stack", game.getStack().size());
        d.add("where", where);
        return d;
    }

    void finishDecision(JsonObject d, Game game, ReplayPlayer p, String type, String text, GraphRecord.Ask ask) {
        if (!p.seat.equals(recordSeat)) return;   // only the recorded seat's decisions are labels
        if (encode) d.add("features", features(game, p, type, text));
        if (graph && ask != null) d.add("graph", GraphRecord.encode(game, p.getId(), ask, perfectInfo));
        if (heuristic) d.addProperty("heuristic", GameStateEvaluator3.evaluateNormalized(p.getId(), game));
        decisions.add(d);
    }

    JsonArray features(Game game, ReplayPlayer p, String type, String text) {
        StateEncoder enc = new StateEncoder();
        UUID me = p.getId();
        enc.setAgent(me);
        enc.setOpponent(game.getOpponent(me).getId());
        enc.perfectInfo = perfectInfo;
        Set<Integer> fv = enc.processState(game, me, ActionEncoder.ActionType.valueOf(type), text);
        JsonArray a = new JsonArray();
        fv.stream().mapToInt(Integer::intValue).sorted().forEach(a::add);
        return a;
    }

    static JsonObject option(String label, int idx) {
        JsonObject o = new JsonObject();
        o.addProperty("label", label);
        o.addProperty("idx", idx);
        return o;
    }

    void fail(RuntimeException e) {
        if (failure == null) failure = e;
    }

    // =================================================================================== state

    /** The name a permanent is compared by: a card's printed name, a token's name without " Token". */
    static String cardKey(Permanent pm) {
        if (pm instanceof PermanentToken) {
            String n = pm.getName();
            return n.endsWith(" Token") ? n.substring(0, n.length() - 6) : n;
        }
        if (pm instanceof PermanentCard) return ((PermanentCard) pm).getCard().getName();
        return pm.getName();
    }

    Map<String, Map<String, Integer>> battlefieldCounts(Game game) {
        Map<String, Map<String, Integer>> out = new LinkedHashMap<>();
        for (String s : Spec.SEATS) out.put(s, new TreeMap<>());
        for (Permanent pm : game.getBattlefield().getAllActivePermanents()) {
            String s = b.seatOf(pm.getControllerId());
            if (s != null) out.get(s).merge(cardKey(pm), 1, Integer::sum);
        }
        return out;
    }

    /** Copies of `name` the scripted seat still has to cast this turn (so they stay in hand until then). */
    int pendingFromHand(String name) {
        return pendingFromHand(seat, name);
    }

    int pendingFromHand(String who, String name) {
        int n = 0;
        for (TurnScript.Item it : (who.equals(seat) ? mine : theirs)) {
            if (!it.done && it.kind.equals("cast") && name.equals(it.name) && it.key.startsWith("Cast ")) n++;
        }
        return n;
    }

    // =================================================================================== card facts

    /** What the window heuristics need from a card, cached per name (database lookups only). */
    static final class Facts {
        private static final Map<String, Optional<Facts>> CACHE = new ConcurrentHashMap<>();
        boolean creature, instant, land, permanent;
        int mv;
        String rules;
        mage.Mana cost;           // the printed mana cost (colored pips feed the mana reservation)

        static Facts of(String name) {
            if (name == null) return null;
            return CACHE.computeIfAbsent(name, n -> {
                try {
                    CardInfo ci = StateInjector.cardInfo(n, null, null);
                    Facts f = new Facts();
                    List<CardType> types = ci.getTypes();
                    f.creature = types.contains(CardType.CREATURE);
                    f.instant = types.contains(CardType.INSTANT);
                    f.land = types.contains(CardType.LAND);
                    f.permanent = types.stream().anyMatch(CardType::isPermanentType);
                    f.mv = ci.getManaValue();
                    f.rules = String.join(" ", ci.getRules()).toLowerCase(Locale.ROOT);
                    try {
                        f.cost = StateInjector.newCard(n, null, null).getManaCost().getMana();
                    } catch (RuntimeException e) {
                        f.cost = null;
                    }
                    return Optional.of(f);
                } catch (RuntimeException e) {
                    return Optional.empty();
                }
            }).orElse(null);
        }
    }
}
