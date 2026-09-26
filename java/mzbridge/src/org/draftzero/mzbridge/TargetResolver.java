package org.draftzero.mzbridge;

import mage.MageObject;
import mage.abilities.Ability;
import mage.cards.Card;
import mage.constants.Outcome;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.Spell;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.target.Target;

import java.util.*;

import static mage.target.TargetImpl.STOP_CHOOSING;

/**
 * Picks the targets of a replayed turn (17lands records none). Every candidate gets a score from
 * the end-of-turn snapshot the replay must reach; the best one wins:
 *
 *  - a permanent of the other seat that is on the battlefield more often than the snapshot allows
 *    ("over") should leave: +10 for an effect that is bad for it (destroy, damage, exile, bounce,
 *    sacrifice), +5 more when its name is in its controller's non-combat deaths. When the other
 *    seat's permanents are candidates too, the chooser's own doomed permanents score only 0.5
 *    (0.8 with the non-combat death) for a bad effect: below any of the other seat's candidates
 *    (1), since they usually leave by the other seat's hand or in combat (a Stab on the user's own
 *    creature the opponent killed would reproduce the end state by accident), above its own
 *    survivors (-8). Among the chooser's permanents only (a sacrifice, "creature you control")
 *    they keep the full score. An effect that is good for it prefers a permanent of the chooser's
 *    that stays (+3, +1 attacking or blocking);
 *  - a card in the scripted seat's hand beyond what the snapshot hand and the rest of the script
 *    need should leave it (discard: +10);
 *  - a card in a library or graveyard whose name the snapshot needs in hand or on the battlefield
 *    more often than now should come (search, raise dead: +10);
 *  - a spell on the stack that must not resolve (a permanent spell the snapshot does not show)
 *    is the one to counter (+10);
 *  - a player: a harmful effect goes to the opponent (+6 while its snapshot life is still lower
 *    than now, else +1), a good one to the chooser.
 *
 * evidence "fate": the best score is at least 10 and no candidate with a different name ties it
 * (copies of one card are interchangeable); "fate_ambiguous": several names tie; "heuristic": no
 * candidate is explained by the snapshot. Ties break on (power, mana value, label, uuid) so the
 * choice is deterministic. Optional targets ("up to", canStop) are skipped without evidence.
 */
final class TargetResolver {

    static final class Pick {
        UUID id;
        String label;
        String evidence;
        double score;
        int options;
    }

    private final ReplayRun run;

    TargetResolver(ReplayRun run) {
        this.run = run;
    }

    /** Card text that makes an effect bad for its target whatever Outcome the card declares (an Aura
     * like Witness Protection declares a "good" attach outcome but takes all abilities away). */
    static final List<String> HARMFUL_TEXT = List.of("loses all abilities", "can't attack", "can't block", "gets -",
            "destroy target", "exile target", "damage to target", "damage to any target", "tap target", "doesn't untap",
            "fights target", "fights another target", "counter target", "-1/-1 counter");

    static boolean harmfulText(Ability source, Game game) {
        if (source == null) return false;
        StringBuilder sb = new StringBuilder(String.valueOf(source.getRule()));
        MageObject o = source.getSourceId() == null ? null : game.getObject(source.getSourceId());
        if (o instanceof Spell) o = ((Spell) o).getCard();
        if (o instanceof Card && !(o instanceof Permanent)) sb.append(' ').append(String.join(" ", ((Card) o).getRules(game)));
        String t = sb.toString().toLowerCase(Locale.ROOT);
        for (String h : HARMFUL_TEXT) if (t.contains(h)) return true;
        return false;
    }

    Pick pick(ReplayPlayer p, Outcome outcome, Target target, Ability source, Game game, Set<UUID> possible, boolean canStop) {
        boolean good = outcome != null && outcome.isGood() && !harmfulText(source, game);
        List<Pick> scored = new ArrayList<>();
        Map<String, Map<String, Integer>> bfNow = run.battlefieldCounts(game);
        boolean mixed = false;           // a permanent of the other seat among the candidates
        for (UUID id : possible) {
            Permanent pm = game.getPermanent(id);
            if (pm != null && !pm.getControllerId().equals(p.getId())) mixed = true;
        }
        for (UUID id : possible) {
            Pick k = new Pick();
            k.id = id;
            k.label = BridgePlayer.targetLabel(game, id, p.getId());
            k.score = score(p, id, good, mixed, game, bfNow);
            scored.add(k);
        }
        // ties: the biggest creature first (policy "low": the smallest), then label and id
        int dir = "low".equals(run.policy.targetOrder) ? 1 : -1;
        scored.sort(Comparator.comparingDouble((Pick x) -> -x.score)
                .thenComparingInt(x -> dir * power(game, x.id))
                .thenComparingInt(x -> dir * manaValue(game, x.id))
                .thenComparing(x -> x.label)
                .thenComparing(x -> x.id.toString()));
        Pick best = scored.get(0);
        best.options = possible.size() + (canStop ? 1 : 0);
        // an optional target ("up to") is taken unless the snapshot argues against every candidate
        if (canStop && best.score < 0) {
            Pick stop = new Pick();
            stop.id = STOP_CHOOSING;
            stop.label = "Stop Choosing";
            stop.evidence = "no_evidence";
            stop.options = best.options;
            return stop;
        }
        if (best.score >= 10) {
            boolean otherName = false;
            for (Pick x : scored) {
                if (x != best && x.score == best.score && !nameOf(game, x.id).equals(nameOf(game, best.id))) otherName = true;
            }
            best.evidence = otherName ? "fate_ambiguous" : "fate";
        } else {
            best.evidence = "heuristic";
        }
        return best;
    }

    /** Should the scripted opponent counter this spell? A permanent spell the snapshot does not show. */
    boolean shouldCounter(Game game, StackObject so, String counteringSeat) {
        if (!(so instanceof Spell)) return false;
        Spell s = (Spell) so;
        String seat = run.b.seatOf(s.getControllerId());
        if (seat == null || seat.equals(counteringSeat)) return false;
        Card c = s.getCard();
        if (c == null) return false;
        if (c.isPermanent(game)) {
            Map<String, Integer> exp = run.script.expected.battlefield.get(seat);
            if (exp == null) return false;
            int now = run.battlefieldCounts(game).getOrDefault(seat, Map.of()).getOrDefault(c.getName(), 0);
            return exp.getOrDefault(c.getName(), 0) <= now;
        }
        return "respond".equals(run.policy.bWindow); // an instant or sorcery: only when an attempt says so
    }

    // ------------------------------------------------------------------ scoring

    private double score(ReplayPlayer p, UUID id, boolean good, boolean mixed, Game game, Map<String, Map<String, Integer>> bfNow) {
        TurnScript.Expected ex = run.script.expected;
        Player pl = game.getPlayer(id);
        if (pl != null) {
            boolean self = id.equals(p.getId());
            String seat = run.b.seatOf(id);
            Integer want = ex.life.get(seat);
            boolean losesMore = want != null && want < pl.getLife();
            boolean gainsMore = want != null && want > pl.getLife();
            if (good) return self ? (gainsMore ? 6 : 3) : -5;
            return self ? -10 : (losesMore ? 6 : 1);
        }
        Permanent pm = game.getPermanent(id);
        if (pm != null) {
            String seat = run.b.seatOf(pm.getControllerId());
            String key = ReplayRun.cardKey(pm);
            int over = 0;
            if (ex.hasBattlefield && seat != null) {
                over = bfNow.getOrDefault(seat, Map.of()).getOrDefault(key, 0)
                        - ex.battlefield.getOrDefault(seat, Map.of()).getOrDefault(key, 0);
            }
            boolean own = pm.getControllerId().equals(p.getId());
            if (!good) {
                if (over > 0) {
                    Map<String, Integer> nc = ex.deaths.getOrDefault(seat, Map.of()).getOrDefault("noncombat", Map.of());
                    boolean died = nc.getOrDefault(key, 0) > 0 && pm.isCreature(game);
                    if (own && mixed) return died ? 0.8 : 0.5;
                    return died ? 15 : 10;
                }
                return own ? -8 : 1;
            }
            if (!own) return -5;
            if (over > 0) return -2;
            double s = 3;
            if (pm.isCreature(game)) s += 1;
            // a pump or an Equipment goes on a creature that attacks (or is about to: the attack
            // plan), most of all one fighting a creature the snapshot says dies
            if (pm.isAttacking() || pm.getBlocking() > 0 || run.plannedAttacker(pm, game)) s += 2;
            s += fightsDoomed(pm, game, bfNow);
            return s;
        }
        StackObject so = game.getStack().getStackObject(id);
        if (so != null) {
            String seat = run.b.seatOf(so.getControllerId());
            String mine = run.b.seatOf(p.getId());
            if (seat != null && seat.equals(mine)) return good ? 3 : -10;
            return shouldCounter(game, so, mine) ? 10 : 2;
        }
        Card card = game.getCard(id);
        if (card != null) {
            Zone zone = game.getState().getZone(id);
            String owner = run.b.seatOf(card.getOwnerId());
            String name = card.getName();
            if (zone == Zone.HAND) {
                if (owner == null) return 0;
                if (!owner.equals(run.seat)) {
                    // the other seat's hidden hand: keep the cards its script still has to play
                    // (its own cleanup discard happens at the start of an end-of-turn entry)
                    return run.pendingFromHand(owner, name) > 0 ? (good ? 5 : -10) : 0;
                }
                if (!ex.hasHand) return 0;
                int inHand = handCount(game, owner, name);
                int keep = ex.handA.getOrDefault(name, 0) + run.pendingFromHand(name);
                int extra = inHand - keep;
                if (!good) return extra > 0 ? 10 : -5;
                return extra > 0 ? 5 : -1;       // put onto the battlefield / cast from hand
            }
            if (zone == Zone.LIBRARY || zone == Zone.GRAVEYARD) {
                if (owner == null) return 0;
                int need = needed(game, owner, name, bfNow);
                double creature = card.isCreature(game) ? 0.5 : 0;
                if (zone == Zone.GRAVEYARD && !good) {
                    // exile from a graveyard: the opponent's cards (creature cards feed "if a creature card was exiled")
                    boolean mineCard = owner.equals(run.b.seatOf(p.getId()));
                    return mineCard ? (need > 0 ? -5 : -1) : 1 + creature;
                }
                return need > 0 ? 10 : creature;
            }
            return 0;
        }
        return 0;
    }

    /**
     * In combat against a creature the snapshot says dies (on the battlefield more often than it
     * allows): +6 when that creature would survive the combat as things stand (the pump is what
     * kills it), +2 when it dies anyway.
     */
    private int fightsDoomed(Permanent pm, Game game, Map<String, Map<String, Integer>> bfNow) {
        if (!run.script.expected.hasBattlefield) return 0;
        mage.game.combat.CombatGroup g = game.getCombat().findGroup(pm.getId());
        if (g == null) g = game.getCombat().findGroupOfBlocker(pm.getId());
        if (g == null) return 0;
        List<UUID> others = new ArrayList<>(pm.isAttacking() ? g.getBlockers() : g.getAttackers());
        int best = 0;
        for (UUID id : others) {
            Permanent o = game.getPermanent(id);
            if (o == null || !o.isCreature(game)) continue;
            String seat = run.b.seatOf(o.getControllerId());
            String key = ReplayRun.cardKey(o);
            int over = bfNow.getOrDefault(seat, Map.of()).getOrDefault(key, 0)
                    - run.script.expected.battlefield.getOrDefault(seat, Map.of()).getOrDefault(key, 0);
            if (over <= 0) continue;
            boolean survives = o.getToughness().getValue() - o.getDamage() > pm.getPower().getValue();
            best = Math.max(best, survives ? 6 : 2);
        }
        return best;
    }

    /** How many more copies of `name` the snapshot needs in `seat`'s hand and on its battlefield than it has now. */
    private int needed(Game game, String seat, String name, Map<String, Map<String, Integer>> bfNow) {
        TurnScript.Expected ex = run.script.expected;
        int want = ex.battlefield.getOrDefault(seat, Map.of()).getOrDefault(name, 0);
        int have = bfNow.getOrDefault(seat, Map.of()).getOrDefault(name, 0);
        if (seat.equals(run.seat) && ex.hasHand) {
            want += ex.handA.getOrDefault(name, 0) + run.pendingFromHand(name);
            have += handCount(game, seat, name);
        }
        return want - have;
    }

    private int handCount(Game game, String seat, String name) {
        Player pl = run.b.players.get(seat);
        int n = 0;
        for (Card c : game.getPlayer(pl.getId()).getHand().getCards(game)) if (c.getName().equals(name)) n++;
        return n;
    }

    private static String nameOf(Game game, UUID id) {
        Permanent pm = game.getPermanent(id);
        if (pm != null) return ReplayRun.cardKey(pm);
        Card c = game.getCard(id);
        if (c != null) return c.getName();
        return String.valueOf(game.getEntityName(id, null));
    }

    private static int power(Game game, UUID id) {
        Permanent pm = game.getPermanent(id);
        return pm != null && pm.isCreature(game) ? pm.getPower().getValue() : 0;
    }

    private static int manaValue(Game game, UUID id) {
        Permanent pm = game.getPermanent(id);
        if (pm != null) return pm.getManaValue();
        Card c = game.getCard(id);
        return c != null ? c.getManaValue() : 0;
    }
}
