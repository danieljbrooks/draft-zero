package org.draftzero.mzbridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import mage.Mana;
import mage.cards.Card;
import mage.counters.Counter;
import mage.counters.CounterType;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentCard;
import mage.game.permanent.PermanentImpl;
import mage.game.permanent.PermanentToken;
import mage.game.stack.StackObject;
import mage.players.Player;

import java.util.*;

/**
 * A StateSpec-shaped JSON view of a live game, for round-trip checks (spec -> build -> dump ->
 * compare) and for looking at the state a decision was taken in. Fields the spec has keep the
 * spec's names and meaning; engine-only details (P/T, canAttack, the full library, the engine's
 * own step) go under "x", which StateSpec.from_dict ignores.
 */
final class Dumper {
    private Dumper() {
    }

    static JsonObject dump(StateInjector.Built b, boolean withLibrary) {
        Game game = b.game;
        Spec spec = b.spec;
        JsonObject d = new JsonObject();
        d.addProperty("version", Spec.SCHEMA_VERSION);
        d.addProperty("turn", game.getTurnNum());
        d.addProperty("activePlayer", b.seatOf(game.getActivePlayerId()));
        boolean atEntry = game.getStep() != null && game.getStep().getType() == b.engineStep && game.getTurnNum() == spec.turn;
        // BEGIN_STEP is held one step early inside the engine (StateInjector.setTurnPosition)
        d.addProperty("phase", atEntry ? spec.phase : name(game.getTurnPhaseType()));
        d.addProperty("step", atEntry ? spec.step : name(game.getTurnStepType()));
        if (atEntry) d.addProperty("enterMode", spec.enterMode);
        d.addProperty("priorityPlayer", b.seatOf(game.getState().getPriorityPlayerId()));
        JsonArray passed = new JsonArray();
        for (String seat : Spec.SEATS) if (b.players.get(seat).isPassed()) passed.add(seat);
        d.add("passedPlayers", passed);
        d.addProperty("startingPlayer", b.seatOf(game.getStartingPlayerId()));

        JsonObject players = new JsonObject();
        for (String seat : Spec.SEATS) players.add(seat, player(b, seat, withLibrary));
        d.add("players", players);

        // stack, bottom to top (SpellStack iterates top first)
        List<StackObject> objs = new ArrayList<>();
        for (StackObject so : game.getStack()) objs.add(so);
        Collections.reverse(objs);
        JsonArray stack = new JsonArray();
        for (StackObject so : objs) {
            JsonObject s = new JsonObject();
            s.addProperty("controller", b.seatOf(so.getControllerId()));
            s.addProperty("card", so.getName());
            JsonArray targets = new JsonArray();
            so.getStackAbility().getTargets().forEach(t -> t.getTargets().forEach(id -> targets.add(ref(b, id))));
            s.add("targets", targets);
            stack.add(s);
        }
        d.add("stack", stack);

        JsonArray attackers = new JsonArray();
        JsonArray blockers = new JsonArray();
        for (CombatGroup g : game.getCombat().getGroups()) {
            for (UUID a : g.getAttackers()) {
                JsonObject at = new JsonObject();
                at.addProperty("attacker", ref(b, a));
                at.addProperty("defender", ref(b, g.getDefenderId()));
                attackers.add(at);
                for (UUID bl : g.getBlockers()) {
                    JsonObject bo = new JsonObject();
                    bo.addProperty("blocker", ref(b, bl));
                    bo.addProperty("attacker", ref(b, a));
                    blockers.add(bo);
                }
            }
        }
        d.add("attackers", attackers);
        d.add("blockers", blockers);

        JsonObject x = new JsonObject();
        x.addProperty("engineStep", name(game.getTurnStepType()));
        x.addProperty("stepPart", game.getStep() == null || game.getStep().getStepPart() == null ? null : game.getStep().getStepPart().name());
        x.addProperty("paused", game.isPaused());
        x.addProperty("gameOver", game.checkIfGameIsOver());
        d.add("x", x);
        return d;
    }

    static JsonObject player(StateInjector.Built b, String seat, boolean withLibrary) {
        Game game = b.game;
        Player p = b.players.get(seat);
        Spec.PlayerState ps = b.spec.players.get(seat);
        JsonObject o = new JsonObject();
        o.addProperty("name", ps.name);
        o.addProperty("life", p.getLife());
        o.add("decklist", strings(ps.decklist));
        o.addProperty("decklistSource", ps.decklistSource);
        o.addProperty("landsPlayed", p.getLandsPlayed());
        o.add("hand", names(p.getHand().getCards(game)));
        o.addProperty("handUnknown", 0);
        o.add("graveyard", names(p.getGraveyard().getCards(game)));
        List<Card> exiled = new ArrayList<>();
        for (Card c : game.getExile().getAllCards(game)) if (c.isOwnedBy(p.getId())) exiled.add(c);
        o.add("exile", names(exiled));
        List<Card> lib = p.getLibrary().getCards(game);
        int k = Math.min(lib.size(), ps.libraryTop.size());
        o.add("libraryTop", names(lib.subList(0, k)));
        o.addProperty("librarySize", lib.size());
        String pool = poolString(p.getManaPool().getMana());
        if (!pool.isEmpty()) o.addProperty("manaPool", pool);
        JsonArray bf = new JsonArray();
        for (Permanent pm : game.getBattlefield().getAllActivePermanents(p.getId())) bf.add(perm(b, pm));
        o.add("battlefield", bf);
        JsonObject x = new JsonObject();
        x.addProperty("turnsTaken", p.getTurns());
        if (withLibrary) x.add("library", names(lib));
        o.add("x", x);
        return o;
    }

    static JsonObject perm(StateInjector.Built b, Permanent pm) {
        Game game = b.game;
        JsonObject o = new JsonObject();
        Spec.Perm origin = b.origin.get(pm.getId());
        if (pm instanceof PermanentToken) {
            if (origin != null && origin.token != null) {
                o.addProperty("token", origin.token);
                if (origin.set != null) o.addProperty("set", origin.set);
            }
            o.addProperty("tokenClass", ((PermanentToken) pm).getToken().getClass().getSimpleName());
        } else {
            // the card's own name: effects can rename a permanent (Witness Protection -> "Legitimate
            // Businessperson"); the effective name is x.name
            o.addProperty("name", pm instanceof PermanentCard ? ((PermanentCard) pm).getCard().getName() : pm.getName());
        }
        if (origin != null && origin.id != null) o.addProperty("id", origin.id);
        o.addProperty("tapped", pm.isTapped());
        o.addProperty("sick", !(boolean) Reflect.get(PermanentImpl.class, pm, "controlledFromStartOfControllerTurn"));
        o.addProperty("damage", pm.getDamage());
        JsonObject counters = new JsonObject();
        TreeMap<String, Integer> sorted = new TreeMap<>();
        for (Counter c : pm.getCounters(game).values()) {
            CounterType ct = CounterType.findByName(c.getName());
            sorted.put(ct == null ? c.getName() : ct.name(), c.getCount());
        }
        sorted.forEach(counters::addProperty);
        o.add("counters", counters);
        if (pm.getAttachedTo() != null) o.addProperty("attachTo", ref(b, pm.getAttachedTo()));
        if (pm.isFaceDown(game)) o.addProperty("faceDown", true);
        JsonObject x = new JsonObject();
        x.addProperty("name", pm.getName());
        x.addProperty("uuid", pm.getId().toString()); // reproducible per (spec, idSeed): see DeterministicIds
        if (pm.isCreature(game)) {
            x.addProperty("power", pm.getPower().getValue());
            x.addProperty("toughness", pm.getToughness().getValue());
            Player opp = game.getOpponent(pm.getControllerId());
            // rules checks as of now: attack the opponent this turn / block anything
            x.addProperty("canAttack", game.isActivePlayer(pm.getControllerId()) && !pm.isTapped() && pm.canAttack(opp.getId(), game));
            x.addProperty("canBlock", !pm.isTapped() && pm.canBlockAny(game));
            if (pm.isAttacking()) x.addProperty("attacking", true);
            if (pm.getBlocking() > 0) x.addProperty("blocking", true);
        }
        if (!pm.getAttachments().isEmpty()) {
            JsonArray att = new JsonArray();
            for (UUID id : pm.getAttachments()) att.add(game.getEntityName(id, null));
            x.add("attachments", att);
        }
        o.add("x", x);
        return o;
    }

    /** "<seat>:<alias>" for aliased permanents, "player:A" for players, else "<seat>:?<name>". */
    static String ref(StateInjector.Built b, UUID id) {
        if (id == null) return null;
        String seat = b.seatOf(id);
        if (seat != null) return "player:" + seat;
        String alias = b.aliasOf(id);
        if (alias != null) return alias;
        Permanent pm = b.game.getPermanent(id);
        if (pm != null) return b.seatOf(pm.getControllerId()) + ":?" + pm.getName();
        return "?" + b.game.getEntityName(id, null);
    }

    static String poolString(Mana m) {
        StringBuilder sb = new StringBuilder();
        sb.append("W".repeat(m.getWhite())).append("U".repeat(m.getBlue())).append("B".repeat(m.getBlack()))
                .append("R".repeat(m.getRed())).append("G".repeat(m.getGreen())).append("C".repeat(m.getColorless()));
        return sb.toString();
    }

    static JsonArray names(Collection<Card> cards) {
        JsonArray a = new JsonArray();
        for (Card c : cards) a.add(c.getName());
        return a;
    }

    static JsonArray strings(Collection<String> xs) {
        JsonArray a = new JsonArray();
        for (String s : xs) a.add(s);
        return a;
    }

    static String name(Enum<?> e) {
        return e == null ? null : e.name();
    }
}
