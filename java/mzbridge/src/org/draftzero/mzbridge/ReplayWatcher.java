package org.draftzero.mzbridge;

import mage.constants.TurnPhase;
import mage.constants.WatcherScope;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.events.ZoneChangeEvent;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import mage.players.Player;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * What a replayed turn needs to remember beyond the final state:
 *  - the non-token creatures that die, as 17lands' *_creatures_killed_* columns count them (tokens
 *    never appear there): "turn|seat|combat|name", seat = the owner (17lands lists a creature
 *    stolen with Involuntary Employment and sacrificed under its owner), combat = it died during
 *    the combat phase;
 *  - the active player's hand as the cleanup step begins, before the discard to hand size:
 *    17lands' end-of-turn snapshot is taken there (a user hand of 8 at the snapshot is followed by
 *    7 at the next one, with no discard recorded);
 *  - the attacks and blocks the engine accepted (ATTACKER_DECLARED, BLOCKER_DECLARED), so the
 *    replay can tell a recorded attack or block that never happened even when the puppet was not
 *    asked (no available attacker left, the block plan refused).
 * Read back with {@code game.getState().getWatcher(ReplayWatcher.class)}: rollbacks restore a copy,
 * so a reference kept from before the replay can be stale.
 *
 * Watcher.copy() rebuilds watchers reflectively from their only constructor and deep-copies the
 * fields, hence one public no-argument constructor and no final fields.
 */
public class ReplayWatcher extends mage.watchers.Watcher {
    private List<String> deaths = new ArrayList<>();
    private List<String> cleanupHands = new ArrayList<>();
    private List<String> combat = new ArrayList<>();

    public ReplayWatcher() {
        super(WatcherScope.GAME);
    }

    @Override
    public void watch(GameEvent event, Game game) {
        if (event.getType() == GameEvent.EventType.ATTACKER_DECLARED) {
            combat.add(game.getTurnNum() + "|att|" + event.getSourceId());          // source = the attacker
            return;
        }
        if (event.getType() == GameEvent.EventType.BLOCKER_DECLARED) {
            combat.add(game.getTurnNum() + "|blk|" + event.getSourceId() + "|" + event.getTargetId()); // blocker|attacker
            return;
        }
        if (event.getType() == GameEvent.EventType.CLEANUP_STEP_PRE) {
            Player a = game.getPlayer(game.getActivePlayerId());
            if (a == null) return;
            StringBuilder sb = new StringBuilder().append(game.getTurnNum()).append('|').append("PlayerA".equals(a.getName()) ? "A" : "B");
            for (mage.cards.Card c : a.getHand().getCards(game)) sb.append('|').append(c.getName());
            cleanupHands.add(sb.toString());
            return;
        }
        if (event.getType() != GameEvent.EventType.ZONE_CHANGE || !(event instanceof ZoneChangeEvent)) return;
        ZoneChangeEvent z = (ZoneChangeEvent) event;
        if (!z.isDiesEvent()) return;
        Permanent p = z.getTarget();
        if (p == null || p instanceof PermanentToken || !p.isCreature(game)) return;
        Player c = game.getPlayer(p.getOwnerId());
        String seat = c != null && "PlayerA".equals(c.getName()) ? "A" : "B";
        boolean combat = game.getTurnPhaseType() == TurnPhase.COMBAT;
        deaths.add(game.getTurnNum() + "|" + seat + "|" + (combat ? "combat" : "noncombat") + "|" + ReplayRun.cardKey(p));
    }

    @Override
    public void reset() {
        // keep the record across the turn boundary (GameState.resetWatchers runs at cleanup)
    }

    List<String> deaths() {
        return deaths;
    }

    /** Ids of the creatures declared as attackers in `turn`. */
    Set<String> attackers(int turn) {
        Set<String> out = new HashSet<>();
        for (String e : combat) {
            String[] f = e.split("\\|");
            if (Integer.parseInt(f[0]) == turn && f[1].equals("att")) out.add(f[2]);
        }
        return out;
    }

    /** "blockerId|attackerId" of every block declared in `turn`. */
    Set<String> blocks(int turn) {
        Set<String> out = new HashSet<>();
        for (String e : combat) {
            String[] f = e.split("\\|");
            if (Integer.parseInt(f[0]) == turn && f[1].equals("blk")) out.add(f[2] + "|" + f[3]);
        }
        return out;
    }

    /** The hand `seat` held as the cleanup step of `turn` began, or null if that cleanup did not start. */
    List<String> handAtCleanup(int turn, String seat) {
        List<String> out = null;
        for (String e : cleanupHands) {
            String[] f = e.split("\\|", -1);
            if (Integer.parseInt(f[0]) != turn || !f[1].equals(seat)) continue;
            out = new ArrayList<>(Arrays.asList(f).subList(2, f.length));
        }
        return out;
    }
}
