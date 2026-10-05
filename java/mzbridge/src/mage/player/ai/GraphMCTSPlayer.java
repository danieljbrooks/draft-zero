package mage.player.ai;

import mage.abilities.Ability;
import mage.cards.Cards;
import mage.constants.Outcome;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.player.ai.encoder.StateEncoder;
import mage.players.ChooseCreatureToBlockAbility;
import mage.target.Target;
import mage.target.common.TargetAttackingCreature;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;

/**
 * MageZero's simulation player (MCTSPlayer), unchanged in what it decides, that also remembers what
 * the graph encoder needs about its latest decision and v0.2's MCTSPlayer doesn't keep: the
 * decision's source object (the graph's DecisionSource edge), a target's candidate cards, and which
 * creature an attack or block question is about (docs/022 §3.3). BenchPlayer builds its searches'
 * games with it when the seat's network is a graph network.
 *
 * The attack and block loops are v0.2's ComputerPlayer.selectAttackersOneAtATime and
 * selectBlockersOneAtATime line for line, with the creature noted before each question.
 */
public class GraphMCTSPlayer extends MCTSPlayer {

    /** the source of the latest target or yes/no question (null: none, or a priority decision) */
    public transient UUID decisionSource;
    /** a target question's candidate cards outside the zones the encoder walks */
    public transient Cards decisionCards;
    /** the creature the current "attack with X?" question is about, and the player it would attack */
    public transient UUID attacker;
    public transient UUID defender;
    /** the creature the current block question is about */
    public transient UUID blocker;

    public GraphMCTSPlayer(UUID id, UUID targetPlayer, StateEncoder encoder) {
        super(id, targetPlayer, encoder);
    }

    public GraphMCTSPlayer(final GraphMCTSPlayer p) {
        super(p);
        this.decisionSource = p.decisionSource;
        this.decisionCards = p.decisionCards;
        this.attacker = p.attacker;
        this.defender = p.defender;
        this.blocker = p.blocker;
    }

    @Override
    public GraphMCTSPlayer copy() {
        return new GraphMCTSPlayer(this);
    }

    @Override
    public void selectAttackersOneAtATime(Game game, UUID attackingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) {
            UUID opponentId = game.getCombat().getDefenders().iterator().next();
            List<Permanent> availableAttackers = getAvailableAttackers(game);
            availableAttackers.sort(Comparator.comparing(Permanent::getId)); // need deterministic order
            for (Permanent a : availableAttackers) {
                attacker = a.getId();
                defender = opponentId;
                boolean willAttack = chooseUse(Outcome.Neutral, "attack with: " + a.getName() + "?", null, game);
                if (willAttack) {
                    this.declareAttacker(a.getId(), opponentId, game, false);
                }
            }
            game.getPlayers().resetPassed();
        }
    }

    @Override
    public void selectBlockersOneAtATime(Ability source, Game game, UUID defendingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
        if (!game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) {
            List<Permanent> blockers = getAvailableBlockers(game);
            blockers.sort(Comparator.comparing(Permanent::getId));
            for (Permanent b : blockers) {
                blocker = b.getId();
                Target attackerTarget = new TargetAttackingCreature(0, 1);
                makeChoice(Outcome.Neutral, attackerTarget, new ChooseCreatureToBlockAbility("choose which creature to block for " + b.getName()), game, null);
                UUID attackerId = attackerTarget.getFirstTarget();
                declareBlocker(defendingPlayerId, b.getId(), attackerId, game);
            }
            game.getPlayers().resetPassed();
        }
    }

    @Override
    protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (!game.isPaused() && !game.checkIfGameIsOver()) {
            decisionSource = source instanceof ChooseCreatureToBlockAbility ? blocker
                    : source == null ? null : source.getSourceId();
            decisionCards = fromCards;
        }
        return super.makeChoice(outcome, target, source, game, fromCards);
    }

    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage, String trueText, String falseText,
                             Ability source, Game game) {
        if (!game.isPaused() && !game.checkIfGameIsOver()) {
            decisionSource = source == null ? null : source.getSourceId();
            decisionCards = null;
        }
        return super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
    }
}
