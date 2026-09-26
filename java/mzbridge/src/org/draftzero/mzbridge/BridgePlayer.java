package org.draftzero.mzbridge;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameImpl;
import mage.player.ai.ComputerPlayerMCTS2;
import mage.player.ai.MCTSNode;
import mage.player.ai.MCTSNode2;
import mage.player.ai.encoder.ActionEncoder;
import mage.players.ChooseCreatureToBlockAbility;
import mage.target.Target;

import java.util.*;
import java.util.stream.Collectors;

import static mage.target.TargetImpl.STOP_CHOOSING;

/**
 * Both seats of a bridge game. A {@code ComputerPlayerMCTS2} (so the anchor copies, histories and
 * MCTS simulations are exactly MageZero's) that plays one of two roles:
 *
 *  PUPPET   the seat that is not deciding. It never acts on its own: it passes priority, answers
 *           "no" to attack questions, declares no blocks, and resolves other forced choices with
 *           XMage's plain heuristics. Every answer is written to the player history in the form
 *           MCTSPlayer replays, so a search anchored before these answers rebuilds the same state
 *           (critique §5: "both seats are puppets during replay").
 *  DECIDER  the seat whose first non-trivial decision we want. Trivial decisions on the way (a
 *           priority window where only Pass is legal, a target with one option) are taken without
 *           search. At the first real decision it either records the legal options (CAPTURE) or
 *           runs one MCTS search with a fresh tree and records the root's children (SEARCH), then
 *           pauses the game.
 *
 * Tree reuse is off: ComputerPlayerMCTS2 re-roots on a matching subtree of the previous search and
 * counts those visits toward the budget (magezero_mcts.md §4.4); the bridge clears the tree
 * before every search so {@code budget} means new simulations.
 */
public class BridgePlayer extends ComputerPlayerMCTS2 {

    public enum Role {PUPPET, DECIDER}

    public enum Mode {CAPTURE, SEARCH}

    /** Called at the decision point, before the game moves on (encode / state dumps happen here). */
    public interface Listener {
        void onDecision(Game game, BridgePlayer player, Decision decision);
    }

    public String seat;
    public Role role = Role.PUPPET;
    public Mode mode = Mode.CAPTURE;
    /**
     * True while GameImpl.init() runs inside StateInjector.build: every yes/no question is answered
     * "no" and not recorded. init() offers opening-hand actions (a Leyline in the first seven cards
     * of the decklist asks "put it onto the battlefield?"); a "yes" there leaves an extra permanent
     * that is not in the spec and takes the card out of the library.
     */
    public boolean setup = true;
    public transient Decision decision;
    public transient Listener listener;
    public transient RuntimeException failure;
    /** priority windows the decider passed before its decision (only Pass legal, or before decideFrom) */
    public transient int passedBefore;

    public BridgePlayer(String name, String seat) {
        super(name, RangeOfInfluence.ONE, 6);
        this.seat = seat;
    }

    public BridgePlayer(final BridgePlayer p) {
        super(p);
        this.seat = p.seat;
        this.role = p.role;
        this.mode = p.mode;
        this.setup = p.setup;
        // decision / listener / failure belong to the live player only; copies (anchors) never act
    }

    @Override
    public BridgePlayer copy() {
        return new BridgePlayer(this);
    }

    /**
     * Optional start of the decision window (turn, step): before it the decider behaves like a
     * puppet (passes, no attacks, heuristic choices). E.g. an end-of-turn 17lands snapshot entered
     * at END_TURN, where only the next main phase's decision is wanted, not an upkeep instant.
     */
    public int fromTurn = 0;
    public int fromStep = -1;

    private boolean deciding(Game game) {
        if (role != Role.DECIDER || decision != null) return false;
        if (game.getTurnNum() < fromTurn) return false;
        if (game.getTurnNum() == fromTurn && fromStep >= 0) {
            PhaseStep st = game.getTurnStepType();
            return st != null && Spec.turnSteps().indexOf(st.name()) >= fromStep;
        }
        return true;
    }

    private static boolean stopped(Game game) {
        return game.isPaused() || game.checkIfGameIsOver();
    }

    // ------------------------------------------------------------------ priority

    @Override
    public boolean priority(Game game) {
        if (stopped(game)) return false;
        if (!deciding(game)) {
            checkpoint(game);
            if (role == Role.DECIDER && decision == null) passedBefore++;
            pass(game);
            return false;
        }
        try {
            game.getState().setPriorityPlayerId(playerId);
            List<ActivatedAbility> playable = getPlayableAbilities(game); // non-mana + Pass (autoTap)
            if (playable.size() < 2) {
                checkpoint(game);
                passedBefore++;
                pass(game); // a lone Pass is not a decision; the history entry lets simulations replay it
                return false;
            }
            game.firePriorityEvent(playerId);
            // the checkpoint ComputerPlayerMCTS.priority takes before searching. It also clears the
            // micro-decision histories, which MCTSPlayer's encoding of this point has empty too; a
            // capture only needs that part (the anchor copy costs a game copy)
            if (mode == Mode.SEARCH) game.setLastPriority(playerId);
            else ((GameImpl) game).clearHistory();
            Decision d = newDecision(game, "PRIORITY", "priority");
            for (ActivatedAbility a : playable) {
                String label = a.toString();
                MageObject src = a.getSourceId() == null ? null : game.getObject(a.getSourceId());
                String srcName = src == null || label.contains(src.getName()) ? null : src.getName();
                d.add(label, actionEncoder.getActionIndex(a, true), srcName);
            }
            reached(game, d, ActionEncoder.ActionType.PRIORITY);
        } catch (RuntimeException e) {
            fail(game, e);
        }
        return false; // never act: the game is paused at the decision
    }

    /**
     * MageZero's forced checkpoints (GameImpl.isCheckPoint: BEGIN_COMBAT, DECLARE_ATTACKERS,
     * DECLARE_BLOCKERS, turn-1 upkeep). ComputerPlayerMCTS.priority re-anchors there even when only
     * Pass is legal, which also clears both players' micro-decision histories. Doing the same
     * keeps the ChosenTargets / UseChoices features of a later combat decision equal to what
     * MageZero encodes in self-play, and keeps the prefix a search replays short (an earlier
     * anchor replays everything since the injection, e.g. a whole untap / upkeep / draw).
     */
    private void checkpoint(Game game) {
        if (!game.isCheckPoint(playerId)) return;
        game.getState().setPriorityPlayerId(playerId);
        if (mode == Mode.SEARCH) game.setLastPriority(playerId);
        else ((GameImpl) game).clearHistory();
    }

    // ------------------------------------------------------------------ yes / no (attacks, "may")

    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage, String trueText, String falseText,
                             Ability source, Game game) {
        if (setup) return false; // opening-hand actions during init(): see the field
        if (stopped(game)) return false;
        if (!deciding(game)) {
            // puppets never attack; other "may" questions get XMage's default answer
            boolean out = !message.startsWith("attack with: ") && outcome != Outcome.AIDontUseIt;
            getPlayerHistory().useSequence.add(out);
            return out;
        }
        try {
            Decision d = newDecision(game, "CHOOSE_USE", message);
            d.add("no", 0, null);
            d.add("yes", 1, null);
            MCTSNode best = reached(game, d, ActionEncoder.ActionType.CHOOSE_USE);
            boolean out = best != null && best.getUseAction();
            getPlayerHistory().useSequence.add(out);
            return out;
        } catch (RuntimeException e) {
            fail(game, e);
            return false;
        }
    }

    // ------------------------------------------------------------------ targets (incl. blocks)

    @Override
    protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (target.getMessage(game).equals("Select a starting player")) {
            target.add(this.getId(), game);
            return true;
        }
        if (stopped(game)) return false;
        // the prechecks MCTSPlayer.makeChoice runs before it consumes a scripted target
        if (fromCards != null && fromCards.isEmpty()) return false;
        UUID controller = target.getAffectedAbilityControllerId(getId());
        if (target.isChoiceCompleted(controller, source, game, fromCards)) return false;
        Set<UUID> possible = target.possibleTargets(controller, source, game, fromCards).stream()
                .filter(id -> !target.contains(id)).collect(Collectors.toSet());
        if (possible.isEmpty()) return false;
        boolean canStop = target.isChosen(game);

        if (!deciding(game)) {
            if (source instanceof ChooseCreatureToBlockAbility) {
                getPlayerHistory().targetSequence.add(STOP_CHOOSING); // puppets never block
                return false;
            }
            return heuristicTarget(outcome, target, source, game, fromCards, controller, possible, canStop);
        }
        int n = possible.size() + (canStop ? 1 : 0);
        if (n == 1) {
            UUID id = possible.iterator().next();
            target.addTarget(id, source, game);
            getPlayerHistory().targetSequence.add(id);
            return true;
        }
        try {
            String text = (source == null ? "null" : source.getRule()) + ":Choose a target:" + target.getTargetName();
            Decision d = newDecision(game, "CHOOSE_TARGET", text);
            List<UUID> opts = new ArrayList<>(possible);
            if (canStop) opts.add(STOP_CHOOSING);
            List<String[]> labelled = new ArrayList<>();
            for (UUID id : opts) labelled.add(new String[]{targetLabel(game, id), id.toString()});
            labelled.sort(Comparator.comparing((String[] x) -> x[0]).thenComparing(x -> x[1]));
            for (String[] l : labelled) d.add(l[0], actionEncoder.getTargetIndex(l[0]), null);
            MCTSNode best = reached(game, d, ActionEncoder.ActionType.CHOOSE_TARGET);
            if (best != null && best.getTargetAction() != null && !best.getTargetAction().equals(STOP_CHOOSING)) {
                target.addTarget(best.getTargetAction(), source, game);
            }
        } catch (RuntimeException e) {
            fail(game, e);
        }
        return false;
    }

    /** ComputerPlayer.makeChoice (the non-MCTS default), recording what MCTSPlayer will replay. */
    private boolean heuristicTarget(Outcome outcome, Target target, Ability source, Game game, Cards fromCards,
                                    UUID controller, Set<UUID> possible, boolean canStop) {
        int n = possible.size() + (canStop ? 1 : 0);
        if (n == 1) {
            UUID id = possible.iterator().next();
            target.addTarget(id, source, game);
            getPlayerHistory().targetSequence.add(id);
            return true;
        }
        boolean out = makeChoiceHelper(outcome, target, source, game, fromCards);
        if (out) {
            getPlayerHistory().targetSequence.addAll(target.getTargets());
            if (!target.isChoiceCompleted(controller, source, game, fromCards)) {
                getPlayerHistory().targetSequence.add(STOP_CHOOSING);
            }
        }
        return out;
    }

    static String targetLabel(Game game, UUID id, UUID perspective) {
        if (STOP_CHOOSING.equals(id)) return "Stop Choosing";
        return game.getEntityName(id, perspective);
    }

    private String targetLabel(Game game, UUID id) {
        return targetLabel(game, id, playerId);
    }

    // ------------------------------------------------------------------ named choices / numbers

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        if (stopped(game)) return chooseFallback(outcome, choice, game);
        // MCTSPlayer.choose answers these without consuming a scripted choice, so record nothing
        if (choice.getChoices().size() == 1) return chooseHelper(outcome, choice, game);
        if (choice.getMessage() != null && (choice.getMessage().equals("Choose creature type")
                || choice.getMessage().equals("Choose a creature type"))) {
            if (chooseCreatureType(outcome, choice, game)) return true;
        }
        Set<String> options = new TreeSet<>(choice.getKeyChoices().keySet());
        if (options.isEmpty()) options = new TreeSet<>(choice.getChoices());
        if (options.isEmpty()) return false;
        boolean byKey = !choice.getKeyChoices().isEmpty();
        if (!deciding(game) || options.size() == 1) {
            String chosen = options.iterator().next();
            if (byKey) choice.setChoiceByKey(chosen);
            else choice.setChoice(chosen);
            getPlayerHistory().choiceSequence.add(chosen);
            return true;
        }
        try {
            Decision d = newDecision(game, "MAKE_CHOICE", choice.getMessage());
            for (String o : options) d.add(o, -1, null);
            choiceOptions = new HashSet<>(options);
            MCTSNode best = reached(game, d, ActionEncoder.ActionType.MAKE_CHOICE);
            String chosen = best != null && best.getChoiceAction() != null ? best.getChoiceAction() : options.iterator().next();
            if (byKey) choice.setChoiceByKey(chosen);
            else choice.setChoice(chosen);
            return true;
        } catch (RuntimeException e) {
            fail(game, e);
            return false;
        }
    }

    @Override
    protected int makeChoiceAmount(int min, int max, Game game, Ability source, boolean isManaPay) {
        if (stopped(game)) return min;
        if (min >= max) return min;
        if (max - min > 64) max = min + 64; // MageZero's clamp
        if (!deciding(game)) {
            getPlayerHistory().numSequence.add(0);
            return min;
        }
        try {
            Decision d = newDecision(game, "CHOOSE_NUM", "choose num for " + source);
            d.numMin = min;
            for (int v = min; v <= max; v++) d.add(String.valueOf(v), -1, null);
            numOptionsSize = max - min + 1;
            MCTSNode best = reached(game, d, ActionEncoder.ActionType.CHOOSE_NUM);
            return best == null ? min : min + best.getAmountAction();
        } catch (RuntimeException e) {
            fail(game, e);
            return min;
        }
    }

    // ------------------------------------------------------------------ decision bookkeeping

    private Decision newDecision(Game game, String type, String text) {
        Decision d = new Decision();
        d.type = type;
        d.text = text;
        d.player = seat;
        d.turn = game.getTurnNum();
        d.phase = game.getTurnPhaseType() == null ? null : game.getTurnPhaseType().name();
        d.step = game.getTurnStepType() == null ? null : game.getTurnStepType().name();
        d.activePlayer = seatOf(game, game.getActivePlayerId());
        d.stackSize = game.getStack().size();
        d.passedBefore = passedBefore;
        return d;
    }

    static String seatOf(Game game, UUID id) {
        mage.players.Player p = id == null ? null : game.getPlayer(id);
        if (p == null) return null;
        return "PlayerA".equals(p.getName()) ? "A" : "B";
    }

    /**
     * The decision point: tell the listener (encode / dump need the state before anything moves),
     * search if asked, then pause the game so GameImpl.playPriority and the step unwind.
     */
    private MCTSNode reached(Game game, Decision d, ActionEncoder.ActionType type) {
        decision = d;
        if (listener != null) listener.onDecision(game, this, d);
        MCTSNode best = null;
        if (mode == Mode.SEARCH) {
            long t0 = System.nanoTime();
            best = getNextAction(game, type);
            d.searchSeconds = (System.nanoTime() - t0) / 1e9;
            if (best == null) throw new IllegalStateException("MCTS returned no action for " + d.type + " '" + d.text + "'");
        }
        game.pause();
        return best;
    }

    private void fail(Game game, RuntimeException e) {
        if (failure == null) failure = e;
        game.pause();
    }

    // ------------------------------------------------------------------ search hooks

    @Override
    protected MCTSNode2 getNextAction(Game game, ActionEncoder.ActionType actionType) {
        Reflect.set(ComputerPlayerMCTS2.class, this, "root", null); // fresh tree: no reused visits
        return super.getNextAction(game, actionType);
    }

    @Override
    protected MCTSNode calculateActions(Game game, ActionEncoder.ActionType action) {
        MCTSNode best = super.calculateActions(game, action);
        MCTSNode root = (MCTSNode) Reflect.get(ComputerPlayerMCTS2.class, this, "root");
        if (decision != null && root != null) {
            List<Decision.Child> kids = new ArrayList<>();
            for (MCTSNode c : root.getChildren()) {
                kids.add(new Decision.Child(childLabel(c, action, game), safeIndex(c, game), c.getVisits(),
                        c.getVisits() > 0 ? c.getMeanScore() : null, (double) Reflect.get(MCTSNode.class, c, "prior")));
            }
            decision.children = kids;
            decision.rootVisits = root.getVisits();
            decision.rootQ = root.getVisits() > 0 ? root.getMeanScore() : null;
            decision.rootValue = root.networkScore;
            decision.best = best == null ? null : childLabel(best, action, game);
            @SuppressWarnings("unchecked")
            Set<Integer> fv = (Set<Integer>) Reflect.get(MCTSNode.class, root, "stateVector");
            decision.rootFeatures = fv;
        }
        return best;
    }

    private String childLabel(MCTSNode c, ActionEncoder.ActionType type, Game game) {
        switch (type) {
            case PRIORITY:
                return c.getPriorityAction() == null ? "?" : c.getPriorityAction().toString();
            case CHOOSE_TARGET:
                return c.getTargetAction() == null ? "?" : targetLabel(game, c.getTargetAction());
            case CHOOSE_USE:
                return c.getUseAction() ? "yes" : "no";
            case CHOOSE_NUM:
                return String.valueOf((decision == null ? 0 : decision.numMin) + c.getAmountAction());
            default:
                return String.valueOf(c.getChoiceAction());
        }
    }

    private static int safeIndex(MCTSNode c, Game game) {
        try {
            return c.getActionIndex(game);
        } catch (RuntimeException e) {
            return -2;
        }
    }
}
