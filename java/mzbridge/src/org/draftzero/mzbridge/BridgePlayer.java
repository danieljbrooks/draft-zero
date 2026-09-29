package org.draftzero.mzbridge;

import mage.MageObject;
import mage.ObjectColor;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.choices.ChoiceColor;
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
import mage.players.PlayerScript;
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

    public enum Mode {CAPTURE, SEARCH, ROOT}

    /** Called at the decision point, before the game moves on (encode / state dumps happen here). */
    public interface Listener {
        void onDecision(Game game, BridgePlayer player, Decision decision);
    }

    public String seat;
    public Role role = Role.PUPPET;
    public Mode mode = Mode.CAPTURE;
    /**
     * True while StateInjector.build runs (GameImpl.init() and the injection): every yes/no
     * question is answered "no" and not recorded. init() offers opening-hand actions (a Leyline in
     * the first seven cards of the decklist asks "put it onto the battlefield?"); a "yes" there
     * leaves an extra permanent that is not in the spec and takes the card out of the library.
     * Injected permanents ask "as this enters" questions: a shock land's "pay 2 life?" answered yes
     * cost its controller 2 life during injection (39 of 105 Arena specs).
     */
    public boolean setup = true;
    /**
     * Set by StateInjector while it injects (after init()): every named choice and target an
     * entering permanent asks for, with the answer given. The spec has no field for them (Heraldic
     * Banner's color, Adaptive Automaton's creature type), so each becomes a build warning instead
     * of a silent guess. Null otherwise (init() and the game itself are not reported).
     */
    public transient List<String> setupNotes;
    public transient Decision decision;
    public transient Listener listener;
    public transient RuntimeException failure;
    /** priority windows the decider passed before its decision (only Pass legal, or before decideFrom) */
    public transient int passedBefore;
    /** Benchmark: play this land (a priority option's label) at the first decision, then decide. */
    public transient String preLand;
    /** ROOT mode: the captured search root and the scripts that rebuild it */
    public transient MCTSNode2 capturedRoot;
    public transient PlayerScript capturedPrefixA;
    public transient PlayerScript capturedPrefixB;
    public transient ActionEncoder.ActionType capturedType;

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
        if (preLand != null && role == Role.DECIDER && decision == null && game.isActivePlayer(playerId)
                && game.getTurnStepType() == PhaseStep.PRECOMBAT_MAIN && game.getStack().isEmpty()
                && game.getTurnNum() >= fromTurn) {
            // the benchmark's decisions come after the human's land drop (docs/012 §2.2): play it at
            // the first main-phase priority, deciding or not (an attack question comes later)
            try {
                game.getState().setPriorityPlayerId(playerId);
                for (ActivatedAbility a : getPlayableAbilities(game)) {
                    if (a.toString().equals(preLand)) {
                        preLand = null;
                        if (activateAbility(a, game)) return true;
                        throw new IllegalStateException("preLand: could not play " + a);
                    }
                }
                throw new IllegalStateException("preLand '" + preLand + "' is not a legal play here");
            } catch (RuntimeException e) {
                fail(game, e);
                return false;
            }
        }
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
            if (mode != Mode.CAPTURE) game.setLastPriority(playerId);
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
        if (mode != Mode.CAPTURE) game.setLastPriority(playerId);
        else ((GameImpl) game).clearHistory();
    }

    // ------------------------------------------------------------------ yes / no (attacks, "may")

    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage, String trueText, String falseText,
                             Ability source, Game game) {
        // opening-hand actions during init(), "as this enters" questions during the injection. Not
        // reported: the spec's own state wins (a shock land's tapped flag), and a "no" that leaves a
        // doomed permanent (Phantasmal Image not copying: a 0/0) is caught by StateInjector.doomed
        if (setup) return false;
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
            boolean out = heuristicTarget(outcome, target, source, game, fromCards, controller, possible, canStop);
            if (setup && possible.size() + (canStop ? 1 : 0) > 1) {
                noteSetup(game, source, target.getMessage(game), target.getTargets().stream()
                        .map(id -> targetLabel(game, id)).collect(Collectors.joining(", ", "[", "]")));
            }
            return out;
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
        if (!setup) return chooseInGame(outcome, choice, game);
        // an "as this enters" choice of an injected permanent. A color is the chooser's main
        // color for a benefit (Heraldic Banner), its opponent's for a drawback, rather than the
        // alphabetical first; either way it is a guess, reported as a build warning
        boolean out;
        if (choice instanceof ChoiceColor && choice.getChoices().size() > 1) {
            choice.setChoice(mainColor(game, outcome.isGood() ? playerId : game.getOpponent(playerId).getId(), choice.getChoices()));
            out = true;
        } else {
            out = chooseInGame(outcome, choice, game);
        }
        if (Math.max(choice.getChoices().size(), choice.getKeyChoices().size()) > 1) {
            noteSetup(game, null, choice.getMessage(), choice.isKeyChoice() ? choice.getChoiceKey() : choice.getChoice());
        }
        return out;
    }

    /** The color with the most cards of that color the player owns (its decklist), ties in WUBRG order. */
    static String mainColor(Game game, UUID owner, Set<String> options) {
        Map<String, Integer> n = new LinkedHashMap<>();
        for (String c : List.of("White", "Blue", "Black", "Red", "Green")) if (options.contains(c)) n.put(c, 0);
        for (Card c : game.getCards()) {
            if (!c.isOwnedBy(owner)) continue;
            ObjectColor col = c.getColor(game);
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

    /** Record a question answered during the injection (see setupNotes). */
    private void noteSetup(Game game, Ability source, String question, String answer) {
        if (setupNotes == null) return;
        MageObject o = source == null || source.getSourceId() == null ? null : game.getObject(source.getSourceId());
        String who = o != null ? o.getName() : game.getPermanentsEntering().values().stream()
                .map(MageObject::getName).collect(Collectors.joining("/"));
        setupNotes.add(seat + ": " + (who.isEmpty() ? "a permanent" : who) + " asked \"" + question
                + "\" as it was injected; answered " + answer + " (the spec cannot say what was chosen)");
    }

    private boolean chooseInGame(Outcome outcome, Choice choice, Game game) {
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
        if (mode == Mode.ROOT) {
            captureRoot(game, type);
        } else if (mode == Mode.SEARCH) {
            long t0 = System.nanoTime();
            best = getNextAction(game, type);
            d.searchSeconds = (System.nanoTime() - t0) / 1e9;
            if (best == null) throw new IllegalStateException("MCTS returned no action for " + d.type + " '" + d.text + "'");
        }
        game.pause();
        return best;
    }

    /**
     * ROOT mode: MageZero's search root for this decision, built exactly as
     * ComputerPlayerMCTS2.getNextAction builds it (a copy of the last priority checkpoint plus both
     * players' scripts since then), validated but neither expanded nor searched. The benchmark's
     * driver (mage.player.ai.BenchSearch) searches it.
     */
    private void captureRoot(Game game, ActionEncoder.ActionType type) {
        if (actionEncoder == null) actionEncoder = new ActionEncoder();
        Game sim = createMCTSGame(game.getLastPriority());
        PlayerScript a = new PlayerScript(getPlayerHistory());
        PlayerScript b = new PlayerScript(game.getOpponent(playerId).getPlayerHistory());
        MCTSNode2 r = new MCTSNode2(this, sim, type, new PlayerScript(a), new PlayerScript(b));
        r.validateState();
        capturedRoot = r;
        capturedPrefixA = a;
        capturedPrefixB = b;
        capturedType = type;
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
