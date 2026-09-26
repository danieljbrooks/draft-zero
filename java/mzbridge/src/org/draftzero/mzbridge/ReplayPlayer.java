package org.draftzero.mzbridge;

import com.google.gson.JsonObject;
import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.cards.Cards;
import mage.constants.Outcome;
import mage.game.Game;
import mage.players.ChooseCreatureToBlockAbility;
import mage.target.Target;

import java.util.List;
import java.util.UUID;

/**
 * A puppet that follows a {@link ReplayRun} script, for both seats of a replay_turn request: the
 * scripted seat plays its recorded lands, spells and activations and declares exactly its recorded
 * attackers; the other seat blocks as recorded and casts its recorded instants in their window.
 * Targets come from the run's {@link TargetResolver}. Everything else (priority outside the
 * replayed turn, "may" questions, modes, X) keeps the {@link BridgePlayer} puppet behaviour, and
 * every answer goes into the player history the way MageZero records it, so the features encoded
 * at each decision are the ones MageZero would see.
 *
 * The run is shared by reference with the copies the engine makes (bookmarks, rollbacks restore
 * the live player in place); copies never act.
 */
public class ReplayPlayer extends BridgePlayer {
    transient ReplayRun run;

    public ReplayPlayer(BridgePlayer p, ReplayRun run) {
        super(p);
        this.run = run;
    }

    public ReplayPlayer(final ReplayPlayer p) {
        super(p);
        this.run = p.run;
    }

    @Override
    public ReplayPlayer copy() {
        return new ReplayPlayer(this);
    }

    private boolean off(Game game) {
        return run == null || setup || game.isPaused() || game.checkIfGameIsOver();
    }

    @Override
    public boolean priority(Game game) {
        if (run == null || setup) return super.priority(game);
        if (game.isPaused() || game.checkIfGameIsOver()) return false;
        try {
            return run.priority(this, game);
        } catch (RuntimeException e) {
            run.fail(e);
            game.pause();
            return false;
        }
    }

    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        if (off(game) || !run.inTurn(game) || !seat.equals(run.seat)) {
            super.selectAttackers(game, attackingPlayerId);
            return;
        }
        try {
            run.selectAttackers(this, game, attackingPlayerId);
        } catch (RuntimeException e) {
            run.fail(e);
            game.pause();
        }
    }

    @Override
    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        if (off(game) || !run.inTurn(game) || seat.equals(run.seat)) {
            super.selectBlockers(source, game, defendingPlayerId);
            return;
        }
        try {
            run.selectBlockers(this, source, game, defendingPlayerId);
        } catch (RuntimeException e) {
            run.fail(e);
            game.pause();
        }
    }

    @Override
    protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (off(game) || source instanceof ChooseCreatureToBlockAbility
                || target.getMessage(game).equals("Select a starting player")) {
            return super.makeChoice(outcome, target, source, game, fromCards);
        }
        try {
            return run.chooseTarget(this, outcome, target, source, game, fromCards);
        } catch (RuntimeException e) {
            run.fail(e);
            game.pause();
            return false;
        }
    }

    @Override
    public boolean chooseUse(Outcome outcome, String message, String secondMessage, String trueText, String falseText,
                             Ability source, Game game) {
        if (off(game)) return super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
        JsonObject d = run.beforeUse(this, message, game);
        boolean out;
        if (run.inTurn(game) && !run.policy.may.equals("default") && !message.startsWith("attack with: ")) {
            out = run.policy.may.equals("yes");          // this attempt's answer to every "may" question
            getPlayerHistory().useSequence.add(out);
        } else {
            out = super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
        }
        run.afterUse(d, out);
        return out;
    }

    @Override
    protected int makeChoiceAmount(int min, int max, Game game, Ability source, boolean isManaPay) {
        if (off(game) || min >= max) return super.makeChoiceAmount(min, max, game, source, isManaPay);
        if (run.inTurn(game)) run.flags.add(isManaPay ? "guessed_x" : "guessed_amount");
        if (isManaPay && run.inTurn(game)) {
            // X: as much as the untapped mana allows (the puppet's default is the minimum, which
            // turns an X creature into a 0/0); recorded as MageZero records a CHOOSE_NUM offset
            int x = Math.max(min, Math.min(max, getAvailableManaProducers(game).size()
                    - (source == null ? 0 : source.getManaCostsToPay().getUnpaid().manaValue())));
            getPlayerHistory().numSequence.add(x - min);
            return x;
        }
        return super.makeChoiceAmount(min, max, game, source, isManaPay);
    }

    @Override
    public mage.abilities.Mode chooseMode(mage.abilities.Modes modes, Ability source, Game game) {
        // ComputerPlayerMCTS lists "no mode" first and the puppet's default answer (index 0) would
        // pick it, which cancels the spell
        if (off(game)) return super.chooseMode(modes, source, game);
        try {
            return run.chooseMode(this, modes, source, game);
        } catch (RuntimeException e) {
            run.fail(e);
            game.pause();
            return null;
        }
    }

    @Override
    public boolean choose(Outcome outcome, mage.choices.Choice choice, Game game) {
        if (!off(game) && choice instanceof mage.choices.ChoiceColor && choice.getChoices().size() > 1) {
            // "choose a color" (Heraldic Banner): the color of most of the creatures it helps (or
            // hurts), not the alphabetical first
            String c = run.color(this, outcome, choice.getChoices(), game);
            choice.setChoice(c);
            getPlayerHistory().choiceSequence.add(c);
            if (run.inTurn(game)) run.flags.add("inferred_color");
            return true;
        }
        if (!off(game) && run.inTurn(game) && choice.getChoices().size() + choice.getKeyChoices().size() > 1) {
            run.flags.add("guessed_choice");
        }
        return super.choose(outcome, choice, game);
    }

    /** Mana producers the auto-tap may use: minus the ones the run reserved for later items (ReplayRun.reserveMana). */
    @Override
    public List<MageObject> getAvailableManaProducers(Game game) {
        List<MageObject> out = super.getAvailableManaProducers(game);
        if (run != null && run.hidden != null && seat.equals(run.hiddenFor)) out.removeIf(o -> run.hidden.contains(o.getId()));
        return out;
    }

    List<MageObject> allManaProducers(Game game) {
        return super.getAvailableManaProducers(game);
    }

    /** MageZero's priority options: the non-mana playables (autoTap) plus Pass. */
    List<ActivatedAbility> playable(Game game) {
        return getPlayableAbilities(game);
    }
}
