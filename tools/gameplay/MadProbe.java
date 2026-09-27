import com.google.gson.Gson;
import mage.abilities.Ability;
import mage.cards.repository.RepositoryUtil;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.player.ai.ComputerPlayer7;
import mage.players.PlayerImpl;
import org.draftzero.mzbridge.Spec;
import org.draftzero.mzbridge.StateInjector;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Runs XMage's MAD AI decision (ComputerPlayer7.calculateActions, the same code "Computer - mad" and
 * MageZero's minimax type use) on a StateSpec built by the mzbridge, as seat A, and prints the chain
 * of abilities it chooses. The probe is not seated in the game: calculateActions only needs the game
 * and the deciding player's id.
 *
 * usage (from a directory holding db/cards.h2.mv.db):
 *   java --enable-final-field-mutation=ALL-UNNAMED -cp "mzbridge.jar:xmage/lib/*" MadProbe.java SEEDS SKILL spec.json...
 */
public class MadProbe {

    static final class Probe extends ComputerPlayer7 {
        Probe(UUID seatId, int skill) throws Exception {
            super("PlayerA", RangeOfInfluence.ONE, skill);
            Field f = PlayerImpl.class.getDeclaredField("playerId");
            f.setAccessible(true);
            f.set(this, seatId);
        }

        List<String> decide(Game game) {
            actions.clear();
            root = null;
            calculateActions(game);
            List<String> out = new ArrayList<>();
            for (Ability a : actions) out.add(a.toString());
            return out;
        }

        Integer chosenScore() {
            return root == null ? null : root.getScore();
        }
    }

    static Spec load(String json) {
        Spec s = new Gson().fromJson(json, Spec.class);
        for (Spec.PlayerState p : s.players.values()) {
            if (p.decklist == null) p.decklist = new ArrayList<>();
            if (p.hand == null) p.hand = new ArrayList<>();
            if (p.graveyard == null) p.graveyard = new ArrayList<>();
            if (p.exile == null) p.exile = new ArrayList<>();
            if (p.libraryTop == null) p.libraryTop = new ArrayList<>();
            if (p.battlefield == null) p.battlefield = new ArrayList<>();
            if (p.decklistSource == null) p.decklistSource = "exact";
            for (Spec.Perm perm : p.battlefield) if (perm.counters == null) perm.counters = new java.util.LinkedHashMap<>();
        }
        if (s.stack == null) s.stack = new ArrayList<>();
        if (s.attackers == null) s.attackers = new ArrayList<>();
        if (s.blockers == null) s.blockers = new ArrayList<>();
        if (s.passedPlayers == null) s.passedPlayers = new ArrayList<>();
        return s;
    }

    public static void main(String[] args) throws Exception {
        System.setOut(System.err); // keep XMage's stray prints off stdout
        java.io.PrintStream out = new java.io.PrintStream(new java.io.FileOutputStream(java.io.FileDescriptor.out), true);
        RepositoryUtil.bootstrapLocalDb();
        int seeds = Integer.parseInt(args[0]);
        int skill = Integer.parseInt(args[1]);
        for (int i = 2; i < args.length; i++) {
            String json = Files.readString(Path.of(args[i]));
            for (int seed = 0; seed < seeds; seed++) {
                StateInjector.Built b = StateInjector.build(load(json), seed, false);
                Game game = b.game;
                UUID a = b.players.get("A").getId();
                UUID current = game.getPlayerList().get();
                Probe p = new Probe(a, skill);
                long t0 = System.nanoTime();
                List<String> chain = p.decide(game);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                String top = game.getPlayer(a).getLibrary().size() > 0
                        ? game.getPlayer(a).getLibrary().getFromTop(game).getName() : "-";
                out.printf("{\"spec\": \"%s\", \"seed\": %d, \"step\": \"%s\", \"currentIsA\": %s, \"aTopCard\": \"%s\", \"chain\": %s, \"score\": %s, \"ms\": %d}%n",
                        Path.of(args[i]).getFileName(), seed, game.getTurnStepType(), current.equals(a), top,
                        new Gson().toJson(chain), p.chosenScore(), ms);
            }
        }
        System.exit(0);
    }
}
