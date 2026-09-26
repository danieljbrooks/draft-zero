package org.draftzero.mzbridge;

import mage.Mana;
import mage.abilities.Ability;
import mage.abilities.common.SimpleStaticAbility;
import mage.abilities.effects.ContinuousEffect;
import mage.abilities.effects.common.InfoEffect;
import mage.abilities.keyword.IndestructibleAbility;
import mage.cards.Card;
import mage.cards.MeldCard;
import mage.cards.decks.Deck;
import mage.cards.repository.CardInfo;
import mage.cards.repository.CardRepository;
import mage.cards.repository.TokenInfo;
import mage.cards.repository.TokenRepository;
import mage.cards.repository.TokenType;
import mage.constants.MultiplayerAttackOption;
import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.constants.SubType;
import mage.constants.TurnPhase;
import mage.constants.Zone;
import mage.counters.CounterType;
import mage.game.Game;
import mage.game.GameImpl;
import mage.game.GameOptions;
import mage.game.GameState;
import mage.game.TwoPlayerDuel;
import mage.game.TwoPlayerMatch;
import mage.game.combat.Combat;
import mage.game.events.ZoneChangeEvent;
import mage.game.match.Match;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentCard;
import mage.game.permanent.PermanentImpl;
import mage.game.permanent.PermanentMeld;
import mage.game.permanent.PermanentToken;
import mage.game.permanent.token.Token;
import mage.game.permanent.token.TokenImpl;
import mage.game.stack.Spell;
import mage.game.turn.Phase;
import mage.game.turn.Step;
import mage.game.turn.Turn;
import mage.game.turn.TurnMod;
import mage.player.ai.encoder.ActionEncoder;
import mage.player.ai.encoder.StateEncoder;
import mage.players.Player;
import mage.players.PlayerImpl;
import mage.util.CardUtil;
import mage.util.RandomUtil;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Builds an XMage game that sits at an arbitrary turn / step with arbitrary zone contents, from a
 * StateSpec v1, without firing the events a real game would have fired to get there.
 *
 * The recipe is the research proof of concept's (xmage_state.md §5-§7):
 *  1. TwoPlayerDuel + two players + a fake TwoPlayerMatch, as ParallelDataGenerator does it;
 *  2. game.start() with stopOnTurn=1/UNTAP runs only GameImpl.init(); the opening hands go back;
 *  3. cards move from the libraries into their zones without events (hand, graveyard, exile,
 *     battlefield via CardUtil.putCardOntoBattlefieldWithEffects's recipe, tokens the same way,
 *     counters, damage, sickness, attachments, life, lands played, mana pool, library order and
 *     size); a control-changed permanent (Perm.owner) comes out of its owner's library;
 *  4. Turn / Phase / Step point at the target step, with Step.stepPart primed for Phase.resumeStep;
 *  5. hygiene: drop the "starting player skips the draw" TurnMod, pending triggers, queued
 *     simultaneous events and watcher history, then applyEffects.
 * {@link #anchor} then pauses the game and re-anchors MageZero's search at the injected state.
 *
 * What changed from the proof of concept (critique.md §3.2, C4, and the gotchas becoming checks):
 *  - decks come from the spec's decklist (names), not from .dck files;
 *  - every random choice is a function of the request seed: a private Random for library order and
 *    hidden hands, RandomUtil and GameState.localRandom for the engine, reproducible UUIDs;
 *  - BEGIN_STEP is entered as "the previous step has just ended" (stepPart=POST), which a paused
 *    MCTS copy resumes exactly like the live game, so a decision made inside a step's turn-based
 *    actions (blocks at DECLARE_BLOCKERS) is searched from the right state (the 3b negative case);
 *  - verify() reads the built game back and fails on any zone, counter, tap, sickness, damage,
 *    attachment, controller or owner that does not match the spec, instead of warning.
 */
public final class StateInjector {

    public static final class Built {
        public Spec spec;
        /** seeds the engine RNGs, the library order and the hidden cards (one determinization) */
        public long seed;
        /** seeds the UUIDs; kept equal across determinizations so decisions come in the same order */
        public long idSeed;
        public Game game;
        public Match match;
        public final Map<String, BridgePlayer> players = new LinkedHashMap<>();
        /** "B:goblin" (and "B:x#2" for count > 1) -> permanent id */
        public final Map<String, UUID> aliases = new LinkedHashMap<>();
        /** permanent id -> the spec entry that made it */
        public final Map<UUID, Spec.Perm> origin = new LinkedHashMap<>();
        public final Map<UUID, String> originSeat = new HashMap<>();
        public final List<String> warnings = new ArrayList<>();
        /** where the engine resumes: equals spec phase/step except for BEGIN_STEP (previous step, POST) */
        public TurnPhase enginePhase;
        public PhaseStep engineStep;
        public Step.StepPart enginePart;

        public BridgePlayer seat(String s) {
            return players.get(s);
        }

        public String seatOf(UUID playerId) {
            for (Map.Entry<String, BridgePlayer> e : players.entrySet()) {
                if (e.getValue().getId().equals(playerId)) return e.getKey();
            }
            return null;
        }

        public String aliasOf(UUID permId) {
            String base = null;
            for (Map.Entry<String, UUID> e : aliases.entrySet()) {
                if (!e.getValue().equals(permId)) continue;
                if (e.getKey().contains("#")) return e.getKey();
                base = e.getKey();
            }
            return base;
        }
    }

    /** A spec the bridge cannot build; the message lists every problem found. */
    public static final class SpecException extends RuntimeException {
        public final List<String> problems;

        SpecException(String what, List<String> problems) {
            super(what + ": " + String.join("; ", problems));
            this.problems = problems;
        }
    }

    private static final Map<String, CardInfo> CARD_INFO = new ConcurrentHashMap<>();
    private static final List<String> PREFERRED_SETS = List.of("FDN", "SPG");
    private static volatile List<TokenInfo> TOKENS;
    private static final ActionEncoder ACTIONS = new ActionEncoder(); // vocabulary from -Dmz.actionVocab

    private StateInjector() {
    }

    // =================================================================================== build

    public static Built build(Spec spec, long seed, boolean lenient) {
        return build(spec, seed, seed, lenient);
    }

    /**
     * @param idSeed seeds the card/permanent UUIDs. Attack and block questions come in UUID order,
     *               so determinizations of one decision share it and differ only in {@code seed}.
     * @param seed   seeds the library order, the hidden hand cards and the engine's RNGs.
     */
    public static Built build(Spec spec, long idSeed, long seed, boolean lenient) {
        List<String> errs = spec.validate();
        if (!errs.isEmpty()) throw new SpecException("invalid spec", errs);
        List<String> unknown = unknownCards(spec);
        if (!unknown.isEmpty()) {
            throw new SpecException("unknown card", List.of(unknown.size() + " card(s) not in the XMage card database: " + unknown
                    + " (options.substitute {\"missing\": \"Plains\"} builds the spec with a stand-in)"));
        }
        warm(spec);

        DeterministicIds.reset(idSeed);
        Random rng = new Random(seed);
        Built out = new Built();
        out.spec = spec;
        out.seed = seed;
        out.idSeed = idSeed;
        if (spec.startingPlayer != null && (spec.turn % 2 == 1) != spec.startingPlayer.equals(spec.activePlayer)) {
            // legal only after an extra turn; far more often a per-player vs global turn mix-up
            out.warnings.add("startingPlayer " + spec.startingPlayer + " with activePlayer " + spec.activePlayer + " on turn "
                    + spec.turn + ": the player on the play takes the odd turns (extra turn, or a turn-numbering error?)");
        }
        for (String seat : Spec.SEATS) {
            String pool = spec.players.get(seat).manaPool;
            if ("BEGIN_STEP".equals(spec.enterMode) && pool != null && !pool.isEmpty()) {
                // BEGIN_STEP ends the previous step first, and mana pools empty between steps
                out.warnings.add(seat + ".manaPool " + pool + " is emptied: a BEGIN_STEP entry starts the step with empty pools");
            }
        }

        // ---- 1. game, players, decks (ParallelDataGenerator.runSingleGame / createLocalPlayer) ----
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
                MulliganType.GAME_DEFAULT.getMulligan(0), 40, 20, 7);
        Match match = new TwoPlayerMatch(new MatchOptions("mzbridge", "mzbridge", false));
        out.game = game;
        out.match = match;
        for (String seat : Spec.SEATS) {
            // G15: the vocabulary's player targets and ComputerPlayer8 expect these names
            out.players.put(seat, new BridgePlayer("Player" + seat, seat));
        }
        // G8: the ComputerPlayerMCTS2 constructor reseeds the thread's RandomUtil with a constant
        RandomUtil.setSeed(seed);
        for (String seat : Spec.SEATS) {
            BridgePlayer p = out.players.get(seat);
            p.setTestMode(true);
            Deck deck = new Deck();
            Map<String, Integer> copies = new HashMap<>();
            for (String name : spec.players.get(seat).decklist) {
                // a card's ids depend on (seat, name, copy) only, not on the rest of the decklist, so
                // determinizations whose belief decklists differ still share every visible card's
                // UUID, and with it the attack/block question order (DeterministicIds)
                int copy = copies.merge(name, 1, Integer::sum);
                DeterministicIds.reset(mix(idSeed, cardSalt(seat, name, copy)));
                deck.getCards().add(newCard(name, null, null));
            }
            DeterministicIds.reset(mix(idSeed, 6 + (seat.equals("A") ? 0 : 1)));
            game.loadCards(deck.getCards(), p.getId());
            game.addPlayer(p, deck);
            match.addPlayer(p, deck); // G11: MCTS2 reads getMatchPlayer().getDeck()
            // useDeck fills the library from Deck.getMaindeckCards(), a HashSet of cards hashed by
            // identity, so copies of the same card land in a different order every build (and
            // takeCard would put a different copy, i.e. a different UUID, onto the battlefield).
            // Restore the decklist order.
            p.getLibrary().clear();
            for (Card c : deck.getCards()) p.getLibrary().putOnBottom(c, game);
        }
        for (String seat : Spec.SEATS) configure(out.players.get(seat), out.players.get(Spec.other(seat)));

        BridgePlayer active = out.players.get(spec.activePlayer);
        BridgePlayer starting = out.players.get(spec.starting());
        game.setStartingPlayerId(starting.getId());

        // ---- 2. run init() only ----
        GameOptions options = new GameOptions();
        options.testMode = true;
        options.skipInitShuffling = true; // libraries are shuffled below, after the known cards leave
        options.stopOnTurn = 1;
        options.stopAtStep = PhaseStep.UNTAP; // GameImpl.checkStopOnTurnOption: play() returns before turn 1
        game.setGameOptions(options);
        game.setLocalRandom(new Random(mix(seed, 1)));
        game.start(starting.getId());
        if (game.checkIfGameIsOver()) throw new IllegalStateException("game ended during init()");
        returnHandsToLibrary(game, out); // G3: GameImpl.drawHand is a JVM-wide static; leave it alone
        // opening-hand actions were declined (BridgePlayer.setup); anything init() still put onto
        // the battlefield would be an extra permanent the spec does not have
        if (!game.getBattlefield().getAllPermanents().isEmpty()) {
            throw new IllegalStateException("init() put permanents onto the battlefield: "
                    + game.getBattlefield().getAllPermanents().stream().map(Permanent::getName).toList());
        }
        // from here on, the choices entering permanents ask for are reported (BridgePlayer.setupNotes)
        for (BridgePlayer p : out.players.values()) p.setupNotes = new ArrayList<>();

        // ---- 3. zones ----
        Ability fake = new SimpleStaticAbility(Zone.OUTSIDE, new InfoEffect("mzbridge state injection"));
        Map<Spec.Perm, List<UUID>> made = new IdentityHashMap<>();
        Map<Spec.StackItem, Card> stackCards = new IdentityHashMap<>();
        for (String seat : Spec.SEATS) injectVisible(game, out, seat, fake, made, lenient);
        for (Spec.StackItem si : spec.stack) {
            // before the library is shuffled and hidden hands are drawn, so a random draw cannot take it
            stackCards.put(si, takeCard(game, out, out.players.get(si.controller), si.card, lenient));
        }
        for (String seat : Spec.SEATS) injectLibraryAndHidden(game, out, seat, rng, fake, lenient);
        for (String seat : Spec.SEATS) attach(game, out, seat, fake, made);

        // ---- 4. turn position ----
        game.getState().setTurnNum(spec.turn);
        game.getState().setActivePlayerId(active.getId());
        game.getState().setPlayerByOrderId(active.getId());
        game.getState().getPlayerList().setCurrent(active.getId());
        String prioSeat = spec.priorityPlayer != null ? spec.priorityPlayer : spec.activePlayer;
        game.getState().setPriorityPlayerId(out.players.get(prioSeat).getId());
        setTurnPosition(game, out, spec, active.getId());
        for (String seat : Spec.SEATS) {
            Player p = out.players.get(seat);
            int want = p == starting ? (spec.turn + 1) / 2 : spec.turn / 2; // Player.getTurns(); few cards read it
            for (int i = p.getTurns(); i < want; i++) p.becomesActivePlayer();
        }
        boolean held = "PRIORITY_HELD".equals(spec.enterMode);
        for (String seat : Spec.SEATS) {
            Reflect.set(PlayerImpl.class, out.players.get(seat), "passed", held && spec.passedPlayers.contains(seat));
        }

        // ---- combat ----
        if (out.enginePhase == TurnPhase.COMBAT) {
            // rule 507.1 set-up that BeginCombatStep.beginStep does; entering inside the phase skips it
            Combat combat = game.getCombat();
            combat.clear();
            combat.setAttacker(active.getId());
            combat.setDefenders(game);
        }
        if (!spec.attackers.isEmpty()) {
            Combat combat = game.getCombat();
            game.getTurn().setDeclareAttackersStepStarted(true);
            for (Spec.Attack at : spec.attackers) {
                UUID atk = out.aliases.get(at.attacker);
                UUID def = at.defender.startsWith("player:") ? out.players.get(at.defender.substring(7)).getId()
                        : out.aliases.get(at.defender);
                Permanent pm = game.getPermanent(atk);
                pm.setTapped(false); // declaring the attack taps it (unless vigilance)
                if (!combat.declareAttacker(atk, def, active.getId(), game)) {
                    throw new SpecException("cannot build combat", List.of("XMage refuses attacker " + at.attacker
                            + " (" + pm.getName() + ") -> " + at.defender + ": summoning sick, can't attack, or wrong defender"));
                }
            }
            for (Spec.Block bl : spec.blockers) {
                UUID blk = out.aliases.get(bl.blocker);
                UUID atk = out.aliases.get(bl.attacker);
                Permanent bp = game.getPermanent(blk);
                Player defender = game.getPlayer(bp.getControllerId());
                defender.declareBlocker(defender.getId(), blk, atk, game);
                if (bp.getBlocking() == 0) {
                    throw new SpecException("cannot build combat", List.of("XMage refuses block " + bl.blocker
                            + " (" + bp.getName() + ") -> " + bl.attacker));
                }
            }
            if (!spec.blockers.isEmpty()) combat.acceptBlockers(game);
        }

        // ---- stack (bottom to top) ----
        for (int i = 0; i < spec.stack.size(); i++) {
            DeterministicIds.reset(mix(out.idSeed, 5000L + i));
            castOntoStack(game, out, spec.stack.get(i), stackCards.get(spec.stack.get(i)));
        }

        // ---- 5. hygiene ----
        game.getState().getTurnMods().clear();          // G2: TwoPlayerDuel.init: starting player skips DRAW
        if (entersBeforeFirstDraw(spec)) {
            // ...which is right when the engine is still to play turn 1's draw step (an Arena
            // mulligan / starting-player decision entered at turn 1 UPKEEP): put it back
            game.getState().getTurnMods().add(new TurnMod(starting.getId()).withSkipStep(PhaseStep.DRAW));
        }
        game.getState().clearTriggeredAbilities();      // G7: triggers fired by injection (attach, cast, counters)
        // ...and the events queued for the next GameState.handleSimultaneousEvent, which runs only
        // on resume and would fire triggers then (how injected tokens used to fire their ETB
        // triggers). The game has not resumed yet, so every queued event comes from the injection.
        // Still queued with tokens entering event-free: TAPPED_BATCH, when an "enters tapped"
        // replacement taps an injected permanent (Authority of the Consuls: 16 of 2,449 real 17lands
        // builds); before BridgePlayer.setup covered the injection, a shock land's "pay 2 life?"
        // answered yes also queued LOST_LIFE_BATCH (39 of 105 Arena specs)
        ((List<?>) Reflect.get(GameState.class, game.getState(), "simultaneousEvents")).clear();
        game.getState().resetWatchers();                // "this turn" watchers start empty (a known limitation)
        game.applyEffects();
        // safety net only: the decision player pauses the game itself (G6: stop options are not
        // checked inside the resumed phase, so this stops at the end of the next turn at the latest)
        game.getOptions().stopOnTurn = spec.turn + 1;
        game.getOptions().stopAtStep = PhaseStep.END_TURN;
        game.setLocalRandom(new Random(mix(seed, 2)));  // C4: in-game shuffles use this RNG, and copies carry it

        // questions asked while injecting ("pay 2 life or enter tapped?" of a shock land) are
        // answered "no" (BridgePlayer.setup): injection must not pay costs or change life totals.
        // Named choices and targets (Heraldic Banner's color) become warnings: the spec cannot say
        for (BridgePlayer p : out.players.values()) {
            p.setup = false;
            out.warnings.addAll(p.setupNotes);
            p.setupNotes = null;
        }
        List<String> problems = verify(out, made);
        if (!problems.isEmpty()) {
            if (!lenient) throw new SpecException("built state does not match the spec", problems);
            out.warnings.addAll(problems);
        }
        out.warnings.addAll(doomed(out));
        return out;
    }

    /**
     * Permanents that the state-based actions (rule 704) remove as soon as the game resumes. Such a
     * spec validates and reads back exactly, yet the decision is taken without them: a warning,
     * since the spec is not wrong about what it lists (e.g. an Aura whose host 17lands does not
     * name, or a Phantasmal Image injected without the creature it copied: a 0/0).
     */
    static List<String> doomed(Built out) {
        Game game = out.game;
        List<String> bad = new ArrayList<>();
        Map<String, Integer> legends = new HashMap<>();
        for (Permanent pm : game.getBattlefield().getAllActivePermanents()) {
            String what = out.seatOf(pm.getControllerId()) + ": " + pm.getName();
            if (pm.isCreature(game)) {
                int t = pm.getToughness().getValue();
                if (t <= 0) {
                    bad.add(what + " has toughness " + t);
                } else if (pm.getDamage() >= t && !pm.hasAbility(IndestructibleAbility.getInstance(), game)) {
                    bad.add(what + " has lethal damage (" + pm.getDamage() + " on toughness " + t + ")");
                }
            }
            if (pm.isPlaneswalker(game) && pm.getCounters(game).getCount(CounterType.LOYALTY) == 0) bad.add(what + " has no loyalty");
            if (pm.hasSubtype(SubType.AURA, game) && pm.getAttachedTo() == null) bad.add(what + " is an Aura attached to nothing");
            if (pm.isLegendary(game) && legends.merge(pm.getControllerId() + "|" + pm.getName(), 1, Integer::sum) == 2) {
                bad.add(what + " is a second legendary permanent of that name (legend rule)");
            }
        }
        List<String> warn = new ArrayList<>();
        for (String b : bad) warn.add(b + ": the state-based actions remove it when the game resumes");
        return warn;
    }

    /**
     * Pause the game and re-anchor MageZero's search at the current state (G4). MCTS rebuilds every
     * tree from game.getLastPriority() plus the player histories since then; after injection that
     * anchor is still the empty pre-injection copy that init() made.
     */
    public static void anchor(Built b, boolean forSearch) {
        Game game = b.game;
        RandomUtil.setSeed(mix(b.seed, 3));
        game.setLocalRandom(new Random(mix(b.seed, 4))); // before the copy, which carries it
        game.pause();
        if (!forSearch) {
            ((GameImpl) game).clearHistory(); // no search will read an anchor: skip the game copy
            return;
        }
        UUID prio = game.getState().getPriorityPlayerId();
        game.setLastPriority(prio);
        Game anchor = game.getLastPriority();
        if (anchor == game || anchor.getTurnNum() != game.getTurnNum() || anchor.getStep() == null
                || anchor.getStep().getType() != b.engineStep) {
            throw new IllegalStateException("MCTS anchor not at the injected state");
        }
    }

    /** Re-draw a seat's whole hand from its library (below the known top cards). One PIMC sample. */
    public static List<String> resampleHand(Built b, String seat, Random rng) {
        Game game = b.game;
        Player p = b.players.get(seat);
        int n = p.getHand().size();
        int known = b.spec.players.get(seat).libraryTop.size();
        List<Card> tops = new ArrayList<>();
        for (int i = 0; i < known && p.getLibrary().size() > 0; i++) tops.add(p.getLibrary().drawFromTop(game));
        List<Card> hand = new ArrayList<>(p.getHand().getCards(game));
        p.getHand().clear();
        for (Card c : hand) p.getLibrary().putOnBottom(c, game);
        p.getLibrary().shuffle(rng);
        List<String> names = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Card c = p.getLibrary().drawFromTop(game);
            c.setZone(Zone.HAND, game);
            p.getHand().add(c);
            names.add(c.getName());
        }
        for (int i = tops.size() - 1; i >= 0; i--) p.getLibrary().putOnTop(tops.get(i), game);
        return names;
    }

    /** Salt of one decklist card's id stream: seat, name (64-bit FNV-1a) and copy number. */
    static long cardSalt(String seat, String name, int copy) {
        long h = 0xcbf29ce484222325L;
        for (byte x : (seat + "|" + name).getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
            h ^= x & 0xff;
            h *= 0x100000001b3L;
        }
        return mix(h, copy);
    }

    /** SplitMix-style mixing so seed, seed+1, ... give unrelated streams. */
    static long mix(long seed, long salt) {
        long z = seed + 0x9E3779B97F4A7C15L * (salt + 1);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    // =================================================================================== players

    /** Defaults for both seats: offline, no priors, no noise, auto-tap. Coach overrides the decider. */
    static void configure(BridgePlayer p, BridgePlayer opp) {
        StateEncoder enc = new StateEncoder();
        enc.setAgent(p.getId());
        enc.setOpponent(opp.getId());
        p.stateEncoder = enc;          // non-null: RLInit (and its HTTP client to 127.0.0.1:50052) is skipped
        p.actionEncoder = ACTIONS;     // non-null: printAllActionsFromDeck is skipped
        p.offlineMode = true;
        p.nn = null;
        p.noPolicyPriority = true;
        p.noPolicyTarget = true;
        p.noPolicyUse = true;
        p.noPolicyOpponent = true;
        p.noNoise = true;
        p.autoTap = true;
        p.allowMulligans = false;
        p.allowDuplicates = true;
        p.searchBudget = 300;
        p.searchTimeout = 60;
    }

    // =================================================================================== cards

    private static final Set<String> WARM = ConcurrentHashMap.newKeySet();

    /**
     * Database lookups and first-time class initialisation (card classes, their static abilities,
     * token classes) can draw UUIDs. Doing them before the id stream is reset keeps the ids of a
     * build independent of what this worker happened to build before.
     */
    static void warm(Spec spec) {
        for (Spec.PlayerState ps : spec.players.values()) {
            for (String name : ps.decklist) {
                if (WARM.add("card|" + name)) newCard(name, null, null);
            }
            for (Spec.Perm perm : ps.battlefield) {
                if (perm.isToken()) {
                    String cls = tokenClassName(perm);
                    if (WARM.add("token|" + cls)) {
                        try {
                            Class.forName(cls).getConstructor().newInstance();
                        } catch (ReflectiveOperationException | RuntimeException ignored) {
                            // reported properly by putToken
                        }
                    }
                } else if (perm.name != null && (perm.set != null || perm.number != null) && WARM.add("card|" + perm.name + "|" + perm.set)) {
                    newCard(perm.name, perm.set, perm.number);
                }
            }
        }
    }

    /** Every card name of the spec the database does not know, sorted (one error lists them all). */
    static List<String> unknownCards(Spec spec) {
        Set<String> names = new TreeSet<>();
        for (Spec.PlayerState ps : spec.players.values()) {
            names.addAll(ps.decklist);
            for (Spec.Perm perm : ps.battlefield) if (!perm.isToken() && perm.name != null) names.add(perm.name);
        }
        for (Spec.StackItem si : spec.stack) if (si.card != null) names.add(si.card);
        List<String> out = new ArrayList<>();
        for (String n : names) if (!Substitutions.known(n)) out.add(n);
        return out;
    }

    static CardInfo cardInfo(String name, String set, String number) {
        String key = name + "|" + set + "|" + number;
        CardInfo ci = CARD_INFO.get(key);
        if (ci != null) return ci;
        if (set != null && number != null) ci = CardRepository.instance.findCard(set, number);
        if (ci == null && name != null) {
            List<CardInfo> all = CardRepository.instance.findCards(name);
            List<String> prefs = new ArrayList<>();
            if (set != null) prefs.add(set);
            prefs.addAll(PREFERRED_SETS);
            outer:
            for (String s : prefs) {
                for (CardInfo c : all) {
                    if (s.equalsIgnoreCase(c.getSetCode())) {
                        ci = c;
                        break outer;
                    }
                }
            }
            // deterministic fallback (CardRepository.findCard(name, true) picks a random printing)
            if (ci == null && !all.isEmpty()) ci = CardRepository.instance.findPreferredCoreExpansionCard(name);
        }
        if (ci == null) {
            throw new SpecException("unknown card", List.of("'" + name + "'" + (set == null ? "" : " (" + set + ":" + number + ")")
                    + " is not in the XMage card database (options.substitute {\"missing\": \"Plains\"} builds the spec with a stand-in)"));
        }
        CARD_INFO.put(key, ci);
        return ci;
    }

    static Card newCard(String name, String set, String number) {
        Card c = cardInfo(name, set, number).createCard();
        if (c == null) throw new SpecException("unknown card", List.of("XMage cannot create '" + name + "'"));
        return c;
    }

    static void returnHandsToLibrary(Game game, Built out) {
        for (Player p : out.players.values()) {
            for (Card c : new ArrayList<>(p.getHand().getCards(game))) {
                p.getHand().remove(c);
                p.getLibrary().putOnTop(c, game);
            }
            if (!p.getHand().isEmpty()) throw new IllegalStateException("opening hand not returned");
        }
    }

    /** Take a card with this name out of the player's library, which keeps "library = deck minus seen". */
    static Card takeCard(Game game, Built out, Player p, String name, boolean lenient) {
        for (Card c : p.getLibrary().getCards(game)) {
            if (c.getName().equals(name)) return p.getLibrary().remove(c.getId(), game);
        }
        if (!lenient) {
            throw new SpecException("card accounting", List.of(p.getName() + ": no '" + name
                    + "' left in the library (decklist minus the cards already placed)"));
        }
        out.warnings.add(p.getName() + ": '" + name + "' not in the remaining library, created a new card");
        Card c = newCard(name, null, null);
        game.loadCards(new HashSet<>(Collections.singletonList(c)), p.getId());
        return c;
    }

    static void injectVisible(Game game, Built out, String seat, Ability fakeTemplate,
                              Map<Spec.Perm, List<UUID>> made, boolean lenient) {
        Spec.PlayerState ps = out.spec.players.get(seat);
        Player p = out.players.get(seat);
        for (Spec.Perm perm : ps.battlefield) {
            List<UUID> ids = new ArrayList<>();
            for (int i = 0; i < perm.count; i++) {
                Permanent pm;
                if (perm.isToken()) {
                    // A token's UUID decides its place in attack/block order. The fork draws it from
                    // game.getLocalRandom() (PermanentToken), which is seeded per determinization, so
                    // tie it to the spec entry instead: the same in every determinization. One mixed
                    // stream per (seat, entry, copy): the former 1000*seat + 31*entry + copy gave copy
                    // 32 of an entry the stream of the next entry's first token (one UUID, two permanents)
                    long salt = mix(mix(seat.equals("A") ? 1 : 2, made.size()), i);
                    DeterministicIds.reset(mix(out.idSeed, salt));
                    game.setLocalRandom(new Random(mix(out.idSeed, salt + 1)));
                    pm = putToken(game, p, perm, fakeTemplate);
                } else {
                    // a control-changed permanent's card comes out of its OWNER's library
                    Player owner = out.players.get(Spec.ownerSeat(seat, perm));
                    Card c = takeCard(game, out, owner, perm.name, lenient);
                    Ability src = fakeTemplate.copy();
                    src.setControllerId(p.getId());
                    src.setSourceId(c.getId());
                    // no ETB event; "enters with counters/tapped" replacements still apply
                    putCardOntoBattlefield(src, game, c, p, perm.tapped);
                    pm = game.getPermanent(c.getId());
                    if (pm == null) throw new SpecException("cannot build battlefield", List.of(seat + ": '" + perm.name + "' did not enter the battlefield"));
                }
                decorate(game, out, seat, perm, pm, i);
                ids.add(pm.getId());
            }
            made.put(perm, ids);
        }
        for (String n : ps.hand) {
            Card c = takeCard(game, out, p, n, lenient);
            c.setZone(Zone.HAND, game);
            p.getHand().add(c);
        }
        for (String n : ps.graveyard) { // bottom -> top
            Card c = takeCard(game, out, p, n, lenient);
            c.setZone(Zone.GRAVEYARD, game);
            p.getGraveyard().add(c);
        }
        for (String n : ps.exile) {
            Card c = takeCard(game, out, p, n, lenient);
            c.setZone(Zone.EXILED, game);
            game.getExile().add(c);
        }
        p.initLife(ps.life);           // initLife: no life gain/loss events (setLife would fire them)
        p.resetLandsPlayed();
        for (int i = 0; i < ps.landsPlayed; i++) p.incrementLandsPlayed();
        if (ps.manaPool != null && !ps.manaPool.isEmpty()) {
            Mana m = new Mana();
            for (char ch : ps.manaPool.toCharArray()) {
                switch (ch) {
                    case 'W': m.increaseWhite(); break;
                    case 'U': m.increaseBlue(); break;
                    case 'B': m.increaseBlack(); break;
                    case 'R': m.increaseRed(); break;
                    case 'G': m.increaseGreen(); break;
                    default: m.increaseColorless();
                }
            }
            p.getManaPool().addMana(m, game, fakeTemplate, true);
        }
    }

    /** Known top cards out, seeded shuffle, hidden hand drawn, library trimmed, known cards back on top. */
    static void injectLibraryAndHidden(Game game, Built out, String seat, Random rng, Ability fakeTemplate, boolean lenient) {
        Spec.PlayerState ps = out.spec.players.get(seat);
        Player p = out.players.get(seat);
        List<Card> tops = new ArrayList<>();
        for (String n : ps.libraryTop) tops.add(takeCard(game, out, p, n, lenient));
        p.getLibrary().shuffle(rng);
        for (int i = 0; i < ps.handUnknown; i++) {
            Card c = p.getLibrary().drawFromTop(game);
            if (c == null) throw new SpecException("hidden hand", List.of(seat + ": library ran out drawing handUnknown"));
            c.setZone(Zone.HAND, game);
            p.getHand().add(c);
        }
        if (ps.librarySize != null) {
            int keep = ps.librarySize - tops.size();
            if (p.getLibrary().size() < keep) {
                out.warnings.add(seat + ": librarySize " + ps.librarySize + " > cards available (" + (p.getLibrary().size() + tops.size()) + ")");
            }
            while (p.getLibrary().size() > Math.max(0, keep)) {
                Card c = p.getLibrary().drawFromBottom(game); // random cards leave: the library is shuffled
                c.setZone(Zone.OUTSIDE, game);
            }
        }
        for (int i = tops.size() - 1; i >= 0; i--) p.getLibrary().putOnTop(tops.get(i), game);
    }

    /**
     * CardUtil.putCardOntoBattlefieldWithEffects without its setOwnerId(player): the card keeps
     * its owner (the library it came from) and enters under {@code controller}'s control, with
     * originalControllerId = controller, as a reanimated creature does (ZonesHandler builds the
     * PermanentCard with the new controller), so resetting control in applyEffects keeps it there.
     * No ENTERS_THE_BATTLEFIELD event; "enters with counters / tapped" replacements still apply.
     */
    static void putCardOntoBattlefield(Ability source, Game game, Card card, Player controller, boolean tapped) {
        Card permCard = CardUtil.getDefaultCardSideForBattlefield(game, card);
        permCard.setZone(Zone.BATTLEFIELD, game);
        PermanentCard permanent = permCard instanceof MeldCard ? new PermanentMeld(permCard, controller.getId(), game)
                : new PermanentCard(permCard, controller.getId(), game);
        // as ZonesHandler does for every permanent that enters: the card's static abilities were
        // registered (GameState.addCard) with its OWNER as controller, so without this a
        // control-changed Crusader of Odric counted its owner's creatures and a stolen anthem
        // pumped the owner's team
        game.getContinuousEffects().setController(permanent.getId(), controller.getId());
        game.getPermanentsEntering().put(permanent.getId(), permanent);
        permCard.applyEnterWithCounters(permanent, source, game);
        permanent.entersBattlefield(source, game, Zone.OUTSIDE, false);
        game.addPermanent(permanent, game.getState().getNextPermanentOrderNumber());
        game.getPermanentsEntering().remove(permanent.getId());
        if (tapped) permanent.setTapped(true);
        permanent.removeSummoningSickness();
        for (ContinuousEffect effect : game.getState().getContinuousEffects().getLayeredEffects(game)) {
            Optional<Ability> ability = game.getState().getContinuousEffects().getLayeredEffectAbilities(effect).stream().findFirst();
            if (ability.isPresent() && permanent.getId().equals(ability.get().getSourceId())) {
                effect.init(ability.get(), game, controller.getId());
            }
        }
    }

    /**
     * A token straight onto the battlefield, event-free like a card. Token.putOntoBattlefield
     * would queue a ZONE_CHANGE and an ENTERS_THE_BATTLEFIELD simultaneous event (and a
     * CREATED_TOKEN one) that GameState handles only when the game resumes, i.e. after the
     * injector cleared the pending triggers: "whenever a creature enters" permanents (Dazzling
     * Angel, Authority of the Consuls) then triggered on every injected token (17lands row 560303
     * user turn 9: 6 triggers on the stack at the decision, +6 life after rollover). It would
     * also run CREATE_TOKEN replacement effects (token doublers). This is
     * TokenImpl.putOntoBattlefieldHelper without the events.
     */
    static Permanent putToken(Game game, Player p, Spec.Perm perm, Ability fakeTemplate) {
        String cls = tokenClassName(perm);
        Token token;
        try {
            token = (Token) Class.forName(cls).getConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            throw new SpecException("unknown token", List.of("no token class " + cls));
        } catch (NoSuchMethodException e) {
            throw new SpecException("unsupported token", List.of(cls + " has no no-argument constructor"));
        } catch (ReflectiveOperationException | ClassCastException e) {
            throw new SpecException("unsupported token", List.of("cannot create " + cls + ": " + e));
        }
        Ability src = fakeTemplate.copy();
        src.setControllerId(p.getId());
        src.setSourceId(token.getId());
        if (token instanceof TokenImpl) {
            // the image / set code TokenImpl.putOntoBattlefield would pick (the dump shows the set)
            TokenInfo info = TokenImpl.generateTokenInfo((TokenImpl) token, game, token.getId());
            token.setExpansionSetCode(info.getSetCode());
            token.setImageNumber(info.getImageNumber());
        }
        PermanentToken pm = new PermanentToken(token, p.getId(), game); // id from game.getLocalRandom()
        game.getState().addCard(pm);
        game.getPermanentsEntering().put(pm.getId(), pm);
        pm.setTapped(perm.tapped);
        pm.updateZoneChangeCounter(game, new ZoneChangeEvent(pm, p.getId(), Zone.OUTSIDE, Zone.BATTLEFIELD));
        game.setScopeRelevant(true);
        pm.entersBattlefield(src, game, Zone.OUTSIDE, false); // replacements apply; no event is queued
        game.setScopeRelevant(false);
        game.addPermanent(pm, game.getState().getNextPermanentOrderNumber());
        pm.setZone(Zone.BATTLEFIELD, game);
        game.getPermanentsEntering().remove(pm.getId());
        return pm;
    }

    static String tokenClassName(Spec.Perm perm) {
        if (perm.tokenClass != null) {
            return perm.tokenClass.contains(".") ? perm.tokenClass : "mage.game.permanent.token." + perm.tokenClass;
        }
        if (TOKENS == null) TOKENS = TokenRepository.instance.getByType(TokenType.TOKEN);
        Set<String> classes = new TreeSet<>();
        for (TokenInfo t : TOKENS) {
            if (t.getName().equals(perm.token) && (perm.set == null || perm.set.equalsIgnoreCase(t.getSetCode()))) {
                classes.add(t.getFullClassFileName());
            }
        }
        if (classes.isEmpty()) throw new SpecException("unknown token", List.of("no token '" + perm.token + "'" + (perm.set == null ? "" : " in set " + perm.set)));
        if (classes.size() > 1) {
            throw new SpecException("ambiguous token", List.of("token '" + perm.token + "'" + (perm.set == null ? "" : "/" + perm.set)
                    + " maps to " + classes + "; give tokenClass instead"));
        }
        return classes.iterator().next();
    }

    static void decorate(Game game, Built out, String seat, Spec.Perm perm, Permanent pm, int index) {
        if (perm.id != null) {
            String base = seat + ":" + perm.id;
            if (index == 0) out.aliases.put(base, pm.getId());
            if (perm.count > 1) out.aliases.put(base + "#" + (index + 1), pm.getId());
        }
        out.origin.put(pm.getId(), perm);
        out.originSeat.put(pm.getId(), seat);
        pm.setTapped(perm.tapped); // the spec's state wins over "enters tapped" replacements
        // counters straight into the map: Permanent.addCounters would fire COUNTER_ADDED triggers.
        // A listed type is set to the spec's count (a planeswalker has already entered with its
        // printed loyalty); types the spec does not list keep what the card entered with.
        for (Map.Entry<String, Integer> e : perm.counters.entrySet()) {
            CounterType ct = counterType(e.getKey());
            int have = pm.getCounters(game).getCount(ct);
            if (have > 0) pm.getCounters(game).removeCounter(ct, have);
            if (e.getValue() > 0) pm.getCounters(game).addCounter(ct.createInstance(e.getValue()));
        }
        // no public setters (G9); Permanent.damage(...) would fire DAMAGED events
        Reflect.set(PermanentImpl.class, pm, "damage", perm.damage);
        Reflect.set(PermanentImpl.class, pm, "controlledFromStartOfControllerTurn", !perm.sick);
        if (perm.faceDown) pm.setFaceDown(true, game); // FDN has no morph/disguise; real ones need the effect
    }

    static CounterType counterType(String key) {
        try {
            return CounterType.valueOf(key);
        } catch (IllegalArgumentException e) {
            CounterType ct = CounterType.findByName(key);
            if (ct != null) return ct;
            throw new SpecException("unknown counter", List.of("counter '" + key + "' (use mage.counters.CounterType names, e.g. P1P1)"));
        }
    }

    static void attach(Game game, Built out, String seat, Ability fakeTemplate, Map<Spec.Perm, List<UUID>> made) {
        Player p = out.players.get(seat);
        for (Spec.Perm perm : out.spec.players.get(seat).battlefield) {
            if (perm.attachTo == null) continue;
            UUID hostId = out.aliases.get(perm.attachTo);
            UUID auraId = made.get(perm).get(0);
            Permanent host = game.getPermanent(hostId);
            Ability src = fakeTemplate.copy();
            src.setControllerId(p.getId());
            src.setSourceId(auraId);
            if (ReanimatedAura.enchantsGraveyardCard(game, game.getPermanent(auraId))) {
                // Animate Dead & co. enchant "creature card in a graveyard" until their ETB trigger
                // (not fired by injection) swaps that for the creature they returned
                ReanimatedAura.install(game, game.getPermanent(auraId), host, src);
                if (!host.getControllerId().equals(p.getId())) {
                    out.warnings.add(seat + ": " + perm.describe() + " enchants " + perm.attachTo + ", which " + seat
                            + " does not control: the creature it returned enters under " + seat + "'s control (list it under "
                            + seat + " with owner " + Spec.other(seat) + ")");
                }
            }
            // before any state-based action check, or an unattached Aura goes to the graveyard
            boolean ok = host.addAttachment(auraId, src, game);
            Permanent aura = game.getPermanent(auraId);
            if (!ok || aura == null || !hostId.equals(aura.getAttachedTo())) {
                throw new SpecException("cannot attach", List.of(seat + ": " + perm.describe() + " -> " + perm.attachTo
                        + " (" + host.getName() + ") was refused by XMage"));
            }
        }
    }

    static void castOntoStack(Game game, Built out, Spec.StackItem si, Card c) {
        Player p = out.players.get(si.controller);
        if (c.getSpellAbility() == null || c.isLand(game)) {
            // Card.cast would throw an NPE in SpellAbility.getSpellAbilityToResolve
            throw new SpecException("cannot build stack", List.of("'" + si.card + "' is not a spell (a land or a card with no spell "
                    + "ability cannot be on the stack; a substituted card on the stack is dropped by options.substitute)"));
        }
        c.setZone(Zone.HAND, game);
        p.getHand().add(c);
        // Card.cast puts the spell on the stack without paying costs or choosing targets (the "stack:" cheat)
        c.cast(game, Zone.HAND, c.getSpellAbility(), p.getId());
        Spell spell = game.getStack().getSpell(c.getId());
        if (spell == null) throw new SpecException("cannot build stack", List.of("casting " + si.card + " failed"));
        int nTargets = spell.getSpellAbility().getTargets().size();
        if (si.targets.size() > nTargets) {
            throw new SpecException("cannot build stack", List.of(si.card + " has " + nTargets + " target slots, spec gives " + si.targets.size()));
        }
        int ti = 0;
        for (String t : si.targets) {
            UUID id = t.startsWith("player:") ? out.players.get(t.substring(7)).getId() : out.aliases.get(t);
            spell.getSpellAbility().getTargets().get(ti++).addTarget(id, spell.getSpellAbility(), game, true);
        }
    }

    // =================================================================================== turn position

    /** Turn 1, and the engine will still run its DRAW step (entry before it, or at it with BEGIN_STEP). */
    static boolean entersBeforeFirstDraw(Spec spec) {
        if (spec.turn != 1) return false;
        List<String> steps = Spec.turnSteps();
        int at = steps.indexOf(spec.step), draw = steps.indexOf("DRAW");
        return at < draw || (at == draw && "BEGIN_STEP".equals(spec.enterMode));
    }

    /** Previous step in turn order (BEGIN_STEP enters as "that step has just ended"). */
    static PhaseStep previousStep(PhaseStep s) {
        List<String> steps = Spec.turnSteps();
        int i = steps.indexOf(s.name());
        if (i <= 0) throw new IllegalArgumentException("no previous step for " + s);
        return PhaseStep.valueOf(steps.get(i - 1));
    }

    static void setTurnPosition(Game game, Built out, Spec spec, UUID activeId) {
        PhaseStep target = PhaseStep.valueOf(spec.step);
        PhaseStep stepType;
        Step.StepPart part;
        switch (spec.enterMode) {
            case "BEGIN_STEP":
                // A paused game always resumes a PRE step through resumeBeginStep (no turn-based
                // actions), and MCTS copies are always paused (G5). POST of the previous step makes
                // both the live game and every copy run postPriority and then play the target step
                // from its start with beginStep.
                stepType = previousStep(target);
                part = Step.StepPart.POST;
                break;
            case "PRIORITY_HELD":
                stepType = target;
                part = Step.StepPart.PRIORITY;
                break;
            default:
                stepType = target;
                part = Step.StepPart.PRE;
        }
        TurnPhase phaseType = TurnPhase.valueOf(Spec.phaseOf(stepType.name()));
        Turn turn = game.getState().getTurn();
        Phase phase = turn.getPhase(phaseType); // the instance inside Turn.phases (resumePlay iterates it)
        @SuppressWarnings("unchecked")
        List<Step> steps = (List<Step>) Reflect.get(Phase.class, phase, "steps");
        Step step = null;
        for (Step s : steps) if (s.getType() == stepType) step = s;
        if (step == null) throw new IllegalStateException("step " + stepType + " not in phase " + phaseType);
        turn.setPhase(phase);
        phase.setStep(step);
        // G1: Phase.resumeStep ends the game (game.end()) when stepPart is null
        Reflect.set(Step.class, step, "stepPart", part);
        Reflect.set(Phase.class, phase, "activePlayerId", activeId);
        out.enginePhase = phaseType;
        out.engineStep = stepType;
        out.enginePart = part;
    }

    // =================================================================================== checks

    /** Read the game back and list every difference from the spec (the gotchas as checks). */
    static List<String> verify(Built out, Map<Spec.Perm, List<UUID>> made) {
        Game game = out.game;
        Spec spec = out.spec;
        List<String> bad = new ArrayList<>();
        // G1
        Step step = game.getStep();
        if (step == null || step.getType() != out.engineStep || step.getStepPart() != out.enginePart) {
            bad.add("turn position: engine at " + (step == null ? null : step.getType() + "/" + step.getStepPart())
                    + ", expected " + out.engineStep + "/" + out.enginePart);
        }
        if (game.getTurnNum() != spec.turn) bad.add("turn " + game.getTurnNum() + " != " + spec.turn);
        if (!out.players.get(spec.activePlayer).getId().equals(game.getActivePlayerId())) bad.add("active player not " + spec.activePlayer);
        // G2
        if (game.getState().getTurnMods().size() != (entersBeforeFirstDraw(spec) ? 1 : 0)) bad.add("TurnMods left after injection");
        for (String seat : Spec.SEATS) {
            BridgePlayer p = out.players.get(seat);
            Spec.PlayerState ps = spec.players.get(seat);
            // G7
            if (!game.getState().getTriggered(p.getId()).isEmpty()) bad.add(seat + ": pending triggers after injection");
            // G11
            if (p.getMatchPlayer() == null || p.getMatchPlayer().getDeck() == null) bad.add(seat + ": match player not wired");
            if (p.getLife() != ps.life) bad.add(seat + ": life " + p.getLife() + " != " + ps.life);
            if (p.getLandsPlayed() != ps.landsPlayed) bad.add(seat + ": landsPlayed " + p.getLandsPlayed() + " != " + ps.landsPlayed);
            int handWant = ps.hand.size() + ps.handUnknown;
            if (p.getHand().size() != handWant) bad.add(seat + ": hand " + p.getHand().size() + " != " + handWant);
            if (p.getGraveyard().size() != ps.graveyard.size()) bad.add(seat + ": graveyard " + p.getGraveyard().size() + " != " + ps.graveyard.size());
            if (ps.librarySize != null && p.getLibrary().size() != ps.librarySize) bad.add(seat + ": library " + p.getLibrary().size() + " != " + ps.librarySize);
            List<Card> lib = p.getLibrary().getCards(game);
            for (int i = 0; i < ps.libraryTop.size(); i++) {
                if (i >= lib.size() || !lib.get(i).getName().equals(ps.libraryTop.get(i))) bad.add(seat + ": library top " + i + " is not " + ps.libraryTop.get(i));
            }
            int permsWant = 0;
            for (Spec.Perm perm : ps.battlefield) permsWant += perm.count;
            int permsHave = game.getBattlefield().getAllActivePermanents(p.getId()).size();
            if (permsHave != permsWant) bad.add(seat + ": " + permsHave + " permanents on the battlefield, spec has " + permsWant);
            for (Spec.Perm perm : ps.battlefield) {
                for (UUID id : made.getOrDefault(perm, List.of())) {
                    Permanent pm = game.getPermanent(id);
                    String what = seat + ": " + perm.describe();
                    if (pm == null) {
                        bad.add(what + " left the battlefield");
                        continue;
                    }
                    boolean attacking = spec.attackers.stream().anyMatch(a -> id.equals(out.aliases.get(a.attacker)));
                    if (!attacking && pm.isTapped() != perm.tapped) bad.add(what + ": tapped=" + pm.isTapped());
                    boolean sick = !(boolean) Reflect.get(PermanentImpl.class, pm, "controlledFromStartOfControllerTurn");
                    if (sick != perm.sick) bad.add(what + ": sick=" + sick); // G9
                    if (pm.getDamage() != perm.damage) bad.add(what + ": damage=" + pm.getDamage());
                    for (Map.Entry<String, Integer> e : perm.counters.entrySet()) {
                        int have = pm.getCounters(game).getCount(counterType(e.getKey()));
                        if (have != e.getValue()) bad.add(what + ": " + e.getKey() + " counters=" + have);
                    }
                    if (perm.attachTo != null && !out.aliases.get(perm.attachTo).equals(pm.getAttachedTo())) bad.add(what + ": not attached to " + perm.attachTo);
                    if (!p.getId().equals(pm.getControllerId())) bad.add(what + ": controlled by " + out.seatOf(pm.getControllerId()));
                    String ownerSeat = Spec.ownerSeat(seat, perm);
                    if (!perm.isToken() && !out.players.get(ownerSeat).getId().equals(pm.getOwnerId())) {
                        bad.add(what + ": owned by " + out.seatOf(pm.getOwnerId()) + ", spec " + ownerSeat);
                    }
                }
            }
        }
        if (game.getStack().size() != spec.stack.size()) bad.add("stack has " + game.getStack().size() + " objects, spec " + spec.stack.size());
        return bad;
    }
}
