# Hidden information: what MageZero sees, what its search sees, and how other games do it

Written 2026-09-26 on branch `gameplay-data`. It answers four questions:

1. How is private information handled? What are the data structures, and where is it masked?
2. How does the search handle private information? Does it leak into the future states the search
   explores?
3. How do other hidden-information games (StarCraft, Stratego, poker, bridge, Mahjong, Magic itself)
   handle it, in search and in training?
4. How does MageZero compare? Which approaches are most promising for a limited agent that has to
   play well under real uncertainty?

A second round, the same day, added three more:

5. Is IS-MCTS the fix? (§6)
6. Do XMage's own search bots leak, or only MageZero? (§7)
7. How should baseline and trained agents get only the information they should? (§8)

**Code audited.**

- Experiment #1's engine: MageZero `bcc76de` and the XMage fork at `5a32441c`.
- Upstream v0.2.0: the fork's `master` at `cb7e9c6fe2` and MageZero `main` at `11a5974668`. It
  behaves the same on every path described here (§3.8).
- Upstream XMage's own AI players, at `magefree/mage` `504f614fe6` (§7).

File references are to the fork unless marked, with `5a32441c`'s line numbers.

**Measured.** Two controlled position pairs (§3.4), each differing only in a card the deciding
player can't see:

- the opponent's hand (a counterspell or not);
- the player's own next draw (a castable card or a dead land).

They were decided by:

- MageZero's own MCTS through the mzbridge, with three evaluators;
- XMage's MAD AI through a small probe (§7).

Scripts: `tools/gameplay/hidden_info_leak.py` and `tools/gameplay/MadProbe.java`.

## Summary

**1. Nothing is masked: experiment #1's network saw the opponent's hand.**

- `StateEncoder` writes the opponent's hand, card by card, into every state vector. Its
  `perfectInfo` flag defaults to true (`StateEncoder.java:57`, `:591`).
- The switch that should turn this off is never read. `configs/game.yml`, here and upstream, writes
  `hidden_info:`, but `Config.java:79` reads `hiddenInfo`, so the key is ignored and the default
  (true) wins (`:145`). docs/008 §6 found this.
- So in experiment #1 **both hands were in every training row and in every network evaluation
  inside the search.**
- Nothing later in the pipeline can mask it. The inference server and trainer receive hashed
  integer ids with no zone labels. The ignore list only drops rare features, and token dropout is a
  regularizer.
- Even with the switch fixed, the encoder only hides a hand from whoever is *not* deciding at that
  node. Inside a search, the nodes where the opponent decides would still encode the opponent's
  real hand.

**2. The search is clairvoyant, independently of the network.**

- Every search starts from an exact copy of the real game (`GameImpl.createSimulationForAI` is
  `this.copy()`). The copy holds both hands, both libraries in order, and both real decks.
  `shuffleUnknowns`, which would re-deal the hidden cards, has no callers (`ComputerPlayerMCTS.java:378`:
  "dont shuffle here").
- The search explores future states by running the real rules engine forward in that copy:
  - the opponent's real hand decides what the opponent can do in the tree;
  - the real library order decides every draw. Draws set an `isRandomTransition` flag that nothing
    reads.
- There are no random rollouts. Each new node is scored by the network, which sees both hands, or
  by the offline heuristic, which reads only hand *size*.
- **Demonstrated (§3.4).** In one public position, A holds Serra Angel and B has 1UU open. B's
  hidden hand is either Refute + Island or Island + Island, and A cannot tell which.

  | Search (8 seeds per world) | B holds Refute | B holds two Islands | Q(cast) gap, paired |
  |---|---|---|---|
  | Offline heuristic, 1,000 simulations | Pass 6/8, Q(cast) −0.06 | Cast 7/8, Q(cast) +0.26 | +0.31 [0.21, 0.41] |
  | gen 33 as experiment #1 ran it, 300 simulations | Pass 7/8, Q(cast) −0.05 | Cast 7/8, Q(cast) +0.28 | +0.32 [0.23, 0.42] |
  | gen 33 with the opponent's hand masked from its input | Pass 6/8, Q(cast) −0.05 | Cast 7/8, Q(cast) +0.27 | +0.32 [0.23, 0.41] |
  | Offline, B's hand re-drawn for each of 16 searches (PIMC) | Cast 12/16 | Cast 12/16 | +0.03 [−0.01, 0.07] |

  The search plays around Refute only when Refute is really there. Masking the network's input
  doesn't change that. Re-dealing the hidden cards does.
- **It also reads its own next card.** Holding a cantrip creature, it values casting it +0.065 higher
  (CI 0.058–0.072, 8 of 8 seeds) when the card it will draw is castable than when it's a dead land.
  The policy target moves 13 points with it.
- **The leak reaches training twice.** Policy targets are the root visit counts. Value targets blend
  the game result with the search's root values. At experiment #1's λ = 0.70 the game result
  carries on average about 2% of a target's weight; the search's own estimates carry the rest.
- **So experiment #1 learned, and measured card strength in, open-hand Magic.** Its largest gaps
  against 17lands are cards whose value depends on the opponent *not* knowing: blue (−8.0 points),
  Refute (45% against 58%), and the combat tricks (Fleeting Flight and Giant Growth have the most
  negative IWD). That fits clairvoyance. The targeting and timing explanation in docs/003 fits
  too. Neither has been tested as the cause.

**3. Other games (§4).** Apart from deliberate cheating variants, every system surveyed keeps the
acting policy a function of what the player could know. The families differ in how they search:

| Family | How it treats hidden state | Examples |
|---|---|---|
| **Determinization (PIMC)** | Sample worlds consistent with what you know, search each as if it were known, average | bridge (GIB), Skat (Kermit), simplified Magic (Cowling et al.), AlphaZero-style Stratego (AlphaZe\*\*, Ataraxos) |
| **Information-set search** | One tree over what a player can distinguish, with a new sample each iteration | IS-MCTS |
| **Belief-state game-theoretic search** | Search over probability distributions of hidden states, computing equilibrium strategies | DeepStack, Libratus, Pluribus (poker); ReBeL; Student of Games; Obscuro (fog-of-war chess) |
| **Model-free RL on partial observations** | No search at play time; memory and equilibrium training instead | AlphaStar, OpenAI Five, DeepNash (Stratego, emergent bluffing), DouZero |
| **Privileged information in training only** | Hidden state reaches a critic, an oracle or a teacher, never the acting policy | AlphaStar's value network, PerfectDou, Suphx's oracle guiding |

Where measured, seeing hidden information is worth a lot:

- In Dou Di Zhu a cheating search won 56.5% against 43.6% for determinized search.
- In AlphaStar, giving the training-only value network the opponent's observations raised the
  average win rate from 22% to 82% in a simplified ablation.

Learning to *imitate* an expert who sees it fails, because the expert doesn't know what the
student can't see (Warrington et al. 2021). MageZero's policy targets are exactly that.

**4. The gap, and what to do (§5).** MageZero is a perfect-information agent in all three places
that matter: its inputs, the world its search explores, and its training targets. It is below
determinization, the simplest hidden-information baseline. The ranked plan:

1. **Fix the observation.** Correct the config key, encode every node from the searcher's
   information, keep cards that are legitimately known, and add an equivalence test. This is
   hours of work, and it doesn't fix the search.
2. **Determinize the search.** Sample the opponent's hand and both libraries' order from the
   searcher's belief at every decision, with one tree per sampled world and no tree reuse across
   worlds. The belief starts uniform over the unseen cards and later comes from a learned belief
   head, as in Ataraxos. This is the cheapest real fix. It makes the caster play around a
   counterspell as often as the counterspell is likely to be there.
3. **Give the in-tree opponent its own information set.** Plain determinization lets the simulated
   opponent see your hand. That erases exactly the value a counterspell has in human games:
   mana held up that an opponent walks into, and bluffs. There are three ways to fix it:
   - an opponent that acts from an imperfect-information policy;
   - re-sampling your own hidden cards at the opponent's nodes;
   - multiple-observer IS-MCTS.
4. **Weight value targets toward real outcomes** (λ ≈ 0.95), so search bias is inherited less.
5. **Choose opponents to match the comparison.** Self-play measures a card against the agent
   itself. 17lands measures it against human players. Human-like opponents from docs/008 close
   that gap.

Equilibrium methods (DeepNash's R-NaD, ReBeL, Student of Games) are the principled end point for
bluffing and mixed play, but they're research-scale for Magic.

**5. IS-MCTS (§6): the right direction, not the first step.**

- **What it fixes.** One information-set tree fixes the searcher's strategy fusion, and spends a
  small budget better than K shallow trees. With a network it beat determinization by +291 Elo in
  Phantom Go.
- **What it doesn't.** Its plain form still lets the simulated opponent see your hand, so it doesn't
  recover a counterspell's ambush and bluff value. That needs the multiple-observer form with
  self-determinization, or a policy opponent.
- **What it costs.** In MageZero it needs world-independent action keys and a state cache per world:
  "particle IS-MCTS", which holds the belief as K sampled worlds.
- **Order.** Ensemble determinization first, then particle SO-IS-MCTS as an A/B at equal budget.

**6. XMage's own bots (§7): the clairvoyant search is MageZero's.**

- **Upstream's Monte Carlo player determinizes.** It re-deals the opponents' hands and shuffles
  every library, per thread and per rollout. MageZero's rewrite dropped that.
- **The MAD AI searches the real copy, but never lets the opponent respond.** It casts the Angel
  whatever B holds (6/6 seeds, same score): blind, not clairvoyant.
- **It leaks its own draws through resolution.** In the cantrip position its choice flips with its
  unseen next card (6/6).

**7. Agents that see only what they should (§8).**

- **One information contract, one set of shared components:** a knowledge tracker, a belief in
  three tiers, and a world sampler with its own RNG, hooked where every XMage bot copies the game.
- **One certification suite** that every agent must pass.
- **Every baseline made fair with the same sampler**, including the raw search that makes gen-0
  data and serves as the yardstick.
- **The first three steps need no GPU:**
  1. the sampler;
  2. the certification suite;
  3. a clairvoyant-against-fair head-to-head, which measures what the leak is worth in Magic.

---

## 1. What is hidden in a game of limited

| Information | Who knows it in a real game | Encoder, experiment #1 | Search, experiment #1 |
|---|---|---|---|
| Your hand | you | every card | real |
| Opponent's hand | the opponent; you know its size and any card revealed or returned to it | **every card** (`perfectInfo`) | **real**: decides the opponent's options |
| Library order (either player) | nobody, except after scry, surveil or "look at the top" effects | count only (`LibraryCount`) | **real**: every draw in the tree is the real next card |
| Your library's contents | you: your decklist minus the cards seen | not encoded | real |
| Opponent's decklist | in limited, only the cards seen so far | not encoded, but the hand is | **real deck** |
| Face-down cards | their controller | the real card's characteristics (not used by FDN: none of its 512 card classes has a face-down mechanic) | real |
| Battlefield, graveyards, face-up exile, stack, life, mana pools, zone sizes | everyone | encoded | real |

"Known hidden" information matters too. A creature bounced to its owner's hand, a revealed tutor
target, or a card seen by a "look at the top card" effect is hidden from the table but known to a
player. XMage keeps no lasting record of it: `GameState.revealed` is cleared whenever all players
pass priority, and `lookedAt` is copied shallowly (MageZero issue #4 found the second). The
encoder marks it `// TODO: revealed cards` (`StateEncoder.java:560`).

## 2. How private information is represented and masked

### 2.1 From game state to network input

```
XMage Game ── StateEncoder.processState(game, decisionPlayer, decisionType, decisionText)
   │            builds a tree of named Features: "Player/Hand/Serra Angel#1/Flying", ...
   │            each name is hashed with its parent's seed into [0, 2,000,000)   (v0.2: [0, 2^31))
   ▼
Set<Integer> stateVector                                    ~1–2k ids per state
   ├── search: every MCTSNode ── ids ──▶ inference server ── ignore list or dense vocab ──▶ model
   └── training: LabeledState(stateVector, root visit counts, root mean Q)
                 ──▶ HDF5 /indices /offsets /row ──▶ H5Indexed ──▶ train.py
```

- `Features.java` hashes `name#occurrence` under the parent's seed. Names never reach Python: the
  only record of which id means what is an optional research log (`FeatureMap`,
  `log_feature_hash`).
- One state vector is computed for every decision node the search creates
  (`MCTSPlayer.freezeState`, `MCTSPlayer.java:102–106`). The same encoder object serves both seats
  inside the searcher's tree (`ComputerPlayerMCTS.java:375`).
- A training row stores the *root's* state vector, the root visit counts and the root mean Q
  (`ComputerPlayerMCTS2.java:338`), with the blended value label added after the game
  (`ParallelDataGenerator.java:437–447`).

### 2.2 How a hand is encoded

`processState` always writes two player blocks: `Player` for the searching agent and `Opponent`
for the other seat, whoever is deciding (`StateEncoder.java:645–650`). Within each block,
`processPlayer` (`:533–609`) writes life, lands-playable, `LibraryCount`, mana pool, battlefield,
graveyard, command zone and watchers. The hand is written in one of two ways:

```java
if (playerId == decisionPlayerId || perfectInfo) {   // StateEncoder.java:591, "keep perspective"
    processHand(hand, ...);        // per card: name#k, card types, colours, subtypes, mana cost,
                                   // each ability's rule text, CanActivate for activated abilities
} else {
    f.addNumericFeature("CardsInHand", hand.size());   // a count only
}
```

The library is always a count. Its order and contents are never encoded.

### 2.3 The switch, and why it was on in experiment #1

| Layer | What it says |
|---|---|
| `StateEncoder.perfectInfo` | `true` by default (`:57`) |
| `ParallelDataGenerator.java:328, :331` | per seat, `perfectInfo = Config…hiddenInfo.opponentHand` |
| `Config.java:79` | reads the YAML key **`hiddenInfo`** |
| `Config.java:145` | `see_opponent_hand` defaults to **`true`** |
| `configs/game.yml:34, :64` (upstream MageZero's: `:39, :72`) | writes **`hidden_info:`** `see_opponent_hand: true` |

The YAML's key doesn't match, so the default applies. The YAML's own value is also true. Either way,
every experiment #1 state had both hands. Setting it to false in `game.yml` would do nothing.

### 2.4 Nothing downstream can mask it

The server and the trainer see only integers. Without the feature names they cannot tell a hand
feature from a battlefield feature, so there is no place in Python to apply a mask:

- **The ignore list** (`dataset.create_redundancy_ignore_list`) and the dense vocab drop features
  seen in ≤ 10 states and exact duplicates. Opponent-hand features are common, so they survive.
- **Token dropout** (30% of tokens during training, `model.py:205`) is random regularization. It
  drops hand features no more than any other feature.

A mask has to be applied in the encoder, before hashing.

### 2.5 What `perfectInfo = false` would and would not do

With the key fixed, a hand is shown only to the player deciding at that node:

- **Training rows are clean.** Each row is the searcher's own decision (`isPlayer` is always true
  for MCTS2 rows), so the opponent's hand becomes a count.
- **Search nodes where the opponent decides are not.** `MCTSPlayer.freezeState` encodes from the
  deciding player's view (`processState(game, playerId, …)`). At an opponent node that player is
  the opponent, so the opponent's hand, the real one, is encoded, and the searcher's hand is
  hidden. The network's value at those nodes still depends on the opponent's real cards. The
  gen 33 run with the input masked (§3.4) had this.
- **Known cards are hidden too.** A bounced creature the searcher saw go back to hand becomes a
  count, which loses legitimate information (§1).
- The test uses `==` on `UUID`s. It works only because XMage's copies share `UUID` objects: fragile,
  but not a leak today.

### 2.6 Other inputs

- **The offline evaluator** (`GameStateEvaluator3`, used by raw-search baselines and by §3.4) scores
  hand *size* only (`GameStateEvaluator3.java:49`).
- **The opponent policy head** predicts the opponent's action from a state vector that includes
  the opponent's hand. It was off in experiment #1.

## 3. How the search handles private information

### 3.1 How one search runs

1. At a real decision, `ComputerPlayerMCTS2.getNextAction` copies the game at the player's last
   priority: `createMCTSGame(game.getLastPriority())` (`ComputerPlayerMCTS2.java:294`). That calls
   `GameImpl.createSimulationForAI()`, which is `this.copy()` (`GameImpl.java:357–363`), and swaps
   both seats for `MCTSPlayer` puppets that share the searcher's encoder
   (`ComputerPlayerMCTS.java:371–384`). The Javadoc above it says the copy re-deals hidden cards.
   The code doesn't: `shuffleUnknowns` (`:385–404`, "TODO: make true stochastic MCTS") has no
   callers. The only re-dealing code, `MCTSNode.randomizePlayers`, sits in the deprecated
   `createSimulation`.
2. Each tree node stores a `GameState`. `MCTSNode.validateState` restores the parent's state into
   one shared simulation game, gives both puppets the actions on the path, and resumes the engine
   until someone faces a decision (`MCTSNode.java:238–284`). **Every future state in the tree is
   produced by the real rules engine from the real hidden state.**
3. The node's children are the deciding player's legal options in that state: playable abilities
   (`MCTSPlayer.java:122`), targets, choices, amounts (`MCTSNode.createChildren`, `:432–480`). Its
   value and priors come from the network, called on that node's state vector
   (`MCTSNode2.java:52–58`).
4. PUCT selection maximizes the searcher's value at its own nodes and minimizes it at the
   opponent's (`MCTSNode.select`, sign −1, `:391–431`).
5. **No rollouts.** A leaf is scored once, by the network or the offline heuristic, and backed up.
6. After the search the root visit counts become the policy target and the root mean Q becomes the
   row's `stateScore` (`ComputerPlayerMCTS2.java:338`). The chosen child's subtree is reused as the
   next decision's root when its state vector matches (`:308–313`).

### 3.2 Where hidden information enters the tree

| Channel | Mechanism | Code | Removed by `perfectInfo = false`? |
|---|---|---|---|
| **The opponent's options** | Children are the opponent's legal plays on the real state, so a Refute it really holds is a child, and one it doesn't hold isn't | `MCTSPlayer.java:122`, `MCTSNode.java:432` | No |
| **Draws** | The copy keeps library order, so each draw in the tree is the real next card, for both players. `drawCards` sets `isRandomTransition` (`MCTSPlayer.java:97–101`), `MCTSNode.setPlayer` copies it to the node (`:291`), and nothing reads it: no chance nodes, no re-sampling | `MCTSPlayer.java:97`, `MCTSNode.java:291` | No |
| **The opponent's deck** | Everything the opponent will draw comes from its real remaining library | copy | No |
| **Network input** | Every node's evaluation includes both hands | `StateEncoder.java:591`, `MCTSNode2.java:52` | At the searcher's nodes only (§2.5) |
| **The in-tree opponent** | Opponent nodes minimize the searcher's value in a world where the opponent sees the searcher's hand. It never blocks into a combat trick it can see | `MCTSNode.select` | No |

Removing the first four channels is what determinization does (§5). The last one survives
determinization (§5.2).

### 3.3 How far down the leak reaches

Neither channel needs a deep tree:

- **The counterspell channel** needs no depth at all. B's decision to counter is the ply directly
  below A's cast.
- **The draw channel** needs a draw inside the tree. A cantrip or an "enters, draw a card" creature
  puts one right below the root, as §3.4's second position measures. A draw step needs a tree that
  crosses into the next turn.

Experiment #1 searched 96 simulations per decision, and reused subtrees add depth. How often its
trees contained a draw was not measured. Counting nodes with `isRandomTransition` set, per search,
would measure it.

### 3.4 Experiments: a counterspell that may not be there, and a card not yet drawn

**Position** (`tools/gameplay/hidden_info_leak.py`). Two real 17lands FDN decks from
`assets/sample/`:

- **A** (WG, deck 02871): turn 7, postcombat main, land already played, six untapped lands, Serra
  Angel in hand.
- **B** (UR, deck 02581): three Islands and two Mountains untapped, two cards in hand.

Everything public is identical in two worlds:

- **World R**: B holds Refute + Island, and can counter the Angel.
- **World N**: B holds Island + Island, and can't.

A's legal options are Cast Serra Angel and Pass. Passing ends A's turn with the Angel still in
hand.

**Method.** Each search is MageZero's `ComputerPlayerMCTS2` on the built game, through the mzbridge
`coach` op, with a fresh tree for each seed. Seeds are paired across the worlds. Conditions:

- **clairvoyant**: 8 seeds per world, searching the world as built. This is what self-play does.
- **PIMC**: 16 searches per world, each with B's hand re-drawn from its unseen cards.

**Evaluators:**

- the offline heuristic at 1,000 simulations, which reads hand *size* only, so any difference comes
  from the tree's transitions alone;
- gen 33 at 300 simulations, with the binary prior and `perfectInfo` on, as experiment #1 ran it;
- gen 33 with `perfectInfo` off.

The last one is outside gen 33's training distribution. It is here to show that masking the input
leaves the transitions leaking.

| Evaluator | Condition | World | Best (seeds) | Q(Cast) | Q(Pass) | Visits to Cast |
|---|---|---|---|---|---|---|
| Offline, 1,000 sims | clairvoyant | R | **Pass** 6/8 | −0.055 | −0.047 | 38% |
| | | N | **Cast** 7/8 | +0.256 | +0.020 | 84% |
| | PIMC | R | Cast 12/16 | +0.074 | −0.096 | 76% |
| | | N | Cast 12/16 | +0.105 | −0.097 | 78% |
| gen 33, perfectInfo on, 300 sims | clairvoyant | R | **Pass** 7/8 | −0.045 | +0.015 | 35% |
| | | N | **Cast** 7/8 | +0.278 | +0.128 | 80% |
| | PIMC | R | Cast 13/16 | +0.045 | −0.051 | 71% |
| | | N | Cast 14/16 | +0.080 | −0.033 | 74% |
| gen 33, perfectInfo off, 300 sims | clairvoyant | R | **Pass** 6/8 | −0.046 | −0.013 | 39% |
| | | N | **Cast** 7/8 | +0.273 | +0.086 | 82% |

Q is the searcher's mean backed-up value in [−1, 1]. Paired over seeds, Q(Cast) in N minus R is:

| Evaluator | Clairvoyant | PIMC |
|---|---|---|
| Offline | +0.312 [0.212, 0.411] | +0.030 [−0.011, 0.072] |
| gen 33, perfectInfo on | +0.323 [0.229, 0.418] | +0.035 [−0.006, 0.075] |
| gen 33, perfectInfo off | +0.320 [0.229, 0.411] | not run |

The best action also changes. With the clairvoyant search, A casts in 1–2 of 8 seeds in world R
and 7 of 8 in world N (Fisher p = 0.01–0.04).

**Reading it.**

- **The clairvoyant search plays around Refute only when Refute is there.** Its verdict on a
  position A cannot distinguish swings by about 0.3 Q, and flips, on a card A cannot see.
- **Masking the network's input changes nothing measurable.** The leak is in the tree's
  transitions.
- **PIMC gives the same answer in both worlds, within noise.** Both worlds draw B's hand from the
  same 32 unseen cards. The real hand only changes where cards sit in the library before the seeded
  shuffle, so 2 of the 16 samples differ between worlds. In one of them, world R's sample holds
  Refute where world N's holds an Island, which accounts for the +0.03. What matters is the
  belief: about 12% of B's possible two-card hands hold a counter (Refute or Essence Scatter), and
  A casts anyway.

**Limits.** This is one constructed position with a pure counter-or-not question. It shows the
mechanism, not how often it decides real games. The PIMC condition samples from B's *true*
decklist, which still leaks the decklist's composition. A real limited player knows only the
cards seen so far.

**A second position: the searcher's own next card.** This tests the draw channel of §3.2.

- **A** (same deck): turn 5, precombat main, land played, Plains and two Forests untapped. Its hand
  is Helpful Hunter ({1}{W} 1/1, draws a card when it enters) and Cathar Commando ({1}{W} 3/1).
- **World E**: A's library top is Llanowar Elves, which it could cast with the Forest left over.
- **World L**: A's library top is a Plains. It has already played its land.

A has not seen either card. Offline evaluator, 1,000 simulations, 8 paired seeds, clairvoyant
search (`--position cantrip`):

| | World E (Elves on top) | World L (Plains on top) | Paired E − L |
|---|---|---|---|
| Best | Hunter 8/8 | Hunter 8/8 | — |
| Q(Cast Helpful Hunter) | +0.016 | −0.049 | **+0.065 [0.058, 0.072]** |
| Q(Cast Cathar Commando) | −0.125 | −0.126 | 0.000 [−0.013, 0.014] |
| Visits to Hunter (the policy target) | 81% | 68% | **+13 points [6, 19]** |

**Every seed values the cantrip higher when the card it will draw is castable.** The line that
draws nothing doesn't move. The best action didn't flip here, but the training targets moved with
a card A can't see.

**A wider measure already exists.** docs/008 §8.3 coached 119 real 17lands decisions twice: once
with the opponent's true hand and deck, as clairvoyant self-play would search them, and once with
belief samples. The best action changed in 19% of decisions, against 11% between two independent
searches. The CIs overlap, and those were coaching positions, not self-play. Still, it suggests
the leak is not confined to contrived positions.

### 3.5 What reaches the training data

- **Policy targets** are the root visit counts of the clairvoyant search. In world R above the
  target says "pass", in world N "cast", from the same public state.
- **Value targets** are `y_i = λ·y_{i+1} + (1 − λ)·h_i`, with the game result z at the end
  (`ParallelDataGenerator.java:437–447`). `h_i` is the clairvoyant search's root mean Q. A row k
  decisions before the end gives z a weight of λ^k:

  | λ | z's weight 10 decisions out | 20 decisions out | Average over a game (~106 rows per player) |
  |---|---|---|---|
  | 0.70 (experiment #1, gens 3+) | 0.028 | 0.0008 | **0.022** |
  | 0.95 (Will's recommendation) | 0.60 | 0.36 | 0.18 |

  At λ = 0.70 the value head learns, almost entirely, what the clairvoyant search thinks.
- **The game results themselves** come from games between two clairvoyant players. Nobody walked
  into a counterspell or a combat trick they could see, and no bluff ever worked. Even a pure
  outcome label measures open-hand Magic.
- **The inputs** contain the opponent's hand (§2). The network can, and at λ = 0.70 has every
  reason to, use it.

### 3.6 Consequences

1. **Experiment #1's agent plays a different game.** Its policy and value are functions of
   information a real player never has. Against a human, or on 17lands data where the opponent's
   hand is unknown in 86% of rows, its input would be out of distribution (docs/008 §6).
2. **Its card statistics describe open-hand Magic.** Which cards lose value there, relative to
   human games?
   - **Counterspells.** The caster sees them and never walks in. Their value from mana held up
     that an opponent walks into, and from bluffs, is zero.
   - **Combat tricks and flash creatures.** The opponent never blocks into what it can see.
   - **Anything whose value is surprise, or making the opponent guess.**

   Experiment #1's largest gaps against 17lands sit exactly there:
   - blue commons, −8.0 points (docs/003 §6.6);
   - Refute, 45.3% against 57.6%;
   - Fleeting Distraction, 45.7% against 54.8%;
   - the most negative IWD: Fleeting Flight −16.7 and Giant Growth −14.9.

   docs/003 attributes these gaps to targeting and timing errors, which it measured, and both can
   be true. The prediction that separates them is in §5.5.
3. **The evaluation is leak-on-leak.** The raw-search baseline is clairvoyant too, so "beats raw
   search 55.8%" compares two open-hand players.
4. **Coaching needs a hidden-information network** (docs/008 §11). The gen 33 coach runs with
   `perfectInfo` on, over belief samples, because that is how gen 33 was trained.

### 3.7 Where this repo already searches without the leak

The coach (docs/008 §8) doesn't search the real hidden state:

- It samples the opponent's deck and hand from a belief model built on 17lands decks, or re-draws
  them from the unseen cards.
- The bridge shuffles each library below the known top cards with a per-sample seed
  (`StateInjector.injectLibraryAndHidden`, `resampleHand`).
- It aggregates Q over K searches.

This is PIMC. Within each sample the search is still clairvoyant about *that sample*, and both
seats see each other's cards in the tree. That flaw is in §5.2. `StateInjector` is the recipe
self-play would need.

### 3.8 Upstream status

- **v0.2.0** changes none of this. The v0.2.0 release notes don't mention hidden information.
  Diffing the four search classes against `5a32441c` shows targeting and mode fixes. Tree reuse now
  also requires matching `stateString`. The `hiddenInfo` / `hidden_info` mismatch,
  `perfectInfo = true`, "dont shuffle here" and the unread `isRandomTransition` are unchanged.
- **[WillWroble/MageZero#4](https://github.com/WillWroble/MageZero/issues/4)** is an open research
  issue by an outside contributor (opened 2026-09-20). It asks the same questions under "Gate A:
  Information / SearchWorld semantics" and makes three findings:
  - the same observation does not mean the same search world;
  - `StateEncoder` equality authorized unsafe reuse across worlds;
  - `GameState.copy` is not a safe knowledge snapshot.

  It reports two fixes as implemented: world-local reuse quarantine and search-control RNG
  isolation. Neither is in v0.2.0's `master`. Still open: world-local engine RNG, an observer
  knowledge snapshot, a perspective-safe determinization, and an end-to-end hidden-information
  regression. Its planned direction matches §5.4: one observer-consistent world, then one
  world-local tree, discarded after the decision.
- ROADMAP D7 (does experiment #2's network see the opponent's hand?) and the "determinization in
  search" question for Will are still open.

## 4. How other hidden-information games are handled

This section summarizes published work, checked against the primary papers. "Preprint" marks
work that wasn't peer-reviewed. Numbers are the authors' own.

### 4.1 Determinization (PIMC): sample the hidden state, then search as if it were known

- **Bridge: GIB** (Ginsberg 2001).
  - Deals about 50 layouts of the unseen hands and keeps only those its bidding model could have
    produced the actual auction from. It weights them by the card locations implied by the play so
    far, solves each with perfect information, and plays the move with the best total.
  - Solved 100 of 180 Bridge Master problems (Bridge Baron: 33).
- **Two failure modes** (Frank & Basin 1998). No number of samples removes either:
  - **Strategy fusion.** The search picks a different continuation in each sampled world, although
    the player can't tell those worlds apart.
  - **Non-locality.** A node's value depends on other parts of the tree, because opponents steer
    play with their private information.

  GIB shows both. It never makes information-gathering plays, and it prefers lines that postpone a
  guess, since every sampled world assumes the guess will later be made correctly.
- **When PIMC works** (Long, Sturtevant, Buro & Furtak 2010). On synthetic trees, low *leaf
  correlation* predicts failure. How quickly play *disambiguates* the hidden state matters too:
  PIMC is near-optimal when disambiguation is fast. Skat and Hearts measured favourable values.
  In Kuhn poker, where nothing is disambiguated, PIMC is exploitable.
- **Inference-weighted sampling** (Skat).
  - Kermit (Buro et al. 2009) reweights sampled deals by how likely each makes the observed
    bidding. That was worth 996 against 779 points per 36-game series, at expert level.
  - Rebstock et al. (2019) weight each sampled layout by the product of the opponents' action
    probabilities under policy networks trained on human games. That beat the previous inference in
    two of three game types.
  - In Rebstock et al., a "cheating" variant that put all the weight on the true deal did *worse*
    in two game types. Judge beliefs by play, not by how accurate they look.
- **AlphaZero with determinization: AlphaZe\*\*** (Blüml, Czech & Kersting 2023; Barrage Stratego,
  DarkHex).
  - Samples 3 worlds per move (12 in DarkHex), runs AlphaZero MCTS in each, and averages the
    *policies*.
  - The network sees the player's information-set encoding: probability planes for unknown enemy
    pieces.
  - "TrueSight learning" trains early games with search on the *true* state, then switches to
    sampling. It was worse during that phase and slightly better at the end.
  - 68–74% against heuristic Stratego bots, but only 4–16% against P2SRO, a stronger learned agent.
  - MAPLE (Li et al., CoG 2026) replaces the separate trees with one information-set tree whose
    leaves are each evaluated on 5 sampled worlds: +291 Elo over AlphaZe\*\* in Phantom Go.
- **Stratego with a learned belief: Ataraxos** (Sokota et al. 2025, preprint).
  - Self-play RL, plus a transformer belief network that samples the hidden pieces (about 1,000
    samples) for test-time search.
  - Beat a top human 15–1–4, for a reported training cost of a few thousand dollars.
  - It is the closest published analogue to what MageZero would become with stage 1 and a learned
    belief (§5.4).

### 4.2 Information-set search: one tree over what the player can actually distinguish

**IS-MCTS** (Cowling, Powley & Whitehouse 2012; the papers write ISMCTS) samples a fresh world at every iteration but keeps
one tree whose nodes are information sets. There are three variants:

- **SO-IS-MCTS** (one tree, from the searcher's view) fixes the searcher's own strategy fusion and
  spends the whole budget on one deeper tree. It still treats the opponent's moves as fully
  visible.
- **SO-IS-MCTS + POM** merges opponent moves the searcher can't tell apart. That treats the opponent
  as choosing at random among them.
- **MO-IS-MCTS** keeps one tree per player, so the simulated opponent decides from its own
  statistics.

Non-locality stays unaddressed. Because the searcher never samples its *own* hidden cards, the
simulated opponent effectively sees them, and it cannot bluff. Letting the search determinize its
own hidden information as the opponent would see it is the known fix, and produced emergent
bluffing and inference (Cowling, Whitehouse & Powley 2015).

**The value of cheating, measured.** Dou Di Zhu, the same 5,000 deals for each player:

| Player | Win rate |
|---|---|
| Cheating UCT (sees all hands) | 56.5% |
| Determinized UCT | 43.6% |
| IS-MCTS | 42.3% |

Cheating mattered in 1,421 of the deals, and those were exactly where IS-MCTS beat
determinization.

§6 evaluates IS-MCTS for MageZero in detail.

### 4.3 Belief-state and game-theoretic search

These methods search over probability distributions of hidden states, not over sampled worlds.
They compute equilibrium, mixed strategies, which is where bluffing lives.

- **Poker**:
  - **DeepStack** (Moravčík et al. 2017) re-solves a depth-limited lookahead at every decision
    from its own range and the opponent's counterfactual values. Its value networks take both
    ranges. It beat professionals.
  - **Libratus** (Brown & Sandholm 2018) precomputes a blueprint strategy and refines it with
    *safe* nested subgame solving: a refined strategy must never give any opponent hand more than
    the blueprint did. Assuming the opponent simply follows the blueprint is exploitable.
  - **Pluribus** (2019) makes six-player play balanced by letting each player choose among four
    continuation strategies at the depth limit. It computes a strategy for every hand it could
    hold, then plays its real one.
- **ReBeL** (Brown, Bakhtin, Lerer & Gong 2020) is AlphaZero-style self-play whose search state is
  a *public belief state*: public information plus each player's distribution over private
  states. It reduces to an AlphaZero-like algorithm when nothing is hidden. The authors name its
  limit: the network's input grows with the number of private states per public state.
- **Student of Games** (Schmid et al. 2023) uses growing-tree CFR and sound self-play in one
  algorithm across chess, Go, poker and Scotland Yard. It too must enumerate private states. The
  authors suggest sampling world states from a generative model instead.
- **Fog-of-war chess: Obscuro** (Zhang & Sandholm, ICLR 2026).
  - Keeps every consistent position (up to about 1M) and samples a few hundred into each subgame.
  - Prunes positions whose falsity is high-order knowledge ("knowledge-limited" solving), so it
    never computes full common knowledge.
  - Solves with CFR, with Stockfish at the leaves.
  - 85% against the previous best AI, 97% against humans rated 1450–2006, and 16–4 against the
    top-rated human.
- **Hanabi** (cooperative):
  - **SPARTA** (Lerer et al. 2020) computes an exact belief from an agreed blueprint policy, then
    searches.
  - **Learned Belief Search** (Hu et al. 2021) samples the hidden hand from an LSTM belief model
    trained on the blueprint's games. It recovers most of exact search's gain at a fraction of the
    cost, but its beliefs are only valid while the others follow the blueprint.

### 4.4 Model-free RL on partial observations, with no search at play time

- **StarCraft II: AlphaStar** (Vinyals et al. 2019).
  - **What the policy sees.** Only the units in view, with an LSTM over the whole history. It moves
    a camera as a human does.
  - **Human data first.** Supervised learning on 971k replays, then RL with a KL penalty toward
    that policy. Without human data, test Elo was 149 against 1,540.
  - **League training** (§4.6).
  - **No search.** Grandmaster in all three races, above 99.8% of ranked players.
- **Dota 2: OpenAI Five** (Berner et al. 2019).
  - About 16,000 human-visible features. Enemies in fog keep their last-seen features. A
    4,096-unit LSTM.
  - The value is a projection of the same LSTM state: no privileged critic. The paper names
    AlphaStar-style full-information value networks as future work.
  - No search. Beat the world champions.
- **Stratego: DeepNash** (Perolat et al. 2022).
  - **What it sees.** Only its information state: its own pieces, the public knowledge of what
    each piece could be, and recent moves. Nothing privileged, not even for the value head.
  - **Why no search.** About 10⁵³⁵ nodes, and too many private configurations per public state for
    equilibrium search.
  - **Training.** Regularized Nash Dynamics (R-NaD) instead: repeatedly penalize the policy toward
    a regularization policy, run the learning dynamics to a fixed point, and make that the new
    regularizer. At scale it reaches an approximate equilibrium rather than cycling.
  - **Play-time tweaks.** Rare actions are thresholded away, with a few play-time filters.
  - **Emergent behaviour.** Bluffing: a Scout chasing as if it were the 10, an unrevealed 10 moving
    like a low piece. Trading material for information.
  - **Results.** 84% of ranked games on Gravon (third-ranked), 97–100% against existing bots.
- **Dou Dizhu: DouZero** (Zha et al. 2021).
  - Deep Monte Carlo with imperfect-information inputs only, and no search. Ranked first of 344
    bots.
  - Its Q-network is also what acts, so it cannot take perfect-information features without
    learning to play "in a cheating style" (PerfectDou's observation). MageZero's value network is
    in the same position, because the search uses it at play time.

### 4.5 Privileged information during training only

These systems use hidden information to *train*, never to act:

- **AlphaStar's value functions** take the opponent's observations as input, only in training
  (55M of 139M weights are needed at inference).
  - In a simplified ablation that raised the average win rate from 22% to 82%.
  - The value also sees the policy's history (its LSTM core), which makes it closer to V(h, s)
    than to V(s).
  - Honor of Kings (Ye et al. 2020) did the same.
- **PerfectDou** (Yang et al. 2022), "perfect-training, imperfect-execution":
  - PPO whose critic sees both opponents' hands, plus a dense reward computed by an oracle.
  - The actor sees only imperfect information. It beat DouZero (win rate 0.54).
  - In ablation the privileged critic helped modestly. The oracle reward mattered more.
- **Suphx** (Mahjong, Li et al. 2020, preprint), oracle guiding:
  1. Train an agent by RL with oracle features (opponents' hands, the wall).
  2. Continue RL while multiplying those features by a Bernoulli mask whose keep probability decays
     from 1 to 0. The same network becomes a normal agent.
  3. Fine-tune at a lower learning rate.

  Plain distillation from the oracle "does not work well". The gain was modest: stable rank 8.24
  to 8.30. It reached 10 dan on Tenhou, above 99.99% of ranked players. Its run-time adaptation
  (sample hidden hands consistent with the agent's own tiles, then fine-tune) won 66% against the
  unadapted policy, but was too slow to use online.
- **Asymmetric actor-critic** (Pinto et al. 2018, robotics): the critic sees the full simulator
  state, the actor images.
- **The bias caveat** (Baisero & Amato 2022). For an agent that acts on its history, a critic
  V(s) on the full state is generally ill-defined or biased relative to the history value V(h).
  The unbiased fix is a history-state critic V(h, s), whose expectation over s given h is V(h).
- **Imitating a privileged expert fails** (Warrington et al. 2021).
  - The expert doesn't know what the student can't see. So the student learns neither to gather
    information nor to hedge.
  - "Learning by Cheating" (Chen et al. 2019) works for driving because the privileged
    information is largely visible to the camera anyway. In card games it isn't.
  - **MageZero's policy targets are exactly this:** visit counts from a search that sees the hidden
    cards (§3.5).

### 4.6 Opponent populations and exploitability

Self-play can chase strategy cycles and learn policies that only beat themselves.

- **AlphaStar's league**:
  - main agents, trained by prioritized fictitious self-play against all past players;
  - main exploiters, which attack the current main agents;
  - league exploiters, which look for weaknesses of the whole league.

  Adding the exploiters raised the minimum win rate against past versions from 46% (pure
  self-play) to 71%, at about the same Elo.
- **Card games show the same.** ByteRL, the champion *Legends of Code and Magic* agent, lost 80–90%
  of games to a behaviour-cloned PPO best response (Haluska & Schmid 2024).
- **Equilibrium methods** (DeepNash, CFR) avoid the problem by construction.

### 4.7 Magic and other collectible card games

- **Magic, simplified.** Two IEEE papers used a Magic reduced to lands and vanilla creatures, with
  identical known decks:
  - **Ward & Cowling (2009)** ran flat Monte Carlo over which cards to play, reshuffling the unseen
    cards in every rollout. It won about 80% against a rules player.
  - **Cowling, Ward & Powley (2012)** used *ensemble determinization*: independent UCT trees, each
    with its own fixed deck order, with root visits summed. The best split was 20–100 trees within
    10k simulations. Plain chance-node MCTS won only 23% against the expert-rule player; the
    ensemble reached about 50%. The authors argue Magic's high leaf correlation suits PIMC. They
    left inference unaddressed and made no cheating comparison.
- **Other Magic work.** No peer-reviewed Magic gameplay agent found handles hidden information
  with a belief model or a privileged critic. A 2026 MTG RL benchmark (da Costa Cunha et al.,
  preprint) uses a partially observed environment with PPO. Its reward shaping deliberately uses
  only observable quantities, so hidden information doesn't leak into training.
- **Hearthstone**:
  - determinized UCT with sampled worlds (Zhang & Buro 2017);
  - per-iteration sampling with a value network trained on games from a *cheating* MCTS
    (Świechowski et al. 2018);
  - predicting the opponent's deck and upcoming cards from replay statistics, over 95% accurate on
    turns 3–5 (Bursztein 2016);
  - an MCTS ensemble over sampled opponent hands (Dockhorn et al. 2018).

  A model-free agent that sees only what a human sees, with a variant allowed to peek at part of
  the opponent's *deck* (never the hand), gained 5.5 points from the peek (Xiao et al. 2023).
- **Legends of Code and Magic,** a drafted card game like limited: determinized search with
  imitation-trained networks and a prior over the opponent's deck built from its draft picks won
  51% against ByteRL, against 27% without search (Rubin 2026, preprint).

### 4.8 The systems side by side

| System | Game | What the acting policy sees | Privileged info in training | Play-time search, and its hidden state |
|---|---|---|---|---|
| GIB, Kermit | bridge, Skat | own cards + inference | — | PIMC over inference-weighted deals |
| Cowling et al. | simplified Magic | own information | — | ensemble determinization |
| IS-MCTS | card and board games | own information | — | a world per iteration, information-set trees |
| DeepStack, Libratus, Pluribus | poker | own range | — | belief-state re-solving |
| ReBeL, Student of Games | poker, Scotland Yard, chess, Go | public belief state | — | CFR over public belief states |
| Obscuro | fog-of-war chess | own information | — | knowledge-limited subgames over sampled positions |
| AlphaZe\*\*, MAPLE | Stratego, DarkHex, Phantom Go | information-set encoding | TrueSight (early clairvoyant games) | AlphaZero MCTS over sampled worlds |
| Ataraxos (preprint) | Stratego | information state | not stated | search over worlds from a learned belief network |
| AlphaStar | StarCraft II | visible units + LSTM memory | value sees the opponent's observations | none |
| OpenAI Five | Dota 2 | human-visible features + LSTM | none | none |
| DeepNash | Stratego | information state | none | none (R-NaD equilibrium) |
| DouZero, PerfectDou | Dou Dizhu | own hand, public cards | PerfectDou: critic sees all hands | none |
| Suphx | Mahjong | own tiles, public tiles | oracle features, annealed away | none online |
| **MageZero, experiment #1** | **Magic, limited** | **both hands** | **everything, and it's used at play time** | **the real game** |

## 5. MageZero against these approaches

### 5.1 Where MageZero sits

| | What the policy sees | World the search explores | Simulated opponent | Training targets |
|---|---|---|---|---|
| **MageZero, experiment #1** | both hands | the real one, draws included | sees everything | clairvoyant search, plus clairvoyant games |
| Determinization / PIMC (§4.1) | own information, or the sampled world | K sampled worlds | sees the sampled world | averaged over samples |
| Multiple-observer IS-MCTS (§4.2) | own information | a new sample every iteration | acts on its own information set | — |
| Belief-state search (§4.3) | public belief state | subgames over beliefs | equilibrium | equilibrium values |
| Model-free RL (§4.4) | own observation plus memory | none at test time | — | outcomes of real games |
| Privileged training (§4.5) | own observation | — | — | oracle inputs to the critic or a teacher, dropped for play |

MageZero is below every row. Every system in §4 keeps the acting policy a function of what the
agent could know. MageZero doesn't do that at the input, in the search, or in the targets.

### 5.2 What "the strength of a counterspell in human games" requires

A counterspell's value in a real game has four parts, against one cost:

- **Answer.** It counters a spell cast into it.
- **Tax.** An opponent who suspects it plays around it: casts the lesser spell first, waits,
  overextends less. That happens whether or not it's really there.
- **Ambush.** An opponent who doesn't suspect it walks in.
- **Bluff.** Mana held up *without* it still taxes an opponent who can't tell.
- **Cost.** Mana held up and not used.

| Regime | Answer | Tax | Ambush | Bluff | Cost of holding |
|---|---|---|---|---|---|
| Human games (what 17lands measures) | yes | yes, imperfectly | yes | yes | yes |
| Open-hand, experiment #1 | only when the caster is forced | paid only when it's really there | none | none | none: the holder knows whether anything is coming |
| PIMC for both players | yes | in proportion to its probability | happens in the real games, but the holder's search assumes it won't | none | yes |
| Information-set opponent | yes | yes | yes, and the holder's search sees it | only with mixed play | yes |
| Equilibrium play | yes | yes | yes | yes, at the right frequency | yes |

That gives three requirements:

1. **The searcher's decisions depend only on its information:** its inputs, and the world it
   searches. Stages 0–1 below.
2. **The simulated opponent decides from *its* information.** PIMC fails this. In each sampled
   world the searcher's own hand is real, and the simulated opponent sees it. So a holder's search
   concludes that holding up Refute only ever taxes: nobody in its tree walks in, and no bluff can
   work. Combat tricks lose their surprise the same way. Stage 2.
3. **The opponents resemble the population you compare against.** A perfect hidden-information
   agent trained by self-play learns a counterspell's value against itself. 17lands measures it
   against human players, who sometimes walk into counters and sometimes play around counters that
   aren't there. Matching 17lands needs human-like opponents in the pool. Stage 4.

### 5.3 Which families fit MageZero

1. **Determinization is the most promising family, and it fits the engine as it is.**
   - A MageZero search already is one world and one tree, so only the source of the world changes.
     The coach already does this (§3.7).
   - It is the standard first step in card games: bridge, Skat, simplified Magic, Hearthstone
     (§4.1, §4.7).
   - It is also how the strongest recent AlphaZero-style results search: AlphaZe\*\* and MAPLE,
     Ataraxos in Stratego (self-play plus a learned belief sampler, reported at a few thousand
     dollars), and determinized search in a drafted card game (Rubin 2026).
   - Its known failures are strategy fusion and non-locality (§4.1), plus the opponent-side problem
     in §5.2.
2. **An information-set opponent is what counterspells, tricks and bluffs need.** There are three
   ways to get one:
   - **A policy opponent (cheap).** At opponent nodes, choose by an imperfect-information policy:
     the network from the opponent's perspective, with the searcher's hand hidden. Don't minimize
     the searcher's Q in a world where the opponent can see its cards. This fits MageZero's tree.
     The simulated opponent is then only as good as the policy head, and a weak one makes the search
     optimistic. SPARTA and Learned Belief Search make the same trade in Hanabi (§4.3).
   - **Self-determinization.** At opponent nodes, also re-sample the searcher's own hidden cards
     from the opponent's point of view. In IS-MCTS this produced bluffing and inference (Cowling,
     Whitehouse & Powley 2015, §4.2).
   - **Multiple-observer IS-MCTS (principled, expensive).** A new world each iteration, with
     statistics per information set (§4.2). MageZero's node design caches one world's `GameState`
     per node. §6 evaluates IS-MCTS in detail and proposes a particle variant that keeps one state
     per world.
3. **Belief inference** decides how good the world samples are:
   - Bridge and Skat programs weight each sampled deal by how likely it makes the opponent's actual
     bidding or play. Kermit's bid inference was worth about 28% more points (§4.1).
   - For Magic, "passed with 1UU open on its own turn" should raise the counterspell's probability.
     Without inference, tells go unread, and bluffs are pointless even for a perfect searcher.
   - Recent systems learn the belief. Ataraxos uses a belief network. A drafted card game used a
     deck prior built from the opponent's draft picks, the limited analogue. Hearthstone bots use
     deck prediction from replay statistics.
   - A cheap start: a **belief head** that predicts the opponent's hand and deck from the public
     state. It is a useful auxiliary target, and it later becomes the sampler.
   - Judge beliefs by play. Rebstock's inference that "cheated" toward the true deal played worse.
   - For the opponent's decklist, start with the true deck's unseen cards: a small leak of its
     composition. Move to belief.py's 17lands deck model later.
4. **Privileged information during training** helps, but it is not a fix (§4.5):
   - AlphaStar's and PerfectDou's critics saw hidden information; Suphx annealed oracle inputs
     away.
   - A network that both acts and sees hidden cards learns to rely on them. That is why DouZero
     couldn't use perfect-information features. MageZero's value network acts, inside the search at
     play time, so the network the search calls must take imperfect-information inputs.
   - A privileged head can be a training-only auxiliary. Make it history-conditioned V(h, s), not
     V(s) (Baisero & Amato).
   - A clairvoyant-first curriculum (AlphaZe\*\*'s TrueSight, Suphx's oracle annealing) is an
     option, with modest gains in both papers. Optional, later.
5. **Equilibrium methods** are the principled answer to bluffing and mixed play (§4.3, §4.4):
   - **R-NaD / DeepNash** reached equilibrium play with emergent bluffing in Stratego without
     search. It would mean giving up MageZero's engine.
   - **ReBeL, DeepStack and Student of Games** search over public belief states. That needs a
     belief over every possible hidden hand. A poker hand has 1,326 possibilities. A limited hand
     has millions even with a known decklist (seven cards from 32 unseen: C(32, 7) ≈ 3.4M), and
     the decklist isn't known.
   - **Obscuro** handles millions of fog-of-war positions by sampling a few hundred per subgame and
     reasoning only about limited knowledge. That is the most scalable route into this family.

   Research-scale for Magic for now.
6. **Opponent populations** (§4.6) guard against a policy that only beats itself:
   - AlphaStar's exploiters, and ByteRL losing 80–90% to a best response, show why.
   - For 17lands-comparable statistics the pool should also hold human-like opponents (docs/008
     §7).
   - AlphaStar's supervised start and its KL penalty toward human play are the same idea on the
     policy side.

### 5.4 A plan for experiment #2

The stages are ordered by value for cost. Each has its own test. Coordinate stages 0–1 with Will
and with MageZero#4, which is working on the same contracts.

**Stage 0: the observation (hours).**

- Make `see_opponent_hand: false` take effect: write the `hiddenInfo:` key, or have `Config.java`
  read both keys. Log the effective `perfectInfo` at JVM start.
- Encode every node from the *searcher's* information set: its own hand shown, the opponent's hand
  as a count, at every node. That stops the network reading the opponent's real hand at opponent
  nodes (§2.5). Replace `==` with `equals`.
- Track legitimately known hidden cards with a watcher, and encode them as known: cards that moved
  from a public zone to a hand, revealed cards, looked-at library cards.
- Add equivalence tests. Two specs that differ only in hidden cards must give identical root
  features (bridge `encode`). After stage 1, they must also give the same search statistics within
  noise (`hidden_info_leak.py`, as in the PIMC rows of §3.4).

Effect: the network stops reading hidden cards. The search still leaks.

**Stage 1: determinized self-play search (days).**

- At each real decision, build K worlds from the searcher's information:
  - re-deal the opponent's unknown hand cards from its unseen cards, with known cards pinned;
  - shuffle both libraries below any known top cards;
  - draw from a search-owned, seeded RNG (MageZero#4's RNG domains).

  The recipe is `StateInjector.resampleHand` / `injectLibraryAndHidden`. `shuffleUnknowns` is a
  starting point, but it uses the engine's RNG and ignores known cards.
- Use one fresh tree per world, with no reuse across decisions. That also removes docs/008's
  "the budget counts reused visits".
- The policy target is the visits summed over worlds. The row's `stateScore` is the visit-weighted
  mean of the root Q.
- **K.** AlphaZe\*\* used 3 worlds per move in Stratego. Cowling et al. found 20–100 worlds best
  at 10k simulations in simplified Magic. At 96–300 simulations a decision, start with K = 3–4
  worlds sharing the budget, and measure in the pilot. K = 1 is valid and cheapest, but a single
  world makes decisions probability-match: A would hold the Angel in the ~12% of searches that
  sample a counter, rather than make the expected-value choice.
- **A belief head** (§5.3) can be trained alongside from the start. It doesn't need to drive
  sampling until it beats uniform sampling in play.

Effect: removes the first four channels of §3.2. The caster plays around a counterspell as often
as one is likely. Cost: about the same number of simulations if K splits the budget, plus one game
copy per world.

**Stage 2: an information-set opponent (weeks; research).**

- Start with the policy opponent from §5.3. At opponent nodes it needs a second encoding from the
  opponent's side: its own sampled hand shown, the searcher's hand as a count. Stage 0's
  searcher-side encoding would hand the opponent's policy the searcher's cards. Gate it on a
  trained opponent-perspective policy head, since an untrained one makes a weak simulated
  opponent.
- Test from the holder's side: B, on its own turn, with Refute and a castable creature. Does B
  hold up mana more often when A holds a bomb it can't see? Run the same kind of test for a combat
  trick.
- Move to multiple-observer IS-MCTS only if the policy opponent proves too weak. Stage 2 can build
  on particle SO-IS-MCTS (§6), which is worth an A/B against stage 1 on its own.

**Stage 3: value targets (config).** Set λ ≈ 0.95, which is also Will's recommendation (ROADMAP
D1). The game result then carries about 18% of a target's average weight instead of 2%. Under
PIMC the real games contain ambushes that the search doesn't model, and the outcome label is where
the value head can learn them.

**Stage 4: opponents.** Add human-like policies (docs/008's imitation heads) to the opponent pool
when the goal is 17lands-comparable card statistics. Add exploiters later, as AlphaStar's league
did (§4.6).

### 5.5 How to tell it worked

- **Equivalence.** After stage 1, the counterspell test gives the same answer in both worlds
  within noise, as the PIMC rows of §3.4 already do.
- **Holder's side.** After stage 2, the holder test shows ambush value: B holds up Refute more
  often when a bomb is plausible.
- **Card statistics, the prediction that separates the explanations in §3.6.** If clairvoyance
  explains part of experiment #1's gaps, a determinized arm should narrow the gaps for:
  - counterspells;
  - instant-speed interaction;
  - combat tricks.

  It should narrow them more than for creatures and sorcery-speed removal, and stage 2 should
  narrow them further. If they don't move, the targeting explanation gains weight.
- **Leak-free evaluation.** Both sides of every evaluation game use the same hidden-information
  search. A leak-trained network evaluated without the leak is out of distribution.

## 6. Evaluating IS-MCTS for MageZero

IS-MCTS (information set MCTS, §4.2) was recommended as the fix. **It is the right direction, but
not the first step:**

- It fixes what determinization can't fix on the searcher's side: strategy fusion, and a small
  budget split across K shallow trees.
- Its plain form (SO-IS-MCTS) still lets the simulated opponent see the searcher's hand. So on its
  own it doesn't recover a counterspell's ambush and bluff value (§5.2). That needs the
  multiple-observer form with self-determinization.
- In MageZero it needs everything stage 1 needs (observation, knowledge tracking, world sampler,
  RNG ownership), plus world-independent action keys and a state cache per world.

Recommendation: build stage 1 as ensemble determinization. Then build *particle* SO-IS-MCTS
behind a flag and A/B it at equal budget.

### 6.1 How it works

- **One tree, whose nodes are the searcher's information sets:** what it can distinguish, not
  concrete states.
- **Each iteration samples a world** consistent with the searcher's information: a determinization.
  It walks down the tree choosing only among actions legal in that world, expands one node,
  evaluates it, and backs the result up.
- **Statistics are shared across worlds.** "Cast Refute" at an opponent node is legal only in the
  worlds where the opponent holds Refute and has the mana. Selection must account for that: a child
  that is rarely *available* shouldn't be penalized as if it were rarely *chosen*. IS-MCTS uses the
  number of times a child was available in place of the parent's visit count (§6.3).

Variants (Cowling, Powley & Whitehouse 2012):

| Variant | What it models | Cost |
|---|---|---|
| **SO-IS-MCTS** (single observer) | one tree from the searcher's view. The searcher's later decisions are one choice per information set, not one per world, so no strategy fusion on its side. Opponent moves are treated as seen | about one tree's worth |
| **SO-IS-MCTS + POM** | also merges opponent moves the searcher can't see | as above |
| **MO-IS-MCTS** (multiple observer) | one tree per player; each player chooses from its own tree, keyed by what it can see | one tree per player, updated together |

What each fixes, next to what MageZero does now and stage 1:

| Problem | MageZero now | Ensemble determinization (stage 1) | SO-IS-MCTS | MO-IS-MCTS + self-determinization |
|---|---|---|---|---|
| Search sees the real hidden cards | yes | fixed | fixed | fixed |
| Searcher plans later moves knowing the sampled world (strategy fusion) | n/a (it knows the real world) | **no** | fixed | fixed |
| Budget split into K shallow trees | — | yes | one deeper tree | one tree per player |
| Draws | the real next card | fixed per tree | re-sampled every iteration | re-sampled every iteration |
| Simulated opponent sees the searcher's hand: no ambush, no bluff | yes | **yes** | **yes** | fixed, to the extent self-determinization samples the searcher's hand as the opponent would |
| Inference from the opponent's play (non-locality) | — | no | no | no: that's the belief's job (§5.3) |
| Equilibrium mixing (bluff *frequencies*) | no | no | no | no guarantee |

Almost every Magic move is public: casting a card reveals it. So SO-IS-MCTS's "opponent moves are
seen" costs little, and POM only matters for hidden choices: scry and surveil order, mulligan
bottoms, face-down cards.

### 6.2 What the evidence says

- **Mixed.** IS-MCTS beat determinization in Lord of the Rings: The Confrontation. In Dou Di Zhu it
  tied overall (42.3% against 43.6%): it won the deals where hidden information mattered and lost
  the rest to opponent-node branching. In Phantom (4,4,4) it fell behind determinization at longer
  thinking times (§6.3).
- **With a network.** MAPLE shares one information-set tree across 5 fixed, well-chosen worlds. It
  beat AlphaZe\*\* (AlphaZero on sampled worlds) by +291 Elo in Phantom Go and +136 in Dark Hex. It
  lost to PIMC when the worlds were chosen at random (§6.3).
- **Bluffing** appeared only with many trees, inference and self-determinization, and only in a
  game with 10 hidden configurations (Cowling, Whitehouse & Powley 2015, §6.3).
- **No equilibrium guarantee.** IS-MCTS, like PIMC, can be exploited (§6.3). Poker-grade
  mixing needs regret-based methods (§4.3).

### 6.3 Details from the literature

**The selection rule** (Cowling, Powley & Whitehouse 2012; pseudocode in Whitehouse's thesis).
Children illegal in the current world are skipped. Among the legal ones, pick the argmax of
r(c)/n(c) + k·√(ln n′(c) / n(c)), with k = 0.7. Here n′(c), the *availability count*, is how often
the parent was visited while c was legal; it replaces the parent's visit count. The authors'
acknowledged cost: each action gets one value averaged over different legal-action subsets.

**Where plain IS-MCTS failed: opponent branching.**

- In Dou Di Zhu, about 1,500 distinct opponent leads appeared across 1,000 sampled worlds, against
  about 88 in any one world. SO-IS-MCTS kept expanding new opponent moves near the root, which is
  why it lost slightly on the deals where hidden information didn't matter.
- **This is the main risk for Magic.** The opponent's legal plays come from its hidden hand, so an
  opponent node shared across worlds collects the union of every possible hand's plays.
- Two remedies are published:
  - key opponent nodes by the opponent's sampled information state (many-tree IS-MCTS; OpenSpiel's
    implementation);
  - treat opponent moves as a chance node sampled from a policy network. DeltaDou does this in Dou
    Dizhu, with worlds drawn by Bayesian inference over the policy. It is the "policy opponent" of
    §5.3.

**Other results.**

- **Lord of the Rings: The Confrontation** (hidden piece identities): all three variants beat
  determinization as the Dark player, despite running 2–4× slower per iteration.
- **Phantom (4,4,4)**: with more than about 1.5 s per move, SO-IS-MCTS and MO-IS-MCTS fell *behind*
  determinization. Assuming an informed opponent, they decided the game was lost and played
  randomly.
- **Whitehouse's thesis** names the mirror of strategy fusion, *strategy fission*: averaging over
  the opponent's possible hands leaves the modelled opponent unable to tailor its play to its own
  hand. In a solvable Dou Di Zhu variant, most of cheating's advantage came from fusion, and
  IS-MCTS beat PIMC 67.2% to 62.7%. IS-MCTS also shipped in a commercial Spades game.

**Bluffing needs more than MO-IS-MCTS** (Cowling, Whitehouse & Powley 2015, The Resistance only).

- **Many-tree IS-MCTS.** One tree per player per root information set.
- **Inference.** Uses the tree itself as the opponent model.
- **Self-determinization.** Some iterations run on worlds the searcher knows are false but the
  opponents might believe. Those update only the opponents' trees and the searcher's hypothetical
  trees, never its real decision tree.
- **Results.** Spies bluffed, recovering about half of what inference took from them. Putting 3/8
  to 7/8 of the budget into false worlds worked best.
- **Scope.** The authors credit the tiny information sets (10 configurations). A Magic hand has
  millions.

**No convergence, and more search can hurt.**

- IS-MCTS settles on stable, non-equilibrium strategies (Ponsen, de Jong & Lanctot 2011).
- In Goofspiel its exploitability *grew* with thinking time. An opponent who knows the algorithm can
  beat it by an almost arbitrary margin (Lisý, Lanctot & Bowling 2015).
- Smooth UCT, which samples the average policy with a decaying probability, approaches equilibrium
  in small poker (Heinrich & Silver 2015). It is the cheapest known repair at the root.

**With networks, fixed particles win.**

- **MAPLE.** One information-set tree shared by k = 5 fixed worlds, chosen from 50 candidates by a
  learned scorer. It replays the path in each world and drops worlds where it's illegal.
  - A child's prior is the mean policy over the worlds where the action is legal.
  - The cost is k×N network calls, as with PIMC. Re-sampling every iteration would cost N×depth,
    because priors depend on the world.
  - With *random* world choice it lost to PIMC in Dark Hex. The world selection mattered.
- **OpenSpiel's `ISMCTSBot`** (`is_mcts.cc`, "lightly tested") has `max_world_samples = K`: a fixed
  particle set.
  - It keys nodes by each player's information-state string. It has no availability counts.
  - Its Python version offers PUCT. A node's prior comes from whichever world reached it first.
- **Commercial Spades IS-MCTS** biased by a human-move network (Baier et al. 2018) needed to delay
  network calls until a node had a few visits, to keep its speed.

### 6.4 How it would fit MageZero's engine

Five things in MageZero's search assume one concrete world:

1. **Each node caches one `GameState`.** A child is reached by one engine step from its parent's
   cached state (`MCTSNode.validateState`). IS-MCTS walks a different world each iteration, and one
   world's cached state is wrong for another. Two options:
   - **Replay from the root every iteration.** Engine steps per iteration grow with depth. XMage's
     copy machinery was already 37% of profile samples (ROADMAP, WillWroble/MageZero#3), so this is
     the expensive option.
   - **Particle IS-MCTS (recommended).**
     - Sample K worlds per decision, as ensemble determinization does. Each iteration picks one.
     - A node caches one `GameState` per world, created lazily by one engine step from the parent's
       state *in that world*.
     - Cost per iteration stays about what it is now, plus a first visit per (node, world) and one
       game copy per world.
     - Memory is at most K× the state cache. At a few hundred simulations per decision that's
       thousands of states, well under `MAX_TREE_NODES` = 100k.
     - It is IS-MCTS with the belief held as K particles, the way POMCP holds its root belief.
2. **Actions are concrete objects from one world.** A child holds an `Ability`, a target `UUID` or a
   string. In another world the opponent's Refute is a different card object. Children need
   world-independent keys, resolved to the concrete ability in each iteration's world. The bridge
   already has such keys: its labels ("Cast Refute", "Play Island", target names, duplicates
   merged) are what the coach aggregates by, close to the `ActionEncoder` indices the policy uses.
3. **Node identity.** An information-set node is identified by the path of keys, optionally split
   by what the searcher observes along it (its own draws). MageZero already has an observation key:
   the state vector, which it compares for transpositions and tree reuse.
   - MageZero#4 found that comparison unsafe *across concrete worlds*.
   - With stage 0's searcher-side encoding it is exactly IS-MCTS's merge rule: same observation,
     same node.
   - The unsafe part is the cached world, which the particle design keys by world.
   - The state vectors of one node in different worlds must match. That doubles as a leak
     detector: any difference is information the encoder shouldn't have.
4. **PUCT with availability.** Renormalize priors over the children available in the current world,
   and use availability counts in the exploration term. The network evaluates the searcher's
   information set, so a node's value is the same in every world, as IS-MCTS assumes.
5. **Targets and reuse.**
   - The root visit counts of an information-set tree are a natural policy target, and its mean Q
     the row's `stateScore`.
   - Tree reuse across decisions is *valid* for information sets, unlike for worlds.
   - The particles are re-sampled at each decision.

What stage 1 already provides and IS-MCTS reuses: the knowledge tracker, belief, world sampler and
world-local RNG (§8.2), the root aggregation by label, and the certification suite. What IS-MCTS
adds: points 1–4.

### 6.5 Verdict

- **Don't start with IS-MCTS.** Ensemble determinization is days of work on the same components.
  It is what upstream XMage's own Monte Carlo player already does (§7), and it is the baseline
  IS-MCTS has to beat. MageZero#4 points the same way. Its planned direction is one
  observer-consistent world per tree, discarded after the decision. It asks that determinization
  not be passed off as full imperfect-information search, and it lists IS-MCTS and POMCP as later
  options without choosing.
- **Then particle SO-IS-MCTS, behind a flag, at equal budget.** Measure:
  - the certification suite (§8.2);
  - head-to-head against ensemble determinization;
  - simulations per second;
  - a K sweep for the ensemble.

  Its advantage should show most at MageZero's small budgets (one tree of 300 against four of 75)
  and where the right play depends on what is learned later.

  Don't share opponent nodes across worlds. Key them by the opponent's sampled information, or
  sample the opponent's move from its policy (DeltaDou), or the union of every possible hand's
  plays will swamp the tree, as in Dou Di Zhu (§6.3). The particle design is MAPLE's and
  OpenSpiel's `max_world_samples`. How the K worlds are chosen matters (§6.3).
- **For counterspells, tricks and bluffs, SO-IS-MCTS is not enough.** Its simulated opponent still
  sees the searcher's hand. The holder test (§5.4, stage 2) decides between MO-IS-MCTS with
  self-determinization and the cheaper policy opponent.
- **Don't expect equilibrium play from any of these.** Bluff *frequencies* need regret-based search
  or training (§4.3, §4.4).

## 7. Do XMage's own bots leak?

**Mostly not. The clairvoyant search is MageZero's rewrite, not XMage.** All of XMage's AI
simulations start from the same call, `GameImpl.createSimulationForAI()`: an exact copy of the
real game, upstream (`magefree/mage` at `504f614fe6`, 2026-09-26) as in this fork. What each bot
*does* with the copy differs, and that decides whether hidden cards reach its choices.

The code was audited for every bot. The MAD AI was also measured, by running its real decision
routine (`ComputerPlayer7.calculateActions`) on §3.4's positions. `tools/gameplay/MadProbe.java`
does this, via `hidden_info_leak.py --decider mad`, 6 build seeds per world, skill 6:

| Position | World | MAD AI's choice | Score |
|---|---|---|---|
| Counterspell | B holds Refute | Cast Serra Angel, 6/6 | 2970 |
| | B holds two Islands | Cast Serra Angel, 6/6 | 2970 |
| Cantrip | A's unseen top card is Llanowar Elves | Cast Helpful Hunter, 6/6 | 1138 |
| | A's unseen top card is a Plains | **Cast Cathar Commando**, 6/6 | 1098 |

| Bot | Search | World it searches | Verdict |
|---|---|---|---|
| **"Computer - monte carlo", upstream** (`ComputerPlayerMCTS`, the original author's MCTS) | UCT with random rollouts, one tree per CPU core (`USE_MULTIPLE_THREADS = true`), root statistics merged | **Re-dealt.** Each thread's copy re-deals every opponent's hand from its library and shuffles every library, the bot's own included (`createMCTSGame`). Each rollout re-deals again (`MCTSNode.randomizePlayers`). | **Determinized**: ensemble determinization, as in §4.1. It still knows the opponent's true decklist. |
| **"Computer - monte carlo", this fork** (`ComputerPlayerMCTS2`, MageZero) | PUCT with a network | exact copy; "dont shuffle here" | **Clairvoyant** (§3) |
| **"Computer - mad"** (`ComputerPlayer6`/`7`, and MageZero's "minimax" type `ComputerPlayer8`) | alpha-beta over the bot's own chain of actions within the current step | exact copy | **Blind to the opponent's responses, rather than clairvoyant. Its own draws leak through resolution** (measured above) |
| **Rule-based** (`ComputerPlayer`, the base of all of the above) | none | — | None found. It reads its own hand and its own library's contents, and other zones only where the rules allow a play from them (e.g. the top of a library) |

**MageZero removed determinization that XMage already had.** The fork's `createMCTSGame` still
carries the original Javadoc ("Swaps all other players hands with random cards from the library")
above the line "dont shuffle here". Upstream XMage still ships the determinizing player as
"Computer - monte carlo". This fork's `config.xml` comments it out in favour of
`ComputerPlayerMCTS2`.

**The MAD AI** (`ComputerPlayer6`, the default "Computer - mad" and the base of MageZero's minimax
opponent):

- **It never simulates an opponent's response.** After each of its own actions that uses the stack
  (and after a pass), the simulation makes every player pass priority, "skip priority for opponents
  before stack resolve" (`ComputerPlayer6.java:558–561`; upstream `:552`). The spell resolves
  unanswered. So the opponent's hand never produces a counterspell, trick or flash blocker in its
  search, real or imagined. **In the counterspell position it casts the Angel with the same score
  whatever B holds** (table above). It can't play around anything.
- **Its evaluator counts hand sizes, not hand contents** (`GameStateEvaluator2.java:114–115`). A
  TODO there proposes scoring the opponent's *revealed* cards.
- **It leaks through resolution.** Spells resolve in the copy (`ComputerPlayer6.java:429`):
  - a simulated cantrip draws the real next card, and the chain can go on to play it;
  - an opponent's forced discard or sacrifice is chosen from its real hand by the rule-based
    logic.

  Because the evaluator counts hand sizes, a discard's *identity* rarely moves its score. The
  own-draw leak can, and did. In the cantrip position the simulated Hunter draws the real Elves
  and the chain casts them (score 1138). With a Plains on top it draws a dead card, and the Commando
  line wins (1098). **The choice flips on a card the player can't see**, in 6 of 6 seeds.
- **Its search stays inside the current step** (depth counts actions, not turns). Future draw steps
  never enter it.

**What the bots share.** `createSimulationForAI()` is one entry point for every bot's
simulations, so it is the natural place for a world sampler that serves all of them (§8.2).

## 8. Agents that see only what they should: baselines and trained agents

### 8.1 One information contract

Every agent should condition on the same things a player at a real table could. For player P:

- **Public state:** battlefield, graveyards, face-up exile, stack, life, mana pools, counters,
  phase, zone sizes, and what has been played.
- **P's own cards:** its hand, and its decklist, so the *contents* of its library but not the
  order.
- **Hidden cards P legitimately knows:** revealed, returned to hand from a public zone, looked at,
  or found by a search.
- **About the opponent:** its hand size, the cards it has shown, and a prior over its deck.

The contract is enforced by two things. Networks get the stage 0 encoding (§5.4). Searches get
sampled worlds, never the real one.

### 8.2 Shared components, built once in the fork

1. **A knowledge tracker.** A watcher that records, per player, which hidden cards they know:
   - cards that moved from a public zone into a hand;
   - reveals;
   - looked-at library cards;
   - scry and surveil positions, forgotten on shuffle.

   XMage has no lasting record of this (§1). The encoder (known cards shown) and the sampler (known
   cards pinned) both read it.
2. **A belief over the opponent's unseen cards,** in three tiers:
   - **T1, true decklist.** The unseen cards are the opponent's real 40 minus what was seen. Cheap
     and symmetric, but it leaks the deck's composition: the searcher knows whether a counterspell
     exists in that deck at all.
   - **T2, deck model.** Sample a decklist from 17lands decks consistent with the cards seen
     (`belief.py`'s deck pool, ported or precomputed), then deal from its unseen part. No leak.
   - **T3, inference.** T2 weighted by how likely each hand makes the opponent's actual plays
     under the policy network, or drawn from a learned belief head (§5.3).
3. **A world sampler.** `sample(game, observer, belief, rng)` returns a copy in which:
   - the opponent's hand is its known cards plus unknown slots filled from the belief;
   - both libraries are shuffled below any known positions, the observer's own included;
   - randomness inside the world comes from a world-local seeded RNG, not the global `RandomUtil`.
     In this fork `Library.shuffle()` draws from `RandomUtil`, which the MageZero constructor
     reseeds to a constant (MageZero#4's RNG domains).

   Reference implementations: the bridge's `StateInjector.injectLibraryAndHidden` /
   `resampleHand`, and upstream XMage's `createMCTSGame`. The hook is a variant of
   `createSimulationForAI` that takes the observer, so every bot gets it (§7).
4. **A certification suite.** Generalize `tools/gameplay/hidden_info_leak.py` from MageZero's search
   to any agent: pairs of worlds identical to the agent's observation must give the same decision
   distribution, within noise. Positions:
   - **counterspell** (§3.4's first position);
   - **combat trick:** A attacks into B's open mana, and B holds Giant Growth or a land;
   - **own draw** (§3.4's cantrip position): A holds a cantrip, and its library's top card is
     either a land or a spell it could cast with the mana left. Whether to cast the cantrip must
     not depend on which;
   - **positive control:** B's hand holds a card A *saw* bounced. The agent should use that, so the
     pair must differ.

   What exists today:
   - the counterspell and own-draw positions (`--position counterspell|cantrip`);
   - MageZero's search through the bridge;
   - the MAD AI through `MadProbe.java`, which runs a bot's own decision routine on a
     bridge-built game with no bridge change. The same trick works for any XMage player class.

   Still needed: the trick and positive-control positions, and a `decider` option in the bridge
   for bots whose decisions need the game to run. Run the suite in CI with small budgets. No
   agent's results count until it passes.

### 8.3 Baselines

| Baseline | Role here | Today | Fair version |
|---|---|---|---|
| **Raw search** (MageZero's MCTS, offline evaluator) | gen-0 training data and the evaluation yardstick | clairvoyant, so experiment #1's gen-0 data and yardstick were too | ensemble determinization with the heuristic evaluator: the same code as the trained agent's stage 1 |
| **MAD AI / minimax** (`ComputerPlayer7`, `8`) | XMage's standard bot, MageZero's minimax opponent | real copy; blind to responses; its own draws leak through resolution (§7) | one sampled world per decision in `createSimulation`. Its search is unchanged, so it still can't play around anything |
| **Upstream Monte Carlo** (`ComputerPlayerMCTS`) | historical reference | determinized, true decklist | already fair at tier T1; weak (random rollouts) |
| **Rule-based** (`ComputerPlayer`) | floor | no search | fair as is |
| **Human-like policy** (docs/008's imitation heads) | a human-style opponent (§5.4, stage 4) | trained on imperfect-information encodings | fair by construction; a shallow search on sampled worlds would add strength |

**Keep the old yardstick during the change.** ROADMAP wants a fixed yardstick (raw search at 96).
Run the clairvoyant raw search next to the fair one for one experiment, so results stay comparable
with experiment #1's 55.8%. A head-to-head between the two measures what the leak is worth in
Magic, at no GPU cost. Dou Di Zhu's gap was 56.5% against 43.6% (§4.2); nobody has measured
Magic's.

A quick prototype for that match: call `shuffleUnknowns` in `createMCTSGame` and disable root
reuse. That gives MageZero with one sampled world per decision, a few lines behind a config flag
(the `hiddenInfo:` block is the key Java actually reads). It still ignores known cards and draws
from `RandomUtil`, so use it only to measure, not to train.

### 8.4 Trained agents

The trained agent is the §5.4 plan with the shared components:

1. **Observation:** the stage 0 encoder, fed by the knowledge tracker.
2. **Search:** ensemble determinization on sampled worlds (stage 1). Then particle IS-MCTS behind a
   flag (§6). Then an opponent that decides from its own information: a policy opponent, or
   multiple-observer IS-MCTS with self-determinization (stage 2).
3. **Data:** fair self-play only. Gen 0 comes from the fair raw search, not the clairvoyant one.
4. **Opponents:** past selves, fair baselines, human-like policies. Evaluate only against fair
   opponents, and report the clairvoyant head-to-head separately.
5. **Belief:** start at T1 for both seats, move to T2 before quoting card statistics against
   17lands (human players don't know the opponent's decklist), then T3.

### 8.5 Order of work

| # | Step | Depends on | Tells you |
|---|---|---|---|
| 1 | Knowledge tracker, world sampler (T1), world-local RNG in the fork | — | — |
| 2 | Bridge `decider` option; certification suite | 1 | which agents are fair. Measured so far (§3.4, §7): MageZero fails the counterspell and cantrip pairs; MAD passes the counterspell pair, since it never models responses, and fails the cantrip pair. Expected: upstream Monte Carlo passes both |
| 3 | Fair raw search; clairvoyant-vs-fair head-to-head | 1 | what the leak is worth in Magic |
| 4 | Stage 0 encoder | 1 | — |
| 5 | Experiment #2 network on fair self-play (stage 1 search, λ ≈ 0.95) | 2–4 | card statistics without the leak (§5.5) |
| 6 | Particle SO-IS-MCTS against ensemble determinization at equal budget | 5 | whether one information-set tree beats K small trees |
| 7 | Policy opponent or MO-IS-MCTS with self-determinization; holder test | 6 | ambush and bluff value (§5.2) |
| 8 | Belief T2, then T3 | 1 | 17lands-comparable statistics |

Steps 1–3 need no GPU. Steps 1 and 2 overlap MageZero#4's open items (observer knowledge snapshot,
world-local engine RNG, determinization, hidden-information regression), so they're worth doing
with that contributor, or at least in the same shape.

## 9. Caveats

- **Two constructed positions.** §3.4 shows the mechanism, not its frequency in self-play. Two
  measurements would give the frequency:
  - count, per search, the nodes whose options or draws depended on hidden cards;
  - run docs/008 §8.3's true-hand-against-belief comparison (19% against 11%) on self-play states.
- **Nothing in §5.4 has been run in self-play.** Its effort and cost figures are estimates.
- **The card-statistics link (§3.6) is a hypothesis.** docs/003's targeting diagnostics are
  measured.
- **gen 33 with `perfectInfo` off is out of distribution.** It is in §3.4 only to show that masking
  the input doesn't remove the transition leak.
- **The PIMC condition knows B's true decklist**, so it leaks the decklist's composition, though
  not the hand.
- **§4 summarizes published work** in paraphrase, with the authors' reported numbers. Read the
  sources before relying on details.
- **Upstream is moving.** MageZero#4 may change the search contract. Check its state before
  implementing stage 1.
- **The MAD probe calls the bot's decision routine directly** on a bridge-built game. The probe
  holds the seat's id but isn't seated. `calculateActions` only needs the game and the id, but the
  real game loop (action cache, timers) was not exercised.
- **In the cantrip position MageZero's best action didn't flip.** The leak shows in its Q values
  and visit targets, and the MAD AI's choice did flip. Both are single constructed positions.
- **Upstream XMage's Monte Carlo player was not measured.** It isn't in this fork's jar. "Passes"
  in §8.5 is read from its code.
- **§6 is analysis, not an implementation.** The particle IS-MCTS cost is an estimate, and nothing
  in §6 has been run.

## References

**MageZero and XMage**

- WillWroble/MageZero#4, "[Research] Harden MCTS Search Semantics Before Advanced Search Variants"
  (open, 2026-09-20): https://github.com/WillWroble/MageZero/issues/4
- MageZero v0.2.0-alpha release notes: https://github.com/WillWroble/MageZero/releases/tag/v0.2.0-alpha
- XMage fork: https://github.com/WillWroble/mage (`master` at `cb7e9c6fe2` is v0.2.0; experiment
  #1 ran `5a32441c`, whose source is in the archive described in ROADMAP)

**Determinization, information sets, inference**

- Ginsberg (2001). GIB: Imperfect information in a computationally challenging game. *JAIR* 14.
  https://arxiv.org/abs/1106.0669
- Frank & Basin (1998). Search in games with incomplete information: a case study using Bridge card
  play. *Artificial Intelligence* 100. https://www.sciencedirect.com/science/article/pii/S0004370297000829
- Long, Sturtevant, Buro & Furtak (2010). Understanding the success of perfect information Monte
  Carlo sampling in game tree search. *AAAI*. https://ojs.aaai.org/index.php/AAAI/article/view/7562
- Buro, Long, Furtak & Sturtevant (2009). Improving state evaluation, inference, and search in
  trick-based card games. *IJCAI*. https://www.ijcai.org/Proceedings/09/Papers/236.pdf
- Rebstock, Solinas, Buro & Sturtevant (2019). Policy based inference in trick-taking card games.
  *IEEE CoG*. https://arxiv.org/abs/1905.10911
- Cowling, Powley & Whitehouse (2012). Information set Monte Carlo tree search. *IEEE TCIAIG* 4(2).
  https://eprints.whiterose.ac.uk/id/eprint/75048/
- Cowling, Whitehouse & Powley (2015). Emergent bluffing and inference with Monte Carlo tree search.
  *IEEE CIG*. http://orangehelicopter.com/academic/papers/cig15.pdf
- Blüml, Czech & Kersting (2023). AlphaZe∗∗: AlphaZero-like baselines for imperfect information
  games are surprisingly strong. *Frontiers in AI* 6.
  https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full
- Li, Guei, Wu & Wu (2026). MAPLE. *IEEE CoG*. https://arxiv.org/abs/2605.24139
- Sokota, Vinitsky, Hu, Kolter & Farina (2025). Superhuman AI for Stratego using self-play
  reinforcement learning and test-time search (Ataraxos). Preprint. https://arxiv.org/abs/2511.07312

**IS-MCTS (§6)**

- Whitehouse (2014). Monte Carlo tree search for games with hidden information and uncertainty.
  PhD thesis, University of York. https://etheses.whiterose.ac.uk/8117/
- Ponsen, de Jong & Lanctot (2011). Computing approximate Nash equilibria and robust best-responses
  using sampling. *JAIR* 42. https://arxiv.org/abs/1401.4591
- Lisý, Lanctot & Bowling (2015). Online Monte Carlo counterfactual regret minimization for search
  in imperfect information games. *AAMAS*. https://www.ifaamas.org/Proceedings/aamas2015/aamas/p27.pdf
- Heinrich & Silver (2015). Smooth UCT search in computer poker. *IJCAI*.
  https://www.ijcai.org/Proceedings/15/Papers/084.pdf
- Jiang et al. (2019). DeltaDou: expert-level Doudizhu AI through self-play. *IJCAI*.
  https://www.ijcai.org/proceedings/2019/0176.pdf
- Baier et al. (2018). Emulating human play in a leading mobile card game. *IEEE Transactions on
  Games*. https://repository.falmouth.ac.uk/2872/1/tog2018-emulating-human-play.pdf
- Silver & Veness (2010). Monte-Carlo planning in large POMDPs (POMCP). *NeurIPS*.
  https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf
- OpenSpiel IS-MCTS: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/is_mcts.cc

**Belief-state and game-theoretic search**

- Moravčík et al. (2017). DeepStack. *Science* 356. https://arxiv.org/abs/1701.01724
- Brown & Sandholm (2018). Superhuman AI for heads-up no-limit poker: Libratus. *Science* 359.
  https://www.science.org/doi/10.1126/science.aao1733
- Brown & Sandholm (2019). Superhuman AI for multiplayer poker (Pluribus). *Science* 365.
  https://www.science.org/doi/10.1126/science.aay2400
- Brown, Bakhtin, Lerer & Gong (2020). Combining deep reinforcement learning and search for
  imperfect-information games (ReBeL). *NeurIPS*. https://arxiv.org/abs/2007.13544
- Schmid et al. (2023). Student of Games. *Science Advances* 9(46). https://arxiv.org/abs/2112.03178
- Zhang & Sandholm (2026). General search techniques without common knowledge for
  imperfect-information games, and application to superhuman Fog of War chess (Obscuro). *ICLR*.
  https://arxiv.org/abs/2506.01242
- Lerer, Hu, Foerster & Brown (2020). Improving policies via search in cooperative partially
  observable games (SPARTA). *AAAI*. https://arxiv.org/abs/1912.02318
- Hu, Lerer, Brown & Foerster (2021). Learned belief search. Preprint.
  https://arxiv.org/abs/2106.09086

**Learning on partial observations, privileged training**

- Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning
  (AlphaStar). *Nature* 575. https://www.nature.com/articles/s41586-019-1724-z
- Berner et al. (2019). Dota 2 with large scale deep reinforcement learning (OpenAI Five). Preprint.
  https://arxiv.org/abs/1912.06680
- Perolat et al. (2022). Mastering the game of Stratego with model-free multiagent reinforcement
  learning (DeepNash). *Science* 378. https://www.science.org/doi/10.1126/science.add4679
- Zha et al. (2021). DouZero. *ICML*. https://arxiv.org/abs/2106.06135
- Yang et al. (2022). PerfectDou: dominating DouDizhu with perfect information distillation.
  *NeurIPS*. https://arxiv.org/abs/2203.16406
- Li et al. (2020). Suphx: mastering Mahjong with deep reinforcement learning. Preprint.
  https://arxiv.org/abs/2003.13590
- Pinto et al. (2018). Asymmetric actor critic for image-based robot learning. *RSS*.
  https://arxiv.org/abs/1710.06542
- Chen, Zhou, Koltun & Krähenbühl (2019). Learning by cheating. *CoRL*. https://arxiv.org/abs/1912.12294
- Baisero & Amato (2022). Unbiased asymmetric reinforcement learning under partial observability.
  *AAMAS*. https://arxiv.org/abs/2105.11674
- Warrington et al. (2021). Robust asymmetric learning in POMDPs. *ICML*.
  https://arxiv.org/abs/2012.15566
- Ye et al. (2020). Towards playing full MOBA games with deep reinforcement learning (Honor of
  Kings). *NeurIPS*. https://arxiv.org/abs/2011.12692

**Magic and other card games**

- Ward & Cowling (2009). Monte Carlo search applied to card selection in Magic: The Gathering.
  *IEEE CIG*. https://dblp.org/rec/conf/cig/WardC09.html
- Cowling, Ward & Powley (2012). Ensemble determinization in Monte Carlo tree search for the
  imperfect information card game Magic: The Gathering. *IEEE TCIAIG* 4(4).
  https://eprints.whiterose.ac.uk/id/eprint/75050/
- da Costa Cunha, Liu, French & Mian (2026). Causal RL for complex card games: a Magic The
  Gathering benchmark. Preprint. https://arxiv.org/abs/2605.06066
- Zhang & Buro (2017). Improving Hearthstone AI by learning high-level rollout policies and
  bucketing chance node events. *IEEE CIG*. https://skatgame.net/mburo/ps/cig17-hsai.pdf
- Świechowski, Tajmajer & Janusz (2018). Improving Hearthstone AI by combining MCTS and supervised
  learning algorithms. *IEEE CIG*. https://arxiv.org/abs/1808.04794
- Bursztein (2016). I am a legend: hacking Hearthstone using statistical learning methods. *IEEE
  CIG*. https://elie.net/publication/i-am-a-legend-hacking-hearthstone-using-statistical-learning-methods
- Dockhorn et al. (2018). Predicting opponent moves for improving Hearthstone AI. *IPMU*.
  https://doi.org/10.1007/978-3-319-91476-3_51
- Xiao et al. (2023). Mastering strategy card game (Hearthstone) with improved techniques. *IEEE
  CoG*. https://arxiv.org/abs/2303.05197
- Haluska & Schmid (2024). Learning to beat ByteRL. *ALA workshop*. https://arxiv.org/abs/2404.16689
- Rubin (2026). Preprint on determinized search with a draft-pick deck prior in Legends of Code and
  Magic (title not checked). https://arxiv.org/abs/2609.06816
