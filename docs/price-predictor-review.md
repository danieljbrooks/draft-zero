# lordofthepigs' Forge simulations: notes for DraftZero

lordofthepigs ([npiguet](https://github.com/npiguet) on GitHub) is doing some great work simulating
limited games. His repo, [npiguet/price-predictor](https://github.com/npiguet/price-predictor),
has played hundreds of thousands of games in the Forge engine. On top of those games he has built
sealed pools, a deck-rating model, deck builders, draft bots and an ability-effect model, and he
writes up every experiment in detail.

What follows are Claude-generated insights describing his work. Claude read his experiment
reports, diaries and code at commit `15d49c8` (2026-09-19) and on 2026-09-25 wrote up what matters
most for DraftZero:
- "We" and "our" mean DraftZero, and "I" is Claude.
- Figures about his work are his reported numbers.
- The last section lists what I checked directly.

## Summary

He has already built the pipeline we'll need later: card list → sealed pools → deck builder →
drafter.
- **Every game is played by Forge's default AI**, and he deliberately optimizes against it: "Forge's
  piloting quirks are the target, not a confound".
- **He doesn't compare anything with 17lands**, by design.
- So his numbers describe how sets play under Forge's AI rather than how they should be played.

What's most useful for us:
- his evidence on how a weak player distorts card stats;
- his evaluation habits;
- the failures he hit building deck and draft bots.

## First: exp #1's ρ = 0.28 was mostly limited by game count

This one is about our own metric, prompted by his reliability checks.

**Method.** I sampled real 17lands FDN Premier Draft games down to [exp #1](fdn-generalist-report.md)'s
sizes, 60 samples per size. For each sample, I correlated the commons' GIH win rates against the
full 791k-game data, counting commons with ≥30 games in hand. That shows what human-quality play
would score at each size:

| Deck results | Best possible ρ, median [5–95%] |
|---|---|
| 1,000 (one ~500-game window) | 0.35 [0.13–0.57] |
| 2,880 (gens 10–33) | 0.49 [0.34–0.61] |
| 10,000 | 0.75 |
| 50,000 | 0.94 |

- **Exp #1's early windows (0.43, 0.45) were already at this ceiling.** The late 0.05 is below it,
  so that decline is real.
- **The [Phase 4](../ROADMAP.md) criterion "beats 0.28" needs restating against the ceiling at
  exp #2's size.**
  - A ceiling of ~0.75 needs about 10k deck results, roughly 5k self-play games since both sides count.
  - He runs the same kind of check on his own stats, using split-half reliability.
- **An earlier 35.7k-game Forge run of ours, on HOB, is far past the ceiling.** Whatever gap it
  showed against 17lands is systematic, not noise. That supports the view that Forge's AI is the
  problem, not the sample size.

## How a weak player distorts card stats

- **Forge's AI leaves the same distortion as our exp #1 agent.**
  - **His Forge-derived card values against human pick orders**
    ([scorer-preferences](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-08-27-scorer-preferences.md)):
    - below for removal (−0.2 SD), card draw (−0.45) and artifacts (−0.63);
    - above for flyers (+0.57);
    - no premium for instant-speed removal.
  - **Colours in his games:** Forge-built decks win most with white and least with blue.
  - **Exp #1's gaps against 17lands GIH win rates:** blue commons averaged 8.0 points below, red 7.3.
    Premium removal was about 12 points low.
  - **Two unrelated weak players showing one pattern suggests a generic weak-player effect.**
    - That gives exp #2 a testable prediction: if weak play is the cause, the removal and blue gaps
      should shrink as search goes from 96 to 300+ simulations.
    - Worth measuring in the Phase 3 budget pilot, pooling the premium removal spells to get enough
      games.
  - **Caveat:** his "human" reference is Forge's own pick-order files, which is partly circular.
- **He splits a card's win rate into three parts**
  ([encoder-preferences](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-08-28-encoder-preferences.md)):
  which decks it's in, how often it gets cast, and how much more often winners cast it than losers.
  - Exp #1 plays 17lands' own decks, so the first part is roughly held fixed. Stab's gap must then
    come from when and on what it gets cast.
  - Our per-game summary logs cards drawn but not cards cast
    ([stats.py](../src/draftzero/stats.py#L4)). Worth adding cast turn and target when that summary
    is rebuilt on v0.2.
- **Draws count as a loss for both sides** ([stats.py](../src/draftzero/stats.py#L83)).
  - He has the reverse problem. His pipeline skips drawn or aborted games, and it restarts its oldest
    worker every 60 seconds, losing whatever match that worker was playing, most often a long one.
  - Long games are where removal and control decks live.
  - In the one local run I could check (162 gen-0 games), 2.5% had no winner.
  - That's small, but it pushes in the same direction as the removal gap. Worth checking exp #1's
    rate and counting draws as half a win.

## Cheap checks that need no human data

- **Deck-quality ladder.** Replay eval decks with k random swaps from each deck's own 17lands
  sideboard. A player whose stats you can trust should lose steadily as k rises.
  - His ladder has four steps: Forge's best build, 3 random swaps, 8 random swaps, and a fully
    random deck.
  - Their match win rates are 65 / 52 / 36 / 15%.
- **Board-erasure test** (designed, not yet run). Blank or relabel the opponent's board and count how
  many attack and block decisions change.
  - His own probes found that his drafter's 4th generation had become a fixed pick order that
    ignored the rest of the table.
  - Pair this with his combat census, which scans every attack and block decision rather than just
    the ones made. That turns "doesn't chump a 2/2 with a 1/1" into a number.
- **Judge the value head by how it ranks sibling positions, not by calibration** (designed). In his
  words, "a value head that reads 0.52 for every candidate is coin-flipping however well calibrated
  it is."
- **A/A test.** Two identical checkpoints once showed him a +0.78 "improvement". Calibrate our
  evals the same way.
- **Fixed shuffles per deck pair, with seats swapped** (designed). Mana screw and flood then cancel
  out. Our games already log a seed; check whether it fixes library order.

## For the future deck-building and draft stages

- **Match results alone didn't teach his deck-rating model card quality.**
  - Accuracy sat at 0.69–0.70 through eight fixes.
  - Per-card labels from game logs lifted it to 0.734. The resulting decks beat Forge's builder 61/39
    over 6,331 games.
  - The dataset also grew at the same time, so the gain is partly confounded.
  - For us: build the deck model on per-card stats from our own games.
- **His deck-rating model ended up simply adding up card values.**
  - It sees no synergy: adding 0→3 enablers moves the score +0.008 ± 0.010.
  - It reads card proportions rather than counts, and can't learn land counts.
  - If the player doesn't exploit synergies, the training data contains none.
  - His probe set is a ready-made audit for any deck model we build.
- **The drafter learns its player's tastes.**
  - **Better-scoring drafters did win more Forge games:** about 72% of best-of-sevens, roughly 60%
    per game
    ([gen-4](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-08-09-draft-agent-gen4-online-grpo.md)).
  - **They also inherited Forge's tastes:** they passed removal and put white in 60–63% of decks
    ([behaviour](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-08-29-draft-agent-behaviour.md)).
    The deck-rating model stayed frozen across every drafter generation, so nothing could correct
    that.
  - **For us, the order is player → deck-value model → drafter**, refitting the value model as the
    player improves.
  - **A learned deck-value reward is unavoidable.** Each drafter generation needed ~3k drafts (~25k
    rated decks), so real games can only be used to validate.
- **One score per finished deck can't teach reading signals or hate-drafting.** Taking a card only
  to deny it is worth about 1/7 of playing it, below what the training signal can detect.
- **Hold the deck builder fixed when comparing drafters.** Swapping the builder alone was worth about
  one generation of drafting (73.9%, n=92).

## Not ready to borrow: card-text and card-effect embeddings

This is the route to building decks and drafting before human games exist. It doesn't generalize
yet.
- **His card-text encoder:**
  - explains 81% of the variance in card outcomes on cards it trained on, but only 37% on unseen
    cards;
  - is nearly matched by a simple model on 135 hand-listed features;
  - misreads mana costs: adding 2 mana *raises* its prediction;
  - hasn't been tested on a held-out set.
- **His ability-effect model:**
  - captures what kind of effect an ability has but not its size (Lightning Bolt ≈ the same line with
    1–10 damage);
  - fails his own quality gate so far, and no downstream gain has been measured yet;
  - would need engine patches to rebuild for XMage.
- **Cheap takeaway:** explicit features (mana value, power/toughness, coloured mana symbols, XMage
  effect classes with their parameters) plus a held-out-set check.
- **His game-agent design keeps per-card identity features out of the policy by default**, for
  exactly this reason
  ([design](https://github.com/npiguet/price-predictor/blob/master/experiments/2026-09-12-game-playing-agent-design.md)).

## Smaller notes

- **His engine-patch pattern fits [Phase 2](../ROADMAP.md)'s "Will's engine stays clean".**
  - Each engine hook is its own commit, and his code compiles against stock Forge.
  - It looks for the hooks at startup and falls back cleanly without them.
  - It tags each record with which mode collected it.
- **"Forge is too weak" is only established for its default AI.** Forge's slower simulation-based AI
  never generated data in his repo. At ~25–46 s per game it's still roughly an order of magnitude
  cheaper than our search, but its strength is unmeasured. Low priority.
- **One number is easy to misread.** His on-the-play win rate of 47.9% includes later games of each
  best-of-seven, where Forge lets the previous game's loser go first. So it doesn't show that Forge
  undervalues tempo.
- **Game length doesn't separate human, Forge and exp #1 games.** All run about 18–20 turns,
  counting each player's turn separately.

## What these numbers rest on

- **His figures.** Most figures are his own reports, mostly from single runs. His write-ups are
  candid about the bugs he found along the way, for example an effect model that once trained on
  all-zero inputs while its loss still fell. So treat any single figure with some caution.
- **What I checked directly:**
  - Forge's first-player rule;
  - which decisions MageZero's search controls: targets and combat are included, so weak targeting
    isn't the removal explanation;
  - the 17lands figures;
  - the noise ceiling.
