//! Stage-1 harness tests (`cargo test --release`).

use dzk::bots::BotSpec;
use dzk::deck::{self, load_deck, Deck, DeckRow};
use dzk::determinize::{self, hidden_zone_multiset, unknown_slots};
use dzk::game::Game;
use dzk::record::{play_game, GameRecord, GameTask};
use dzk::rng::SplitMix64;
use dzk::runner::{game_tasks, run_play, run_summarize, summarize, PlayArgs, Summary, SummaryRec};
use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::ids::PlayerId;
use mtg_kernel::rl::{make_legal_action_v5, observe_policy_v5_without_projection_hash_v1, ActionSemanticV1};
use mtg_kernel::state::GameState;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const UG: &str = "FDN_top_04956_UG";
const WG: &str = "FDN_top_20626_WG";

fn decks_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/sample/decks")
}

fn fixture() -> (Deck, Deck) {
    (load_deck(&decks_dir(), UG).unwrap(), load_deck(&decks_dir(), WG).unwrap())
}

fn scratch(name: &str) -> PathBuf {
    // under target/tmp (cargo's CARGO_TARGET_TMPDIR), replaced on every run
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("dzk-test-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn fixture_decks_resolve_with_fdn_reference_names() {
    let fdn = deck::fdn_reference_names();
    for d in [load_deck(&decks_dir(), UG).unwrap(), load_deck(&decks_dir(), WG).unwrap()] {
        assert_eq!(d.ids.len(), 40, "{}", d.stem);
        for row in &d.rows {
            let id = card_id_by_name(&row.name).unwrap();
            assert_eq!(CARD_DEFS[id as usize].name, row.name, "registry name");
            assert!(fdn.contains(&row.name), "{} not in FDN_gih.json", row.name);
        }
    }
    let supported: Vec<&String> = fdn
        .iter()
        .filter(|n| {
            card_id_by_name(n).is_some_and(|id| {
                let c = &CARD_DEFS[id as usize];
                c.has_full_support() && !c.is_token
            })
        })
        .collect();
    let nonbasic = supported.iter().filter(|n| !deck::BASICS.contains(&n.as_str())).count();
    eprintln!("FDN reference names with full engine support: {} ({} non-basic)", supported.len(), nonbasic);
    // Registry names of every supported FDN reference card are exactly the reference names.
    for n in &supported {
        assert_eq!(CARD_DEFS[card_id_by_name(n).unwrap() as usize].name, n.as_str());
    }
}

#[test]
fn fdn_rule_refuses_non_fdn_pauper_cards() {
    let deck =
        |extra: &str| vec![DeckRow { count: 39, name: "Island".into() }, DeckRow { count: 1, name: extra.into() }];
    // Brainstorm: full support in the Pauper registry, not an FDN card.
    let e = deck::resolve_rows(&deck("Brainstorm")).unwrap_err();
    assert!(e.contains("FDN reference list"), "{e}");
    // Tolarian Terror: a Pauper-base definition that is also an FDN card.
    assert_eq!(deck::resolve_rows(&deck("Tolarian Terror")).unwrap().len(), 40);
    // Too small, unknown: mtg-kernel's own refusals.
    assert!(deck::resolve_rows(&[DeckRow { count: 39, name: "Island".into() }]).is_err());
    assert!(deck::resolve_rows(&deck("Not A Card")).is_err());
}

fn play_args(dir: &Path, bot1: &str, bot2: &str, seed: u64, n_pairs: Option<usize>) -> PlayArgs {
    PlayArgs {
        decks_dir: decks_dir(),
        pairs_file: dir.join("pairs.tsv"),
        bot1: bot1.into(),
        bot2: bot2.into(),
        seed,
        threads: 2,
        out: dir.join("games.jsonl"),
        n_pairs,
        pair_range: None,
        shard: None,
        resume: false,
        data_out: None,
        limits: Default::default(),
    }
}

fn lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn total(m: &BTreeMap<String, u32>) -> u32 {
    m.values().sum()
}

#[test]
fn twenty_random_games_end_natural() {
    let (ug, wg) = fixture();
    // uniform mulligans too, so the London mulligan bookkeeping is exercised
    let random = BotSpec::parse("random,mull=random").unwrap();
    let pairs = vec![(UG.to_string(), WG.to_string()), (WG.to_string(), UG.to_string())];
    let tasks = game_tasks(&pairs, 10, 7, true);
    assert_eq!(tasks.len(), 20);
    let mut turns = Vec::new();
    let mut mulligans = 0;
    for t in &tasks {
        let (d1, d2) = if t.deck1 == UG { (&ug, &wg) } else { (&wg, &ug) };
        let (rec, _) = play_game(t, d1, d2, ("random,mull=random", &random), ("random,mull=random", &random));
        assert_eq!(rec.error, None, "pair {} swap {}", t.pair, t.swap);
        assert_eq!(rec.terminal.classification.as_deref(), Some("natural"), "{:?}", rec.terminal);
        assert!(rec.winner.is_some() || rec.terminal.outcome.as_deref() == Some("draw"));
        // the opening hand is the last seven-card deal (17lands); the kept hand is 7 minus mulligans and part of it
        for s in ["A", "B"] {
            let seat = &rec.seats[s];
            assert_eq!(total(&seat.opening_hand), 7, "{s}: {seat:?}");
            assert_eq!(total(&seat.kept_hand), 7 - u32::from(rec.mulligans[s]), "{s}: {seat:?}");
            for (name, &n) in &seat.kept_hand {
                assert!(seat.opening_hand.get(name).copied().unwrap_or(0) >= n, "{s}: kept {name} not dealt");
            }
            for (name, &n) in &seat.in_hand {
                let want =
                    seat.opening_hand.get(name).copied().unwrap_or(0) + seat.drawn.get(name).copied().unwrap_or(0);
                assert_eq!(n, want, "{s}: inHand {name}");
            }
        }
        assert!(
            rec.rounds * 2 >= rec.turns && rec.turns + 1 >= rec.rounds * 2,
            "turns {} rounds {}",
            rec.turns,
            rec.rounds
        );
        mulligans += rec.mulligans.values().map(|&m| u32::from(m)).sum::<u32>();
        turns.push(rec.turns);
    }
    assert!(mulligans > 0, "uniform mulligan choices should take some");
    eprintln!("20 random games ({mulligans} mulligans), turns {turns:?}");
}

fn strip_seconds(line: &str) -> String {
    let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
    v.as_object_mut().unwrap().remove("seconds");
    serde_json::to_string(&v).unwrap()
}

#[test]
fn records_are_deterministic() {
    let dir = scratch("determinism");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n{WG}\t{UG}\n")).unwrap();
    let mut runs = Vec::new();
    for run in 0..2 {
        let out = dir.join(format!("run{run}/games.jsonl"));
        let args = PlayArgs {
            decks_dir: decks_dir(),
            pairs_file: dir.join("pairs.tsv"),
            bot1: "mcts:20,worlds=2".into(),
            bot2: "random".into(),
            seed: 11,
            threads: 2 + run, // different thread counts, same records
            out: out.clone(),
            n_pairs: Some(2),
            pair_range: None,
            shard: None,
            resume: false,
            data_out: None,
            limits: Default::default(),
        };
        run_play(&args).unwrap();
        let mut lines: Vec<String> = std::fs::read_to_string(&out).unwrap().lines().map(strip_seconds).collect();
        lines.sort();
        runs.push(lines);
    }
    assert_eq!(runs[0].len(), 4);
    assert_eq!(runs[0], runs[1], "same seed and bots must give identical records");
    // and a different seed gives different games
    let (ug, wg) = fixture();
    let random = BotSpec::parse("random").unwrap();
    let t = |seed: u64| GameTask {
        pair: 0,
        swap: false,
        deck1: UG.into(),
        deck2: WG.into(),
        game_seed: seed,
        env_seed: seed,
    };
    let a = play_game(&t(1), &ug, &wg, ("random", &random), ("random", &random)).0;
    let b = play_game(&t(1), &ug, &wg, ("random", &random), ("random", &random)).0;
    let c = play_game(&t(2), &ug, &wg, ("random", &random), ("random", &random)).0;
    let norm = |r: &GameRecord| strip_seconds(&serde_json::to_string(r).unwrap());
    assert_eq!(norm(&a), norm(&b));
    assert_ne!(norm(&a), norm(&c));
}

#[test]
fn resume_skips_finished_games_and_cuts_a_torn_line() {
    let dir = scratch("resume");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let out = dir.join("games.jsonl");
    let mut args = PlayArgs {
        decks_dir: decks_dir(),
        pairs_file: dir.join("pairs.tsv"),
        bot1: "random".into(),
        bot2: "first".into(),
        seed: 3,
        threads: 1,
        out: out.clone(),
        n_pairs: Some(2),
        pair_range: Some((0, 1)),
        shard: None,
        resume: false,
        data_out: None,
        limits: Default::default(),
    };
    run_play(&args).unwrap();
    assert_eq!(std::fs::read_to_string(&out).unwrap().lines().count(), 2);
    assert!(run_play(&args).is_err(), "an existing output needs --resume");
    // a torn last line, as a kill mid-write would leave
    let mut text = std::fs::read_to_string(&out).unwrap();
    text.push_str("{\"pair\": 1, \"sw");
    std::fs::write(&out, text).unwrap();
    args.resume = true;
    args.pair_range = None;
    let s = run_play(&args).unwrap();
    let lines: Vec<String> = std::fs::read_to_string(&out).unwrap().lines().map(String::from).collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(s["games"], 4);
    let mut keys: Vec<(u64, bool)> = lines
        .iter()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            (v["pair"].as_u64().unwrap(), v["swap"].as_bool().unwrap())
        })
        .collect();
    keys.sort();
    assert_eq!(keys, vec![(0, false), (0, true), (1, false), (1, true)]);
    assert!(dir.join("summary.json").exists());
}

/// Every referenced object's definition refreshed from `state`, and the stable ids recomputed.
fn recomputed_stable_ids(game: &Game, state: &GameState) -> Vec<String> {
    (0..game.num_legal())
        .map(|i| game.legal_action(i))
        .map(|a| {
            let mut v = serde_json::to_value(&a.semantic).unwrap();
            fn walk(v: &mut serde_json::Value, state: &GameState) {
                match v {
                    serde_json::Value::Object(m) => {
                        if let (Some(id), true) =
                            (m.get("arena_id").and_then(|x| x.as_u64()), m.contains_key("card_db_id"))
                        {
                            let def = state.objects.get(mtg_kernel::ids::ObjectId(id as u32)).card_def;
                            m.insert("card_db_id".into(), serde_json::json!(def));
                        }
                        for x in m.values_mut() {
                            walk(x, state);
                        }
                    }
                    serde_json::Value::Array(a) => a.iter_mut().for_each(|x| walk(x, state)),
                    _ => {}
                }
            }
            walk(&mut v, state);
            let sem: ActionSemanticV1 = serde_json::from_value(v).unwrap();
            make_legal_action_v5(a.selected_index, sem, None).unwrap().stable_id
        })
        .collect()
}

fn defs(state: &GameState, ids: &[mtg_kernel::ids::ObjectId]) -> Vec<u16> {
    ids.iter().map(|&id| state.objects.get(id).card_def).collect()
}

#[test]
fn redeal_keeps_the_actors_view_and_each_owners_cards() {
    let (ug, wg) = fixture();
    let mut checked = 0;
    let mut changed_worlds = 0;
    let mut refused = 0;
    let mut reseeded = 0;
    for seed in [5u64, 6, 7] {
        let mut game = Game::new(&ug, &wg, seed, seed, false).unwrap();
        let mut rng = SplitMix64::new(seed);
        let mut step = 0;
        while !game.is_terminal() {
            if step % 3 == 0 {
                let viewer = game.actor().unwrap();
                let root = game.state();
                for w in 0..3u64 {
                    let world = match determinize::redeal(&game, &mut SplitMix64::new(seed * 100 + w + step)) {
                        Ok(world) => world,
                        Err(_) => {
                            refused += 1;
                            continue;
                        }
                    };
                    let ws = world.state();
                    // the same cards per owner, in the same zones' sizes
                    assert_eq!(hidden_zone_multiset(root), hidden_zone_multiset(ws));
                    for p in 0..2 {
                        let (a, b) = (&root.players[p], &ws.players[p]);
                        assert_eq!(a.hand.len(), b.hand.len());
                        assert_eq!(a.library.len(), b.library.len());
                        assert_eq!(defs(root, &a.battlefield), defs(ws, &b.battlefield));
                        assert_eq!(defs(root, &a.graveyard), defs(ws, &b.graveyard));
                        assert_eq!(a.life, b.life);
                    }
                    assert_eq!(defs(root, &root.exile), defs(ws, &ws.exile));
                    // the viewer's own hand is never re-dealt
                    let own = &root.players[viewer.index()].hand;
                    assert_eq!(defs(root, own), defs(ws, own));
                    // the observation (rebuilt inside redeal) and the legal action ids, recomputed from the
                    // world's own objects
                    let at = game.decision_coords().unwrap();
                    let mut again = observe_policy_v5_without_projection_hash_v1(
                        ws,
                        world.session().policy_surface_v1(),
                        viewer,
                        at.step,
                        at.physical_decision_id,
                        at.substep_index,
                        at.substep_count,
                    )
                    .unwrap();
                    let obs = game.observation().unwrap();
                    again.visible_projection_hash = obs.visible_projection_hash;
                    assert_eq!(again, *obs);
                    assert_eq!(
                        recomputed_stable_ids(&game, ws),
                        (0..game.num_legal()).map(|i| game.legal_action(i).stable_id.clone()).collect::<Vec<_>>()
                    );
                    // the future shuffle stream is re-seeded
                    assert_ne!(root.legacy_rng(), None);
                    if ws.legacy_rng() != root.legacy_rng() {
                        reseeded += 1;
                    }
                    let opp = viewer.opponent();
                    let slots = unknown_slots(root, viewer);
                    if slots.iter().flatten().any(|&id| root.objects.get(id).card_def != ws.objects.get(id).card_def) {
                        changed_worlds += 1;
                    }
                    // known cards stay put
                    for e in root.known_hand_cards(viewer, opp) {
                        assert_eq!(root.objects.get(e.object).card_def, ws.objects.get(e.object).card_def);
                    }
                    for owner in [PlayerId::P0, PlayerId::P1] {
                        for e in root.known_library_cards(viewer, owner) {
                            assert_eq!(root.objects.get(e.object).card_def, ws.objects.get(e.object).card_def);
                        }
                    }
                    // the world plays on
                    let mut w2 = world.clone_untracked();
                    w2.play(0).unwrap();
                    checked += 1;
                }
            }
            let k = rng.below(game.num_choices() as u64) as usize;
            game.play(k).unwrap();
            step += 1;
        }
    }
    eprintln!(
        "re-deal: {checked} worlds checked, {changed_worlds} changed hidden cards, {reseeded} re-seeded the RNG, \
         {refused} refused"
    );
    assert!(checked > 50);
    assert!(changed_worlds * 2 > checked, "re-deals should usually move hidden cards");
    assert_eq!(reseeded, checked, "every world gets its own future shuffle stream");
}

/// The masked index of the London mulligan announcement's `mulligan: want` answer.
fn mulligan_choice(game: &Game, want: bool) -> usize {
    (0..game.num_choices())
        .find(|&k| matches!(game.choice(k).semantic, ActionSemanticV1::ChooseLondonMulligan { mulligan, .. } if mulligan == want))
        .expect("a London mulligan announcement")
}

#[test]
fn a_world_does_not_know_which_cards_a_mulligan_brings_back() {
    // With the real future RNG, a world's mulligan redraw returns exactly the real game's objects (the
    // reviewers' leak); re-seeded, it usually does not.
    let (ug, wg) = fixture();
    let (mut same, mut n) = (0, 0);
    for seed in 0..20u64 {
        let game = Game::new(&ug, &wg, seed, seed, false).unwrap();
        assert_eq!(game.actor(), Some(PlayerId::P0)); // P0's first announcement
        let world = determinize::redeal(&game, &mut SplitMix64::new(seed + 1000)).unwrap();
        let redraw = |mut g: Game| {
            g.play(mulligan_choice(&g, true)).unwrap(); // P0 mulligans
            g.play(mulligan_choice(&g, false)).unwrap(); // P1 keeps: P0 redraws seven
            g.state().players[0].hand.clone()
        };
        let (real, sampled) = (redraw(game.clone_untracked()), redraw(world));
        assert_eq!(real.len(), 7);
        same += usize::from(real == sampled);
        n += 1;
    }
    eprintln!("mulligan redraws identical to the real game's in {same} of {n} worlds");
    assert!(same * 4 < n, "{same} of {n}");
}

#[test]
fn mcts50_beats_random_in_a_small_paired_sample() {
    let (ug, wg) = fixture();
    let mcts = BotSpec::parse("mcts:50").unwrap();
    let random = BotSpec::parse("random").unwrap();
    let pairs = vec![(UG.to_string(), WG.to_string()), (WG.to_string(), UG.to_string())];
    let tasks = game_tasks(&pairs, 4, 21, false);
    let mut score = 0.0;
    let mut n = 0;
    for t in &tasks {
        let (d1, d2) = if t.deck1 == UG { (&ug, &wg) } else { (&wg, &ug) };
        let (rec, _) = play_game(t, d1, d2, ("mcts:50", &mcts), ("random", &random));
        assert_eq!(rec.error, None);
        // the default mulligan policy keeps, as the Java baseline does (random's too)
        for seat in ["A", "B"] {
            assert_eq!(rec.mulligans[seat], 0);
            assert_eq!(rec.search[seat]["mulligan_decisions"], 1);
        }
        let seat = if t.swap { "B" } else { "A" };
        assert_eq!(rec.search[seat]["world_fallbacks"], 0);
        score += match rec.winner_role.as_deref() {
            Some("bot1") => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        };
        n += 1;
    }
    eprintln!("mcts:50 vs random: score {score}/{n} = {:.3}", score / n as f64);
}

#[test]
fn uncached_nodes_replay_to_the_same_search() {
    // cache=2: almost every node drops its game copy and is re-derived by replay; the games must not change.
    let (ug, wg) = fixture();
    let random = BotSpec::parse("random").unwrap();
    let t = GameTask { pair: 0, swap: false, deck1: UG.into(), deck2: WG.into(), game_seed: 77, env_seed: 77 };
    let run = |spec: &str| {
        let s = BotSpec::parse(spec).unwrap();
        play_game(&t, &ug, &wg, (spec, &s), ("random", &random)).0
    };
    let a = run("mcts:60,worlds=2");
    let b = run("mcts:60,worlds=2,cache=2");
    assert_eq!(a.error, None);
    assert_eq!(a.winner, b.winner);
    assert_eq!(a.decisions, b.decisions);
    assert_eq!(a.seats, b.seats);
    let (sa, sb) = (&a.search["A"], &b.search["A"]);
    assert_eq!(sa["simulations"], sb["simulations"]);
    assert_eq!(sa["replay_steps"], 0);
    assert!(sb["replay_steps"].as_u64().unwrap() > 0, "{sb}");
    eprintln!("cache=2 replayed {} steps over {} simulations", sb["replay_steps"], sb["simulations"]);
}

#[test]
fn seen_in_hand_covers_every_card_ever_held() {
    // Every card in a hand at any decision after the mulligans was in the opening hand or drawn since.
    let (ug, wg) = fixture();
    let mut mulligans = 0;
    for seed in 30..40u64 {
        let mut game = Game::new(&ug, &wg, seed, seed, true).unwrap();
        let mut rng = SplitMix64::new(seed);
        while !game.is_terminal() {
            if let Some(t) = game.tracker().filter(|t| t.kept.is_some()) {
                let s = game.state();
                for p in 0..2 {
                    let mut seen = std::collections::BTreeMap::<u16, i64>::new();
                    for &d in t.kept.as_ref().unwrap()[p].iter().chain(&t.drawn[p]).chain(&t.tutored[p]) {
                        *seen.entry(d).or_default() += 1;
                    }
                    for &id in &s.players[p].hand {
                        let def = s.objects.get(id).card_def;
                        let n = seen.entry(def).or_default();
                        *n -= 1;
                        assert!(*n >= 0, "seed {seed}: {} in hand was never seen", CARD_DEFS[def as usize].name);
                    }
                }
            }
            let k = rng.below(game.num_choices() as u64) as usize;
            game.play(k).unwrap();
        }
        let t = game.tracker().unwrap();
        let m = game.mulligans();
        for p in 0..2 {
            let kept = &t.kept.as_ref().unwrap()[p];
            assert_eq!(kept.len(), 7 - usize::from(m[p]));
            // the deal is seven cards, the kept hand among them, the rest bottomed
            assert_eq!(t.deal[p].len(), 7);
            let mut rest = t.deal[p].clone();
            for d in kept {
                let i = rest.iter().position(|x| x == d).expect("a kept card was dealt");
                rest.remove(i);
            }
            assert_eq!(rest.len(), usize::from(m[p]));
            mulligans += usize::from(m[p]);
        }
    }
    assert!(mulligans > 0);
}

#[test]
fn resume_refuses_a_file_from_another_run() {
    let dir = scratch("resume-mismatch");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let mut args = PlayArgs {
        decks_dir: decks_dir(),
        pairs_file: dir.join("pairs.tsv"),
        bot1: "random".into(),
        bot2: "first".into(),
        seed: 3,
        threads: 1,
        out: dir.join("games.jsonl"),
        n_pairs: Some(1),
        pair_range: None,
        shard: None,
        resume: false,
        data_out: None,
        limits: Default::default(),
    };
    run_play(&args).unwrap();
    args.resume = true;
    args.bot2 = "greedy1".into();
    assert!(run_play(&args).unwrap_err().contains("was played as"));
    args.bot2 = "first".into();
    args.seed = 4;
    assert!(run_play(&args).unwrap_err().contains("env_seed"));
    args.seed = 3;
    // the same stems in the other seats
    std::fs::write(dir.join("pairs.tsv"), format!("{WG}\t{UG}\n")).unwrap();
    assert!(run_play(&args).unwrap_err().contains("was played as"));
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    assert_eq!(run_play(&args).unwrap()["games"], 2); // nothing left to play
}

#[test]
fn an_alias_is_a_mirror() {
    let dir = scratch("mirror-alias");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let s = run_play(&play_args(&dir, "mcts:2", "heuristic@2", 8, Some(1))).unwrap();
    assert_eq!(s["config"]["mirror"], true);
    let g = lines(&dir.join("games.jsonl"));
    let env: Vec<u64> = g.iter().map(|r| r["env_seed"].as_u64().unwrap()).collect();
    assert_eq!(env.iter().max().unwrap() - env.iter().min().unwrap(), 500_000, "{env:?}");
}

#[test]
fn a_locked_output_is_refused() {
    let dir = scratch("lock");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let args = play_args(&dir, "random", "first", 1, Some(1));
    let held = std::fs::OpenOptions::new().create(true).append(true).open(&args.out).unwrap();
    held.lock().unwrap();
    assert!(run_play(&args).unwrap_err().contains("locked"));
    drop(held);
    assert_eq!(run_play(&args).unwrap()["games"], 2);
}

#[test]
fn summary_counts_a_repeated_game_once_and_splits_exactly() {
    let rec = |pair: usize, swap: bool, role: Option<&str>, error: Option<&str>| -> SummaryRec {
        serde_json::from_value(serde_json::json!({
            "pair": pair, "swap": swap, "bot1": "x", "bot2": "y", "winner_role": role, "error": error,
            "winner": role.map(|r| if (r == "bot1") != swap { "A" } else { "B" }),
            "halt_step": error.map(|_| serde_json::json!({"seat": "B", "role": "bot2", "forced": false})),
            "terminal": {"classification": if error.is_some() { "halted" } else { "natural" }},
            "turns": 10, "seconds": 1.0,
        }))
        .unwrap()
    };
    let mut s = Summary::default();
    for r in [
        rec(0, false, Some("bot1"), None),
        rec(0, true, Some("bot2"), None), // split
        rec(1, false, None, None),
        rec(1, true, None, None), // two draws: not a split
        rec(2, false, Some("bot1"), None),
        rec(2, true, Some("bot1"), None),
        rec(3, false, None, Some("halted: fail_closed: x")),
        rec(3, true, Some("bot1"), None),
    ] {
        assert!(s.add(&r));
    }
    assert!(!s.add(&rec(0, false, Some("bot2"), None)), "a repeated (pair, swap) is ignored");
    let v = s.finish();
    assert_eq!(v["games"], 8);
    assert_eq!(v["duplicates_ignored"], 1);
    assert_eq!(v["valid_games"], 7);
    assert_eq!(v["pairs_split"], 1);
    assert_eq!(v["pairs_both_won_by_bot1"], 1);
    assert_eq!(v["halts"]["by_role_of_last_step"]["bot2"], 1);
    assert_eq!(v["terminals"]["halted"], 1);
    assert!((v["bot1_score"].as_f64().unwrap() - 5.0 / 7.0).abs() < 1e-12);
}

#[test]
fn resume_counts_a_duplicated_record_once() {
    let dir = scratch("dups");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let mut args = play_args(&dir, "random", "first", 2, Some(2));
    run_play(&args).unwrap();
    let text = std::fs::read_to_string(&args.out).unwrap();
    let first = text.lines().next().unwrap().to_string();
    std::fs::write(&args.out, format!("{text}{first}\n")).unwrap();
    args.resume = true;
    let s = run_play(&args).unwrap();
    assert_eq!(s["games"], 4);
    assert_eq!(s["duplicates_ignored"], 1);
}

#[test]
fn pair_range_past_the_pairs_file_plays_and_shards_merge() {
    let dir = scratch("range");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let mut args = play_args(&dir, "random", "first", 1, None);
    args.pair_range = Some((3, 5));
    assert_eq!(run_play(&args).unwrap()["games"], 4);
    let pairs: Vec<u64> = lines(&args.out).iter().map(|r| r["pair"].as_u64().unwrap()).collect();
    assert!(pairs.iter().all(|&p| p == 3 || p == 4), "{pairs:?}");
    // an empty selection is an error, not an empty run
    args.pair_range = Some((5, 5));
    args.out = dir.join("empty.jsonl");
    assert!(run_play(&args).unwrap_err().contains("no games"));
    // shards: their own summaries, and `summarize` over both is the whole run
    let mut shard_files = Vec::new();
    for i in 0..2 {
        let mut a = play_args(&dir, "random", "first", 4, Some(3));
        a.shard = Some((i, 2));
        a.out = dir.join(format!("s{i}/games.jsonl"));
        run_play(&a).unwrap();
        assert!(dir.join(format!("s{i}/summary.shard-{i}-of-2.json")).exists());
        shard_files.push(a.out);
    }
    let whole =
        run_play(&PlayArgs { out: dir.join("whole/games.jsonl"), ..play_args(&dir, "random", "first", 4, Some(3)) })
            .unwrap();
    let merged = run_summarize(&shard_files).unwrap();
    assert_eq!(merged["games"], 6);
    assert_eq!(merged["bot1_score"], whole["bot1_score"]);
}

#[test]
fn a_halted_game_names_its_last_step() {
    // env_seed 1746 (`--seed 1`, pair 746 of the fixture pairs): uniform play including mulligans reaches the
    // observation layer's Ward fail-close.
    let (ug, wg) = fixture();
    let spec = BotSpec::parse("random,mull=random").unwrap();
    let t = GameTask { pair: 746, swap: false, deck1: UG.into(), deck2: WG.into(), game_seed: 1746, env_seed: 1746 };
    let (rec, _) = play_game(&t, &ug, &wg, ("random,mull=random", &spec), ("random,mull=random", &spec));
    let e = rec.error.clone().unwrap_or_default();
    assert!(e.starts_with("halted:") && e.contains("Ward"), "{e}");
    let h = rec.halt_step.clone().expect("a halt step");
    assert!(h.seat == "A" || h.seat == "B");
    assert_eq!(h.role, if h.seat == "A" { "bot1" } else { "bot2" });
    let s = summarize(&[rec]);
    assert_eq!(s["valid_games"], 0);
    assert_eq!(s["halts"]["by_role_of_last_step"][h.role.as_str()], 1);
}
