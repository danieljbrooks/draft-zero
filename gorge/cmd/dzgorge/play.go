package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/adams-shaun/gorge/botpolicy"
	"github.com/adams-shaun/gorge/bots"
	"github.com/adams-shaun/gorge/cards"
	"github.com/adams-shaun/gorge/decision"
	"github.com/adams-shaun/gorge/events"
	"github.com/adams-shaun/gorge/internal/azmcts"
	gbench "github.com/adams-shaun/gorge/internal/bench"
	"github.com/adams-shaun/gorge/internal/policynet"
	"github.com/adams-shaun/gorge/internal/searchseat"
	"github.com/adams-shaun/gorge/internal/spellbench/builtins"
	"github.com/adams-shaun/gorge/rules"
	"github.com/adams-shaun/gorge/seat"
	"github.com/adams-shaun/gorge/state"
	"github.com/adams-shaun/gorge/view"
)

// gameRec is one game, one JSON line of -out.
type gameRec struct {
	Game    int       `json:"game"`
	Pair    int       `json:"pair"`
	Leg     int       `json:"leg"` // 0: A in seat 0; 1: the policies swapped
	Seed    uint64    `json:"seed"`
	Decks   [2]string `json:"decks"`
	Side    [2]string `json:"side"` // "A" or "B" per seat
	Pols    [2]string `json:"pols"`
	Winner  int       `json:"winner"` // the winning seat, -1 for a draw, stall or error
	Result  string    `json:"result"` // "A", "B", "draw", "stall" or "error"
	Stall   string    `json:"stall,omitempty"`
	Turns   int32     `json:"turns"`
	Intents int       `json:"intents"`
	Starter int       `json:"starter"`
	MS      float64   `json:"ms"`
	// InHand is every card each seat had in hand: its kept opening hand and every card it
	// drew or put into its hand from its library (17lands' "game in hand").
	InHand [2][]string `json:"in_hand"`
	Mull   [2]int      `json:"mull"`
	Visits int         `json:"visits,omitempty"` // visit records this game added to -corpus
	Err    string      `json:"error,omitempty"`
}

type job struct {
	game, pair, leg int
	seed            uint64
	decks           [2]string
	sides           [2]int // index into specs per seat
}

type azAgg struct {
	asked, searched, sims, refused, noWorld, allFailed, priorFallbacks atomic.Int64
	msX1000                                                            atomic.Int64
}

var azStats azAgg

func runPlay(args []string) int {
	fs := flag.NewFlagSet("play", flag.ExitOnError)
	cardsDir := fs.String("cards", ".cards", "gorge corpus directory")
	decksDir := fs.String("decks", "", "pool deck directory (gorge/decks.py)")
	poolPath := fs.String("pool", "", "pool.tsv (gorge/decks.py)")
	split := fs.String("split", "eval", "pool split the decks come from: train, eval or all")
	onlyPath := fs.String("only", "", "optional list of allowed decks (dzgorge coverage -out)")
	aSpec := fs.String("a", "bot", "policy A (see policy.go)")
	bSpec := fs.String("b", "bot", "policy B")
	pairs := fs.Int("pairs", 100, "deck pairs; each is played twice (policies swapped) unless -a == -b or -single")
	single := fs.Bool("single", false, "play each pair once, A in seat 0")
	firstPair := fs.Int("first-pair", 0, "index of the first pair (to extend a run with fresh pairs on the same seed)")
	seed := fs.Uint64("seed", 1, "base seed: the deck pairs and every game's shuffles derive from it")
	workers := fs.Int("workers", 0, "games in parallel (0 = GOMAXPROCS)")
	out := fs.String("out", "", "games JSONL (required)")
	corpus := fs.String("corpus", "", "write every decision an az seat searched as a gzip visit corpus (policytrain -visits-corpus)")
	maxTurns := fs.Int("max-turns", 100, "turn cap (gorge counts both players' turns); a capped game is a stall")
	maxIntents := fs.Int("max-intents", 20000, "decision cap per game; a capped game is a stall")
	// Off by default: gorge's bot answers keep/mulligan with a fixed 1/3 chance of a mulligan,
	// whatever the hand holds (botpolicy/policy.go), and every seat here delegates that ask to it.
	mulligans := fs.Int("mulligans", 0, "London mulligans each player may take (0 skips the round)")
	progress := fs.Int("progress", 0, "print progress every N games (0 = about 20 times)")
	fs.Parse(args)
	if *out == "" {
		fmt.Fprintln(os.Stderr, "play: -out is required")
		return 2
	}
	specs := make([]*policySpec, 2)
	for i, s := range []string{*aSpec, *bSpec} {
		p, err := parsePolicy(s)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
		specs[i] = p
	}
	reg, err := cards.SharedCorpus(*cardsDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	pool, err := readPool(*poolPath, *split)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if *onlyPath != "" {
		raw, err := os.ReadFile(*onlyPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		ok := map[string]bool{}
		for _, l := range bytes.Split(raw, []byte("\n")) {
			ok[string(bytes.TrimSpace(l))] = true
		}
		kept := pool[:0]
		for _, p := range pool {
			if ok[p.Name] {
				kept = append(kept, p)
			}
		}
		pool = kept
	}
	if len(pool) < 2 {
		fmt.Fprintf(os.Stderr, "play: %d decks in split %q\n", len(pool), *split)
		return 1
	}
	legs := 2
	if *aSpec == *bSpec || *single {
		legs = 1
	}
	// Deck pairs and seeds: a pure function of -seed and the pair index.
	var jobs []job
	for p := *firstPair; p < *firstPair+*pairs; p++ {
		r := rand.New(rand.NewPCG(*seed, uint64(p)))
		i := r.IntN(len(pool))
		j := r.IntN(len(pool) - 1)
		if j >= i {
			j++
		}
		ps := r.Uint64()
		for leg := 0; leg < legs; leg++ {
			jb := job{game: len(jobs), pair: p, leg: leg, seed: ps, decks: [2]string{pool[i].Name, pool[j].Name}, sides: [2]int{0, 1}}
			if leg == 1 {
				jb.sides = [2]int{1, 0}
			}
			jobs = append(jobs, jb)
		}
	}
	dc := &deckCache{reg: reg, dir: *decksDir, m: map[string][]*cards.Card{}, names: map[*cards.Card]string{}}
	for _, jb := range jobs {
		for _, d := range jb.decks {
			if _, err := dc.get(d); err != nil {
				fmt.Fprintln(os.Stderr, err)
				return 1
			}
		}
	}
	nw := *workers
	if nw <= 0 {
		nw = runtime.GOMAXPROCS(0)
	}
	t0 := time.Now()
	azmcts.Millis = func() float64 { return float64(time.Since(t0).Microseconds()) / 1000 }
	azmcts.Watch = func(d azmcts.Diag) {
		if d.Kind == "" {
			return
		}
		azStats.asked.Add(1)
		azStats.refused.Add(int64(d.Stats.RedealRefused))
		azStats.noWorld.Add(int64(d.Stats.NoWorld))
		azStats.allFailed.Add(int64(d.Stats.AllFailed))
		azStats.priorFallbacks.Add(int64(d.Stats.PriorFallbacks))
		if d.Searched {
			azStats.searched.Add(1)
			azStats.sims.Add(int64(d.Stats.Completed))
			azStats.msX1000.Add(int64(d.MS * 1000))
		}
	}
	outF, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer outF.Close()
	w := bufio.NewWriter(outF)
	var corpusF *os.File
	if *corpus != "" {
		if corpusF, err = os.Create(*corpus); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer corpusF.Close()
	}
	every := *progress
	if every <= 0 {
		every = max(1, len(jobs)/20)
	}
	var mu sync.Mutex
	var tally struct {
		done, a, b, draw, stall, errs, visits int
		turns                                 int64
	}
	ch := make(chan job)
	var wg sync.WaitGroup
	for k := 0; k < nw; k++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for jb := range ch {
				rec, member := playOne(jb, specs, dc, reg, *maxTurns, *maxIntents, *mulligans, *corpus != "")
				line, _ := json.Marshal(rec)
				mu.Lock()
				w.Write(line)
				w.WriteByte('\n')
				if member != nil {
					if _, err := corpusF.Write(member); err != nil {
						panic(err)
					}
				}
				tally.done++
				tally.visits += rec.Visits
				switch rec.Result {
				case "A":
					tally.a++
				case "B":
					tally.b++
				case "draw":
					tally.draw++
				case "stall":
					tally.stall++
				default:
					tally.errs++
				}
				tally.turns += int64(rec.Turns)
				if tally.done%every == 0 || tally.done == len(jobs) {
					el := time.Since(t0).Seconds()
					fmt.Fprintf(os.Stderr, "[%d/%d %.0fs] %.1f games/s  A %d  B %d  draw %d  stall %d  error %d  A score %.3f\n",
						tally.done, len(jobs), el, float64(tally.done)/el, tally.a, tally.b, tally.draw, tally.stall, tally.errs,
						score(tally.a, tally.b, tally.draw+tally.stall))
				}
				mu.Unlock()
			}
		}()
	}
	for _, jb := range jobs {
		ch <- jb
	}
	close(ch)
	wg.Wait()
	if err := w.Flush(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	el := time.Since(t0).Seconds()
	sum := map[string]any{
		"a":                         specs[0].Raw,
		"b":                         specs[1].Raw,
		"split":                     *split,
		"pairs":                     *pairs,
		"first_pair":                *firstPair,
		"legs":                      legs,
		"seed":                      *seed,
		"workers":                   nw,
		"games":                     tally.done,
		"a_wins":                    tally.a,
		"b_wins":                    tally.b,
		"draws":                     tally.draw,
		"stalls":                    tally.stall,
		"errors":                    tally.errs,
		"a_score":                   score(tally.a, tally.b, tally.draw+tally.stall),
		"wall_s":                    el,
		"games_per_hour":            float64(tally.done) / el * 3600,
		"games_per_hour_per_worker": float64(tally.done) / el * 3600 / float64(nw),
		"mean_turns":                float64(tally.turns) / float64(max(1, tally.done)),
		"visit_records":             tally.visits,
		"max_turns":                 *maxTurns,
		"mulligans":                 *mulligans,
	}
	if n := azStats.searched.Load(); n > 0 {
		sum["az_decisions_asked"] = azStats.asked.Load()
		sum["az_decisions_searched"] = n
		sum["az_ms_per_searched"] = float64(azStats.msX1000.Load()) / 1000 / float64(n)
		sum["az_completed_sims_per_searched"] = float64(azStats.sims.Load()) / float64(n)
	}
	if azStats.asked.Load() > 0 {
		sum["az_redeal_refused"] = azStats.refused.Load()
		sum["az_no_world_sims"] = azStats.noWorld.Load()
		sum["az_all_failed"] = azStats.allFailed.Load()
		sum["az_prior_fallbacks"] = azStats.priorFallbacks.Load()
	}
	js, _ := json.MarshalIndent(sum, "", " ")
	if err := os.WriteFile(*out+".summary.json", js, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	fmt.Println(string(js))
	return 0
}

// score is A's share of the points, a draw or stall counting half.
func score(a, b, half int) float64 {
	n := a + b + half
	if n == 0 {
		return 0
	}
	return (float64(a) + 0.5*float64(half)) / float64(n)
}

var sparePool sync.Pool

func playOne(jb job, specs []*policySpec, dc *deckCache, reg *cards.Registry, maxTurns, maxIntents, mulligans int, record bool) (gameRec, []byte) {
	rec := gameRec{Game: jb.game, Pair: jb.pair, Leg: jb.leg, Seed: jb.seed, Decks: jb.decks, Winner: -1}
	seats := make([]seat.Seat, 2)
	search := false
	var vrecs []policynet.VisitRecord
	for s := 0; s < 2; s++ {
		sp := specs[jb.sides[s]]
		rec.Side[s] = string("AB"[jb.sides[s]])
		rec.Pols[s] = sp.Raw
		st, err := sp.seat(jb.seed ^ uint64(s+1))
		if err != nil {
			rec.Result, rec.Err = "error", err.Error()
			return rec, nil
		}
		seats[s] = st
		if sp.searches() {
			search = true
		}
		if az, ok := st.(*azmcts.Seat); ok && record && sp.Kind == "az" {
			opp := specs[jb.sides[1-s]].Raw
			pairKey := jb.decks[0] + "|" + jb.decks[1]
			gid := fmt.Sprintf("p%07dg%d", jb.pair, jb.leg)
			az.SetRecorder(func(r policynet.VisitRecord) {
				r.GameID, r.Deck, r.Seed, r.Opponent = gid, pairKey, jb.seed, opp
				vrecs = append(vrecs, r)
			})
		}
	}
	d0, _ := dc.get(jb.decks[0])
	d1, _ := dc.get(jb.decks[1])
	cfg := rules.Config{
		Seed: jb.seed, Names: []string{"p0", "p1"}, Decks: [][]*cards.Card{d0, d1},
		Tokens: reg.Tokens, NameUniverse: reg.Cards, Mulligans: mulligans,
	}
	spare, _ := sparePool.Get().(*rules.Spare)
	if spare == nil {
		spare = new(rules.Spare)
	}
	cfg.Spare = spare
	hooks := gbench.Hooks{
		Setup: func(e *rules.Engine) {
			// The decision arena recycles decision storage; a search seat reads the live
			// engine as its root, so a game with one keeps it off (as botbench does).
			if !search {
				e.SetDecisionArena(true)
			}
			for _, st := range seats {
				if b, ok := st.(*builtins.Seat); ok {
					b.SetPlanner(e)
				}
			}
		},
		Decision: func(seatIdx int, d *decision.Decision, in decision.Intent, _ *botpolicy.Board) error {
			if d.Kind == decision.KMulligan && len(d.Options) > 0 && d.Options[0].Kind != "bottom" &&
				len(in.Choices) > 0 && optionKind(d, in.Choices[0]) == "mulligan" {
				rec.Mull[seatIdx]++
			}
			return nil
		},
		Submit: submitWithFallback(seats),
	}
	t0 := time.Now()
	o, e, err := gbench.PlayGame(cfg, seats, maxTurns, maxIntents, hooks)
	rec.MS = float64(time.Since(t0).Microseconds()) / 1000
	rec.Turns, rec.Intents, rec.Starter = o.Turns, o.Intents, o.Starter
	switch {
	case err != nil:
		rec.Result, rec.Err = "error", err.Error()
	case gbench.IsAbort(o.StallOn):
		rec.Result, rec.Stall, rec.Err = "error", o.StallOn, firstLine(o.Livelock)
	case o.IsStalled():
		rec.Result, rec.Stall = "stall", o.StallOn
	case o.Draw:
		rec.Result = "draw"
	default:
		rec.Winner = o.WinnerSeat
		rec.Result = rec.Side[o.WinnerSeat]
	}
	if e != nil {
		rec.InHand = inHand(e, dc)
	}
	var member []byte
	if len(vrecs) > 0 {
		member = corpusMember(vrecs, o, err)
		rec.Visits = len(vrecs)
	}
	if err == nil && e != nil && !gbench.IsAbort(o.StallOn) {
		*spare = e.Release()
		sparePool.Put(spare)
	}
	return rec, member
}

func optionKind(d *decision.Decision, idx int) string {
	for _, o := range d.Options {
		if o.Index == idx {
			return o.Kind
		}
	}
	return ""
}

func firstLine(s string) string {
	if i := bytes.IndexByte([]byte(s), '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// inHand folds the event log into each seat's cards seen in hand: every card moved
// from its owner's library to its hand, minus the cards returned to the library
// before the first turn (a mulligan's shuffle and London bottoming).
func inHand(e *rules.Engine, dc *deckCache) [2][]string {
	var held [2]map[state.ObjID]bool
	held[0], held[1] = map[state.ObjID]bool{}, map[state.ObjID]bool{}
	started := false
	for _, ev := range e.L.Events {
		switch ev.Kind {
		case events.TurnChange:
			started = true
		case events.Draw, events.MoveZone:
			o := e.G.Obj(ev.Obj)
			if o == nil || o.Card == nil || o.Owner > 1 {
				continue
			}
			if ev.To == state.ZHand && (ev.Kind == events.Draw || ev.From == state.ZLibrary) {
				held[o.Owner][ev.Obj] = true
			} else if !started && ev.From == state.ZHand && ev.To == state.ZLibrary {
				delete(held[o.Owner], ev.Obj)
			}
		}
	}
	var out [2][]string
	for p := 0; p < 2; p++ {
		for id := range held[p] {
			out[p] = append(out[p], dc.cardName(e.G.Obj(id).Card))
		}
		sort.Strings(out[p])
	}
	return out
}

// submitWithFallback keeps a builtin seat's refused answer from halting the game:
// the builtin's own refusal answer, then gorge's generic fallbacks (botbench's
// sbSubmitWithFallback).
func submitWithFallback(seats []seat.Seat) func(*rules.Engine, int, *decision.Decision, decision.Intent) (bool, error) {
	return func(e *rules.Engine, seatIdx int, d *decision.Decision, in decision.Intent) (bool, error) {
		err := e.Submit(in)
		if err == nil {
			return true, nil
		}
		b, ok := seats[seatIdx].(*builtins.Seat)
		if !ok {
			return true, err
		}
		v := view.Project(e.G, e, d.Player, d)
		v.Round = view.RoundOf(e.G, e.L.Events)
		if e.Submit(b.Refused(v, *d, in)) == nil {
			return true, nil
		}
		brd := botpolicy.BoardFromGame(e.G, e, d.Player)
		var err3 error
		for _, fb := range bots.Fallbacks(d, brd, seatIdx) {
			if err3 = e.Submit(fb); err3 == nil {
				return true, nil
			}
		}
		return true, fmt.Errorf("answer refused (%v); fallbacks refused (%v)", err, err3)
	}
}

// corpusMember stamps each visit record with its seat's result and encodes the game's
// records as one gzip member (members concatenate into the corpus file).
func corpusMember(recs []policynet.VisitRecord, o gbench.Outcome, err error) []byte {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	enc := json.NewEncoder(zw)
	for i := range recs {
		r := &recs[i]
		switch {
		case err != nil || gbench.IsAbort(o.StallOn) || o.IsStalled():
		case o.Draw:
			r.Outcome, r.OutcomeKnown = 0.5, true
		case o.WinnerSeat == r.Seat:
			r.Outcome, r.OutcomeKnown = 1, true
		default:
			r.Outcome, r.OutcomeKnown = 0, true
		}
		if e := enc.Encode(r); e != nil {
			panic(e)
		}
	}
	if e := zw.Close(); e != nil {
		panic(e)
	}
	return buf.Bytes()
}

var _ searchseat.SearchSeat = (*azmcts.Seat)(nil)
