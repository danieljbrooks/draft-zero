package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/adams-shaun/gorge/internal/policynet"
)

// runGonetCheck is the hidden subcommand gonet-check: the Go network against
// PyTorch's outputs on the same states (python -m dzg.gonet_parity), and its speed.
//
//	dzgorge gonet-check -net best.dzgw -parity parity.json [-tol 1e-3] [-bench 2000]
func runGonetCheck(args []string) int {
	fs := flag.NewFlagSet("gonet-check", flag.ExitOnError)
	netPath := fs.String("net", "", ".dzgw network (python -m dzg.export)")
	parity := fs.String("parity", "", "parity JSON (python -m dzg.gonet_parity)")
	tol := fs.Float64("tol", 1e-3, "largest allowed |Go - PyTorch| on option logits and values")
	bench := fs.Int("bench", 0, "also time this many evaluations of the parity states (single goroutine)")
	fs.Parse(args)
	if *netPath == "" || *parity == "" {
		fmt.Fprintln(os.Stderr, "gonet-check: -net and -parity are required")
		return 2
	}
	gonetCache, gonetCardCache = 0, 0 // parity runs every forward from scratch
	t0 := time.Now()
	g, err := loadGoNet(*netPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	load := time.Since(t0)
	recs, err := readParity(*parity)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	sc := new(gnScratch)
	var maxScore, maxLogit, maxValue float64
	var worstScore, worstValue, nopt int
	for i, r := range recs {
		scores := make([]float32, len(r.opts))
		logit := g.eval(sc, &r.st, r.opts, scores)
		if d := math.Abs(float64(logit) - r.valueLogit); d > maxLogit {
			maxLogit = d
		}
		if d := math.Abs(float64(sigmoid32(logit)) - r.value); d > maxValue {
			maxValue, worstValue = d, i
		}
		for j := range scores {
			if d := math.Abs(float64(scores[j]) - r.scores[j]); d > maxScore {
				maxScore, worstScore = d, i
			}
		}
		nopt += len(scores)
		// the public path (cache off here) answers the same
		if v := g.Value(r.st); v != sigmoid32(logit) {
			fmt.Fprintf(os.Stderr, "gonet-check: record %d: Value %v != eval %v\n", i, v, sigmoid32(logit))
			return 1
		}
	}
	ok := maxScore < *tol && maxValue < *tol
	out := map[string]any{"net": *netPath, "states": len(recs), "options": nopt, "load_ms": load.Seconds() * 1000,
		"max_abs_logit_diff": maxScore, "max_abs_value_diff": maxValue, "max_abs_value_logit_diff": maxLogit,
		"worst_logit_record": recs[worstScore].index, "worst_value_record": recs[worstValue].index,
		"tol": *tol, "pass": ok}
	if *bench > 0 {
		for k, v := range gonetBench(g, recs, *bench) {
			out[k] = v
		}
	}
	js, _ := json.MarshalIndent(out, "", " ")
	fmt.Println(string(js))
	if !ok {
		return 1
	}
	return 0
}

// gonetBench times n evaluations cycling over recs on one goroutine: Score (the trunk,
// the value head and every option) and Value (the trunk and the value head), first
// with no card cache, then with a warm one (a search's states share most cards).
func gonetBench(g *goNet, recs []parityRec, n int) map[string]any {
	sc := new(gnScratch)
	scores := make([]float32, 64)
	run := func(withOpts bool) float64 {
		t0 := time.Now()
		for i := 0; i < n; i++ {
			r := &recs[i%len(recs)]
			if withOpts {
				g.eval(sc, &r.st, r.opts, scores[:len(r.opts)])
			} else {
				g.eval(sc, &r.st, nil, nil)
			}
		}
		return float64(time.Since(t0).Microseconds()) / float64(n)
	}
	out := map[string]any{"bench_evals": n}
	run(true) // warm up
	g.cards, g.pas, g.pbs = nil, nil, nil
	out["us_per_score_cold_cards"] = run(true)
	out["us_per_value_cold_cards"] = run(false)
	g.cards, g.pas, g.pbs = newCardCache(1<<14, g.cd+g.d), newCardCache(1<<14, g.d), newCardCache(1<<14, g.d)
	run(true)
	out["us_per_score_warm_cards"] = run(true)
	out["us_per_value_warm_cards"] = run(false)
	var cards, opts int
	for _, r := range recs {
		cards += len(r.st.Cards)
		opts += len(r.opts)
	}
	out["mean_cards"] = float64(cards) / float64(len(recs))
	out["mean_options"] = float64(opts) / float64(len(recs))
	return out
}

type parityRec struct {
	index      int
	st         policynet.State
	opts       []policynet.Option
	scores     []float64
	value      float64
	valueLogit float64
}

func readParity(path string) ([]parityRec, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Records []struct {
			Index  int          `json:"index"`
			Dense  []float32    `json:"dense"`
			Sparse [][2]float64 `json:"sparse"`
			Cards  []struct {
				Group uint8        `json:"group"`
				Raw   []float32    `json:"raw"`
				Rows  [][2]float64 `json:"rows"`
			} `json:"cards"`
			Options []struct {
				Slots  [][2]float64 `json:"slots"`
				Hashed [][2]float64 `json:"hashed"`
				Dense  []float32    `json:"dense"`
				Bot    int          `json:"bot"`
				EA     int32        `json:"ea"`
				EB     int32        `json:"eb"`
			} `json:"options"`
			Scores     []float64 `json:"scores"`
			Value      float64   `json:"value"`
			ValueLogit float64   `json:"value_logit"`
		} `json:"records"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	feats := func(ps [][2]float64) []policynet.Feature {
		out := make([]policynet.Feature, len(ps))
		for i, p := range ps {
			out[i] = policynet.Feature{Row: uint16(p[0]), Value: float32(p[1])}
		}
		return out
	}
	var recs []parityRec
	for _, r := range doc.Records {
		pr := parityRec{index: r.Index, scores: r.Scores, value: r.Value, valueLogit: r.ValueLogit}
		pr.st = policynet.State{Dense: r.Dense, Sparse: feats(r.Sparse)}
		for _, c := range r.Cards {
			pr.st.Cards = append(pr.st.Cards, policynet.EntityCard{Group: c.Group, Raw: c.Raw, Rows: feats(c.Rows)})
		}
		for _, o := range r.Options {
			pr.opts = append(pr.opts, policynet.Option{Slots: feats(o.Slots), Hashed: feats(o.Hashed), Dense: o.Dense,
				BotPick: o.Bot != 0, EntA: o.EA, EntB: o.EB})
		}
		if len(pr.scores) != len(pr.opts) {
			return nil, fmt.Errorf("%s: record %d: %d scores for %d options", path, r.Index, len(pr.scores), len(pr.opts))
		}
		recs = append(recs, pr)
	}
	if len(recs) == 0 {
		return nil, fmt.Errorf("%s: no records", path)
	}
	return recs, nil
}
