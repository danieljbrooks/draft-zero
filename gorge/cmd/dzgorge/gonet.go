package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/maphash"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/adams-shaun/gorge/internal/policynet"
)

// An in-process network (the policy key gonet=PATH): dzg's mlp (gorge/dzg/models.py
// MLPNet) evaluated in Go from a .dzgw file (python -m dzg.export), as a
// policynet.External, in place of a socket round trip to python -m dzg.serve. The
// forward pass is MLPNet.forward in eval mode for one state and its options:
//
//	card    c = relu(card_phi(relu(card_raw(raw) + card_rows(bag(identity rows)))))
//	pools   per group (0..3) the 0.25-scaled sum and the max of c (0 for an empty group)
//	state   s = dense_in(dense) + sparse_proj(bag(sparse)) + pool_proj(pools),
//	        then x + fc2(relu(fc1(LN(x)))) per block, then LN
//	value   sigmoid(value MLP(s))
//	option  o = LN(bag(slots) + hproj(bag(hashed)) + dense(opt dense) + bot·bot_vec
//	              + ea_proj(c_ea or none_a) + eb_proj(c_eb or none_b))
//	score   policy MLP([s ‖ o]), a logit
//
// Every game in the process shares one network per path. Answers are cached by the
// encoded state like remote.go's (a search revisits positions), and card vectors by
// the card's encoding (a search's states share most of their cards). Plain float32
// loops; scratch buffers come from a sync.Pool, so the hot path allocates only the
// returned score slice.

// gonet knobs (play's -gonet-* flags).
var (
	gonetCache     = 1 << 20 // cached evaluations (values and score lists); 0 = none
	gonetCardCache = 1 << 14 // cached card vectors (and their projections); 0 = none
)

var (
	gonetsMu sync.Mutex
	gonets   = map[string]*goNet{}
)

// gonetModel is the shared network at path wrapped as a policynet.Model encoding under
// the entity feature set.
func gonetModel(path string) (*policynet.Model, error) {
	g, err := sharedGoNet(path)
	if err != nil {
		return nil, err
	}
	return policynet.NewExternal(g, policynet.FeaturesEntity), nil
}

func sharedGoNet(path string) (*goNet, error) {
	gonetsMu.Lock()
	defer gonetsMu.Unlock()
	g, ok := gonets[path]
	if !ok {
		var err error
		if g, err = loadGoNet(path); err != nil {
			return nil, err
		}
		gonets[path] = g
	}
	return g, nil
}

// gonetStats sums every network's counters (for the run summary).
func gonetStats() map[string]any {
	gonetsMu.Lock()
	defer gonetsMu.Unlock()
	if len(gonets) == 0 {
		return nil
	}
	var calls, hits, evals, cardHits, cardMiss, ns, scoreCalls, scoreHits, scoreSeen, evalOpts, evalCards int64
	for _, g := range gonets {
		evalOpts += g.evalOpts.Load()
		evalCards += g.evalCards.Load()
		scoreCalls += g.scoreCalls.Load()
		scoreHits += g.scoreHits.Load()
		scoreSeen += g.scoreSeen.Load()
		calls += g.calls.Load()
		hits += g.hits.Load()
		evals += g.evals.Load()
		cardHits += g.cardHits.Load()
		cardMiss += g.cardMiss.Load()
		ns += g.ns.Load()
	}
	out := map[string]any{"gonet_calls": calls, "gonet_cache_hits": hits, "gonet_evals": evals,
		"gonet_score_calls": scoreCalls, "gonet_score_hits": scoreHits, "gonet_score_miss_value_cached": scoreSeen}
	if evals > 0 {
		out["gonet_us_per_eval"] = float64(ns) / 1000 / float64(evals)
		out["gonet_options_per_eval"] = float64(evalOpts) / float64(evals)
		out["gonet_cards_per_eval"] = float64(evalCards) / float64(evals)
	}
	if n := cardHits + cardMiss; n > 0 {
		out["gonet_card_hit_rate"] = float64(cardHits) / float64(n)
	}
	return out
}

// ---- the weights -------------------------------------------------------------

type gnLinear struct {
	in, out int
	w       []float32 // out×in, row-major (a PyTorch Linear's weight)
	b       []float32 // out, nil without a bias
}

type gnNorm struct {
	w, b []float32
	eps  float64
}

type gnBlock struct {
	norm     gnNorm
	fc1, fc2 gnLinear
}

type goNet struct {
	path                string
	d, ff, dEmb, cd, hh int
	table               []float32 // TableRows×dEmb
	cardRaw, cardRows   gnLinear
	cardPhi             gnLinear
	denseIn, sparseProj gnLinear
	poolProj            gnLinear
	blocks              []gnBlock
	norm                gnNorm
	slots               []float32 // OptionSlotWidth×d
	hproj, odense       gnLinear
	bot                 []float32
	eaProj, ebProj      gnLinear
	noneA, noneB        []float32 // ea_proj(none_a), eb_proj(none_b)
	optNorm             gnNorm
	pol0s, pol0o        gnLinear   // policy.0's state and option halves; pol0s has the bias
	polRest             []gnLinear // policy.2, policy.4, ... (the last has out 1)
	value               []gnLinear // value.0, value.2, ... (the last has out 1)

	htab  []float32 // TableRows×d: hproj(table[r])
	pool  sync.Pool // *gnScratch
	cache *evalCache
	cards *cardCache // card entries [c ‖ W_sum(group)·c]
	pas   *cardCache // ea_proj(c)
	pbs   *cardCache // eb_proj(c)
	seed  maphash.Seed

	calls, hits, evals, cardHits, cardMiss, ns atomic.Int64
	scoreCalls, scoreHits, scoreSeen           atomic.Int64 // scoreSeen: a Score miss whose state's value was cached
	evalOpts, evalCards                        atomic.Int64
}

const (
	gnMagic   = "DZGW0001"
	gnAlign   = 64
	gnGroups  = 4
	gnPoolW   = 2 * gnGroups
	gnRows    = 16384 // policynet.TableRows
	gnSlotRow = policynet.OptionSlotWidth
)

type gnHeader struct {
	Format  string             `json:"format"`
	Arch    string             `json:"arch"`
	Config  map[string]float64 `json:"config"`
	Consts  map[string]int     `json:"consts"`
	Eps     map[string]float64 `json:"layernorm_eps"`
	Tensors []struct {
		Name   string `json:"name"`
		Shape  []int  `json:"shape"`
		Offset int64  `json:"offset"`
		NBytes int64  `json:"nbytes"`
	} `json:"tensors"`
}

func loadGoNet(path string) (*goNet, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 16 || string(raw[:8]) != gnMagic {
		return nil, fmt.Errorf("gonet %s: not a .dzgw file (python -m dzg.export)", path)
	}
	hl := int64(binary.LittleEndian.Uint64(raw[8:16]))
	if hl <= 0 || 16+hl > int64(len(raw)) {
		return nil, fmt.Errorf("gonet %s: bad header length %d", path, hl)
	}
	var h gnHeader
	if err := json.Unmarshal(raw[16:16+hl], &h); err != nil {
		return nil, fmt.Errorf("gonet %s: header: %w", path, err)
	}
	if h.Format != "dzgw-1" {
		return nil, fmt.Errorf("gonet %s: format %q, want dzgw-1", path, h.Format)
	}
	if h.Arch != "mlp" {
		return nil, fmt.Errorf("gonet %s: arch %q: only mlp has a Go forward pass", path, h.Arch)
	}
	want := map[string]int{"DENSE_W": policynet.DenseWidth, "RAW_W": policynet.EntityRawWidth,
		"OPT_DENSE_W": policynet.OptionDenseWidth, "TABLE_ROWS": gnRows, "SLOT_ROWS": gnSlotRow, "GROUPS": gnGroups}
	for k, v := range want {
		if h.Consts[k] != v {
			return nil, fmt.Errorf("gonet %s: %s = %d, this build has %d", path, k, h.Consts[k], v)
		}
	}
	start := (16 + hl + gnAlign - 1) / gnAlign * gnAlign
	tensors := map[string][]float32{}
	shapes := map[string][]int{}
	for _, t := range h.Tensors {
		n := int64(1)
		for _, d := range t.Shape {
			n *= int64(d)
		}
		if t.NBytes != 4*n || start+t.Offset+t.NBytes > int64(len(raw)) || t.Offset < 0 {
			return nil, fmt.Errorf("gonet %s: tensor %s: bad extent", path, t.Name)
		}
		b := raw[start+t.Offset : start+t.Offset+t.NBytes]
		f := make([]float32, n)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		tensors[t.Name], shapes[t.Name] = f, t.Shape
	}
	cfg := func(k string) int { return int(h.Config[k]) }
	g := &goNet{path: path, d: cfg("d"), ff: cfg("ff"), dEmb: cfg("d_emb"), cd: cfg("card_d"), hh: cfg("head_hidden"),
		seed: maphash.MakeSeed()}
	var errs []error
	get := func(name string, shape ...int) []float32 {
		t, ok := tensors[name]
		if !ok {
			errs = append(errs, fmt.Errorf("missing tensor %s", name))
			return nil
		}
		s := shapes[name]
		if len(s) != len(shape) {
			errs = append(errs, fmt.Errorf("tensor %s: shape %v, want %v", name, s, shape))
			return nil
		}
		for i := range s {
			if s[i] != shape[i] {
				errs = append(errs, fmt.Errorf("tensor %s: shape %v, want %v", name, s, shape))
				return nil
			}
		}
		return t
	}
	lin := func(name string, in, out int, bias bool) gnLinear {
		l := gnLinear{in: in, out: out, w: get(name+".weight", out, in)}
		if bias {
			l.b = get(name+".bias", out)
		}
		return l
	}
	norm := func(name string, d int) gnNorm {
		eps, ok := h.Eps[name]
		if !ok {
			errs = append(errs, fmt.Errorf("no layernorm_eps for %s", name))
		}
		return gnNorm{w: get(name+".weight", d), b: get(name+".bias", d), eps: eps}
	}
	d, e, cd, hh := g.d, g.dEmb, g.cd, g.hh
	g.table = get("table.weight", gnRows, e)
	g.cardRaw = lin("card_raw", policynet.EntityRawWidth, cd, true)
	g.cardRows = lin("card_rows", e, cd, false)
	g.cardPhi = lin("card_phi", cd, cd, true)
	g.denseIn = lin("dense_in", policynet.DenseWidth, d, true)
	g.sparseProj = lin("sparse_proj", e, d, false)
	g.poolProj = lin("pool_proj", gnPoolW*cd, d, true)
	for i := 0; i < cfg("layers"); i++ {
		p := fmt.Sprintf("blocks.%d", i)
		g.blocks = append(g.blocks, gnBlock{norm: norm(p+".norm", d), fc1: lin(p+".fc1", d, g.ff, true), fc2: lin(p+".fc2", g.ff, d, true)})
	}
	g.norm = norm("norm", d)
	g.slots = get("opt_in.slots.weight", gnSlotRow, d)
	g.hproj = lin("opt_in.hproj", e, d, false)
	g.odense = lin("opt_in.dense", policynet.OptionDenseWidth, d, true)
	g.bot = get("opt_in.bot", d)
	g.eaProj = lin("ea_proj", cd, d, true)
	g.ebProj = lin("eb_proj", cd, d, true)
	noneA, noneB := get("none_a", cd), get("none_b", cd)
	g.optNorm = norm("opt_norm", d)
	// heads: _mlp(d_in, hidden, layers): Linear at indices 0, 2, ..., 2*layers
	head := func(name string, dIn, layers int) []gnLinear {
		var out []gnLinear
		in := dIn
		for i := 0; i <= layers; i++ {
			o := hh
			if i == layers {
				o = 1
			}
			out = append(out, lin(fmt.Sprintf("%s.%d", name, 2*i), in, o, true))
			in = o
		}
		return out
	}
	pol := head("policy", 2*d, cfg("policy_layers"))
	g.value = head("value", d, cfg("value_layers"))
	if len(errs) > 0 {
		return nil, fmt.Errorf("gonet %s: %v", path, errs)
	}
	// policy.0 on [s ‖ o] = W[:, :d]·s + W[:, d:]·o + b
	p0 := pol[0]
	g.pol0s = gnLinear{in: d, out: p0.out, w: make([]float32, p0.out*d), b: p0.b}
	g.pol0o = gnLinear{in: d, out: p0.out, w: make([]float32, p0.out*d)}
	for o := 0; o < p0.out; o++ {
		copy(g.pol0s.w[o*d:(o+1)*d], p0.w[o*2*d:o*2*d+d])
		copy(g.pol0o.w[o*d:(o+1)*d], p0.w[o*2*d+d:(o+1)*2*d])
	}
	g.polRest = pol[1:]
	g.noneA, g.noneB = make([]float32, d), make([]float32, d)
	g.eaProj.apply(g.noneA, noneA)
	g.ebProj.apply(g.noneB, noneB)
	if gonetCache > 0 {
		g.cache = newEvalCache(gonetCache)
	}
	if gonetCardCache > 0 {
		g.cards = newCardCache(gonetCardCache, cd+d)
		g.pas = newCardCache(gonetCardCache, d)
		g.pbs = newCardCache(gonetCardCache, d)
	}
	// hproj(table[r]) for every row: an option's hashed rows become row gathers
	g.htab = make([]float32, gnRows*d)
	g.hproj.applyN(g.htab, g.table, gnRows, true)
	g.pool.New = func() any { return new(gnScratch) }
	return g, nil
}

// ---- policynet.External ------------------------------------------------------

// Score is the option logits for st's options.
func (g *goNet) Score(st policynet.State, opts []policynet.Option) []float32 {
	g.calls.Add(1)
	g.scoreCalls.Add(1)
	sc := g.pool.Get().(*gnScratch)
	defer g.pool.Put(sc)
	var sh, key uint64
	if g.cache != nil {
		sh = g.hashState(sc, &st)
		key = g.hashOpts(sc, sh, opts)
		if s, ok := g.cache.scores(key); ok {
			g.hits.Add(1)
			g.scoreHits.Add(1)
			return s
		}
		if _, ok := g.cache.value(sh); ok {
			g.scoreSeen.Add(1)
		}
	}
	out := make([]float32, len(opts))
	v := sigmoid32(g.eval(sc, &st, opts, out))
	if g.cache != nil {
		g.cache.put(sh, key, v, out)
	}
	return out
}

// Value is the searching seat's win probability at st.
func (g *goNet) Value(st policynet.State) float32 {
	g.calls.Add(1)
	sc := g.pool.Get().(*gnScratch)
	defer g.pool.Put(sc)
	var sh uint64
	if g.cache != nil {
		sh = g.hashState(sc, &st)
		if v, ok := g.cache.value(sh); ok {
			g.hits.Add(1)
			return v
		}
	}
	v := sigmoid32(g.eval(sc, &st, nil, nil))
	if g.cache != nil {
		g.cache.put(sh, 0, v, nil)
	}
	return v
}

// ---- the forward pass --------------------------------------------------------

type gnScratch struct {
	ent, u, bags []float32 // per card: [c ‖ W_sum(group)·c] (cd + d), card_raw out (cd), identity bag (dEmb)
	pools        []float32 // gnPoolW×cd
	sp           []float32 // dEmb
	s, t         []float32 // d
	h            []float32 // ff
	hs, vh, vh2  []float32 // head hidden
	o            []float32 // per option: d
	ho, ho2      []float32 // per option: hh
	pa, pb       []float32 // per card: ea_proj(c), eb_proj(c) (memo)
	paOK, pbOK   []bool
	miss         []int
	keys         []uint64
	buf          []byte
}

func grow(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n, n+n/4)
	}
	return s[:n]
}

func growB(s []bool, n int) []bool {
	if cap(s) < n {
		return make([]bool, n, n+n/4)
	}
	s = s[:n]
	clear(s)
	return s
}

func sigmoid32(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// eval runs the network on st, writes the option logits into scores (len(opts)) and
// returns the value logit.
func (g *goNet) eval(sc *gnScratch, st *policynet.State, opts []policynet.Option, scores []float32) float32 {
	t0 := time.Now()
	g.evals.Add(1)
	g.evalOpts.Add(int64(len(opts)))
	g.evalCards.Add(int64(len(st.Cards)))
	d, cd, e := g.d, g.cd, g.dEmb
	nc := len(st.Cards)

	// cards: per card [c ‖ W_sum(group)·c] (gnScratch.ent)
	cw := cd + d
	sc.ent = grow(sc.ent, nc*cw)
	g.cardVectors(sc, st.Cards)

	// pool_proj(pools), pools = [0.25·sum per group ‖ max per group]: the sum half is
	// linear, so it is 0.25·Σ_k W_sum(group k)·c_k from the cards' entries; the max half
	// runs on the groups that have cards (c >= 0, so an empty group's max is 0, as
	// scatter_reduce amax over a zero tensor with include_self=False leaves it)
	sc.s = grow(sc.s, d)
	s := sc.s
	copy(s, g.poolProj.b)
	sc.pools = grow(sc.pools, gnGroups*cd)
	clear(sc.pools)
	var used [gnGroups]bool
	sc.t = grow(sc.t, d)
	psum := sc.t
	clear(psum)
	for k := 0; k < nc; k++ {
		gr := int(st.Cards[k].Group)
		if gr >= gnGroups {
			continue
		}
		used[gr] = true
		axpy(psum, sc.ent[k*cw+cd:(k+1)*cw], 1)
		mx := sc.pools[gr*cd : (gr+1)*cd]
		for j, x := range sc.ent[k*cw : k*cw+cd] {
			if x > mx[j] {
				mx[j] = x
			}
		}
	}
	axpy(s, psum, 0.25)
	pw := g.poolProj.in
	for gr, ok := range used {
		if !ok {
			continue
		}
		mx := sc.pools[gr*cd : (gr+1)*cd]
		col := (gnGroups + gr) * cd
		for o := range s {
			s[o] += dot(g.poolProj.w[o*pw+col:o*pw+col+cd], mx)
		}
	}

	// state trunk
	sc.sp = grow(sc.sp, e)
	g.bag(sc.sp, st.Sparse)
	dense := fit(&sc.t, st.Dense, policynet.DenseWidth)
	g.denseIn.addTo(s, dense)
	g.sparseProj.addTo(s, sc.sp)
	sc.t = grow(sc.t, d)
	sc.h = grow(sc.h, g.ff)
	for i := range g.blocks {
		b := &g.blocks[i]
		b.norm.apply(sc.t, s)
		b.fc1.apply(sc.h, sc.t)
		relu(sc.h)
		b.fc2.addTo(s, sc.h)
	}
	g.norm.apply(s, s)

	v := g.mlp(sc, g.value, s)
	if len(opts) > 0 {
		g.options(sc, st, s, opts, scores)
	}
	g.ns.Add(int64(time.Since(t0)))
	return v
}

// mlp runs a head (Linear, ReLU, ..., Linear to 1) on x and returns the scalar.
func (g *goNet) mlp(sc *gnScratch, layers []gnLinear, x []float32) float32 {
	sc.vh = grow(sc.vh, g.hh)
	sc.vh2 = grow(sc.vh2, g.hh)
	a, b := sc.vh, sc.vh2
	in := x
	for i := range layers {
		l := &layers[i]
		if i == len(layers)-1 {
			return l.b[0] + dot(l.w[:l.in], in)
		}
		l.apply(a[:l.out], in)
		relu(a[:l.out])
		in = a[:l.out]
		a, b = b, a
	}
	return 0
}

// options writes the logit of each option.
func (g *goNet) options(sc *gnScratch, st *policynet.State, s []float32, opts []policynet.Option, scores []float32) {
	d, hh := g.d, g.hh
	nc, no := len(st.Cards), len(opts)
	sc.hs = grow(sc.hs, g.pol0s.out)
	g.pol0s.apply(sc.hs, s)
	sc.o = grow(sc.o, no*d)
	sc.pa = grow(sc.pa, nc*d)
	sc.pb = grow(sc.pb, nc*d)
	sc.paOK = growB(sc.paOK, nc)
	sc.pbOK = growB(sc.pbOK, nc)
	var odense [policynet.OptionDenseWidth]float32
	for j := range opts {
		op := &opts[j]
		o := sc.o[j*d : (j+1)*d]
		// bag(slots) + hproj(bag(hashed)) + dense(opt_dense) + bot·bot_vec
		copy(odense[:], op.Dense)
		for k := len(op.Dense); k < len(odense); k++ {
			odense[k] = 0
		}
		g.odense.apply(o, odense[:])
		for _, f := range op.Slots {
			if r := int(f.Row); r < gnSlotRow {
				axpy(o, g.slots[r*d:(r+1)*d], f.Value)
			}
		}
		for _, f := range op.Hashed { // hproj(bag(hashed)) = Σ hproj(table[row])·value
			if r := int(f.Row); r < gnRows {
				axpy(o, g.htab[r*d:(r+1)*d], f.Value)
			}
		}
		if op.BotPick {
			axpy(o, g.bot, 1)
		}
		// the referenced cards' projections (per card: memoised in this call, and cached)
		if k := int(op.EntA) - 1; k >= 0 && k < nc {
			p := sc.pa[k*d : (k+1)*d]
			if !sc.paOK[k] {
				g.cardProj(sc, &g.eaProj, g.pas, k, p)
				sc.paOK[k] = true
			}
			axpy(o, p, 1)
		} else {
			axpy(o, g.noneA, 1)
		}
		if k := int(op.EntB) - 1; k >= 0 && k < nc {
			p := sc.pb[k*d : (k+1)*d]
			if !sc.pbOK[k] {
				g.cardProj(sc, &g.ebProj, g.pbs, k, p)
				sc.pbOK[k] = true
			}
			axpy(o, p, 1)
		} else {
			axpy(o, g.noneB, 1)
		}
		g.optNorm.apply(o, o)
	}
	// policy.0: relu(hs + W_o·o), then the rest of the head, every option at once
	h0 := g.pol0o.out
	sc.ho = grow(sc.ho, no*max(h0, hh))
	sc.ho2 = grow(sc.ho2, no*max(h0, hh))
	g.pol0o.applyN(sc.ho, sc.o, no, true) // no bias: hs has policy.0's
	for j := 0; j < no; j++ {
		row := sc.ho[j*h0 : (j+1)*h0]
		for i := range row {
			row[i] += sc.hs[i]
		}
	}
	if len(g.polRest) == 0 {
		copy(scores, sc.ho[:no])
		return
	}
	a, b := sc.ho, sc.ho2
	for i := range g.polRest {
		l := &g.polRest[i]
		relu(a[:no*l.in])
		l.applyN(b, a, no, true)
		a, b = b, a
	}
	copy(scores, a[:no])
}

// cardProj writes l(c_k) into p, from cache when it has it (keyed by the card).
func (g *goNet) cardProj(sc *gnScratch, l *gnLinear, cache *cardCache, k int, p []float32) {
	if cache != nil && cache.get(sc.keys[k], p) {
		return
	}
	cw := g.cd + g.d
	l.apply(p, sc.ent[k*cw:k*cw+g.cd])
	if cache != nil {
		cache.put(sc.keys[k], p)
	}
}

// cardVectors fills sc.ent with each card's entry [c ‖ W_sum(group)·c], from the card
// cache when it has it.
func (g *goNet) cardVectors(sc *gnScratch, cards []policynet.EntityCard) {
	cd, e, d := g.cd, g.dEmb, g.d
	cw := cd + d
	nc := len(cards)
	if cap(sc.keys) < nc {
		sc.keys = make([]uint64, nc, nc+nc/4)
	}
	sc.keys = sc.keys[:nc]
	sc.miss = sc.miss[:0]
	if g.cards != nil {
		for k := range cards {
			sc.keys[k] = g.hashCard(sc, &cards[k])
			if !g.cards.get(sc.keys[k], sc.ent[k*cw:(k+1)*cw]) {
				sc.miss = append(sc.miss, k)
			}
		}
		g.cardHits.Add(int64(nc - len(sc.miss)))
		g.cardMiss.Add(int64(len(sc.miss)))
	} else {
		for k := range cards {
			sc.miss = append(sc.miss, k)
		}
	}
	if len(sc.miss) == 0 {
		return
	}
	m := len(sc.miss)
	sc.u = grow(sc.u, m*cd)
	sc.bags = grow(sc.bags, m*e)
	sc.t = grow(sc.t, m*policynet.EntityRawWidth)
	raw := sc.t
	for i, k := range sc.miss {
		c := &cards[k]
		r := raw[i*policynet.EntityRawWidth : (i+1)*policynet.EntityRawWidth]
		n := copy(r, c.Raw)
		clear(r[n:])
		g.bag(sc.bags[i*e:(i+1)*e], c.Rows)
	}
	// u = relu(card_raw(raw) + card_rows(bag)); c = relu(card_phi(u))
	g.cardRaw.applyN(sc.u, raw, m, true)
	g.cardRows.applyN(sc.u, sc.bags, m, false)
	relu(sc.u[:m*cd])
	cv := sc.bags // reuse as the output when it is large enough
	if cap(cv) < m*cd {
		cv = make([]float32, m*cd)
	}
	cv = cv[:m*cd]
	g.cardPhi.applyN(cv, sc.u, m, true)
	relu(cv)
	pw := g.poolProj.in
	for i, k := range sc.miss {
		ent := sc.ent[k*cw : (k+1)*cw]
		c := ent[:cd]
		copy(c, cv[i*cd:(i+1)*cd])
		ps := ent[cd:]
		if gr := int(cards[k].Group); gr < gnGroups {
			col := gr * cd
			for o := range ps {
				ps[o] = dot(g.poolProj.w[o*pw+col:o*pw+col+cd], c)
			}
		} else {
			clear(ps)
		}
		if g.cards != nil {
			g.cards.put(sc.keys[k], ent)
		}
	}
	sc.bags = cv[:0]
}

// bag is Σ table[row]·value over fs (rows outside the table are skipped).
func (g *goNet) bag(dst []float32, fs []policynet.Feature) {
	clear(dst)
	e := g.dEmb
	for _, f := range fs {
		if r := int(f.Row); r < gnRows {
			axpy(dst, g.table[r*e:(r+1)*e], f.Value)
		}
	}
}

// fit returns x as exactly w floats (zero-padded or truncated, as remote.go's packer
// sends it), using *buf when x has another length.
func fit(buf *[]float32, x []float32, w int) []float32 {
	if len(x) == w {
		return x
	}
	*buf = grow(*buf, w)
	b := *buf
	n := copy(b, x)
	clear(b[n:])
	return b
}

// ---- kernels -----------------------------------------------------------------

// apply: y = W·x + b.
func (l *gnLinear) apply(y, x []float32) {
	x = x[:l.in]
	y = y[:l.out]
	for o := range y {
		v := dot(l.w[o*l.in:(o+1)*l.in], x)
		if l.b != nil {
			v += l.b[o]
		}
		y[o] = v
	}
}

// addTo: y += W·x + b.
func (l *gnLinear) addTo(y, x []float32) {
	x = x[:l.in]
	y = y[:l.out]
	for o := range y {
		v := dot(l.w[o*l.in:(o+1)*l.in], x)
		if l.b != nil {
			v += l.b[o]
		}
		y[o] += v
	}
}

// applyN: for each of n inputs X[k] (stride in), Y[k] (stride out) = W·X[k] + b, or
// += W·X[k] (no bias) when set is false. Four inputs share each pass over a weight row.
func (l *gnLinear) applyN(Y, X []float32, n int, set bool) {
	in, out := l.in, l.out
	X = X[:n*in]
	Y = Y[:n*out]
	for o := 0; o < out; o++ {
		w := l.w[o*in : (o+1)*in]
		var bias float32
		if set && l.b != nil {
			bias = l.b[o]
		}
		k := 0
		for ; k+4 <= n; k += 4 {
			a0, a1, a2, a3 := dot4(w, X[k*in:(k+1)*in], X[(k+1)*in:(k+2)*in], X[(k+2)*in:(k+3)*in], X[(k+3)*in:(k+4)*in])
			if set {
				Y[k*out+o] = a0 + bias
				Y[(k+1)*out+o] = a1 + bias
				Y[(k+2)*out+o] = a2 + bias
				Y[(k+3)*out+o] = a3 + bias
			} else {
				Y[k*out+o] += a0
				Y[(k+1)*out+o] += a1
				Y[(k+2)*out+o] += a2
				Y[(k+3)*out+o] += a3
			}
		}
		for ; k < n; k++ {
			a := dot(w, X[k*in:(k+1)*in])
			if set {
				Y[k*out+o] = a + bias
			} else {
				Y[k*out+o] += a
			}
		}
	}
}

// dot is Σ a[i]·b[i] over len(a) (len(b) >= len(a)), eight partial sums.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= len(a); i += 8 {
		aa := a[i : i+8 : i+8]
		bb := b[i : i+8 : i+8]
		s0 += aa[0] * bb[0]
		s1 += aa[1] * bb[1]
		s2 += aa[2] * bb[2]
		s3 += aa[3] * bb[3]
		s4 += aa[4] * bb[4]
		s5 += aa[5] * bb[5]
		s6 += aa[6] * bb[6]
		s7 += aa[7] * bb[7]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

// dot4 is dot(w, x_k) for four x_k at once (each len(x_k) >= len(w)).
func dot4(w, x0, x1, x2, x3 []float32) (float32, float32, float32, float32) {
	n := len(w)
	x0, x1, x2, x3 = x0[:n], x1[:n], x2[:n], x3[:n]
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	for ; i+2 <= n; i += 2 {
		ww := w[i : i+2 : i+2]
		y0 := x0[i : i+2 : i+2]
		y1 := x1[i : i+2 : i+2]
		y2 := x2[i : i+2 : i+2]
		y3 := x3[i : i+2 : i+2]
		a0 += ww[0] * y0[0]
		b0 += ww[1] * y0[1]
		a1 += ww[0] * y1[0]
		b1 += ww[1] * y1[1]
		a2 += ww[0] * y2[0]
		b2 += ww[1] * y2[1]
		a3 += ww[0] * y3[0]
		b3 += ww[1] * y3[1]
	}
	if i < n {
		a0 += w[i] * x0[i]
		a1 += w[i] * x1[i]
		a2 += w[i] * x2[i]
		a3 += w[i] * x3[i]
	}
	return a0 + b0, a1 + b1, a2 + b2, a3 + b3
}

// axpy: y += a·x.
func axpy(y, x []float32, a float32) {
	x = x[:len(y)]
	for i := range y {
		y[i] += a * x[i]
	}
}

func relu(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// apply: y = LN(x) (y may be x).
func (n *gnNorm) apply(y, x []float32) {
	y = y[:len(x)]
	var m float64
	for _, v := range x {
		m += float64(v)
	}
	m /= float64(len(x))
	var q float64
	for _, v := range x {
		dv := float64(v) - m
		q += dv * dv
	}
	q /= float64(len(x))
	inv := 1 / math.Sqrt(q+n.eps)
	w, b := n.w[:len(x)], n.b[:len(x)]
	for i, v := range x {
		y[i] = float32((float64(v)-m)*inv)*w[i] + b[i]
	}
}

// ---- hashing and the card cache ----------------------------------------------

func appendF32s(b []byte, xs []float32) []byte {
	for _, x := range xs {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func appendFeats(b []byte, fs []policynet.Feature) []byte {
	b = binary.LittleEndian.AppendUint32(b, uint32(len(fs)))
	for _, f := range fs {
		b = binary.LittleEndian.AppendUint16(b, f.Row)
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f.Value))
	}
	return b
}

// hashState hashes every encoded field the network reads (as remote.go's).
func (g *goNet) hashState(sc *gnScratch, st *policynet.State) uint64 {
	b := sc.buf[:0]
	b = binary.LittleEndian.AppendUint32(b, uint32(len(st.Dense)))
	b = appendF32s(b, st.Dense)
	b = appendFeats(b, st.Sparse)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(st.Cards)))
	for i := range st.Cards {
		c := &st.Cards[i]
		b = append(b, c.Group)
		b = binary.LittleEndian.AppendUint32(b, uint32(len(c.Raw)))
		b = appendF32s(b, c.Raw)
		b = appendFeats(b, c.Rows)
	}
	sc.buf = b
	return maphash.Bytes(g.seed, b)
}

func (g *goNet) hashOpts(sc *gnScratch, sh uint64, opts []policynet.Option) uint64 {
	b := binary.LittleEndian.AppendUint64(sc.buf[:0], sh)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(opts)))
	for i := range opts {
		o := &opts[i]
		b = appendFeats(b, o.Slots)
		b = appendFeats(b, o.Hashed)
		b = binary.LittleEndian.AppendUint32(b, uint32(len(o.Dense)))
		b = appendF32s(b, o.Dense)
		if o.BotPick {
			b = append(b, 1)
		} else {
			b = append(b, 0)
		}
		b = binary.LittleEndian.AppendUint32(b, uint32(o.EntA))
		b = binary.LittleEndian.AppendUint32(b, uint32(o.EntB))
	}
	sc.buf = b
	return maphash.Bytes(g.seed, b) | 1 // never 0: 0 is evalCache.put's "no scores"
}

// hashCard hashes what a card's entry reads: raw, the identity rows and the group.
func (g *goNet) hashCard(sc *gnScratch, c *policynet.EntityCard) uint64 {
	b := append(sc.buf[:0], c.Group)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(c.Raw)))
	b = appendF32s(b, c.Raw)
	b = appendFeats(b, c.Rows)
	sc.buf = b
	return maphash.Bytes(g.seed, b)
}

// cardCache maps a card's hash to w floats (in one slab per shard). A full shard
// starts over.
type cardCache struct {
	shards [64]cardShard
	cap    int
	w      int
}

type cardShard struct {
	mu   sync.Mutex
	idx  map[uint64]int32
	slab []float32
}

func newCardCache(capacity, w int) *cardCache {
	c := &cardCache{cap: max(1, capacity/64), w: w}
	for i := range c.shards {
		c.shards[i].idx = map[uint64]int32{}
	}
	return c
}

func (c *cardCache) get(k uint64, dst []float32) bool {
	s := &c.shards[k%64]
	s.mu.Lock()
	i, ok := s.idx[k]
	if ok {
		copy(dst, s.slab[int(i)*c.w:(int(i)+1)*c.w])
	}
	s.mu.Unlock()
	return ok
}

func (c *cardCache) put(k uint64, v []float32) {
	s := &c.shards[k%64]
	s.mu.Lock()
	if _, ok := s.idx[k]; !ok {
		if len(s.idx) >= c.cap {
			clear(s.idx)
			s.slab = s.slab[:0]
		}
		s.idx[k] = int32(len(s.idx))
		s.slab = append(s.slab, v[:c.w]...)
	}
	s.mu.Unlock()
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
