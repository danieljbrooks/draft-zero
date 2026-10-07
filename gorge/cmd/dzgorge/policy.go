package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/adams-shaun/gorge/internal/azmcts"
	"github.com/adams-shaun/gorge/internal/policynet"
	"github.com/adams-shaun/gorge/internal/spellbench/builtins"
	"github.com/adams-shaun/gorge/seat"
)

// policySpec is one side's policy, parsed from "name[:key=value...]":
//
//	random                       uniform over the legal options, mana paid automatically (sb-uniform)
//	bot                          gorge's production heuristic, no search
//	az:sims=100                  AlphaZero-style PUCT search over the seat's own decisions
//	                             (internal/azmcts), honest worlds: each simulation re-deals the
//	                             cards the seat cannot see. Generation 0 without net=: a uniform
//	                             prior and gorge's heuristic leaf
//	az:sims=100:net=gen1.gpol    the same, the network's prior and value head
//	   :leaf=heuristic           keep the heuristic leaf (the net is the prior only)
//	   :prior=uniform            keep the uniform prior (the net is the leaf only)
//	   :worlds=K                 K deals per decision instead of one per simulation
//	   :explore[:nonoise]        generation: sample moves by visits on turns 1-4 (+ root noise)
//	   :cpuct=1.5 :fpu=0.1 :cands=N
//	prior:net=gen1.gpol          the network's policy alone: the argmax of its prior over the
//	                             candidates the search would build, no simulation
//	az:sims=100:remote=unix:/tmp/dzg.sock
//	                             the network served by python -m dzg.serve (gorge/dzg): an MLP,
//	                             transformer or GNN reading gorge's entity encoding, in place
//	                             of a .gpol (also prior:remote=...)
type policySpec struct {
	Raw, Kind string
	Sims      int
	Net       string
	Remote    string
	model     *policynet.Model
	HeurLeaf  bool
	UniPrior  bool
	Worlds    int
	Explore   bool
	NoNoise   bool
	CPUCT     float64
	FPU       float64
	Cands     int
}

func parsePolicy(s string) (*policySpec, error) {
	// remote=unix:/path and remote=tcp:host:port contain colons: take the rest of the spec.
	rest := ""
	if i := strings.Index(s, ":remote="); i >= 0 {
		s, rest = s[:i], s[i+len(":remote="):]
	}
	parts := strings.Split(s, ":")
	p := &policySpec{Raw: s, Kind: parts[0], Sims: azmcts.DefaultOptions().Sims, CPUCT: -1, FPU: -1, Cands: -1}
	switch p.Kind {
	case "random", "bot", "az", "prior":
	default:
		return nil, fmt.Errorf("policy %q: want random, bot, az or prior", s)
	}
	for _, kv := range parts[1:] {
		k, v, _ := strings.Cut(kv, "=")
		var err error
		switch k {
		case "sims":
			p.Sims, err = strconv.Atoi(v)
		case "net":
			p.Net = v
		case "leaf":
			if v != "heuristic" && v != "net" {
				return nil, fmt.Errorf("policy %q: leaf=%s, want heuristic or net", s, v)
			}
			p.HeurLeaf = v == "heuristic"
		case "prior":
			if v != "uniform" && v != "net" {
				return nil, fmt.Errorf("policy %q: prior=%s, want uniform or net", s, v)
			}
			p.UniPrior = v == "uniform"
		case "worlds":
			p.Worlds, err = strconv.Atoi(v)
		case "explore":
			p.Explore = true
		case "nonoise":
			p.NoNoise = true
		case "cpuct":
			p.CPUCT, err = strconv.ParseFloat(v, 64)
		case "fpu":
			p.FPU, err = strconv.ParseFloat(v, 64)
		case "cands":
			p.Cands, err = strconv.Atoi(v)
		default:
			return nil, fmt.Errorf("policy %q: unknown key %q", s, k)
		}
		if err != nil {
			return nil, fmt.Errorf("policy %q: %s: %w", s, k, err)
		}
	}
	if rest != "" {
		p.Raw, p.Remote = s+":remote="+rest, rest
	}
	if p.Kind == "prior" && p.Net == "" && p.Remote == "" {
		return nil, fmt.Errorf("policy %q: prior needs net= or remote=", s)
	}
	if p.Net != "" && p.Remote != "" {
		return nil, fmt.Errorf("policy %q: net= and remote= are exclusive", s)
	}
	if p.Remote != "" {
		m, err := remoteModel(p.Remote)
		if err != nil {
			return nil, fmt.Errorf("policy %q: %w", s, err)
		}
		p.model = m
	}
	if p.Net != "" {
		m, err := policynet.LoadCheckpointFile(p.Net)
		if err != nil {
			return nil, fmt.Errorf("policy %q: %w", s, err)
		}
		p.model = m
	}
	if p.Kind == "az" || p.Kind == "prior" {
		if _, err := p.seat(1); err != nil {
			return nil, fmt.Errorf("policy %q: %w", s, err)
		}
	}
	return p, nil
}

// recordFeatures is the encoding visit corpora are written in (play -record-features).
var recordFeatures = policynet.FeaturesMZ

func (p *policySpec) searches() bool { return p.Kind == "az" || p.Kind == "prior" }

func (p *policySpec) azConfig() azmcts.SeatConfig {
	cfg := azmcts.DefaultSeatConfig()
	cfg.Search.Sims = p.Sims
	cfg.Search.Kinds = azmcts.AllKinds()
	cfg.Search.HeuristicLeaf = p.HeurLeaf
	cfg.Search.UniformPrior = p.UniPrior
	if p.CPUCT >= 0 {
		cfg.Search.CPUCT = p.CPUCT
	}
	if p.FPU >= 0 {
		cfg.Search.FPU = p.FPU
	}
	if p.Cands > 0 {
		cfg.Search.Limit = p.Cands
	}
	// Honest worlds only: the seat never searches the real engine's hidden zones. The
	// prior-only student asks for no world at all; redeal just satisfies NewSeat.
	cfg.World = azmcts.WorldRedeal
	cfg.Worlds = p.Worlds
	cfg.Explore, cfg.NoNoise = p.Explore, p.NoNoise
	cfg.PriorOnly = p.Kind == "prior"
	cfg.RecordFeatures = recordFeatures
	return cfg
}

// seat builds one seat of one game from its per-seat seed.
func (p *policySpec) seat(seed uint64) (seat.Seat, error) {
	switch p.Kind {
	case "random":
		return builtins.New(builtins.Uniform, builtins.AutoPay, seed^builtins.UniformSeed), nil
	case "bot":
		return seat.NewBot(seed), nil
	default:
		return azmcts.NewSeat(seed, p.model, p.azConfig())
	}
}
