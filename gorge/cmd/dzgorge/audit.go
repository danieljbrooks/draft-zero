package main

// play -audit-oracle SPEC: an oracle search beside side A's az seat. At a sampled share of side A's decisions
// (-audit-rate), the oracle (normally the same network with many more simulations and no top-k cut) searches the
// same position first, then side A decides as it would have; both search records go to -audit-out, one JSON line
// per decision. docs/031's "how good is the bot" measure reads them: how often the bot's choice is the oracle's,
// and how much win chance (by the oracle's estimates) its choice gives up, by decision kind.
//
// The oracle only reads the position: the game, side A's choice and every other seat play exactly as without the
// audit (play's output is unchanged; a test with -audit-rate 1 checks it).

import (
	"bufio"
	"context"
	"encoding/json"
	"math/rand"
	"os"
	"sync"

	"github.com/adams-shaun/gorge/internal/azmcts"
	"github.com/adams-shaun/gorge/decision"
	"github.com/adams-shaun/gorge/internal/policynet"
	"github.com/adams-shaun/gorge/internal/searchseat"
)

var (
	auditOracle string  // -audit-oracle: the oracle's policy spec (an az spec)
	auditRate   float64 // -audit-rate: the share of side A's searched decisions audited
	auditPath   string  // -audit-out
	auditSpec   *policySpec
	auditW      *auditWriter
)

type auditWriter struct {
	mu sync.Mutex
	f  *os.File
	w  *bufio.Writer
}

func newAuditWriter(path string) (*auditWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &auditWriter{f: f, w: bufio.NewWriter(f)}, nil
}

func (a *auditWriter) write(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	a.mu.Lock()
	a.w.Write(b)
	a.w.WriteByte('\n')
	a.w.Flush() // one line per audited decision: cheap next to the oracle's search, and nothing is lost to a kill
	a.mu.Unlock()
}

func (a *auditWriter) close() {
	a.mu.Lock()
	a.w.Flush()
	a.f.Close()
	a.mu.Unlock()
}

// auditSide is one search record, trimmed to what the audit reads.
type auditSide struct {
	Sims      int       `json:"sims"`
	Cands     [][]int   `json:"cands"`
	Visits    []int     `json:"visits"`
	Prior     []float64 `json:"prior"`
	Q         []float64 `json:"q"`
	RootValue float64   `json:"root_value"`
	Choice    int       `json:"choice"`
}

func trimRecord(r *policynet.VisitRecord) auditSide {
	return auditSide{Sims: r.Sims, Cands: r.Cands, Visits: r.Visits, Prior: r.Prior, Q: r.Q, RootValue: r.RootValue,
		Choice: r.Choice}
}

type auditLine struct {
	Pair    int           `json:"pair"`
	Leg     int           `json:"leg"`
	Seat    int           `json:"seat"`
	Turn    int32         `json:"turn"`
	Seq     uint64        `json:"sequence"`
	Kind    decision.Kind `json:"kind"`
	Options int           `json:"options"`
	Agent   auditSide     `json:"agent"`
	Oracle  auditSide     `json:"oracle"`
}

// auditSeat is side A's search seat with the oracle beside it.
type auditSeat struct {
	*azmcts.Seat
	oracle      *azmcts.Seat
	rng         *rand.Rand
	pair, leg   int
	seat        int
	agent, orac *policynet.VisitRecord
}

func newAuditSeat(agent, oracle *azmcts.Seat, seed uint64, pair, leg, seat int) *auditSeat {
	a := &auditSeat{Seat: agent, oracle: oracle, rng: rand.New(rand.NewSource(int64(seed ^ 0xa0d17))), pair: pair,
		leg: leg, seat: seat}
	agent.SetRecorder(func(r policynet.VisitRecord) { a.agent = &r })
	oracle.SetRecorder(func(r policynet.VisitRecord) { a.orac = &r })
	return a
}

func (a *auditSeat) DecideSearch(ctx context.Context, env searchseat.Env, d decision.Decision) (decision.Intent, error) {
	if a.rng.Float64() >= auditRate {
		a.agent = nil
		return a.Seat.DecideSearch(ctx, env, d)
	}
	a.agent, a.orac = nil, nil
	if _, err := a.oracle.DecideSearch(ctx, env, d); err != nil {
		a.orac = nil
	}
	in, err := a.Seat.DecideSearch(ctx, env, d)
	if err == nil && a.agent != nil && a.orac != nil {
		auditW.write(auditLine{Pair: a.pair, Leg: a.leg, Seat: a.seat, Turn: a.agent.Turn, Seq: a.agent.Sequence,
			Kind: d.Kind, Options: len(d.Options), Agent: trimRecord(a.agent), Oracle: trimRecord(a.orac)})
	}
	return in, err
}
