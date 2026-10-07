package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"io"
	"math"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/adams-shaun/gorge/internal/policynet"
)

// A remote network (gorge/dzg/SPEC.md): a policynet.External that asks a dzg server
// (python -m dzg.serve) for option scores and values over a socket. Every game in the
// process shares one client per address: requests from all games are batched onto a
// few connections, and answers are cached by the encoded state, so a search revisiting
// a position it already evaluated costs no round trip.

// Remote batching knobs (play's -remote-* flags).
var (
	remoteConns = 4
	remoteBatch = 256
	remoteWait  = 300 * time.Microsecond
	remoteCache = 1 << 20
)

var (
	remotesMu sync.Mutex
	remotes   = map[string]*remoteNet{}
)

// remoteModel is the shared client for addr wrapped as a policynet.Model encoding
// under the entity feature set.
func remoteModel(addr string) (*policynet.Model, error) {
	remotesMu.Lock()
	defer remotesMu.Unlock()
	r, ok := remotes[addr]
	if !ok {
		var err error
		if r, err = dialRemote(addr, remoteConns); err != nil {
			return nil, err
		}
		remotes[addr] = r
	}
	return policynet.NewExternal(r, policynet.FeaturesEntity), nil
}

// remoteStats sums every client's counters (for the run summary).
func remoteStats() map[string]any {
	remotesMu.Lock()
	defer remotesMu.Unlock()
	if len(remotes) == 0 {
		return nil
	}
	var calls, hits, batches, states int64
	var wait time.Duration
	for _, r := range remotes {
		calls += r.calls.Load()
		hits += r.hits.Load()
		batches += r.batches.Load()
		states += r.states.Load()
		wait += time.Duration(r.waitNs.Load())
	}
	out := map[string]any{"remote_calls": calls, "remote_cache_hits": hits, "remote_batches": batches,
		"remote_states": states}
	if batches > 0 {
		out["remote_states_per_batch"] = float64(states) / float64(batches)
	}
	if miss := calls - hits; miss > 0 {
		out["remote_wait_ms_per_miss"] = wait.Seconds() * 1000 / float64(miss)
	}
	return out
}

type remoteNet struct {
	reqs  chan *rreq
	cache *evalCache
	seed  maphash.Seed

	calls, hits, batches, states, waitNs atomic.Int64
}

type rreq struct {
	st     *policynet.State
	opts   []policynet.Option
	want   bool
	done   chan struct{}
	value  float32
	scores []float32
}

// dialRemote opens conns connections to each server in addr, a comma-separated list
// of unix:/path, tcp:host:port or a bare socket path. Every connection serves the
// same request queue.
func dialRemote(addr string, conns int) (*remoteNet, error) {
	r := &remoteNet{reqs: make(chan *rreq, 4096), cache: newEvalCache(remoteCache), seed: maphash.MakeSeed()}
	for _, a := range strings.Split(addr, ",") {
		network, where := "unix", a
		switch {
		case strings.HasPrefix(a, "unix:"):
			where = a[len("unix:"):]
		case strings.HasPrefix(a, "tcp:"):
			network, where = "tcp", a[len("tcp:"):]
		}
		for i := 0; i < conns; i++ {
			var c net.Conn
			var err error
			for try := 0; try < 120; try++ { // the server may still be loading its model
				if c, err = net.Dial(network, where); err == nil {
					break
				}
				time.Sleep(time.Second)
			}
			if err != nil {
				return nil, fmt.Errorf("remote %s: %w", a, err)
			}
			go r.loop(c)
		}
	}
	return r, nil
}

// Score is the option logits for st's options (policynet.External).
func (r *remoteNet) Score(st policynet.State, opts []policynet.Option) []float32 {
	r.calls.Add(1)
	sh := r.hashState(&st)
	key := r.hashOpts(sh, opts)
	if sc, ok := r.cache.scores(key); ok {
		r.hits.Add(1)
		return sc
	}
	q := r.ask(&st, opts, true)
	r.cache.put(sh, key, q.value, q.scores)
	return q.scores
}

// Value is the searching seat's win probability at st (policynet.External).
func (r *remoteNet) Value(st policynet.State) float32 {
	r.calls.Add(1)
	sh := r.hashState(&st)
	if v, ok := r.cache.value(sh); ok {
		r.hits.Add(1)
		return v
	}
	q := r.ask(&st, nil, false)
	r.cache.put(sh, 0, q.value, nil)
	return q.value
}

func (r *remoteNet) ask(st *policynet.State, opts []policynet.Option, want bool) *rreq {
	q := &rreq{st: st, opts: opts, want: want, done: make(chan struct{})}
	t0 := time.Now()
	r.reqs <- q
	<-q.done
	r.waitNs.Add(int64(time.Since(t0)))
	return q
}

// loop serves one connection: gather a batch, send it, scatter the answers.
func (r *remoteNet) loop(c net.Conn) {
	bw := bufio.NewWriterSize(c, 1<<20)
	br := bufio.NewReaderSize(c, 1<<20)
	var enc packer
	batch := make([]*rreq, 0, remoteBatch)
	timer := time.NewTimer(time.Hour)
	for {
		batch = append(batch[:0], <-r.reqs)
	gather:
		for len(batch) < remoteBatch {
			select {
			case q := <-r.reqs:
				batch = append(batch, q)
				continue
			default:
			}
			timer.Reset(remoteWait)
			select {
			case q := <-r.reqs:
				if !timer.Stop() {
					<-timer.C
				}
				batch = append(batch, q)
			case <-timer.C:
				break gather
			}
		}
		if err := r.roundTrip(&enc, batch, bw, br); err != nil {
			fmt.Fprintf(os.Stderr, "dzgorge: remote network: %v\n", err)
			os.Exit(1)
		}
		r.batches.Add(1)
		r.states.Add(int64(len(batch)))
		for _, q := range batch {
			close(q.done)
		}
	}
}

func (r *remoteNet) roundTrip(enc *packer, batch []*rreq, bw *bufio.Writer, br *bufio.Reader) error {
	enc.reset()
	for _, q := range batch {
		if q.want {
			enc.add(q.st, q.opts)
		} else {
			enc.add(q.st, nil)
		}
	}
	want := make([]byte, len(batch))
	for i, q := range batch {
		if q.want {
			want[i] = 1
		}
	}
	if _, err := bw.Write(enc.request(want)); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	var hdr [20]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return err
	}
	n := binary.LittleEndian.Uint32(hdr[0:])
	magic := binary.LittleEndian.Uint32(hdr[4:])
	b := int(binary.LittleEndian.Uint32(hdr[8:]))
	no := int(binary.LittleEndian.Uint32(hdr[12:]))
	if magic != respMagic || b != len(batch) || no != len(enc.optDense)/policynet.OptionDenseWidth {
		return fmt.Errorf("bad response header: magic %08x B %d (want %d) no %d (want %d)", magic, b, len(batch), no, len(enc.optDense)/policynet.OptionDenseWidth)
	}
	body := make([]byte, int(n)-16)
	if _, err := io.ReadFull(br, body); err != nil {
		return err
	}
	vals := body[:4*b]
	scores := body[pad8(4*b):]
	o := 0
	for i, q := range batch {
		q.value = math.Float32frombits(binary.LittleEndian.Uint32(vals[4*i:]))
		if q.want {
			q.scores = make([]float32, len(q.opts))
			for j := range q.scores {
				q.scores[j] = math.Float32frombits(binary.LittleEndian.Uint32(scores[4*(o+j):]))
			}
			o += len(q.opts)
		}
	}
	return nil
}

// hashState hashes every encoded field the server reads.
func (r *remoteNet) hashState(st *policynet.State) uint64 {
	var h maphash.Hash
	h.SetSeed(r.seed)
	var b [8]byte
	f := func(x float32) { binary.LittleEndian.PutUint32(b[:4], math.Float32bits(x)); h.Write(b[:4]) }
	u := func(x uint32) { binary.LittleEndian.PutUint32(b[:4], x); h.Write(b[:4]) }
	u(uint32(len(st.Dense)))
	for _, x := range st.Dense {
		f(x)
	}
	u(uint32(len(st.Sparse)))
	for _, ft := range st.Sparse {
		u(uint32(ft.Row))
		f(ft.Value)
	}
	u(uint32(len(st.Cards)))
	for _, c := range st.Cards {
		u(uint32(c.Group))
		for _, x := range c.Raw {
			f(x)
		}
		u(uint32(len(c.Rows)))
		for _, ft := range c.Rows {
			u(uint32(ft.Row))
			f(ft.Value)
		}
	}
	return h.Sum64()
}

func (r *remoteNet) hashOpts(sh uint64, opts []policynet.Option) uint64 {
	var h maphash.Hash
	h.SetSeed(r.seed)
	var b [8]byte
	f := func(x float32) { binary.LittleEndian.PutUint32(b[:4], math.Float32bits(x)); h.Write(b[:4]) }
	u := func(x uint32) { binary.LittleEndian.PutUint32(b[:4], x); h.Write(b[:4]) }
	binary.LittleEndian.PutUint64(b[:], sh)
	h.Write(b[:])
	u(uint32(len(opts)))
	for _, o := range opts {
		u(uint32(len(o.Slots)))
		for _, ft := range o.Slots {
			u(uint32(ft.Row))
			f(ft.Value)
		}
		u(uint32(len(o.Hashed)))
		for _, ft := range o.Hashed {
			u(uint32(ft.Row))
			f(ft.Value)
		}
		for _, x := range o.Dense {
			f(x)
		}
		bot := uint32(0)
		if o.BotPick {
			bot = 1
		}
		u(bot)
		u(uint32(o.EntA))
		u(uint32(o.EntB))
	}
	return h.Sum64() | 1 // never 0: 0 is put's "no scores"
}

// evalCache maps state hashes to values and (state, options) hashes to option
// scores. Sharded; a full shard is cleared rather than evicted entry by entry.
type evalCache struct {
	shards [64]cacheShard
	cap    int
}

type cacheShard struct {
	mu     sync.Mutex
	vals   map[uint64]float32
	scores map[uint64][]float32
}

func newEvalCache(capacity int) *evalCache {
	c := &evalCache{cap: capacity / 64}
	for i := range c.shards {
		c.shards[i].vals = map[uint64]float32{}
		c.shards[i].scores = map[uint64][]float32{}
	}
	return c
}

func (c *evalCache) value(k uint64) (float32, bool) {
	s := &c.shards[k%64]
	s.mu.Lock()
	v, ok := s.vals[k]
	s.mu.Unlock()
	return v, ok
}

func (c *evalCache) scores(k uint64) ([]float32, bool) {
	s := &c.shards[k%64]
	s.mu.Lock()
	v, ok := s.scores[k]
	s.mu.Unlock()
	return v, ok
}

func (c *evalCache) put(sh, key uint64, v float32, sc []float32) {
	s := &c.shards[sh%64]
	s.mu.Lock()
	if len(s.vals) >= c.cap {
		s.vals = map[uint64]float32{}
	}
	s.vals[sh] = v
	s.mu.Unlock()
	if key == 0 {
		return
	}
	s = &c.shards[key%64]
	s.mu.Lock()
	if len(s.scores) >= c.cap {
		s.scores = map[uint64][]float32{}
	}
	s.scores[key] = sc
	s.mu.Unlock()
}

// ---- packed arrays (SPEC §2-3) ----------------------------------------------

const (
	reqMagic  = 0x31425A47 // "GZB1"
	respMagic = 0x31525A47 // "GZR1"
)

func pad8(n int) int { return (n + 7) &^ 7 }

// packer accumulates states and their options as SPEC §2's columns. Offsets are
// kept relative to the packer (they start at 0).
type packer struct {
	dense                []float32
	spOff                []int32
	spRow                []uint16
	spVal                []float32
	cardOff              []int32
	cardGroup            []uint8
	cardRaw              []float32
	crOff                []int32
	crRow                []uint16
	crVal                []float32
	optOff               []int32
	optDense             []float32
	optBot               []uint8
	optEA, optEB         []int32
	osOff                []int32
	osRow                []uint16
	osVal                []float32
	ohOff                []int32
	ohRow                []uint16
	ohVal                []float32
	buf                  []byte
}

func (p *packer) reset() {
	p.dense, p.spRow, p.spVal = p.dense[:0], p.spRow[:0], p.spVal[:0]
	p.spOff = append(p.spOff[:0], 0)
	p.cardOff = append(p.cardOff[:0], 0)
	p.cardGroup, p.cardRaw, p.crRow, p.crVal = p.cardGroup[:0], p.cardRaw[:0], p.crRow[:0], p.crVal[:0]
	p.crOff = append(p.crOff[:0], 0)
	p.optOff = append(p.optOff[:0], 0)
	p.optDense, p.optBot, p.optEA, p.optEB = p.optDense[:0], p.optBot[:0], p.optEA[:0], p.optEB[:0]
	p.osOff = append(p.osOff[:0], 0)
	p.ohOff = append(p.ohOff[:0], 0)
	p.osRow, p.osVal, p.ohRow, p.ohVal = p.osRow[:0], p.osVal[:0], p.ohRow[:0], p.ohVal[:0]
}

func (p *packer) states() int { return len(p.spOff) - 1 }

// add appends one state and its options.
func (p *packer) add(st *policynet.State, opts []policynet.Option) {
	dense := st.Dense
	if len(dense) != policynet.DenseWidth {
		dense = make([]float32, policynet.DenseWidth)
		copy(dense, st.Dense)
	}
	p.dense = append(p.dense, dense...)
	for _, f := range st.Sparse {
		p.spRow = append(p.spRow, f.Row)
		p.spVal = append(p.spVal, f.Value)
	}
	p.spOff = append(p.spOff, int32(len(p.spRow)))
	for _, c := range st.Cards {
		p.cardGroup = append(p.cardGroup, c.Group)
		raw := c.Raw
		if len(raw) != policynet.EntityRawWidth {
			raw = make([]float32, policynet.EntityRawWidth)
			copy(raw, c.Raw)
		}
		p.cardRaw = append(p.cardRaw, raw...)
		for _, f := range c.Rows {
			p.crRow = append(p.crRow, f.Row)
			p.crVal = append(p.crVal, f.Value)
		}
		p.crOff = append(p.crOff, int32(len(p.crRow)))
	}
	p.cardOff = append(p.cardOff, int32(len(p.cardGroup)))
	for i := range opts {
		o := &opts[i]
		d := o.Dense
		if len(d) != policynet.OptionDenseWidth {
			d = make([]float32, policynet.OptionDenseWidth)
			copy(d, o.Dense)
		}
		p.optDense = append(p.optDense, d...)
		bot := uint8(0)
		if o.BotPick {
			bot = 1
		}
		p.optBot = append(p.optBot, bot)
		p.optEA = append(p.optEA, o.EntA)
		p.optEB = append(p.optEB, o.EntB)
		for _, f := range o.Slots {
			p.osRow = append(p.osRow, f.Row)
			p.osVal = append(p.osVal, f.Value)
		}
		p.osOff = append(p.osOff, int32(len(p.osRow)))
		for _, f := range o.Hashed {
			p.ohRow = append(p.ohRow, f.Row)
			p.ohVal = append(p.ohVal, f.Value)
		}
		p.ohOff = append(p.ohOff, int32(len(p.ohRow)))
	}
	p.optOff = append(p.optOff, int32(len(p.optBot)))
}

// request is SPEC §3's request message for the packed states, want[i] per state.
func (p *packer) request(want []byte) []byte {
	b := p.buf[:0]
	b = append(b, 0, 0, 0, 0) // nbytes, filled below
	for _, x := range []int{reqMagic, p.states(), len(p.spRow), len(p.cardGroup), len(p.crRow), len(p.optBot), len(p.osRow), len(p.ohRow)} {
		b = binary.LittleEndian.AppendUint32(b, uint32(x))
	}
	b = appendPadded(b, want)
	b = appendF32(b, p.dense)
	b = appendI32(b, p.spOff)
	b = appendU16(b, p.spRow)
	b = appendF32(b, p.spVal)
	b = appendI32(b, p.cardOff)
	b = appendPadded(b, p.cardGroup)
	b = appendF32(b, p.cardRaw)
	b = appendI32(b, p.crOff)
	b = appendU16(b, p.crRow)
	b = appendF32(b, p.crVal)
	b = appendI32(b, p.optOff)
	b = appendF32(b, p.optDense)
	b = appendPadded(b, p.optBot)
	b = appendI32(b, p.optEA)
	b = appendI32(b, p.optEB)
	b = appendI32(b, p.osOff)
	b = appendU16(b, p.osRow)
	b = appendF32(b, p.osVal)
	b = appendI32(b, p.ohOff)
	b = appendU16(b, p.ohRow)
	b = appendF32(b, p.ohVal)
	binary.LittleEndian.PutUint32(b[0:], uint32(len(b)-4))
	p.buf = b
	return b
}

func appendPadded(b []byte, x []byte) []byte {
	b = append(b, x...)
	for len(b)%8 != 4 { // the 4-byte nbytes prefix sits before the 8-aligned payload
		b = append(b, 0)
	}
	return b
}

func appendF32(b []byte, x []float32) []byte {
	for _, v := range x {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return appendPadded(b, nil)
}

func appendI32(b []byte, x []int32) []byte {
	for _, v := range x {
		b = binary.LittleEndian.AppendUint32(b, uint32(v))
	}
	return appendPadded(b, nil)
}

func appendU16(b []byte, x []uint16) []byte {
	for _, v := range x {
		b = binary.LittleEndian.AppendUint16(b, v)
	}
	return appendPadded(b, nil)
}
