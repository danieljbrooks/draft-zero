package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/adams-shaun/gorge/cards"
	"github.com/adams-shaun/gorge/deck"
	"github.com/adams-shaun/gorge/effects"
)

// poolDeck is one row of data/gorge/pool.tsv (gorge/decks.py).
type poolDeck struct {
	Name, Split, Colors string
}

func readPool(path, split string) ([]poolDeck, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []poolDeck
	sc := bufio.NewScanner(f)
	for line := 0; sc.Scan(); line++ {
		if line == 0 {
			continue
		}
		v := strings.Split(sc.Text(), "\t")
		if len(v) < 3 {
			continue
		}
		if split == "all" || v[1] == split {
			out = append(out, poolDeck{Name: v[0], Split: v[1], Colors: v[2]})
		}
	}
	return out, sc.Err()
}

// deckCache resolves pool decks against the corpus once each. It is filled before
// any game starts and read-only afterwards.
type deckCache struct {
	reg *cards.Registry
	dir string
	m   map[string][]*cards.Card
	// names maps a compiled card back to the 17lands name the deck file used, so
	// game records name cards exactly as 17lands' reference does.
	names map[*cards.Card]string
}

func (c *deckCache) get(name string) ([]*cards.Card, error) {
	if d, ok := c.m[name]; ok {
		return d, nil
	}
	f, d, err := deck.Load(c.reg, filepath.Join(c.dir, name+".json"))
	if err != nil {
		return nil, fmt.Errorf("deck %s: %w", name, err)
	}
	c.m[name] = d
	if c.names != nil {
		for _, e := range f.Cards {
			if card, ok := c.reg.Lookup(e.Name); ok {
				c.names[card] = e.Name
			}
		}
	}
	return d, nil
}

// cardName is the deck file's name for a compiled card, else its front face's.
func (c *deckCache) cardName(card *cards.Card) string {
	if n, ok := c.names[card]; ok {
		return n
	}
	if len(card.Faces) > 0 {
		return card.Faces[0].Name
	}
	return card.Path
}

// unsupported is the set of pool card names gorge cannot fully play (a primitive
// its IR references is not implemented), with the missing primitives.
func unsupported(reg *cards.Registry, names []string) map[string][]string {
	sup := effects.Supported()
	out := map[string][]string{}
	for _, n := range names {
		c, ok := reg.Lookup(n)
		if !ok {
			out[n] = []string{"not in corpus"}
			continue
		}
		if m := reg.Unsupported(c, sup); len(m) > 0 {
			out[n] = m
		}
	}
	return out
}

// runCoverage checks every pool deck against the corpus: which cards resolve and
// are fully supported, and how many decks play only supported cards.
func runCoverage(args []string) int {
	fs := flag.NewFlagSet("coverage", flag.ExitOnError)
	cardsDir := fs.String("cards", ".cards", "gorge corpus directory")
	decksDir := fs.String("decks", "", "pool deck directory (gorge/decks.py)")
	poolPath := fs.String("pool", "", "pool.tsv (gorge/decks.py)")
	out := fs.String("out", "", "write the supported-deck list here (one deck per line)")
	fs.Parse(args)
	reg, err := cards.SharedCorpus(*cardsDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	pool, err := readPool(*poolPath, "all")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	dc := &deckCache{reg: reg, dir: *decksDir, m: map[string][]*cards.Card{}}
	names := map[string]int{}
	decksWith := map[string][]string{}
	failed := 0
	for _, p := range pool {
		f, err := os.ReadFile(filepath.Join(*decksDir, p.Name+".json"))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		file, err := deck.Parse(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, p.Name, err)
			failed++
			continue
		}
		for _, n := range file.CardNames() {
			names[n]++
			decksWith[n] = append(decksWith[n], p.Name)
		}
		if _, err := dc.get(p.Name); err != nil {
			fmt.Fprintln(os.Stderr, err)
			failed++
		}
		delete(dc.m, p.Name) // coverage only: do not hold 31k decks
	}
	all := make([]string, 0, len(names))
	for n := range names {
		all = append(all, n)
	}
	sort.Strings(all)
	gaps := unsupported(reg, all)
	bad := map[string]bool{}
	for n, m := range gaps {
		fmt.Printf("unsupported: %-36s in %5d decks  missing %v\n", n, len(decksWith[n]), m)
		for _, d := range decksWith[n] {
			bad[d] = true
		}
	}
	fmt.Printf("%d distinct cards in %d decks; %d unsupported; %d decks failed to resolve; %d of %d decks play only supported cards\n",
		len(all), len(pool), len(gaps), failed, len(pool)-len(bad)-failed, len(pool))
	if *out != "" {
		var b strings.Builder
		for _, p := range pool {
			if !bad[p.Name] {
				b.WriteString(p.Name + "\n")
			}
		}
		if err := os.WriteFile(*out, []byte(b.String()), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	}
	return 0
}
