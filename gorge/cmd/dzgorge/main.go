// Command dzgorge plays DraftZero's FDN Limited decks on the gorge rules engine.
//
// It lives in DraftZero (gorge/cmd/dzgorge) and is copied into a pinned gorge
// checkout to build (gorge/build.sh), because it uses gorge's internal packages:
// the AlphaZero-style search (internal/azmcts), the pure-Go network
// (internal/policynet) and the game runner (internal/bench).
//
//	dzgorge play     -a <policy> -b <policy> -pairs N -out games.jsonl [-corpus visits.jsonl.gz]
//	dzgorge coverage                                  which pool decks gorge fully supports
//	dzgorge pack     -in visits.jsonl.gz -out DIR     a visit corpus as dzg training shards (gorge/dzg)
//
// A game draws two different decks from the pool's split. With -a != -b each deck
// pair is played twice on the same seed, the policies swapping seats, so each
// policy plays each deck once and goes first once (DraftZero's paired games).
// With -a == -b each pair is played once.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "play":
		os.Exit(runPlay(os.Args[2:]))
	case "coverage":
		os.Exit(runCoverage(os.Args[2:]))
	case "pack":
		os.Exit(runPack(os.Args[2:]))
	case "gonet-check": // hidden: the in-process network against PyTorch (gonet_check.go)
		os.Exit(runGonetCheck(os.Args[2:]))
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: dzgorge play|coverage|pack [flags]  (-h for each)")
	os.Exit(2)
}
