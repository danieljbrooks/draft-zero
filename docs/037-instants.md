# Instants

## Introduction

The current algorithm plays instant speed effects poorly. 

In general, instants are played eagerly at improper times and sometimes with improper targets. 

Card winrate statistics show instants performing around 5-10% below the expected GIHWR. 

As a result, we see the optimal draft/sealed metagame for the system skew toward creature based aggro decks with few choices. 

Increases in compute budget (~hundreds GPU hours) driven by autoresearch lead to only modest improvements - suggesting a fundamental issue with the current algorithm. 

This document describes an issue in the search algorithm, illustrate with examples, and discuss potentials solutions. 

## Challenging Search Space

Sorcery speed spells have two timing windows per turn cycle. 

Instants have around an order of magnitude more. 

Instants are also more likely to have targets, which further increases the decision space. 

Most of these decisions are degenerate and close in value. 

For example, against a tapped-out opponent, playing an instant in a priority stop during the player's turn yield the same outcome. 

From a MCTS perspective however, these decisions are exhaustively searched over. 

With search noise, the system frequently selects non-optimal timings. 

## How Humans Play

Humans search play options much more efficiently than the current MCTS algorithm. 

They keep a working understanding of the game state. 

This understanding is updated only when hidden information changes (e.g. a card is drawn, an opponent plays a card, etc). 

This is particularly vexing for the current system because the representation of hidden information is assumed to be only approximately correct - and the exact distribution that unknown information is sampled has varied between experiments. 

The current implement of MCTS is similar to a human whose memory is wiped after every priority stop. 

## Goal

Given the goal of playing high level Magic without prior information (tabula rasa), candidate approaches should strive to do several things: 
1. Reduce the number of degenerate timings for playing instants. 
2. Not rely on heavy search for miniscule differences (e.g. playing a spell on upkeep versus draw step). 
3. Result in human-like instant timing principles (e.g. defer when possible, 
4. Be tolerant of search estimates have some noise. 
5. Apply to general formats using learnable parameters rather than reference past human games. 

## Brainstorming - Wacky Ideas

1. Have one auxiliary network decide whether to play the instant, then another decide decide when to play the instant. 
2. Simply trimming search trees (e.g. start on the first main phase). 
3. Come up with a clever filter to reduce redundant search. E.g. have a network that learns whether or not play the instant on the current step. If the network says no, no search is performed. 
4. Explicitly develop rules for changes in information. (E.g. drawing a card from you library -> less hidden information). Then use changes in information to information as a cue to refresh planning. 
5. By default, MCTS builds a new search tree every step by default. What if we hacked the search process to keep the old tree as a base?

- Daniel Brooks
