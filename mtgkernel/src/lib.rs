//! dzk: draft-zero's FDN self-play harness on mtg-kernel (stage 1: engine wrapper, bots, game records,
//! bench; stage 2: decision encoder, policy/value network, network bots, training data). See README.md.

pub mod bots;
pub mod data;
pub mod deck;
pub mod determinize;
pub mod encode;
pub mod eval;
pub mod game;
pub mod nn;
pub mod nntools;
pub mod record;
pub mod rng;
pub mod runner;

/// The mtg-kernel commit `setup.sh` pins (the patched clone in `.deps/mtg-kernel`).
pub const MTG_KERNEL_COMMIT: &str = "a4e1474b492f2f14ce1ed9ef8e40309fb70e4e8e";
