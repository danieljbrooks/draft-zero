//! SplitMix64: the bots' own seeded generator (independent of the engine's shuffle stream).

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }

    /// Uniform in [0, bound), without modulo bias (bound > 0).
    pub fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return x % bound;
            }
        }
    }
}

/// SplitMix64's finalizer (a bijection on u64).
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of a bot's generator, from the game seed and the bot's seat (0 = A, 1 = B). The salt keeps
/// it apart from the engine's shuffle stream, which is seeded with the raw `env_seed`.
pub fn bot_seed(game_seed: u64, seat: usize) -> u64 {
    mix64(mix64(game_seed ^ 0xD6E8_FEB8_6659_FD93) ^ (seat as u64 + 1).wrapping_mul(0xA076_1D64_78BD_642F))
}
