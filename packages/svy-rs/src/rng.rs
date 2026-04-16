// src/rng.rs
//
// Shared xoshiro256** RNG used across sampling and weighting modules.
// No external rand crate needed — this is consistent with the rest of
// the codebase (weighting/replication.rs uses the same algorithm).

/// Fast, seedable xoshiro256** pseudo-random number generator.
///
/// Used for reproducible random sampling. Seeded from a u64; child
/// generators for independent strata are derived via wrapping addition
/// on the seed so each stratum gets a deterministic but independent stream.
#[derive(Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Create a new RNG from a 64-bit seed using SplitMix64 initialisation.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Rng { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform random integer in [0, n).
    pub fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for a uniform double
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_index(i + 1);
            slice.swap(i, j);
        }
    }

    /// Weighted random choice: draw one index from `weights` (unnormalised).
    /// Returns the chosen index. Panics if weights is empty or all-zero.
    pub fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut u = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            u -= w;
            if u <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

}
