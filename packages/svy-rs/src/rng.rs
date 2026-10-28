//! Seedable xoshiro256** RNG shared by sampling and weighting, with no external rand crate.

/// xoshiro256** generator: a given seed always yields the same stream, so draws are reproducible.
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

    /// Uniform integer in [0, n) without modulo bias (Lemire's nearly-divisionless method).
    pub fn next_index(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "next_index requires n > 0");
        let n = n as u64;
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut low = m as u64;
        if low < n {
            // Rejection threshold: 2^64 mod n
            let t = n.wrapping_neg() % n;
            while low < t {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                low = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Uniform random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_index(i + 1);
            slice.swap(i, j);
        }
    }

    /// One index drawn proportional to unnormalised `weights`; `None` unless the total is positive.
    pub fn weighted_choice(&mut self, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return None;
        }
        let mut u = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            u -= w;
            if u <= 0.0 {
                return Some(i);
            }
        }
        Some(weights.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values: the public-domain reference splitmix64.c and
    // xoshiro256starstar.c, compiled; the seeding also matches Java's SplittableRandom.
    #[test]
    fn seeding_is_splitmix64() {
        let rng = Rng::new(42);
        assert_eq!(
            rng.state,
            [
                0xbdd732262feb6e95,
                0x28efe333b266f103,
                0x47526757130f9f52,
                0x581ce1ff0e4ae394,
            ]
        );
    }

    #[test]
    fn next_u64_matches_reference_xoshiro256starstar() {
        let cases: [(u64, [u64; 4]); 3] = [
            (
                0,
                [
                    0x99ec5f36cb75f2b4,
                    0xbf6e1f784956452a,
                    0x1a5f849d4933e6e0,
                    0x6aa594f1262d2d2c,
                ],
            ),
            (
                42,
                [
                    0x15780b2e0c2ec716,
                    0x6104d9866d113a7e,
                    0xae17533239e499a1,
                    0xecb8ad4703b360a1,
                ],
            ),
            (
                u64::MAX,
                [
                    0x8f5520d52a7ead08,
                    0xc476a018caa1802d,
                    0x81de31c0d260469e,
                    0xbf658d7e065f3c2f,
                ],
            ),
        ];
        for (seed, expected) in cases {
            let mut rng = Rng::new(seed);
            let got: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
            assert_eq!(got, expected, "seed {seed}");
        }
    }

    // Expected: (x >> 11) * 2^-53 and Lemire's bounded draw on the reference stream.
    #[test]
    fn next_f64_and_next_index_match_reference() {
        let mut rng = Rng::new(7);
        let floats: Vec<f64> = (0..3).map(|_| rng.next_f64()).collect();
        assert_eq!(
            floats,
            [0.7005764821796896, 0.2787512294737843, 0.8396274618764198]
        );
        let indices: Vec<usize> = (0..8).map(|_| rng.next_index(10)).collect();
        assert_eq!(indices, [9, 9, 8, 0, 1, 4, 1, 5]);

        let mut rng = Rng::new(123);
        let indices: Vec<usize> = (0..8).map(|_| rng.next_index(3)).collect();
        assert_eq!(indices, [0, 2, 1, 0, 1, 2, 1, 1]);
    }

    #[test]
    fn next_index_stays_in_range() {
        let mut rng = Rng::new(42);
        for n in [1usize, 2, 3, 7, 100] {
            for _ in 0..1000 {
                assert!(rng.next_index(n) < n);
            }
        }
    }

    #[test]
    fn next_index_is_close_to_uniform() {
        // Expected 20k per bucket with sd ~115: a 600 bound (~5 sd) is not flaky.
        let mut rng = Rng::new(7);
        let n = 3usize;
        let draws = 60_000;
        let mut counts = [0usize; 3];
        for _ in 0..draws {
            counts[rng.next_index(n)] += 1;
        }
        let expected = draws / n;
        for (i, &c) in counts.iter().enumerate() {
            let dev = (c as f64 - expected as f64).abs();
            assert!(dev < 600.0, "bucket {i}: {c} vs expected {expected}");
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let a: Vec<usize> = {
            let mut r = Rng::new(123);
            (0..16).map(|_| r.next_index(10)).collect()
        };
        let b: Vec<usize> = {
            let mut r = Rng::new(123);
            (0..16).map(|_| r.next_index(10)).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn weighted_choice_is_none_without_positive_total() {
        let mut rng = Rng::new(1);
        let cases: [&[f64]; 4] = [&[], &[0.0, 0.0], &[-1.0, 0.5], &[f64::NAN, 1.0]];
        for weights in cases {
            assert_eq!(rng.weighted_choice(weights), None, "{weights:?}");
        }
    }

    #[test]
    fn weighted_choice_never_draws_a_zero_weight() {
        let mut rng = Rng::new(9);
        for _ in 0..1000 {
            assert_eq!(rng.weighted_choice(&[0.0, 2.5, 0.0]), Some(1));
        }
        assert_eq!(rng.weighted_choice(&[3.0]), Some(0));
    }
}
