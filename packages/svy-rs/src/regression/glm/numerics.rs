#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Kahan {
    sum: f64,
    c: f64,
}

impl Kahan {
    #[inline]
    pub(super) fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    #[inline]
    pub(super) fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline]
    pub(super) fn value(self) -> f64 {
        self.sum
    }
}
