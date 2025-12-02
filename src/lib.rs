use core::f32::consts::PI;

/// Linear interpolation.
///
/// Gives exact result when t == 1.
const fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (1.0 - t) * a + t * b
}

fn sinc(x: f32) -> f32 {
    let pi_x = PI * x;
    pi_x.sin() / pi_x
}

// https://en.wikipedia.org/wiki/Lanczos_resampling
fn lanczos_kernel<const A: usize>(x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    let a = A as f32;
    a * sinc(x) * sinc(x / a)
}

/// Interpolates `lanczos_kernel` in the range `[-A; A]` on a grid of `N` points using cubic
/// Hermite splines with second-order finite differences at spline endpoints.
///
/// See <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>.
pub struct LanczosKernel<const N: usize, const A: usize> {
    kernel: [f32; N],
}

impl<const N: usize, const A: usize> LanczosKernel<N, A> {
    const _CHECK_1: () = assert!(A >= 1);
    const _CHECK_2: () = assert!(
        N >= 2,
        "LanczosKernel needs at least two points for interpolation."
    );

    pub const X_MIN: f32 = -(A as f32);
    pub const X_MAX: f32 = A as f32;
    pub const X_RANGE: f32 = (2 * A) as f32;

    pub fn new() -> Self {
        let mut kernel = [0.0; N];
        for i in 0..N {
            let x = lerp(Self::X_MIN, Self::X_MAX, i as f32 / (N - 1) as f32);
            kernel[i] = lanczos_kernel::<A>(x);
        }
        Self { kernel }
    }

    pub fn interpolate(&self, x: f32) -> f32 {
        debug_assert!(Self::X_MIN <= x && x <= Self::X_MAX);
        if x == 0.0 {
            return 1.0;
        }
        let n = (N - 1) as f32;
        // Interpolate using cubic Hermite spline.
        // https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        // 1. Find 2-4 closest points.
        let (i00, i0, i1, i11) = {
            let tmp = (x - Self::X_MIN) / Self::X_RANGE;
            let mut i0 = (tmp * n).floor() as usize;
            if i0 == N - 1 {
                i0 -= 1;
            }
            let i1 = i0 + 1;
            let i00 = (i0 > 0).then(|| i0 - 1);
            let i2 = (i1 < N - 1).then(|| i1 + 1);
            (i00, i0, i1, i2)
        };
        debug_assert!(i0 < N);
        debug_assert!(i1 < N);
        let x0 = lerp(Self::X_MIN, Self::X_MAX, i0 as f32 / n);
        let x1 = lerp(Self::X_MIN, Self::X_MAX, i1 as f32 / n);
        // 2. Compute coefficients.
        let dx = x1 - x0;
        let t = (x - x0) / dx;
        let t1 = (1.0 - t) * (1.0 - t);
        let t2 = t * t;
        let h00 = (1.0 + t + t) * t1;
        let h10 = t * t1;
        let h01 = t2 * (3.0 - (t + t));
        let h11 = t2 * (t - 1.0);
        // Finite differences.
        let dx2 = dx + dx;
        let m0 = (self.kernel[i1] - self.kernel[i00.unwrap_or(i1)]) / dx2;
        let m1 = (self.kernel[i11.unwrap_or(i0)] - self.kernel[i0]) / dx2;
        // 3. Interpolate.
        let p = h00 * self.kernel[i0] + h10 * dx * m0 + h01 * self.kernel[i1] + h11 * dx * m1;
        p
    }
}

#[test]
fn lanczos_kernel_works() {
    // `LanczosKernel` should interpolate to the exact same values as the original `lanczos_kernel`
    // function in every node.
    const A: usize = 3;
    let a = A as f32;
    let n = 100;
    let lanczos = LanczosKernel::<100, A>::new();
    for i in 0..n {
        let x = lerp(-a, a, i as f32 / (n - 1) as f32);
        let expected = lanczos_kernel::<A>(x);
        let actual = lanczos.interpolate(x);
        assert_eq!(
            expected,
            actual,
            "{x:20.6} {expected:20.6} {actual:20.6} {eps:20.6}",
            eps = (expected - actual).abs(),
        );
    }
}

pub struct LanczosFilter<const N: usize, const A: usize> {
    kernel: LanczosKernel<N, A>,
}

impl<const N: usize, const A: usize> LanczosFilter<N, A> {
    pub fn new() -> Self {
        let kernel = LanczosKernel::new();
        Self { kernel }
    }

    // TODO Add interpolate_with_history
    pub fn interpolate(&self, x: f32, samples: &[f32]) -> f32 {
        debug_assert!(!samples.is_empty());
        let i = x.floor() as usize;
        debug_assert!(i < samples.len());
        let i_from = i.saturating_sub(A) + 1;
        let i_to = (i + A).min(samples.len() - 1);
        let mut sum = 0.0;
        for j in i_from..=i_to {
            sum += samples[j] * self.kernel.interpolate(x - j as f32);
        }
        sum
    }
}

#[test]
fn lanczos_filter_works() {
    // `LanczosFilter` should interpolate to the exact same values as the original function
    // in every node.
    const A: usize = 3;
    const NUM_SAMPLES: usize = 10;
    let mut xs = [0.0; NUM_SAMPLES];
    let mut samples = [0.0; NUM_SAMPLES];
    for i in 0..NUM_SAMPLES {
        let x0 = 0.0;
        let x1 = (NUM_SAMPLES - 1) as f32;
        let x = lerp(x0, x1, i as f32 / x1);
        xs[i] = x;
        samples[i] = x.sin();
    }
    let lanczos = LanczosFilter::<100, A>::new();
    for i in 0..NUM_SAMPLES {
        let x = xs[i];
        let expected = samples[i];
        let actual = lanczos.interpolate(x, &samples);
        let eps = (expected - actual).abs();
        assert!(
            eps < 1e-4,
            "{x:20.6} {expected:20.6} {actual:20.6} {eps:20.6}",
        );
    }
}

pub struct LanczosResampler<const N: usize, const A: usize> {
    filter: LanczosFilter<N, A>,
    input_sample_rate: usize,
    output_sample_rate: usize,
    output_len_remainder: f64,
    prev_samples: [f32; A],
}

impl<const N: usize, const A: usize> LanczosResampler<N, A> {
    pub fn new(input_sample_rate: usize, output_sample_rate: usize) -> Self {
        assert!(input_sample_rate > 0, "Input sample rate must be non-zero");
        assert!(
            output_sample_rate > 0,
            "Output sample rate must be non-zero"
        );
        let filter = LanczosFilter::new();
        Self {
            filter,
            input_sample_rate,
            output_sample_rate,
            output_len_remainder: 0.0,
            prev_samples: [0.0; A],
        }
    }

    pub fn output_len(&self, input_len: usize) -> f64 {
        let numerator = input_len * self.output_sample_rate;
        let remainder = numerator % self.input_sample_rate;
        if remainder == 0 {
            return (numerator / self.input_sample_rate) as f64;
        }
        let ratio = self.output_sample_rate as f64 / self.input_sample_rate as f64;
        input_len as f64 * ratio
    }

    // TODO return how many samples were read as well
    pub fn resample_into(&mut self, samples: &[f32], output: &mut [f32]) -> usize {
        let output_len = self.output_len(samples.len());
        if output_len == 0.0 {
            return 0;
        }
        self.output_len_remainder += output_len.fract();
        let mut output_len = output_len.floor() as usize;
        if self.output_len_remainder >= 1.0 {
            output_len += 1;
            self.output_len_remainder -= 1.0;
        }
        assert!(output_len <= output.len());
        let x0 = 0.0;
        let x1 = (output_len - 1) as f32;
        for i in 0..output_len {
            let x = lerp(x0, x1, i as f32 / x1);
            output[i] = self.filter.interpolate(x, samples);
        }
        let n = samples.len().min(A);
        if n < A {
            self.prev_samples.copy_within(A - n.., 0);
        }
        self.prev_samples[A - n..].copy_from_slice(&samples[samples.len() - n..]);
        output_len
    }
}

// TODO check that streaming resampling gives the same result as resampling in one go
