use core::f32::consts::PI;
use core::mem::MaybeUninit;

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
    // TODO We need only half of the points because Lanczos kernel is symmetric.
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

    pub fn interpolate_with_prev_samples(
        &self,
        x: f32,
        samples: &[f32],
        prev_samples: &[f32],
    ) -> f32 {
        let i = x.floor() as usize;
        let mut sum = 0.0;
        if i < A {
            let n = prev_samples.len();
            let i_from = n.saturating_sub(A - i - 1);
            for j in i_from..n {
                let k = n - j;
                sum += prev_samples[j] * self.kernel.interpolate(x + k as f32);
            }
        }
        let i_from = i.saturating_sub(A) + 1;
        let i_to = (i + A).min(samples.len() - 1);
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
    output_len_remainder: usize,
    prev_samples: [f32; A],
}

impl<const N: usize, const A: usize> LanczosResampler<N, A> {
    pub fn new(input_sample_rate: usize, output_sample_rate: usize) -> Self {
        Self {
            filter: LanczosFilter::new(),
            input_sample_rate,
            output_sample_rate,
            output_len_remainder: 0,
            prev_samples: [0.0; A],
        }
    }

    fn adjust_lengths(&mut self, input_len: usize, output_len: usize) -> (usize, usize) {
        let (input_len, output_len, remainder) = adjust_lengths(
            input_len,
            output_len,
            self.input_sample_rate,
            self.output_sample_rate,
            self.output_len_remainder,
        );
        self.output_len_remainder = remainder;
        (input_len, output_len)
    }

    pub fn resample_into(&mut self, input: &[f32], output: &mut [impl WriteF32]) -> (usize, usize) {
        // Determine how many input sampels we can process and how many output samples we can
        // produce.
        let (input_len, output_len) = self.adjust_lengths(input.len(), output.len());
        if input_len == 0 || output_len == 0 {
            return (0, 0);
        }
        let input = &input[..input_len];
        let output = &mut output[..output_len];
        let x0 = 0.0;
        let x1 = (output_len - 1) as f32;
        for (i, out) in output.iter_mut().enumerate() {
            let x = lerp(x0, x1, i as f32 / x1);
            out.write(
                self.filter
                    .interpolate_with_prev_samples(x, input, &self.prev_samples[..]),
            );
        }
        let n = input.len().min(A);
        if n < A {
            self.prev_samples.copy_within(A - n.., 0);
        }
        self.prev_samples[A - n..].copy_from_slice(&input[input.len() - n..]);
        (input_len, output_len)
    }
}

pub const fn checked_output_len(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Option<usize> {
    if input_len == 0 || input_sample_rate == 0 || output_sample_rate == 0 {
        return Some(0);
    }
    if input_sample_rate == output_sample_rate {
        return Some(input_len);
    }
    match input_len.checked_mul(output_sample_rate) {
        Some(numerator) => Some(numerator / input_sample_rate),
        None => (input_len / input_sample_rate).checked_mul(output_sample_rate),
    }
}

#[test]
fn checked_output_len_works() {
    assert_eq!(Some(48000), checked_output_len(44100, 44100, 48000));
    assert_eq!(None, checked_output_len(usize::MAX, 44100, 48000));
    assert_eq!(
        Some(usize::MAX),
        checked_output_len(44100, 44100, usize::MAX)
    );
    assert_eq!(
        Some(48000),
        checked_output_len(usize::MAX, usize::MAX, 48000)
    );
}

pub fn resample<const N: usize, const A: usize>(
    input: &[f32],
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Vec<f32> {
    let output_len = checked_output_len(input.len(), input_sample_rate, output_sample_rate)
        .expect("Overflow while determining output length");
    let mut output = Vec::with_capacity(output_len);
    do_resample_into::<N, A>(input, output.spare_capacity_mut());
    // SAFETY: We initialize all elements in `do_resample_into`.
    unsafe { output.set_len(output_len) }
    output
}

fn do_resample_into<const N: usize, const A: usize>(input: &[f32], output: &mut [impl WriteF32]) {
    let filter = LanczosFilter::<N, A>::new();
    let output_len = output.len();
    let x0 = 0.0;
    let x1 = (output_len - 1) as f32;
    for (i, out) in output.iter_mut().enumerate() {
        let x = lerp(x0, x1, i as f32 / x1);
        out.write(filter.interpolate(x, input));
    }
}

pub trait WriteF32 {
    fn write(&mut self, value: f32);
}

impl WriteF32 for f32 {
    fn write(&mut self, value: f32) {
        *self = value;
    }
}

impl WriteF32 for MaybeUninit<f32> {
    fn write(&mut self, value: f32) {
        MaybeUninit::<f32>::write(self, value);
    }
}

// TODO check that streaming resampling gives the same result as resampling in one go

#[inline]
const fn adjust_lengths(
    mut input_len: usize,
    mut output_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
    mut remainder: usize,
) -> (usize, usize, usize) {
    if input_len == 0 || output_len == 0 || input_sample_rate == 0 || output_sample_rate == 0 {
        return (0, 0, remainder);
    }
    // Clamp input length.
    let max_input_len = (usize::MAX / output_sample_rate).saturating_sub(remainder);
    if input_len > max_input_len {
        input_len = max_input_len;
    }
    if input_len == 0 {
        return (0, 0, remainder);
    }
    // Clamp output length.
    let max_output_len = usize::MAX / input_sample_rate;
    if output_len > max_output_len {
        output_len = max_output_len;
    }
    if output_len == 0 {
        return (0, 0, remainder);
    }
    // Do at most two steps of fixed-point iteration to determine output length.
    //eprintln!("    step 0 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}");
    let lhs = input_len * output_sample_rate + remainder;
    let rhs = output_len * input_sample_rate;
    if lhs < rhs {
        // One step was enough.
        output_len = lhs / input_sample_rate;
        remainder = lhs % input_sample_rate;
        //eprintln!("    step 1 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}");
        return (input_len, output_len, remainder);
    }
    // TODO test
    // Do the second step with the new input length.
    input_len = rhs / output_sample_rate;
    //eprintln!("    step 1 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}");
    let lhs = input_len * output_sample_rate + remainder;
    output_len = lhs / input_sample_rate;
    remainder = lhs % input_sample_rate;
    //eprintln!("    step 2 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}");
    (input_len, output_len, remainder)
}

#[test]
fn adjust_lengths_works() {
    assert_eq!(
        (44100, 48000, 0),
        adjust_lengths(44100, 48000, 44100, 48000, 0)
    );
    assert_eq!(
        (44100, 48000, 0),
        adjust_lengths(2 * 44100, 48000, 44100, 48000, 0)
    );
    assert_eq!(
        (44100, 48000, 0),
        adjust_lengths(44100, 2 * 48000, 44100, 48000, 0)
    );
    assert_eq!(
        (2 * 44100, 2 * 48000, 0),
        adjust_lengths(2 * 44100, 2 * 48000, 44100, 48000, 0)
    );
    assert_eq!(
        (44100 / 3, 48000 / 3, 0),
        adjust_lengths(44100 / 3, 48000, 44100, 48000, 0)
    );
    assert_eq!(
        (44100 / 3, 48000 / 3, 0),
        adjust_lengths(44100, 48000 / 3, 44100, 48000, 0)
    );
    assert_eq!(
        (44100, 48000, 0),
        adjust_lengths(usize::MAX, 48000, 44100, 48000, 0)
    );
    assert_eq!(
        (44100, 48000, 0),
        adjust_lengths(44100, usize::MAX, 44100, 48000, 0)
    );
    assert_eq!(
        (1, 1, 0),
        adjust_lengths(usize::MAX, usize::MAX, usize::MAX, usize::MAX, 0)
    );
    assert_eq!(
        (0, 0, 0),
        adjust_lengths(usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbtest::arbtest;

    fn max_sample_rate() -> usize {
        (usize::MAX as f64).sqrt().ceil() as usize
    }

    #[test]
    fn adjust_lengths_remainder_works() {
        arbtest(|u| {
            let max_sample_rate = max_sample_rate();
            let output_sample_rate: usize = u.int_in_range(1..=max_sample_rate)?;
            let input_sample_rate: usize =
                u.int_in_range(1..=max_sample_rate - output_sample_rate)?;
            let input_len = input_sample_rate;
            let output_len = output_sample_rate;
            let num_chunks: usize = u.int_in_range(1..=10.min(input_len))?;
            let max_chunk_len = input_len.div_ceil(num_chunks);
            // TODO output len should also be arbitrary
            let mut total_input_chunks_len = 0;
            let mut total_output_chunks_len = 0;
            let mut output_len_remainder = 0;
            let mut offset = 0;
            for i in 0..num_chunks {
                let chunk_len: usize = if i == num_chunks - 1 {
                    input_len - offset
                } else {
                    u.int_in_range(0..=max_chunk_len.min(input_len - offset))?
                };
                let (input_chunk_len, output_chunk_len, output_rem) = adjust_lengths(
                    chunk_len,
                    output_len,
                    input_sample_rate,
                    output_sample_rate,
                    output_len_remainder,
                );
                //eprintln!("{i} {num_chunks} {offset} adjust({chunk_len}, {output_len}, {input_sample_rate}, {output_sample_rate}, {output_len_remainder}) -> {input_chunk_len} {output_chunk_len} {output_rem}");
                output_len_remainder = output_rem;
                total_input_chunks_len += input_chunk_len;
                total_output_chunks_len += output_chunk_len;
                offset += input_chunk_len;
            }
            assert_eq!(input_len, total_input_chunks_len);
            assert_eq!(output_len, total_output_chunks_len);
            assert_eq!(0, output_len_remainder);
            Ok(())
        });
    }

    #[test]
    fn resample_works() {
        arbtest(|u| {
            let input: Vec<f32> = u.arbitrary()?;
            let input_sample_rate = input.len();
            let output_sample_rate = u.int_in_range(1..=max_sample_rate())?;
            let expected_output = resample::<11, 3>(&input, input_sample_rate, output_sample_rate);
            // TODO
            Ok(())
        });
    }
}
