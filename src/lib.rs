use core::f32::consts::PI;
use core::mem::MaybeUninit;

#[cfg(test)]
mod tests;

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
fn lanczos_kernel<const A: usize>(mut x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    // Ensure function symmetry.
    x = x.abs();
    if x > A as f32 {
        return 0.0;
    }
    let a = A as f32;
    sinc(x) * sinc(x / a)
}

fn dynamic_lanczos_kernel(mut x: f32, a: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    // Ensure function symmetry.
    x = x.abs();
    if x > a {
        return 0.0;
    }
    sinc(x) * sinc(x / a)
}

/// Interpolates `lanczos_kernel` in the range `[-A; A]` on a grid of `2 * N - 1` points using cubic
/// Hermite splines with second-order finite differences at spline endpoints.
///
/// See <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>.
pub struct LanczosKernel<const N: usize, const A: usize> {
    // We need only half of the points because Lanczos kernel is symmetric.
    //
    // Hence we store only values for `x >= 0`.
    kernel: [f32; N],
}

impl<const N: usize, const A: usize> LanczosKernel<N, A> {
    const _CHECK_1: () = assert!(A >= 1, "`A` can't be zero!");
    const _CHECK_2: () = assert!(
        N >= 2,
        "`LanczosKernel` needs at least two points for interpolation."
    );

    const X_MIN: f32 = -(A as f32);
    const X_MAX: f32 = A as f32;

    pub fn new() -> Self {
        let mut kernel = [0.0; N];
        for i in 0..N {
            let x = lerp(0.0, Self::X_MAX, i as f32 / (N - 1) as f32);
            kernel[i] = lanczos_kernel::<A>(x);
        }
        Self { kernel }
    }

    pub fn interpolate(&self, mut x: f32) -> f32 {
        debug_assert!(
            Self::X_MIN <= x && x <= Self::X_MAX,
            "x = {x}, x_min = {}, x_max = {}",
            Self::X_MIN,
            Self::X_MAX
        );
        if x == 0.0 {
            return 1.0;
        }
        // Ensure kernel symmetry.
        x = x.abs();
        let i_max = (N - 1) as f32;
        // Interpolate using cubic Hermite spline.
        // https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        // 1. Find 2-4 closest points.
        let (i00, i0, i1, i11) = {
            let tmp = x / Self::X_MAX;
            let mut i0 = (tmp * i_max).floor() as usize;
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
        let x0 = lerp(0.0, Self::X_MAX, i0 as f32 / i_max);
        let x1 = lerp(0.0, Self::X_MAX, i1 as f32 / i_max);
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
        h00 * self.kernel[i0] + h10 * dx * m0 + h01 * self.kernel[i1] + h11 * dx * m1
    }
}

struct DynamicLanczosKernel {
    kernel: Vec<f32>,
    a: f32,
}

impl DynamicLanczosKernel {
    fn new(n: usize, a: f32) -> Self {
        assert!(a >= 1.0, "A can't be zero!");
        assert!(
            n >= 2,
            "LanczosKernel needs at least two points for interpolation."
        );
        let mut kernel = vec![0.0; n];
        let x_max = a;
        for i in 0..n {
            let x = lerp(0.0, x_max, i as f32 / (n - 1) as f32);
            kernel[i] = dynamic_lanczos_kernel(x, a);
        }
        Self { kernel, a }
    }

    fn interpolate(&self, mut x: f32) -> f32 {
        let x_min = -self.a;
        let x_max = self.a;
        debug_assert!(
            x_min <= x && x <= x_max,
            "x = {x}, x_min = {x_min}, x_max = {x_max}",
        );
        if x == 0.0 {
            return 1.0;
        }
        // Ensure kernel symmetry.
        x = x.abs();
        let n = self.kernel.len();
        let i_max = (n - 1) as f32;
        // Interpolate using cubic Hermite spline.
        // https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        // 1. Find 2-4 closest points.
        let (i00, i0, i1, i11) = {
            let tmp = x / x_max;
            let mut i0 = (tmp * i_max).floor() as usize;
            if i0 == n - 1 {
                i0 -= 1;
            }
            let i1 = i0 + 1;
            let i00 = (i0 > 0).then(|| i0 - 1);
            let i2 = (i1 < n - 1).then(|| i1 + 1);
            (i00, i0, i1, i2)
        };
        debug_assert!(i0 < n);
        debug_assert!(i1 < n);
        let x0 = lerp(0.0, x_max, i0 as f32 / i_max);
        let x1 = lerp(0.0, x_max, i1 as f32 / i_max);
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
        h00 * self.kernel[i0] + h10 * dx * m0 + h01 * self.kernel[i1] + h11 * dx * m1
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
        let i_from = (i + 1).saturating_sub(A);
        let i_to = (i + A).min(samples.len() - 1);
        let mut sum = 0.0;
        for j in i_from..=i_to {
            sum += samples[j] * self.kernel.interpolate(x - j as f32);
        }
        sum
    }

    pub fn interpolate_chunk(&self, x: f32, chunk: &[f32], prev_chunk: &[f32]) -> f32 {
        let i = x.floor() as usize;
        let mut sum = 0.0;
        let n = prev_chunk.len();
        if i < A {
            let i_from = n.saturating_sub(A - i - 1);
            for j in i_from..n {
                let k = n - j;
                sum += prev_chunk[j] * self.kernel.interpolate(x + k as f32);
            }
        }
        let i_from = (i + 1).saturating_sub(A);
        let i_to = (i + A).min(chunk.len() - 1);
        for j in i_from..=i_to {
            sum += chunk[j] * self.kernel.interpolate(x - j as f32);
        }
        sum
    }
}

pub fn resample<const N: usize, const A: usize>(
    input: &[f32],
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Vec<f32> {
    let input_len = input.len();
    if input_len == 0 {
        return Vec::new();
    }
    let output_len = checked_output_len(input_len, input_sample_rate, output_sample_rate)
        .expect("Overflow while determining output length");
    if output_len == 0 {
        return Vec::new();
    }
    if input_len == 1 {
        return vec![input[0]; output_len];
    }
    if output_len == 1 {
        return vec![mean(input); output_len];
    }
    let mut output = Vec::with_capacity(output_len);
    do_resample_into::<N, A>(input, output.spare_capacity_mut());
    // SAFETY: We initialize all elements in `do_resample_into`.
    unsafe { output.set_len(output_len) }
    output
}

fn do_resample_into<const N: usize, const A: usize>(input: &[f32], output: &mut [impl WriteF32]) {
    let filter = LanczosFilter::<N, A>::new();
    let x0 = 0.0;
    let x1 = (input.len() - 1) as f32;
    let i_max = (output.len() - 1) as f32;
    for (i, out) in output.iter_mut().enumerate() {
        let x = lerp(x0, x1, i as f32 / i_max);
        out.write(filter.interpolate(x, input).clamp(-1.0, 1.0));
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

pub struct LanczosResampler<const N: usize, const A: usize> {
    filter: LanczosFilter<N, A>,
    input_sample_rate: usize,
    output_sample_rate: usize,
    remainder: usize,
    // TODO we only need A - 1 points...
    prev_chunk: [f32; A],
    /// The length is counted from the back of the array.
    prev_chunk_len: usize,
}

impl<const N: usize, const A: usize> LanczosResampler<N, A> {
    pub fn new(input_sample_rate: usize, output_sample_rate: usize) -> Self {
        Self {
            filter: LanczosFilter::new(),
            input_sample_rate,
            output_sample_rate,
            remainder: 0,
            prev_chunk: [0.0; A],
            prev_chunk_len: 0,
        }
    }

    pub fn set_output_sample_rate(&mut self, value: usize) {
        self.output_sample_rate = value;
    }

    fn adjust_lengths(&self, input_len: usize, output_len: usize) -> (usize, usize, usize) {
        adjust_lengths(
            input_len,
            output_len,
            self.input_sample_rate,
            self.output_sample_rate,
            self.remainder,
        )
    }

    pub fn resample_into(&mut self, input: &[f32], output: &mut [impl WriteF32]) -> (usize, usize) {
        // Determine how many input sampels we can process and how many output samples we can
        // produce.
        let (input_len, output_len, remainder) = self.adjust_lengths(input.len(), output.len());
        if input_len <= 1 || output_len <= 1 {
            return (0, 0);
        }
        self.remainder = remainder;
        let input = &input[..input_len];
        let output = &mut output[..output_len];
        let x0 = 0.0;
        let x1 = (input_len - 1) as f32;
        let i_max = (output_len - 1) as f32;
        for (i, out) in output.iter_mut().enumerate() {
            let x = lerp(x0, x1, i as f32 / i_max);
            let prev = &self.prev_chunk[A - self.prev_chunk_len..];
            let y = self
                .filter
                .interpolate_chunk(x, input, prev)
                .clamp(-1.0, 1.0);
            out.write(y);
        }
        let n = input.len().min(A - 1);
        self.push_prev(&input[input.len() - n..]);
        (input_len, output_len)
    }

    #[inline]
    fn push_prev(&mut self, samples: &[f32]) {
        let n = samples.len();
        if n < self.prev_chunk_len {
            // Shift left by `n`.
            let i = (A - self.prev_chunk_len).max(n);
            self.prev_chunk.copy_within(i.., i - n);
        }
        self.prev_chunk[A - n..].copy_from_slice(samples);
        self.prev_chunk_len += n;
        if self.prev_chunk_len > A {
            self.prev_chunk_len = A;
        }
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

/// Uses Welford's algorithm.
fn mean(xs: &[f32]) -> f32 {
    let mut avg = 0.0;
    let mut n = 1;
    for x in xs {
        avg += (x - avg) / n as f32;
        n += 1;
    }
    avg
}

fn adjust_lengths(
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
    //eprintln!(
    //    "    step 0 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}"
    //);
    let lhs = input_len * output_sample_rate + remainder;
    let rhs = output_len * input_sample_rate;
    if lhs < rhs {
        // One step is enough.
        output_len = lhs / input_sample_rate;
        remainder = lhs % input_sample_rate;
        //        eprintln!(
        //        "    step 1 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}"
        //    );
        return (input_len, output_len, remainder);
    }
    // TODO test
    // Do the second step with the new input length.
    input_len = (rhs / output_sample_rate).min(input_len);
    //eprintln!(
    //    "    step 1 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}"
    //);
    let lhs = input_len * output_sample_rate + remainder;
    output_len = lhs / input_sample_rate;
    remainder = lhs % input_sample_rate;
    //eprintln!(
    //    "    step 2 {input_len} {output_len} {input_sample_rate} {output_sample_rate} {remainder}"
    //);
    (input_len, output_len, remainder)
}
