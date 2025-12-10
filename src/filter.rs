use crate::LanczosKernel;
use crate::floor;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[derive(Clone, Default)]
pub struct LanczosFilter<const N: usize, const A: usize> {
    kernel: LanczosKernel<N, A>,
}

impl<const N: usize, const A: usize> LanczosFilter<N, A> {
    pub fn new() -> Self {
        let kernel = LanczosKernel::new();
        Self { kernel }
    }

    pub fn interpolate(&self, x: f32, samples: &[f32]) -> f32 {
        debug_assert_ne!(0, samples.len());
        let i = floor(x) as usize;
        debug_assert!(i < samples.len());
        let mut sum = 0.0;
        self.do_interpolate(i, x, samples, &mut sum);
        sum
    }

    pub fn interpolate_chunk(&self, x: f32, chunk: &[f32], prev_chunk: &[f32]) -> f32 {
        let i = floor(x) as usize;
        let mut sum = 0.0;
        if i < A {
            let n = prev_chunk.len();
            let i_from = n.saturating_sub(A - i - 1);
            for (j, y) in prev_chunk.iter().enumerate().take(n).skip(i_from) {
                let k = n - j;
                sum += y * self.kernel.interpolate(x + k as f32);
            }
        }
        self.do_interpolate(i, x, chunk, &mut sum);
        sum
    }

    #[inline]
    fn do_interpolate(&self, i: usize, x: f32, samples: &[f32], sum: &mut f32) {
        let i_from = (i + 1).saturating_sub(A);
        let i_to = (i + 1 + A).min(samples.len());
        for (j, sample) in samples.iter().enumerate().take(i_to).skip(i_from) {
            *sum += sample * self.kernel.interpolate(x - j as f32);
        }
    }

    pub fn interpolate_interleaved(
        &self,
        x: f32,
        input: &[f32],
        num_channels: usize,
        output_frame: &mut [f32],
    ) {
        let frames = input.chunks_exact(num_channels);
        let i = floor(x) as usize;
        let i_from = (i + 1).saturating_sub(A);
        let i_to = (i + 1 + A).min(frames.len());
        output_frame.fill(0.0);
        for (j, frame) in frames.enumerate().take(i_to).skip(i_from) {
            let kernel = self.kernel.interpolate(x - j as f32);
            for (sample, out) in frame.iter().zip(output_frame.iter_mut()) {
                *out += sample * kernel;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lerp;
    use crate::tests::*;
    use alloc::vec;
    use arbtest::arbtest;

    parameterize! {
        interpolation
        chunked
    }

    fn interpolation<const N: usize, const A: usize>() {
        // `LanczosFilter` should interpolate to the values
        // that are near the values of the original function in every node.
        arbtest(|u| {
            let n: usize = u.int_in_range(2..=1000)?;
            let mut xs = vec![0.0; n];
            let mut samples = vec![0.0; n];
            for i in 0..n {
                let x0 = 0.0;
                let x1 = (n - 1) as f32;
                let x = lerp(x0, x1, i as f32 / x1);
                xs[i] = x;
                samples[i] = x.sin();
            }
            let lanczos = LanczosFilter::<N, A>::new();
            for i in 0..n {
                let x = xs[i];
                let expected = samples[i];
                let actual = lanczos.interpolate(x, &samples[..]);
                let eps = (expected - actual).abs();
                assert!(
                    eps < 1e-3,
                    "{x:20.6} {expected:20.6} {actual:20.6} {eps:20.6}",
                );
            }
            Ok(())
        });
    }

    fn chunked<const N: usize, const A: usize>() {
        arbtest(|u| {
            let input_len = u.int_in_range(2 + A..=100)?;
            let input = arbitrary_samples(u, input_len)?;
            let lanczos = LanczosFilter::<N, A>::new();
            let i_max = (input.len() - 1) as f32;
            for prev_len in 0..A {
                for i in 0..input_len {
                    let x0 = prev_len as f32;
                    let x1 = (input.len() - 1) as f32;
                    let x = lerp(x0, x1, i as f32 / i_max);
                    let y1 = lanczos.interpolate(x, &input[..]);
                    let (prev, chunk) = input.split_at(prev_len);
                    let x0 = 0.0;
                    let x1 = (chunk.len() - 1) as f32;
                    let x = lerp(x0, x1, i as f32 / i_max);
                    let y2 = lanczos.interpolate_chunk(x, chunk, prev);
                    assert_near!(y1, y2, 1e-4, "prev len = {prev_len}, i = {i}, A = {A}");
                }
            }
            Ok(())
        });
    }
}
