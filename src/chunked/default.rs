use crate::DEFAULT_A;
use crate::DEFAULT_N;
use crate::Input;
use crate::LanczosFilter;
use crate::Output;
use crate::lerp;
use crate::whole::default::output_len;

/// A [`BasicChunkedResampler`] with default parameters: _N = 16, A = 3_.
pub type ChunkedResampler = BasicChunkedResampler<DEFAULT_N, DEFAULT_A>;

/// A resampler that processes audio input in chunks.
///
/// This struct uses [Lanczos kernel](https://en.wikipedia.org/wiki/Lanczos_resampling)
/// approximated by _2⋅N - 1_ points and defined on interval _[-A; A]_. The kernel is interpolated
/// using cubic Hermite splines with second-order finite differences at spline endpoints. The
/// output is clamped to _[-1; 1]_.
///
/// # Parameters
///
/// The recommended parameters are _N = 16, A = 3_. Using _A = 2_ might improve performance a
/// little bit. Using larger _N_ will techincally improve precision, but precision isn't a good
/// metric for audio signal. With _N = 16_ the kernel fits into exactly 64 B (the size of a cache line).
///
/// # Limitations
///
/// `ChunkedResampler` produces slightly different output compared to processing the whole input at once.
/// If this is undesired, consider using [`WholeResampler`](crate::WholeResampler).
#[derive(Clone)]
pub struct BasicChunkedResampler<const N: usize, const A: usize> {
    filter: LanczosFilter<N, A>,
    input_sample_rate: usize,
    output_sample_rate: usize,
    remainder: usize,
    // TODO we only need A - 1 points...
    prev_chunk: [f32; A],
    /// The length is counted from the back of the array.
    prev_chunk_len: usize,
}

impl<const N: usize, const A: usize> BasicChunkedResampler<N, A> {
    /// Creates new instance of resampler with the specified input and output sample rates.
    #[inline]
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

    /// Returns input sample rate in Hz.
    #[inline]
    pub fn input_sample_rate(&self) -> usize {
        self.input_sample_rate
    }

    /// Returns output sample rate in Hz.
    #[inline]
    pub fn output_sample_rate(&self) -> usize {
        self.output_sample_rate
    }

    /// Dynamically changes output sample rate.
    ///
    /// After changing the sample rate you should consider updating buffer size to
    /// [`max_output_chunk_len`](Self::max_output_chunk_len).
    #[inline]
    pub fn set_output_sample_rate(&mut self, value: usize) {
        self.output_sample_rate = value;
    }

    /// Returns maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as [`output_len`](crate::output_len) plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// [`set_output_sample_rate`](Self::set_output_sample_rate).
    #[inline]
    pub fn max_output_chunk_len(&self, input_chunk_len: usize) -> usize {
        output_len(
            input_chunk_len,
            self.input_sample_rate,
            self.output_sample_rate,
        ) + 1
    }

    /// Resets internal state.
    ///
    /// Erases any information about the previous chunk.
    ///
    /// Use this method when you want to reuse resampler for another audio stream.
    #[inline]
    pub fn reset(&mut self) {
        self.remainder = 0;
        self.prev_chunk_len = 0;
    }

    /// Resamples input signal chunk from the source to the target sample rate and appends the
    /// resulting signal to the output.
    ///
    /// Returns the number of processed input samples. The output is clamped to _[-1; 1]_.
    ///
    /// For each [`input_sample_rate`](Self::input_sample_rate) input samples this method produces exactly
    /// [`output_sample_rate`](Self::output_sample_rate) output samples  even if it is called multiple times with a smaller
    /// amount of input samples; the only exception is when the output sample rate was changed in the process.
    ///
    /// # Edge cases
    ///
    /// Returns 0 when either the input length or output length is less than 2, adjusted in
    /// accordance with sample rate ratio.
    ///
    /// # Limitations
    ///
    /// The output depends on the chunk size, hence resampling the same audio track all at once and
    /// in chunks will produce slightly different results. This a consequence of the fact that Lanczos kernel
    /// isn't an interpolation function, but a filter. To minimize such discrepancies chunk size should
    /// be much larger than _2⋅A + 1_.
    pub fn resample_chunk(
        &mut self,
        chunk: &(impl Input + ?Sized),
        output: &mut impl Output,
    ) -> usize {
        // Determine how many input samples we can process and how many output samples we can
        // produce.
        let (chunk_len, output_len, remainder) =
            self.adjust_lengths(chunk.len(), output.remaining().unwrap_or(usize::MAX));
        if chunk_len <= 1 || output_len <= 1 {
            return 0;
        }
        self.remainder = remainder;
        let chunk = chunk.slice(0..chunk_len);
        let x0 = 0.0;
        let x1 = (chunk_len - 1) as f32;
        let i_max = (output_len - 1) as f32;
        for i in 0..output_len {
            let x = lerp(x0, x1, i as f32 / i_max);
            let prev = &self.prev_chunk[A - self.prev_chunk_len..];
            let y = self
                .filter
                .interpolate_chunk(x, &chunk, prev)
                .clamp(-1.0, 1.0);
            output.write(y);
        }
        let chunk_len = chunk.len();
        let n = chunk_len.min(A - 1);
        for i in chunk_len - n..chunk_len {
            let sample = chunk.get(i);
            self.push_prev(sample);
        }
        chunk_len
    }

    #[inline]
    fn push_prev(&mut self, sample: f32) {
        if 1 < self.prev_chunk_len {
            // Shift left by 1.
            let i = (A - self.prev_chunk_len).max(1);
            self.prev_chunk.copy_within(i.., i - 1);
        }
        self.prev_chunk[A - 1] = sample;
        if self.prev_chunk_len < A {
            self.prev_chunk_len += 1;
        }
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
}

#[inline]
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
    let lhs = input_len * output_sample_rate + remainder;
    let rhs = output_len * input_sample_rate;
    if lhs < rhs {
        // One step is enough.
        output_len = lhs / input_sample_rate;
        remainder = lhs % input_sample_rate;
        return (input_len, output_len, remainder);
    }
    // Do the second step with the new input length.
    input_len = (rhs / output_sample_rate).min(input_len);
    let lhs = input_len * output_sample_rate + remainder;
    output_len = lhs / input_sample_rate;
    remainder = lhs % input_sample_rate;
    (input_len, output_len, remainder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use crate::whole::default::BasicWholeResampler;
    use alloc::vec;
    use alloc::vec::Vec;

    parameterize_impl! {
        (resample_streaming_simple (16 2) (16 3))
        (resample_streaming (16 2) (16 3))
    }

    fn resample_streaming_simple<const N: usize, const A: usize>() {
        const M: usize = 10;
        const O: usize = 20;
        let input = sine_wave(M);
        let whole_resampler = BasicWholeResampler::<N, A>::new();
        let expected_output = whole_resampler.resample_whole(&input[..], M, O);
        let mut resampler = BasicChunkedResampler::<N, A>::new(M, O);
        let mut output = [f32::NAN; O];
        let mut output = &mut output[..];
        let num_read = resampler.resample_chunk(&input[..M / 2], &mut output);
        assert_eq!(M / 2, num_read);
        let num_read = resampler.resample_chunk(&input[M / 2..], &mut output);
        assert_eq!(M / 2, num_read);
        assert_vec_f32_near_relative!(expected_output, output, 0.20);
    }

    fn resample_streaming<const N: usize, const A: usize>() {
        // Check that streaming resampling gives the same result as resampling in one go.
        let whole_resampler = BasicWholeResampler::<N, A>::new();
        arbtest(|u| {
            let input_sample_rate = 2 * u.int_in_range(1..=100)?;
            let input_len: usize = input_sample_rate * u.int_in_range(1..=10)?;
            let input = sine_wave(input_len);
            let output_sample_rate = 2 * input_sample_rate;
            let expected_output =
                whole_resampler.resample_whole(&input[..], input_sample_rate, output_sample_rate);
            let min_chunk_len = input_sample_rate;
            let max_chunk_len = input_sample_rate;
            let mut num_chunks = input_len / max_chunk_len;
            let mut chunks = Vec::with_capacity(num_chunks);
            {
                let mut offset = 0;
                for i in 0..num_chunks {
                    let chunk_len: usize = if i == num_chunks - 1 {
                        input.len() - offset
                    } else {
                        let max = max_chunk_len.min(input.len() - offset);
                        let min = min_chunk_len.min(max);
                        u.int_in_range(min..=max)?
                    };
                    chunks.push(chunk_len);
                    offset += chunk_len;
                }
                while let Some(chunk) = chunks.last() {
                    let chunk_len = *chunk;
                    if chunk_len >= min_chunk_len {
                        break;
                    }
                    chunks.pop();
                    if let Some(chunk) = chunks.last_mut() {
                        *chunk += chunk_len;
                    }
                }
                num_chunks = chunks.len();
                if num_chunks == 0 {
                    return Ok(());
                }
            }
            let mut resampler =
                BasicChunkedResampler::<N, A>::new(input_sample_rate, output_sample_rate);
            let mut actual_output = vec![f32::NAN; expected_output.len()];
            let mut input_slice = &input[..];
            let mut output_slice = &mut actual_output[..];
            for chunk_len in chunks.iter().copied() {
                let num_read =
                    resampler.resample_chunk(&input_slice[..chunk_len], &mut output_slice);
                input_slice = &input_slice[num_read..];
            }
            assert_eq!(&[] as &[f32], input_slice);
            assert_eq!(&[] as &[f32], output_slice);
            assert_vec_f32_near_relative!(expected_output, actual_output, 0.30);
            Ok(())
        });
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
            (0, 0, usize::MAX),
            adjust_lengths(usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
        );
    }

    fn max_sample_rate() -> usize {
        (usize::MAX as f64).sqrt().ceil() as usize
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
}
