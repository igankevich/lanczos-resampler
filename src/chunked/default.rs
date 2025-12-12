use crate::DEFAULT_A;
use crate::DEFAULT_N;
use crate::LanczosFilter;
use crate::Output;
use crate::lerp;
use crate::whole::default::num_output_frames;

#[cfg(any(feature = "alloc", test))]
use alloc::vec::Vec;

/// A [`BasicChunkedResampler`] with default parameters: _N = 16, A = 3_.
pub type ChunkedResampler = BasicChunkedResampler<DEFAULT_N, DEFAULT_A>;

/// A resampler that processes audio input in chunks.
///
/// Use it to process audio streams.
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
    // We only need A - 1 points...
    prev_chunk: [f32; A],
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
    /// [`max_num_output_frames`](Self::max_num_output_frames).
    #[inline]
    pub fn set_output_sample_rate(&mut self, value: usize) {
        self.output_sample_rate = value;
    }

    /// Returns maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as [`num_output_frames`](crate::num_output_frames) plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// [`set_output_sample_rate`](Self::set_output_sample_rate).
    #[inline]
    pub fn max_num_output_frames(&self, num_input_frames: usize) -> usize {
        num_output_frames(
            num_input_frames,
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
    /// Returns 0 when either the input length is less than _max(2, A-1)_ or output length is less
    /// than 2, adjusted in accordance with sample rate ratio.
    ///
    /// # Limitations
    ///
    /// The output depends on the chunk size, hence resampling the same audio track as a whole and
    /// in chunks will produce slightly different results. This a consequence of the fact that Lanczos kernel
    /// isn't an interpolation function, but a filter. To minimize such discrepancies chunk size should
    /// be much larger than _2⋅A + 1_.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lanczos_resampler::ChunkedResampler;
    ///
    /// let n = 1024;
    /// let chunk = vec![0.1; n];
    /// let mut resampler = ChunkedResampler::new(44100, 48000);
    /// let mut output: Vec<f32> = Vec::with_capacity(resampler.max_num_output_frames(n));
    /// let num_processed = resampler.resample(&chunk[..], &mut output);
    /// assert_eq!(n, num_processed);
    /// ```
    pub fn resample(&mut self, chunk: &[f32], output: &mut impl Output) -> usize {
        // Determine how many input samples we can process and how many output samples we can
        // produce.
        let (chunk_len, output_len, remainder) =
            self.adjust_lengths(chunk.len(), output.remaining().unwrap_or(usize::MAX));
        if chunk_len < 2.max(A - 1) || output_len < 2 {
            return 0;
        }
        self.remainder = remainder;
        let chunk = &chunk[0..chunk_len];
        let x0 = 0.0;
        let x1 = (chunk_len - 1) as f32;
        let i_max = (output_len - 1) as f32;
        for i in 0..output_len {
            let x = lerp(x0, x1, i as f32 / i_max);
            let prev = &self.prev_chunk[1..];
            let y = self
                .filter
                .interpolate_chunk(x, chunk, prev)
                .clamp(-1.0, 1.0);
            output.write(y);
        }
        self.prev_chunk[1..].copy_from_slice(&chunk[chunk_len - (A - 1)..]);
        chunk_len
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

/// A [`BasicChunkedInterleavedResampler`] with default parameters: _N = 16, A = 3_.
#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ChunkedInterleavedResampler = BasicChunkedInterleavedResampler<DEFAULT_N, DEFAULT_A>;

/// A resampler that processes audio input in chunks; the channels are interleaved with each other.
///
/// Use it to process audio streams.
///
/// # Limitations
///
/// `ChunkedInterleavedResampler` produces slightly different output compared to processing the whole input at once.
/// If this is undesired, consider using [`WholeResampler`](crate::WholeResampler).
#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Clone)]
pub struct BasicChunkedInterleavedResampler<const N: usize, const A: usize> {
    filter: LanczosFilter<N, A>,
    input_sample_rate: usize,
    output_sample_rate: usize,
    num_channels: usize,
    remainder: usize,
    prev_chunk: Vec<f32>,
}

#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<const N: usize, const A: usize> BasicChunkedInterleavedResampler<N, A> {
    /// Creates new instance of resampler with the specified input and output sample rates and the
    /// number of channels.
    #[inline]
    pub fn new(input_sample_rate: usize, output_sample_rate: usize, num_channels: usize) -> Self {
        Self {
            filter: LanczosFilter::new(),
            input_sample_rate,
            output_sample_rate,
            num_channels,
            remainder: 0,
            prev_chunk: Vec::with_capacity((A - 1) * num_channels),
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

    /// Returns number of channels.
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Dynamically changes output sample rate.
    ///
    /// After changing the sample rate you should consider updating buffer size to
    /// [`max_num_output_frames`](Self::max_num_output_frames).
    #[inline]
    pub fn set_output_sample_rate(&mut self, value: usize) {
        self.output_sample_rate = value;
    }

    /// Returns maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as [`num_output_frames`](crate::num_output_frames) plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// [`set_output_sample_rate`](Self::set_output_sample_rate).
    #[inline]
    pub fn max_num_output_frames(&self, num_input_frames: usize) -> usize {
        if self.num_channels == 0 {
            return 0;
        }
        num_output_frames(
            num_input_frames,
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
        self.prev_chunk.clear();
    }

    /// Resamples input signal chunk from the source to the target sample rate and appends the
    /// resulting signal to the output.
    ///
    /// Returns the number of processed input frames. The output is clamped to _[-1; 1]_.
    ///
    /// For each [`input_sample_rate`](Self::input_sample_rate) input samples this method produces exactly
    /// [`output_sample_rate`](Self::output_sample_rate) output samples  even if it is called multiple times with a smaller
    /// amount of input samples; the only exception is when the output sample rate was changed in the process.
    ///
    /// # Edge cases
    ///
    /// Returns 0 when either the input length is less than _max(2, A-1)_ or output length is less
    /// than 2, adjusted in accordance with sample rate ratio.
    ///
    /// # Panics
    ///
    /// Panics when `chunk.len()` isn't evenly divisible by the number of channels.
    ///
    /// # Limitations
    ///
    /// The output depends on the chunk size, hence resampling the same audio track as a whole and
    /// in chunks will produce slightly different results. This a consequence of the fact that Lanczos kernel
    /// isn't an interpolation function, but a filter. To minimize such discrepancies chunk size should
    /// be much larger than _2⋅A + 1_.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lanczos_resampler::ChunkedInterleavedResampler;
    ///
    /// let n = 2 * 1024;
    /// let chunk = vec![0.1; n];
    /// let mut resampler = ChunkedInterleavedResampler::new(44100, 48000, 2);
    /// let mut output: Vec<f32> = Vec::with_capacity(resampler.max_num_output_frames(n));
    /// let num_processed = resampler.resample(&chunk[..], &mut output);
    /// assert_eq!(n, num_processed);
    /// ```
    pub fn resample(&mut self, chunk: &[f32], output: &mut impl Output) -> usize {
        if self.num_channels == 0 {
            return 0;
        }
        // Determine how many input samples we can process and how many output samples we can
        // produce.
        let num_input_samples = chunk.len();
        assert_eq!(0, num_input_samples % self.num_channels);
        let num_input_frames = num_input_samples / self.num_channels;
        let num_output_frames = output.remaining().unwrap_or(usize::MAX) / self.num_channels;
        let (num_input_frames, num_output_frames, remainder) =
            self.adjust_lengths(num_input_frames, num_output_frames);
        if num_input_frames < 2.max(A - 1) || num_output_frames < 2 {
            return 0;
        }
        self.remainder = remainder;
        let num_input_samples = num_input_frames * self.num_channels;
        let chunk = &chunk[0..num_input_samples];
        let x0 = 0.0;
        let x1 = (num_input_frames - 1) as f32;
        let i_max = (num_output_frames - 1) as f32;
        for i in 0..num_output_frames {
            let x = lerp(x0, x1, i as f32 / i_max);
            output.write_frame(self.num_channels, |output_frame| {
                self.filter.interpolate_chunk_interleaved(
                    x,
                    chunk,
                    &self.prev_chunk[..],
                    self.num_channels,
                    output_frame,
                );
                for sample in output_frame.iter_mut() {
                    *sample = sample.clamp(-1.0, 1.0);
                }
            });
        }
        let num_new_samples = (A - 1) * self.num_channels;
        self.prev_chunk.clear();
        self.prev_chunk
            .extend_from_slice(&chunk[num_input_samples - num_new_samples..num_input_samples]);
        num_input_samples
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
        (resample_interleaved (16 2) (16 3))
    }

    fn resample_streaming_simple<const N: usize, const A: usize>() {
        const M: usize = 10;
        const O: usize = 20;
        let input = sine_wave(M);
        let whole_resampler = BasicWholeResampler::<N, A>::new();
        let expected_output = whole_resampler.resample(&input[..], M, O);
        let mut resampler = BasicChunkedResampler::<N, A>::new(M, O);
        let mut output = [f32::NAN; O];
        let mut output = &mut output[..];
        let num_read = resampler.resample(&input[..M / 2], &mut output);
        assert_eq!(M / 2, num_read);
        let num_read = resampler.resample(&input[M / 2..], &mut output);
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
                whole_resampler.resample(&input[..], input_sample_rate, output_sample_rate);
            let min_chunk_len = input_sample_rate.max(A - 1);
            let chunks = arbitrary_chunks(u, input_len, min_chunk_len, min_chunk_len)?;
            if chunks.is_empty() {
                return Ok(());
            }
            let mut resampler =
                BasicChunkedResampler::<N, A>::new(input_sample_rate, output_sample_rate);
            let mut actual_output = vec![f32::NAN; expected_output.len()];
            let mut input_slice = &input[..];
            let mut output_slice = &mut actual_output[..];
            for chunk_len in chunks.iter().copied() {
                let num_read = resampler.resample(&input_slice[..chunk_len], &mut output_slice);
                assert_ne!(0, num_read, "chunk_len = {chunk_len}");
                input_slice = &input_slice[num_read..];
            }
            assert_eq!(&[] as &[f32], input_slice);
            assert_eq!(&[] as &[f32], output_slice);
            assert_vec_f32_near_relative!(expected_output, actual_output, 0.30);
            Ok(())
        });
    }

    fn resample_interleaved<const N: usize, const A: usize>() {
        arbtest(|u| {
            let num_channels = u.int_in_range(1..=10)?;
            let input_sample_rate = u.int_in_range(1..=100)?;
            let output_sample_rate = 2 * input_sample_rate;
            let min_chunk_len = input_sample_rate.max(A - 1).max(2) * num_channels;
            let input_len = u.int_in_range(min_chunk_len..=1000)?;
            let input = arbitrary_channels(u, input_len, num_channels)?;
            let chunks = arbitrary_chunks(u, input[0].len(), min_chunk_len, min_chunk_len)?;
            let interleaved_input = interleave(&input);
            let expected = interleave(
                &input
                    .iter()
                    .map(|channel| {
                        if chunks.is_empty() {
                            return Ok(Vec::new());
                        }
                        let mut resampler = BasicChunkedResampler::<N, A>::new(
                            input_sample_rate,
                            output_sample_rate,
                        );
                        let mut output = Vec::new();
                        let mut input_slice = &channel[..];
                        for chunk_len in chunks.iter().copied() {
                            let num_read =
                                resampler.resample(&input_slice[..chunk_len], &mut output);
                            assert_ne!(
                                0, num_read,
                                "chunk_len = {chunk_len}, \
                                sample_rate = {input_sample_rate} / {output_sample_rate}, \
                                num_channels = {num_channels}"
                            );
                            input_slice = &input_slice[num_read..];
                        }
                        Ok(output)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let mut resampler = BasicChunkedInterleavedResampler::<N, A>::new(
                input_sample_rate,
                output_sample_rate,
                num_channels,
            );
            let expected_output_len =
                num_output_frames(input_len, input_sample_rate, output_sample_rate) * num_channels;
            let mut output = Vec::new();
            let mut input_slice = &interleaved_input[..];
            let mut num_processed = 0;
            for mut chunk_len in chunks.iter().copied() {
                chunk_len *= num_channels;
                let num_read = resampler.resample(&input_slice[..chunk_len], &mut output);
                assert_ne!(0, num_read, "chunk_len = {chunk_len}");
                input_slice = &input_slice[num_read..];
                num_processed += num_read;
            }
            assert_eq!(
                expected_output_len,
                output.len(),
                "Input length = {input_len}"
            );
            assert_eq!(expected, output);
            if !expected.is_empty() {
                assert_eq!(input_len * num_channels, num_processed);
            }
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
                u.int_in_range(1..=(max_sample_rate - output_sample_rate).max(1))?;
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
