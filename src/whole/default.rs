use crate::DEFAULT_A;
use crate::DEFAULT_N;
use crate::LanczosFilter;
use crate::Output;
use crate::lerp;

#[cfg(any(feature = "alloc", test))]
use alloc::vec::Vec;

/// Calculates resampled length of the input for given input/output sample
/// rates.
///
/// # Panics
///
/// Panics when the input length or the output sample rate is too large.
///
/// # Limitations
///
/// This function shouldn't be used when processing audio track in chunks;
/// use [`ChunkedResampler`](crate::ChunkedResampler) instead.
pub const fn output_len(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> usize {
    checked_output_len(input_len, input_sample_rate, output_sample_rate)
        .expect("Input length or output sample rate is too large")
}

/// Calculates resampled length of the input for given input/output sample
/// rates.
///
/// Returns `None` when input length or output sample rate is too large.
///
/// # Limitations
///
/// This function shouldn't be used when processing audio track in chunks;
/// use [`ChunkedResampler`](crate::ChunkedResampler) instead.
pub const fn checked_output_len(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Option<usize> {
    if input_len <= 1 || input_sample_rate == 0 || output_sample_rate == 0 {
        return Some(0);
    }
    if input_sample_rate == output_sample_rate {
        return Some(input_len);
    }
    let output_len = match input_len.checked_mul(output_sample_rate) {
        Some(numerator) => Some(numerator / input_sample_rate),
        None => (input_len / input_sample_rate).checked_mul(output_sample_rate),
    };
    match output_len {
        Some(output_len) if output_len <= 1 => Some(0),
        output_len => output_len,
    }
}

/// A [`BasicWholeResampler`] with default parameters: _N = 16, A = 3_.
pub type WholeResampler = BasicWholeResampler<DEFAULT_N, DEFAULT_A>;

/// A resampler that processes audio input as a whole.
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
/// `WholeResampler` shouldn't be used to process audio track in chunks;
/// use [`ChunkedResampler`](crate::ChunkedResampler) instead.
#[derive(Clone, Default)]
pub struct BasicWholeResampler<const N: usize, const A: usize> {
    filter: LanczosFilter<N, A>,
}

impl<const N: usize, const A: usize> BasicWholeResampler<N, A> {
    /// Creates new instance of resampler.
    #[inline]
    pub fn new() -> Self {
        let filter = LanczosFilter::new();
        Self { filter }
    }

    /// Resamples input signal from the source to the target sample rate and
    /// returns the resulting output signal as a vector.
    ///
    /// # Edge cases
    ///
    /// Returns an empty vector when either the input length or calculated output length is less than 2.
    ///
    /// # Panics
    ///
    /// Panics when either the input length or the output sample rate is too large.
    ///
    /// # Limitations
    ///
    /// This function shouldn't be used when processing audio track in chunks;
    /// use [`ChunkedResampler::resample_chunk`](crate::ChunkedResampler::resample_chunk) instead.
    #[cfg(any(feature = "alloc", test))]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn resample_whole(
        &self,
        input: &[f32],
        input_sample_rate: usize,
        output_sample_rate: usize,
    ) -> Vec<f32> {
        // TODO Don't resample when rates are equal. Need to change tests after that.
        let input_len = input.len();
        if input_len <= 1 {
            return Vec::new();
        }
        let output_len = output_len(input_len, input_sample_rate, output_sample_rate);
        if output_len <= 1 {
            return Vec::new();
        }
        let mut output = Vec::with_capacity(output_len);
        self.do_resample_into(input, output_len, &mut output.spare_capacity_mut());
        // SAFETY: We initialize all elements in `do_resample_into`.
        unsafe { output.set_len(output_len) }
        output
    }

    /// This is a variant of [`resample`](Self::resample) that doesn't use memory allocation.
    ///
    /// Returns the number of samples read from the input. Currently this is either 0 (see "Panics") or
    /// the input length.
    ///
    /// # Edge cases
    ///
    /// Returns 0 when either the input length or remaining output length is less than 2.
    ///
    /// # Panics
    ///
    /// - Panics when the output isn't large enough to hold all the resampled points.
    ///   Use [`output_len`] to ensure that the buffer size is sufficient.
    /// - Panics when the output is unbounded, i.e. [`Output::remaining`] returns `None`.
    ///
    /// # Limitations
    ///
    /// This function shouldn't be used when processing audio track in chunks;
    /// use [`ChunkedResampler::resample_chunk`](crate::ChunkedResampler::resample_chunk) instead.
    pub fn resample_whole_into(&self, input: &[f32], output: &mut impl Output) -> usize {
        let input_len = input.len();
        if input_len <= 1 {
            return 0;
        }
        let output_len = output
            .remaining()
            .expect("`resample_whole_into` doesn't support unbounded outputs");
        if output_len <= 1 {
            return 0;
        }
        self.do_resample_into(input, output_len, output);
        input_len
    }

    #[doc(hidden)]
    pub fn resample_interleaved_into(
        &self,
        input: &[f32],
        channels: usize,
        output: &mut &mut [f32],
    ) -> usize {
        if channels == 1 {
            return self.resample_whole_into(input, output);
        }
        assert_ne!(0, channels);
        let input_len = input.len();
        assert_eq!(0, input_len % channels);
        let num_input_frames = input_len / channels;
        if num_input_frames <= 1 {
            return 0;
        }
        let output_len = output.len();
        assert_eq!(0, output_len % channels);
        let num_output_frames = output_len / channels;
        if num_output_frames <= 1 {
            return 0;
        }
        let x0 = 0.0;
        let x1 = (num_input_frames - 1) as f32;
        let i_max = (num_output_frames - 1) as f32;
        for (i, output_frame) in output.chunks_exact_mut(channels).enumerate() {
            let x = lerp(x0, x1, i as f32 / i_max);
            self.filter
                .interpolate_interleaved(x, input, channels, output_frame);
            for sample in output_frame.iter_mut() {
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        let slice = core::mem::take(output);
        *output = &mut slice[num_output_frames * channels..];
        num_input_frames
    }

    pub(crate) fn do_resample_into_scalar(
        &self,
        input: &[f32],
        output_len: usize,
        output: &mut impl Output,
    ) {
        let x0 = 0.0;
        let x1 = (input.len() - 1) as f32;
        let i_max = (output_len - 1) as f32;
        for i in 0..output_len {
            let x = lerp(x0, x1, i as f32 / i_max);
            output.write(self.filter.interpolate(x, input).clamp(-1.0, 1.0));
        }
    }

    #[inline]
    fn do_resample_into(&self, input: &[f32], output_len: usize, output: &mut impl Output) {
        // TODO Current SIMD implementation is slow. Should consider using SIMD for interleaved
        // data...
        /*
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("sse4.1")
            && std::is_x86_feature_detected!("sse")
        {
            return self.do_resample_into_avx::<N, A>(input, output);
        }
        */
        self.do_resample_into_scalar(input, output_len, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use alloc::vec;

    parameterize! {
        resample_into_works
        resample_interleaved_into_works
    }

    parameterize_impl! {
        (resample_works (16 2) (16 3))
    }

    fn resample_works<const N: usize, const A: usize>() {
        let resampler = BasicWholeResampler::<N, A>::new();
        assert_eq!(
            vec![0.0; 100],
            resampler.resample_whole(vec![0.0; 100].as_slice(), 100, 100)
        );
        for x in [0.1, -1.0, 1.0, 0.33] {
            assert_vec_f32_near!(
                vec![x; 200],
                resampler.resample_whole(vec![x; 100].as_slice(), 100, 200),
                1e-1
            );
            assert_vec_f32_near!(
                vec![x; 100],
                resampler.resample_whole(vec![x; 200].as_slice(), 200, 100),
                1e-2
            );
            assert_vec_f32_near!(
                vec![x; 100],
                resampler.resample_whole(vec![x; 100].as_slice(), 100, 100),
                1e-2
            );
        }
    }

    fn resample_into_works<const N: usize, const A: usize>() {
        let resampler = BasicWholeResampler::<N, A>::new();
        arbtest(|u| {
            let input_len = u.int_in_range(0..=1000)?;
            let input = arbitrary_samples(u, input_len)?;
            let input_sample_rate = u.int_in_range(1..=100)?;
            let output_sample_rate = u.int_in_range(1..=100)?;
            let expected =
                resampler.resample_whole(&input[..], input_sample_rate, output_sample_rate);
            let mut actual =
                vec![f32::NAN; output_len(input_len, input_sample_rate, output_sample_rate)];
            let mut output = &mut actual[..];
            let num_processed = resampler.resample_whole_into(&input[..], &mut output);
            assert_eq!(0, output.len(), "Input length = {input_len}");
            assert_eq!(expected, actual);
            if !expected.is_empty() {
                assert_eq!(input_len, num_processed);
            }
            Ok(())
        });
    }

    fn resample_interleaved_into_works<const N: usize, const A: usize>() {
        let resampler = BasicWholeResampler::<N, A>::new();
        arbtest(|u| {
            let num_channels = u.int_in_range(1..=10)?;
            let input_len = u.int_in_range(0..=1000)?;
            let input = arbitrary_channels(u, input_len, num_channels)?;
            let interleaved_input = interleave(&input);
            let input_sample_rate = u.int_in_range(1..=100)?;
            let output_sample_rate = u.int_in_range(1..=100)?;
            let expected = interleave(
                &input
                    .iter()
                    .map(|channel| {
                        resampler.resample_whole(
                            &channel[..],
                            input_sample_rate,
                            output_sample_rate,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let mut actual = vec![
                f32::NAN;
                output_len(input_len, input_sample_rate, output_sample_rate)
                    * num_channels
            ];
            let mut output = &mut actual[..];
            let num_processed = resampler.resample_interleaved_into(
                &interleaved_input[..],
                num_channels,
                &mut output,
            );
            assert_eq!(0, output.len(), "Input length = {input_len}");
            assert_eq!(expected, actual);
            if !expected.is_empty() {
                assert_eq!(input_len, num_processed);
            }
            Ok(())
        });
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        arbtest(|u| {
            let input_len = u.arbitrary()?;
            let input_sample_rate = input_len;
            let output_sample_rate = u.arbitrary()?;
            assert_eq!(
                Some(output_sample_rate),
                checked_output_len(input_len, input_sample_rate, output_sample_rate)
            );
            Ok(())
        });
        arbtest(|u| {
            let input_len = u.arbitrary()?;
            let input_sample_rate = u.arbitrary()?;
            let output_sample_rate = input_sample_rate;
            assert_eq!(
                Some(input_len),
                checked_output_len(input_len, input_sample_rate, output_sample_rate)
            );
            Ok(())
        });
    }
}
