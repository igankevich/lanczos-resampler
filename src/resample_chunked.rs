use crate::LanczosFilter;
use crate::Output;
use crate::lerp;

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

    pub fn resample_into(&mut self, input: &[f32], output: &mut impl Output) -> usize {
        // Determine how many input sampels we can process and how many output samples we can
        // produce.
        let (input_len, output_len, remainder) =
            self.adjust_lengths(input.len(), output.remaining());
        if input_len <= 1 || output_len <= 1 {
            return 0;
        }
        self.remainder = remainder;
        let input = &input[..input_len];
        let x0 = 0.0;
        let x1 = (input_len - 1) as f32;
        let i_max = (output_len - 1) as f32;
        for i in 0..output_len {
            let x = lerp(x0, x1, i as f32 / i_max);
            let prev = &self.prev_chunk[A - self.prev_chunk_len..];
            let y = self
                .filter
                .interpolate_chunk(x, input, prev)
                .clamp(-1.0, 1.0);
            output.write(y);
        }
        let n = input.len().min(A - 1);
        self.push_prev(&input[input.len() - n..]);
        input_len
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample;
    use crate::tests::*;

    parameterize_impl! {
        (resample_streaming_simple (16 2) (16 3))
        (resample_streaming (16 2) (16 3))
    }

    fn resample_streaming_simple<const N: usize, const A: usize>() {
        const M: usize = 10;
        const O: usize = 20;
        let input = sine_wave(M);
        let expected_output = resample::<N, A>(&input[..], M, O);
        let mut resampler = LanczosResampler::<N, A>::new(M, O);
        let mut output = [f32::NAN; O];
        let mut output = &mut output[..];
        let num_read = resampler.resample_into(&input[..M / 2], &mut output);
        assert_eq!(M / 2, num_read);
        let num_read = resampler.resample_into(&input[M / 2..], &mut output);
        assert_eq!(M / 2, num_read);
        assert_vec_f32_near_relative!(expected_output, output, 0.20);
    }

    fn resample_streaming<const N: usize, const A: usize>() {
        // Check that streaming resampling gives the same result as resampling in one go.
        arbtest(|u| {
            let input_sample_rate = 2 * u.int_in_range(1..=100)?;
            let input_len: usize = input_sample_rate * u.int_in_range(1..=10)?;
            let input = sine_wave(input_len);
            let output_sample_rate = 2 * input_sample_rate;
            let expected_output =
                resample::<N, A>(&input[..], input_sample_rate, output_sample_rate);
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
                LanczosResampler::<N, A>::new(input_sample_rate, output_sample_rate);
            let mut actual_output = vec![f32::NAN; expected_output.len()];
            let mut input_slice = &input[..];
            let mut output_slice = &mut actual_output[..];
            for chunk_len in chunks.iter().copied() {
                let num_read =
                    resampler.resample_into(&input_slice[..chunk_len], &mut output_slice);
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
