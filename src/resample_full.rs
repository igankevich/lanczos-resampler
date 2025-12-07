use crate::lerp;
use crate::Input;
use crate::LanczosFilter;
use crate::Output;

#[cfg(target_arch = "x86_64")]
mod x86_64;

/// Panics when input length or output sample rate is too large.
pub const fn output_len(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> usize {
    checked_output_len(input_len, input_sample_rate, output_sample_rate)
        .expect("Input length or output sample rate is too large")
}

/// Calculates resampled length of the input for given input/output sample rates.
///
/// This function doesn't track division remainders and should only be used when resampling the
/// whole audio track.
///
/// Returns `None` when input length or output sample rate is too large.
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
fn mean(input: &(impl Input + ?Sized)) -> f32 {
    let mut avg = 0.0;
    let mut n = 1;
    for i in 0..input.len() {
        avg += (input[i] - avg) / n as f32;
        n += 1;
    }
    avg
}

/// Panics when input length or output sample rate is too large.
pub fn resample<const N: usize, const A: usize>(
    input: &(impl Input + ?Sized),
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Vec<f32> {
    let input_len = input.len();
    if input_len == 0 {
        return Vec::new();
    }
    let output_len = output_len(input_len, input_sample_rate, output_sample_rate);
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
    do_resample_into::<N, A>(input, &mut output.spare_capacity_mut());
    // SAFETY: We initialize all elements in `do_resample_into`.
    unsafe { output.set_len(output_len) }
    output
}

/// Panics when input length or output sample rate is too large.
pub fn resample_into<const N: usize, const A: usize>(
    input: &(impl Input + ?Sized),
    input_sample_rate: usize,
    output_sample_rate: usize,
    output: &mut impl Output,
) -> usize {
    let input_len = input.len();
    if input_len == 0 {
        return 0;
    }
    let output_len =
        output_len(input_len, input_sample_rate, output_sample_rate).min(output.remaining());
    if output_len == 0 {
        return 0;
    }
    if input_len == 1 {
        for _ in 0..output_len {
            output.write(input[0]);
        }
        return 1;
    }
    if output_len == 1 {
        output.write(mean(input));
        return input_len;
    }
    do_resample_into::<N, A>(input.take(input_len), output);
    input_len
}

pub(crate) fn do_resample_into_scalar<const N: usize, const A: usize>(
    input: &(impl Input + ?Sized),
    output: &mut impl Output,
) {
    let filter = LanczosFilter::<N, A>::new();
    let x0 = 0.0;
    let x1 = (input.len() - 1) as f32;
    let n = output.remaining();
    let i_max = (n - 1) as f32;
    for i in 0..n {
        let x = lerp(x0, x1, i as f32 / i_max);
        output.write(filter.interpolate(x, input).clamp(-1.0, 1.0));
    }
}

fn do_resample_into<const N: usize, const A: usize>(
    input: &(impl Input + ?Sized),
    output: &mut impl Output,
) {
    /*
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx")
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("sse4.1")
        && std::is_x86_feature_detected!("sse")
    {
        return x86_64::do_resample_into_avx::<N, A>(input, output);
    }
    */
    do_resample_into_scalar::<N, A>(input, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    parameterize! {
        resample_into_works
    }

    parameterize_impl! {
        (resample_works (16 2) (16 3))
    }

    fn resample_works<const N: usize, const A: usize>() {
        assert_eq!(
            vec![0.0; 100],
            resample::<N, A>(vec![0.0; 100].as_slice(), 100, 100)
        );
        for x in [0.1, -1.0, 1.0, 0.33] {
            assert_vec_f32_near!(
                vec![x; 200],
                resample::<N, A>(vec![x; 100].as_slice(), 100, 200),
                1e-1
            );
            assert_vec_f32_near!(
                vec![x; 100],
                resample::<N, A>(vec![x; 200].as_slice(), 200, 100),
                1e-2
            );
            assert_vec_f32_near!(
                vec![x; 100],
                resample::<N, A>(vec![x; 100].as_slice(), 100, 100),
                1e-2
            );
        }
    }

    fn resample_into_works<const N: usize, const A: usize>() {
        arbtest(|u| {
            let input_len = u.int_in_range(0..=1000)?;
            let input = arbitrary_samples(u, input_len)?;
            let input_sample_rate = u.int_in_range(1..=100)?;
            let output_sample_rate = u.int_in_range(1..=100)?;
            let expected = resample::<N, A>(&input[..], input_sample_rate, output_sample_rate);
            let mut actual =
                vec![f32::NAN; output_len(input_len, input_sample_rate, output_sample_rate)];
            let mut output = &mut actual[..];
            let num_read = resample_into::<N, A>(
                &input[..],
                input_sample_rate,
                output_sample_rate,
                &mut output,
            );
            assert_eq!(0, output.len());
            assert_eq!(expected, actual);
            if !expected.is_empty() {
                assert_eq!(input_len, num_read);
            }
            Ok(())
        });
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

    #[test]
    fn mean_works() {
        assert_eq!(1.0, mean(&[1.0, 1.0]));
        assert_eq!(2.0, mean(&[2.0, 2.0]));
    }
}
