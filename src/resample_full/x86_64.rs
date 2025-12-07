#![allow(unused)]

use super::*;
use crate::lerp_avx;
use crate::M256Ext;
use crate::F256;
use core::arch::x86_64::*;

pub(crate) fn do_resample_into_avx<const N: usize, const A: usize>(
    input: &(impl Input + ?Sized),
    output: &mut impl Output,
) {
    unsafe {
        let filter = LanczosFilter::<N, A>::new();
        let x0 = _mm256_setzero_ps();
        let x1 = _mm256_set1_ps((input.len() - 1) as f32);
        let n = output.remaining();
        let i_max = _mm256_set1_ps((n - 1) as f32);
        let steps = n / 8;
        for i0 in (0..steps * 8).step_by(8) {
            let i0 = i0 as i32;
            let i = _mm256_setr_epi32(i0, i0 + 1, i0 + 2, i0 + 3, i0 + 4, i0 + 5, i0 + 6, i0 + 7);
            let x = lerp_avx(x0, x1, _mm256_div_ps(_mm256_cvtepi32_ps(i), i_max));
            let mut y = filter.interpolate_avx(x, input);
            y = _mm256_max_ps(_mm256_set1_ps(-1.0), y);
            y = _mm256_min_ps(_mm256_set1_ps(1.0), y);
            output.write_slice(y.as_f32_array().as_slice());
        }
        let remainder = n % 8;
        if remainder != 0 {
            let i0 = (n - remainder) as i32;
            let mut i = F256::from_scalar(i0 as f32);
            for (index, j) in (i0..n as i32).enumerate() {
                i[index] = j as f32;
            }
            let x = lerp_avx(x0, x1, _mm256_div_ps(i.into(), i_max));
            let mut y = filter.interpolate_avx(x, input);
            y = _mm256_max_ps(_mm256_set1_ps(-1.0), y);
            y = _mm256_min_ps(_mm256_set1_ps(1.0), y);
            output.write_slice(&y.as_f32_array()[..remainder]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::do_resample_into_scalar;
    use crate::tests::*;

    parameterize! {
        do_resample_into_works
    }

    fn do_resample_into_works<const N: usize, const A: usize>() {
        if skip() {
            return;
        }
        arbtest(|u| {
            let input_len = u.int_in_range(0..=20)?;
            let input = arbitrary_samples(u, input_len)?;
            let input_sample_rate = u.int_in_range(1..=100)?;
            let output_sample_rate = u.int_in_range(1..=100)?;
            let expected =
                resample_scalar::<N, A>(&input[..], input_sample_rate, output_sample_rate);
            let actual = resample_avx::<N, A>(&input[..], input_sample_rate, output_sample_rate);
            assert_vec_f32_near!(expected, actual, 2.0 * f32::EPSILON);
            Ok(())
        });
    }

    fn resample_avx<const N: usize, const A: usize>(
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
        do_resample_into_avx::<N, A>(input, &mut output.spare_capacity_mut());
        // SAFETY: We initialize all elements in `do_resample_into_avx`.
        unsafe { output.set_len(output_len) }
        output
    }

    fn resample_scalar<const N: usize, const A: usize>(
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
        do_resample_into_scalar::<N, A>(input, &mut output.spare_capacity_mut());
        // SAFETY: We initialize all elements in `do_resample_into_scalar`.
        unsafe { output.set_len(output_len) }
        output
    }

    fn skip() -> bool {
        !(std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("sse4.1")
            && std::is_x86_feature_detected!("sse"))
    }
}
