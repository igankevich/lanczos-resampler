#![allow(unused)]

use super::*;
use crate::M256Ext;
use crate::F256;
use crate::I256;
use core::arch::x86_64::*;
use seq_macro::seq;

impl<const N: usize, const A: usize> LanczosFilter<N, A> {
    pub fn interpolate_avx(&self, x: __m256, samples: &(impl Input + ?Sized)) -> __m256 {
        unsafe {
            // let i = x.floor() as usize;
            let i = _mm256_cvtps_epi32(_mm256_floor_ps(x));
            // let mut sum = 0.0;
            let mut sum = F256::zero();
            let initial_index = _mm256_set1_epi32(0);
            self.do_interpolate_avx(i, x, samples, initial_index, &mut sum);
            sum.into()
        }
    }

    #[inline]
    fn do_interpolate_avx(
        &self,
        i: __m256i,
        x: __m256,
        samples: &(impl Input + ?Sized),
        initial_index: __m256i,
        sum: &mut F256,
    ) {
        unsafe {
            // let i_from = (i + 1).saturating_sub(A);
            let mut i_from = _mm256_sub_epi32(
                _mm256_add_epi32(i, _mm256_set1_epi32(1)),
                _mm256_set1_epi32(A as i32),
            );
            let mask = _mm256_cmpgt_epi32(i_from, _mm256_set1_epi32(0));
            i_from = _mm256_and_si256(mask, i_from);
            // let i_to = (i + A).min(samples.len() - 1);
            let mut i_to = _mm256_add_epi32(i, _mm256_set1_epi32(A as i32));
            let samples_len_sub_1 = _mm256_set1_epi32((samples.len() - 1) as i32);
            let mask = _mm256_cmpgt_epi32(i_to, samples_len_sub_1);
            let mask_inv = _mm256_xor_si256(mask, _mm256_set1_epi32(-1));
            i_to = _mm256_or_si256(
                _mm256_and_si256(mask, samples_len_sub_1),
                _mm256_and_si256(mask_inv, i_to),
            );
            // for j in i_from..=i_to {
            //     sum += samples[j] * self.kernel.interpolate(x - j as f32);
            // }
            seq!(k in 0..8 {
                let i_from_k = _mm256_extract_epi32(i_from, k);
                let i_to_k = _mm256_extract_epi32(i_to, k);
                let x_f = core::mem::transmute::<__m256, F256>(x)[k];
                // TODO Add Input::as_ptr() to be able to gather?
                let mut sample = F256::zero();
                let mut x_sub_j = F256::zero();
                let mut index = _mm256_extract_epi32(initial_index, k) as usize;
                for j in i_from_k..=i_to_k {
                    x_sub_j[index] = x_f - j as f32;
                    sample[index] = samples[j as usize];
                    index += 1;
                }
                sum[k] += _mm256_mul_ps(sample.into(), self.kernel.interpolate_avx(x_sub_j.into())).sum();
            });
        }
    }

    #[allow(unused)]
    pub fn interpolate_chunk_avx(&self, x: __m256, chunk: &[f32], prev_chunk: &[f32]) -> __m256 {
        unsafe {
            // let i = x.floor() as usize;
            let i: I256 = _mm256_cvtps_epi32(_mm256_floor_ps(x)).into();
            // let mut sum = 0.0;
            let mut sum = F256::zero();
            let mut initial_index = I256::zero();
            seq!(k in 0..8 {
                let i_k = i[k];
                let x_f = core::mem::transmute::<__m256, F256>(x)[k];
                let mut sample = F256::zero();
                let mut x_arg = F256::zero();
                let mut index = 0;
                if i_k < A as i32 {
                    let n = prev_chunk.len();
                    let i_from_k = n.saturating_sub(A - i_k as usize - 1);
                    for (j, s) in prev_chunk.iter().enumerate().take(n).skip(i_from_k) {
                        x_arg[index] = x_f + (n - j) as f32;
                        sample[index] = *s;
                        index += 1;
                    }
                    initial_index[k] = index as i32;
                }
                sum[k] += _mm256_mul_ps(sample.into(), self.kernel.interpolate_avx(x_arg.into())).sum();
            });
            self.do_interpolate_avx(i.into(), x, chunk, initial_index.into(), &mut sum);
            sum.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::F256;
    use crate::tests::*;

    parameterize! {
        interpolate
        interpolate_chunk
    }

    fn interpolate<const N: usize, const A: usize>() {
        if skip() {
            return;
        }
        let lanczos = LanczosFilter::<N, A>::new();
        arbtest(|u| unsafe {
            let input_len = u.int_in_range(2 + A..=100)?;
            let input = arbitrary_samples(u, input_len)?;
            let x_max = (input_len - 1) as f32;
            let x = F256([
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
            ]);
            let actual = {
                let x = _mm256_load_ps(x.as_ptr());
                let y = lanczos.interpolate_avx(x, &input[..]);
                let mut actual = F256([f32::NAN; 8]);
                _mm256_store_ps(actual.as_mut_ptr(), y);
                actual
            };
            let expected = {
                let mut expected = F256([f32::NAN; 8]);
                for (i, x) in x.0.iter().copied().enumerate() {
                    expected.0[i] = lanczos.interpolate(x, &input[..]);
                }
                expected
            };
            // Summation is done in different order, hence `near`.
            assert_vec_f32_near!(expected, actual, 2.0 * f32::EPSILON);
            Ok(())
        });
    }

    fn interpolate_chunk<const N: usize, const A: usize>() {
        if skip() {
            return;
        }
        let lanczos = LanczosFilter::<N, A>::new();
        arbtest(|u| unsafe {
            let input_len = u.int_in_range(2 + A..=100)?;
            let input = arbitrary_samples(u, input_len)?;
            let prev_input_len = u.int_in_range(0..=(A - 1))?;
            let prev_input = arbitrary_samples(u, prev_input_len)?;
            let x_max = (input_len - 1) as f32;
            let x = F256([
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
                arbitrary_f32(u, 0.0, x_max)?,
            ]);
            let actual = {
                let x = _mm256_load_ps(x.as_ptr());
                let y = lanczos.interpolate_chunk_avx(x, &input[..], &prev_input[..]);
                let mut actual = F256([f32::NAN; 8]);
                _mm256_store_ps(actual.as_mut_ptr(), y);
                actual
            };
            let expected = {
                let mut expected = F256([f32::NAN; 8]);
                for (i, x) in x.0.iter().copied().enumerate() {
                    expected.0[i] = lanczos.interpolate_chunk(x, &input[..], &prev_input[..]);
                }
                expected
            };
            // Summation is done in different order, hence `near`.
            assert_vec_f32_near!(expected, actual, 2.0 * f32::EPSILON);
            Ok(())
        });
    }

    fn skip() -> bool {
        !(std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("sse4.1")
            && std::is_x86_feature_detected!("sse"))
    }
}
