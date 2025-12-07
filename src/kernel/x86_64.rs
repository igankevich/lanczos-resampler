use super::*;
use core::arch::x86_64::*;
use core::mem::size_of;

pub(crate) fn lerp_avx(a: __m256, b: __m256, t: __m256) -> __m256 {
    // (1.0 - t) * a + t * b
    unsafe {
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), t), a),
            _mm256_mul_ps(t, b),
        )
    }
}

impl<const N: usize, const A: usize> LanczosKernel<N, A> {
    pub fn interpolate_avx(&self, x: __m256) -> __m256 {
        unsafe {
            // x = x.abs();
            let sign_bit = _mm256_set1_ps(-0.0);
            let x = _mm256_andnot_ps(sign_bit, x);
            // let i_max = (N - 1) as f32;
            let i_max = _mm256_set1_ps((N - 1) as f32);
            let (i00, i0, i1, i11) = {
                // let tmp = x / Self::X_MAX;
                let tmp = _mm256_div_ps(x, _mm256_set1_ps(Self::X_MAX));
                // let mut i0 = (tmp * i_max).floor() as usize;
                let mut i0 = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(tmp, i_max)));
                // if i0 == N - 1 {
                //     i0 -= 1;
                // }
                let n1 = _mm256_set1_epi32((N - 1) as u32 as i32);
                let mask = _mm256_cmpeq_epi32(i0, n1);
                i0 = _mm256_sub_epi32(i0, _mm256_and_si256(mask, _mm256_set1_epi32(1)));
                // let i1 = i0 + 1;
                let i1 = _mm256_add_epi32(i0, _mm256_set1_epi32(1));
                // let i00 = if i0 > 0 { i0 - 1 } else { i1 };
                let mask = _mm256_cmpgt_epi32(i0, _mm256_set1_epi32(0));
                let mask_inv = _mm256_xor_si256(mask, _mm256_set1_epi32(-1));
                let i00 = _mm256_or_si256(
                    _mm256_and_si256(mask, _mm256_sub_epi32(i0, _mm256_set1_epi32(1))),
                    _mm256_and_si256(mask_inv, i1),
                );
                // let i11 = if i1 < N - 1 { i1 + 1 } else { i0 };
                let mask = _mm256_cmpgt_epi32(n1, i1);
                let mask_inv = _mm256_xor_si256(mask, _mm256_set1_epi32(-1));
                let i11 = _mm256_or_si256(
                    _mm256_and_si256(mask, _mm256_add_epi32(i1, _mm256_set1_epi32(1))),
                    _mm256_and_si256(mask_inv, i0),
                );
                (i00, i0, i1, i11)
            };
            // let x0 = lerp(0.0, Self::X_MAX, i0 as f32 / i_max);
            let x0 = lerp_avx(
                _mm256_set1_ps(0.0),
                _mm256_set1_ps(Self::X_MAX),
                _mm256_div_ps(_mm256_cvtepi32_ps(i0), i_max),
            );
            // let x1 = lerp(0.0, Self::X_MAX, i1 as f32 / i_max);
            let x1 = lerp_avx(
                _mm256_set1_ps(0.0),
                _mm256_set1_ps(Self::X_MAX),
                _mm256_div_ps(_mm256_cvtepi32_ps(i1), i_max),
            );
            // let dx = x1 - x0;
            let dx = _mm256_sub_ps(x1, x0);
            // let t = (x - x0) / dx;
            let t = _mm256_div_ps(_mm256_sub_ps(x, x0), dx);
            // let t1 = (1.0 - t) * (1.0 - t);
            let one_sub_t = _mm256_sub_ps(_mm256_set1_ps(1.0), t);
            let t1 = _mm256_mul_ps(one_sub_t, one_sub_t);
            // let t2 = t * t;
            let t2 = _mm256_mul_ps(t, t);
            // let h00 = (1.0 + t + t) * t1;
            let tt = _mm256_add_ps(t, t);
            let h00 = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0), tt), t1);
            // let h10 = t * t1;
            let h10 = _mm256_mul_ps(t, t1);
            // let h01 = t2 * (3.0 - tt);
            let h01 = _mm256_mul_ps(t2, _mm256_sub_ps(_mm256_set1_ps(3.0), tt));
            // let h11 = t2 * (-one_sub_t);
            let h11 = _mm256_mul_ps(t2, _mm256_sub_ps(_mm256_set1_ps(0.0), one_sub_t));
            // let dx2 = 1.0 / (dx + dx);
            let dx2 = _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(dx, dx));
            // let m0 = (self.kernel[i1] - self.kernel[i00]) * dx2;
            let m0 = _mm256_mul_ps(
                _mm256_sub_ps(
                    _mm256_i32gather_ps(self.kernel.as_ptr(), i1, size_of::<f32>() as i32),
                    _mm256_i32gather_ps(self.kernel.as_ptr(), i00, size_of::<f32>() as i32),
                ),
                dx2,
            );
            // let m1 = (self.kernel[i11] - self.kernel[i0]) * dx2;
            let m1 = _mm256_mul_ps(
                _mm256_sub_ps(
                    _mm256_i32gather_ps(self.kernel.as_ptr(), i11, size_of::<f32>() as i32),
                    _mm256_i32gather_ps(self.kernel.as_ptr(), i0, size_of::<f32>() as i32),
                ),
                dx2,
            );
            // h00 * self.kernel[i0] + h10 * dx * m0 + h01 * self.kernel[i1] + h11 * dx * m1
            let s1 = _mm256_add_ps(
                _mm256_mul_ps(
                    h00,
                    _mm256_i32gather_ps(self.kernel.as_ptr(), i0, size_of::<f32>() as i32),
                ),
                _mm256_mul_ps(_mm256_mul_ps(h10, dx), m0),
            );
            let s2 = _mm256_add_ps(
                _mm256_mul_ps(
                    h01,
                    _mm256_i32gather_ps(self.kernel.as_ptr(), i1, size_of::<f32>() as i32),
                ),
                _mm256_mul_ps(_mm256_mul_ps(h11, dx), m1),
            );
            _mm256_add_ps(s1, s2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::F256;
    use crate::tests::*;

    parameterize! {
        same_as_scalar_regular_grid
        same_as_scalar_random
    }

    fn same_as_scalar_regular_grid<const N: usize, const A: usize>() {
        if skip() {
            return;
        }
        unsafe {
            let a = A as f32;
            let i_max = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                _mm256_set1_epi32(N as i32),
                _mm256_set1_epi32(1),
            ));
            let lanczos = LanczosKernel::<N, A>::new();
            assert_eq!(0, N % 8);
            for offset in (0..N).step_by(8) {
                let i = _mm256_setr_epi32(
                    (offset + 0) as i32,
                    (offset + 1) as i32,
                    (offset + 2) as i32,
                    (offset + 3) as i32,
                    (offset + 4) as i32,
                    (offset + 5) as i32,
                    (offset + 6) as i32,
                    (offset + 7) as i32,
                );
                let x = lerp_avx(
                    _mm256_set1_ps(0.0),
                    _mm256_set1_ps(a),
                    _mm256_div_ps(_mm256_cvtepi32_ps(i), i_max),
                );
                let actual_reg = lanczos.interpolate_avx(x);
                let mut actual = F256([f32::NAN; 8]);
                _mm256_store_ps(actual.as_mut_ptr(), actual_reg);
                let mut expected = F256([f32::NAN; 8]);
                for i in offset..offset + 8 {
                    let x = lerp(0.0, a, i as f32 / (N - 1) as f32);
                    expected.0[i - offset] = lanczos.interpolate(x);
                }
                assert_eq!(expected, actual, "{x:?} {expected:?} {actual:?}",);
            }
        }
    }

    fn same_as_scalar_random<const N: usize, const A: usize>() {
        if skip() {
            return;
        }
        let lanczos = LanczosKernel::<N, A>::new();
        let a = A as f32;
        arbtest(|u| unsafe {
            let x = F256([
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
                arbitrary_f32(u, -a, a)?,
            ]);
            let actual = {
                let x = _mm256_load_ps(x.as_ptr());
                let y = lanczos.interpolate_avx(x);
                let mut actual = F256([f32::NAN; 8]);
                _mm256_store_ps(actual.as_mut_ptr(), y);
                actual
            };
            let expected = {
                let mut expected = F256([f32::NAN; 8]);
                for (i, x) in x.0.iter().copied().enumerate() {
                    expected.0[i] = lanczos.interpolate(x);
                }
                expected
            };
            assert_eq!(expected, actual, "{x:?} {expected:?} {actual:?}",);
            Ok(())
        });
    }

    fn skip() -> bool {
        !(std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("avx2"))
    }
}
