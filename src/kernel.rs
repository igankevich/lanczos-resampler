use core::f32::consts::PI;

use crate::floor;
use crate::sin;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "x86_64")]
pub(crate) use self::x86_64::*;

/// Linear interpolation.
///
/// Gives exact result when t == 1.
pub const fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (1.0 - t) * a + t * b
}

// https://en.wikipedia.org/wiki/Lanczos_resampling
pub fn lanczos_kernel<const A: usize>(mut x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    // Ensure symmetry.
    x = x.abs();
    if x >= A as f32 {
        return 0.0;
    }
    let a = A as f32;
    let pi_x = PI * x;
    let pi_x_a = pi_x / a;
    sin(pi_x) * sin(pi_x_a) / (pi_x * pi_x_a)
}

/// Interpolates `lanczos_kernel` in the range _[-A; A]_ on a grid of `2 * N - 1` points using cubic
/// Hermite splines with second-order finite differences at spline endpoints.
///
/// See <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>.
pub struct LanczosKernel<const N: usize, const A: usize> {
    // We need only half of the points because Lanczos kernel is symmetric.
    //
    // Hence we store only values for _x_ in _[0; A]_.
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
        for (i, y) in kernel.iter_mut().enumerate() {
            let x = lerp(0.0, Self::X_MAX, i as f32 / (N - 1) as f32);
            *y = lanczos_kernel::<A>(x);
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
        // Ensure kernel symmetry.
        x = x.abs();
        let i_max = (N - 1) as f32;
        // Interpolate using cubic Hermite spline.
        // https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        // 1. Find 2-4 closest points.
        let (i00, i0, i1, i11) = {
            let tmp = x / Self::X_MAX;
            let mut i0 = floor(tmp * i_max) as usize;
            if i0 == N - 1 {
                i0 -= 1;
            }
            let i1 = i0 + 1;
            let i00 = if i0 > 0 { i0 - 1 } else { i1 };
            let i11 = if i1 < N - 1 { i1 + 1 } else { i0 };
            (i00, i0, i1, i11)
        };
        debug_assert!(i0 < N);
        debug_assert!(i1 < N);
        let x0 = lerp(0.0, Self::X_MAX, i0 as f32 / i_max);
        let x1 = lerp(0.0, Self::X_MAX, i1 as f32 / i_max);
        // 2. Compute coefficients.
        let dx = x1 - x0;
        let t = (x - x0) / dx;
        let one_sub_t = 1.0 - t;
        let t1 = one_sub_t * one_sub_t;
        let t2 = t * t;
        let tt = t + t;
        let h00 = (1.0 + tt) * t1;
        let h10 = t * t1;
        let h01 = t2 * (3.0 - tt);
        let h11 = t2 * (-one_sub_t);
        // Finite differences.
        let dx2 = 1.0 / (dx + dx);
        let m0 = (self.kernel[i1] - self.kernel[i00]) * dx2;
        let m1 = (self.kernel[i11] - self.kernel[i0]) * dx2;
        // 3. Interpolate.
        (h00 * self.kernel[i0] + h10 * dx * m0) + (h01 * self.kernel[i1] + h11 * dx * m1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    parameterize! {
        interpolation
        symmetry
        edge_cases
    }

    fn interpolation<const N: usize, const A: usize>() {
        // `LanczosKernel` should interpolate to the exact same values as the original `lanczos_kernel`
        // function in every node.
        let a = A as f32;
        let lanczos = LanczosKernel::<N, A>::new();
        for i in 0..N {
            let x = lerp(0.0, a, i as f32 / (N - 1) as f32);
            let expected = lanczos_kernel::<A>(x);
            let actual = lanczos.interpolate(x);
            assert_eq!(
                expected,
                actual,
                "{x:20.6} {expected:20.6} {actual:20.6} {eps:20.6}",
                eps = (expected - actual).abs(),
            );
        }
    }

    fn symmetry<const N: usize, const A: usize>() {
        arbtest(|u| {
            let x = u.int_in_range(0..=A)? as f32;
            assert_eq!(lanczos_kernel::<A>(x), lanczos_kernel::<A>(-x));
            let lanczos = LanczosKernel::<N, A>::new();
            assert_eq!(lanczos.interpolate(x), lanczos.interpolate(-x));
            Ok(())
        });
    }

    fn edge_cases<const N: usize, const A: usize>() {
        assert_eq!(1.0, lanczos_kernel::<A>(0.0));
        assert_eq!(0.0, lanczos_kernel::<A>(-(A as f32)));
        assert_eq!(0.0, lanczos_kernel::<A>(A as f32));
    }
}
