use crate::lerp;

use arbitrary::Unstructured;
use core::f32::consts::PI;
pub(crate) use arbtest::arbtest;

/*
fn do_lanczos_kernel_precision(n: usize, a: f32) -> f32 {
    let lanczos = DynamicLanczosKernel::new(n, a);
    let mut rng = rand::rng();
    let mut max_residual = 0.0;
    for _ in 0..100_000 {
        let x = rng.random_range(0.0..=a);
        let residual = (dynamic_lanczos_kernel(x, a) - lanczos.interpolate(x)).abs();
        if residual > max_residual {
            max_residual = residual;
        }
    }
    eprintln!("{n} {a} {max_residual}");
    max_residual
}

#[test]
fn lanczos_kernel_precision() {
    for a in [2.0] {
        for n in 2..10000 {
            let residual = do_lanczos_kernel_precision(n, a);
            if residual <= 1e-6 {
                break;
            }
        }
    }
}
*/

pub fn write_samples(filename: &str, data: &[f32]) {
    use std::io::Write;
    let mut file = std::fs::File::create(filename).unwrap();
    for x in data {
        writeln!(&mut file, "{x}").unwrap();
    }
}

pub fn arbitrary_f32(u: &mut Unstructured<'_>, min: f32, max: f32) -> arbitrary::Result<f32> {
    let i: u32 = u.arbitrary()?;
    let t = i as f32 / u32::MAX as f32;
    Ok(lerp(min, max, t))
}

pub fn arbitrary_samples(u: &mut Unstructured<'_>, len: usize) -> arbitrary::Result<Vec<f32>> {
    let mut samples: Vec<f32> = Vec::with_capacity(len);
    for _ in 0..len {
        samples.push(arbitrary_f32(u, -1.0, 1.0)?);
    }
    Ok(samples)
}

pub fn sine_wave(num_samples: usize) -> Vec<f32> {
    assert!(num_samples > 1);
    (0..num_samples)
        .map(|i| lerp(0.0, 4.0 * PI, i as f32 / (num_samples - 1) as f32).cos())
        .collect()
}

#[allow(unused)]
pub fn gnuplot(expected: &[f32], actual: &[f32]) {
    write_samples("expected", expected);
    write_samples("actual", actual);
    std::process::Command::new("gnuplot")
        .arg("plot.gnuplot")
        .status()
        .unwrap();
}

macro_rules! assert_near {
    ($lhs: expr, $rhs: expr, $eps: expr $(, $arg: expr)*) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let max_eps = $eps;
        let eps = (lhs - rhs).abs();
        if eps > max_eps {
            let extra = format!($($arg,)*);
            panic!("Assertion |lhs - rhs| < eps failed:\n lhs: {lhs}\n rhs: {rhs}\ndiff: {eps}, eps = {max_eps}\n{extra}");
        }
    }};
}

macro_rules! assert_vec_f32_near {
    ($lhs: expr, $rhs: expr, $eps: expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let max_eps = $eps;
        let len = lhs.len().min(rhs.len());
        let mut diff = Vec::with_capacity(len);
        for i in 0..len {
            let mut eps = (rhs[i] - lhs[i]).abs();
            if eps <= max_eps {
                eps = 0.0;
            }
            diff.push(eps);
        }
        if !diff.iter().copied().all(|eps| eps == 0.0) {
            panic!(
                "Assertion |lhs - rhs| < eps failed:\n lhs: {lhs:?}\n rhs: {rhs:?}\ndiff: {diff:?}"
            );
        }
    }};
}

macro_rules! assert_vec_f32_near_relative {
    ($lhs: expr, $rhs: expr, $max_relative_eps: expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let max_relative_eps = $max_relative_eps;
        let len = lhs.len().min(rhs.len());
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for x in lhs.iter().copied() {
            if x > max {
                max = x;
            }
            if x < min {
                min = x;
            }
        }
        let magnitude = max - min;
        let mut diff = Vec::with_capacity(len);
        for i in 0..len {
            let mut eps = (rhs[i] - lhs[i]).abs();
            if magnitude != 0.0 {
                eps /= magnitude;
            }
            if eps <= max_relative_eps {
                eps = 0.0;
            }
            diff.push(eps);
        }
        if !diff.iter().copied().all(|eps| eps == 0.0) {
            write_samples("expected", &lhs);
            write_samples("actual", &rhs);
            std::process::Command::new("gnuplot")
                .arg("plot.gnuplot")
                .status()
                .unwrap();
            panic!(
                "Assertion |lhs - rhs| < eps failed:\n lhs: {lhs:?}\n rhs: {rhs:?}\ndiff: {diff:?}"
            );
        }
    }};
}

macro_rules! parameterize_impl {
    ($(($function: ident $(($n: literal $a: literal))+))+) => {
        paste::paste! {
            $(
                $(
                    #[cfg_attr(not(target_arch = "wasm32"), test)]
                    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
                    fn [<$function _ $n _ $a>]() {
                        $function::<$n, $a>()
                    }
                )+
            )+
        }
    };
}

macro_rules! parameterize {
    ($($function: ident)+) => {
        $(
            $crate::tests::parameterize_impl!{
                ($function (16 1) (16 2) (16 3))
            }
        )+
    };
}

pub(crate) use assert_near;
pub(crate) use assert_vec_f32_near;
pub(crate) use assert_vec_f32_near_relative;
pub(crate) use parameterize;
pub(crate) use parameterize_impl;
