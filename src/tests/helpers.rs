use super::*;
use arbitrary::Unstructured;

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

pub(crate) use assert_near;
pub(crate) use assert_vec_f32_near;
pub(crate) use assert_vec_f32_near_relative;
