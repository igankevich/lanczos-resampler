use super::*;
use arbtest::arbtest;
use rand::Rng;

mod helpers;
use self::helpers::*;

#[test]
fn lanczos_kernel_works() {
    // `LanczosKernel` should interpolate to the exact same values as the original `lanczos_kernel`
    // function in every node.
    const A: usize = 3;
    let a = A as f32;
    let n = 100;
    let lanczos = LanczosKernel::<100, A>::new();
    for i in 0..n {
        let x = lerp(0.0, a, i as f32 / (n - 1) as f32);
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

#[test]
fn lanczos_kernel_symmetry() {
    const A: usize = 3;
    arbtest(|u| {
        let x = u.int_in_range(0..=A)? as f32;
        assert_eq!(lanczos_kernel::<A>(x), lanczos_kernel::<A>(-x));
        let lanczos = LanczosKernel::<100, A>::new();
        assert_eq!(lanczos.interpolate(x), lanczos.interpolate(-x));
        Ok(())
    });
}

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

#[test]
fn lanczos_filter_works() {
    // `LanczosFilter` should interpolate to the exact same values as the original function
    // in every node.
    const A: usize = 3;
    const NUM_SAMPLES: usize = 10;
    let mut xs = [0.0; NUM_SAMPLES];
    let mut samples = [0.0; NUM_SAMPLES];
    for i in 0..NUM_SAMPLES {
        let x0 = 0.0;
        let x1 = (NUM_SAMPLES - 1) as f32;
        let x = lerp(x0, x1, i as f32 / x1);
        xs[i] = x;
        samples[i] = x.sin();
    }
    let lanczos = LanczosFilter::<100, A>::new();
    for i in 0..NUM_SAMPLES {
        let x = xs[i];
        let expected = samples[i];
        let actual = lanczos.interpolate(x, &samples);
        let eps = (expected - actual).abs();
        assert!(
            eps < 1e-4,
            "{x:20.6} {expected:20.6} {actual:20.6} {eps:20.6}",
        );
    }
}

#[test]
fn lanczos_filter_chunked_works() {
    const A: usize = 2;
    const N: usize = 16;
    arbtest(|u| {
        let input_len = u.int_in_range(2 + A..=100)?;
        let input = arbitrary_samples(u, input_len)?;
        let lanczos = LanczosFilter::<N, A>::new();
        let i_max = (input.len() - 1) as f32;
        for prev_len in 0..A {
            for i in 0..input_len {
                let x0 = prev_len as f32;
                let x1 = (input.len() - 1) as f32;
                let x = lerp(x0, x1, i as f32 / i_max);
                let y1 = lanczos.interpolate(x, &input);
                let (prev, chunk) = input.split_at(prev_len);
                let x0 = 0.0;
                let x1 = (chunk.len() - 1) as f32;
                let x = lerp(x0, x1, i as f32 / i_max);
                let y2 = lanczos.interpolate_chunk(x, chunk, prev);
                assert_near!(y1, y2, 1e-4, "prev len = {prev_len}, i = {i}, A = {A}");
            }
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
}

#[test]
fn mean_works() {
    assert_eq!(1.0, mean(&[1.0, 1.0]));
    assert_eq!(2.0, mean(&[2.0, 2.0]));
}

#[test]
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

#[test]
fn adjust_lengths_remainder_works() {
    arbtest(|u| {
        let max_sample_rate = max_sample_rate();
        let output_sample_rate: usize = u.int_in_range(1..=max_sample_rate)?;
        let input_sample_rate: usize = u.int_in_range(1..=max_sample_rate - output_sample_rate)?;
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

#[test]
fn resample_works() {
    assert_eq!(vec![0.0; 100], resample::<17, 3>(&vec![0.0; 100], 100, 100));
    for x in [0.1, -1.0, 1.0, 0.33] {
        assert_vec_f32_near!(
            vec![x; 200],
            resample::<16, 2>(&vec![x; 100], 100, 200),
            1e-1
        );
        assert_vec_f32_near!(
            vec![x; 100],
            resample::<16, 3>(&vec![x; 200], 200, 100),
            1e-2
        );
        assert_vec_f32_near!(
            vec![x; 100],
            resample::<16, 2>(&vec![x; 100], 100, 100),
            1e-2
        );
    }
}

#[test]
fn resample_streaming_simple_works() {
    const A: usize = 3;
    const N: usize = 16;
    const M: usize = 10;
    const O: usize = 20;
    let input = sine_wave(M);
    let expected_output = resample::<N, A>(&input, M, O);
    let mut resampler = LanczosResampler::<N, A>::new(M, O);
    let mut output = vec![f32::NAN; O];
    let (num_read, num_written) = resampler.resample_into(&input[..M / 2], &mut output[..O / 2]);
    assert_eq!(M / 2, num_read);
    assert_eq!(O / 2, num_written);
    let (num_read, num_written) = resampler.resample_into(&input[M / 2..], &mut output[O / 2..]);
    assert_eq!(M / 2, num_read);
    assert_eq!(O / 2, num_written);
    write_samples("expected", &expected_output);
    write_samples("actual", &output);
    std::process::Command::new("gnuplot")
        .arg("plot.gnuplot")
        .status()
        .unwrap();
    assert_vec_f32_near_relative!(expected_output, output, 0.20);
}

#[test]
fn resample_streaming_works() {
    // Check that streaming resampling gives the same result as resampling in one go.
    const A: usize = 2;
    const N: usize = 16;
    arbtest(|u| {
        let input_sample_rate = 2 * u.int_in_range(1..=100)?;
        let input_len: usize = input_sample_rate * u.int_in_range(1..=10)?;
        let input = sine_wave(input_len);
        let output_sample_rate = 2 * input_sample_rate;
        let expected_output = resample::<N, A>(&input, input_sample_rate, output_sample_rate);
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
        let mut resampler = LanczosResampler::<N, A>::new(input_sample_rate, output_sample_rate);
        let mut actual_output = vec![f32::NAN; expected_output.len()];
        let mut input_slice = &input[..];
        let mut output_slice = &mut actual_output[..];
        for i in 0..num_chunks {
            let chunk_len = chunks[i];
            let (num_read, num_written) =
                resampler.resample_into(&input_slice[..chunk_len], output_slice);
            input_slice = &input_slice[num_read..];
            output_slice = &mut output_slice[num_written..];
        }
        assert_eq!(&[] as &[f32], input_slice);
        assert_eq!(&[] as &[f32], output_slice);
        assert_vec_f32_near_relative!(expected_output, actual_output, 0.30);
        //assert_eq!(expected_output, actual_output);
        Ok(())
    }); //.seed(0x671a0fb600000236);
}
