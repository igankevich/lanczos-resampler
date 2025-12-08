#![allow(missing_docs)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use lanczos_resampler::ChunkedResampler;
use lanczos_resampler::WholeResampler;
use rand::Rng;
use rand::rng;
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("resample whole", |b| {
        b.iter_batched(
            || {
                let n = 44100 * 6;
                let mut input: Vec<f32> = Vec::with_capacity(n);
                for _ in 0..n {
                    input.push(rng().random_range(-1.0..=1.0));
                }
                let output = vec![f32::NAN; n];
                let resampler = WholeResampler::new();
                (resampler, input, output)
            },
            |(resampler, input, mut output)| {
                resampler
                    .resample_whole_into(black_box(&input[..]), black_box(&mut &mut output[..]));
            },
            BatchSize::SmallInput,
        )
    });
    let n = 1024;
    let mut resampler = ChunkedResampler::new(44100, 48000);
    let input = vec![0.1; n];
    let output_len = lanczos_resampler::output_len(
        n,
        resampler.input_sample_rate(),
        resampler.output_sample_rate(),
    );
    let mut output = vec![0.0; output_len];
    c.bench_function("resample chunk", |b| {
        b.iter(|| {
            resampler.resample_chunk(black_box(&input[..]), black_box(&mut &mut output[..]));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
