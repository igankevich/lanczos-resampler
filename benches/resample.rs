use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::Rng;
use rand::rng;
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("resample 16/3", |b| {
        b.iter_batched(
            || {
                let n = 44100 * 2 * 6;
                let mut input: Vec<f32> = Vec::with_capacity(n);
                for _ in 0..n {
                    input.push(rng().random_range(-1.0..=1.0));
                }
                let output = vec![f32::NAN; n];
                (input, output)
            },
            |(input, mut output)| {
                lanczos_resampler::resample_into::<16, 2>(
                    black_box(&input[..]),
                    44100,
                    44100,
                    black_box(&mut &mut output[..]),
                );
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
