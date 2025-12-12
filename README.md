# Lanczos resampler

[![Crates.io Version](https://img.shields.io/crates/v/lanczos-resampler)](https://crates.io/crates/lanczos-resampler)
[![Docs](https://docs.rs/lanczos-resampler/badge.svg)](https://docs.rs/lanczos-resampler)
[![dependency status](https://deps.rs/repo/github/igankevich/lanczos-resampler/status.svg)](https://deps.rs/repo/github/igankevich/lanczos-resampler)

An audio resampler that uses [Lanczos filter](https://en.wikipedia.org/wiki/Lanczos_resampling)
as an alternative to traditional windowed sinc filters.
The main advantage of such approach is small number of coefficients required to store the filter state;
this results in small memory footprint and high performance.


## Features

#### Small memory footprint

The library doesn't use memory allocation by default,
and resampler's internal state occupies less than a hundred bytes.

#### High performance

Thanks to small kernel size the processing time of a typical audio chunk is very fast (below 100 μs on a typical laptop).

#### Robustness

When you're resampling from _N_ Hz to _M_ Hz, for each _N_ input samples you will get exactly _M_ output samples[^1].
This results in predictable audio stream playback
and simplifies time synchronization between different streams (e.g. video and audio).

[^1]: Seriously, why other libraries don't have this feature?

#### JS-compatible

This library can be used in web browsers and in general in any JS engine that supports WASM.
All of the abovementioned features are inherent to both Rust and WASM versions of the library.


## Usage

### Kernel parameters

This library uses [Lanczos kernel](https://en.wikipedia.org/wiki/Lanczos_resampling)
approximated by _2N - 1_ points and defined on interval _[-A; A]_. The kernel is interpolated
using cubic Hermite splines with second-order finite differences at spline endpoints. The
output is clamped to _[-1; 1]_.

The recommended parameters are _N = 16, A = 3_. Using _A = 2_ might improve performance a
little bit. Using larger _N_ will techincally improve precision, but precision isn't a good
metric for audio signal. With _N = 16_ the kernel fits into exactly 64 B (the size of a cache line).

### Interleaved vs. non-interleaved format

Non-interleaved format means that audio samples for each channel are stored in separate arrays.
To resample such data you need to call `resample` for each channel individually.

Interleaved format on the other hand means that samples for each channel are stored in a single array using frames;
a frame is a sequence of samples, one sample for each channel.
To resample such data you need to call `resample` only once.

Usually resampling interleaved data is much faster than processing each channel individually
because a CPU can process such data efficiently with SIMD instructions.


### Rust

#### Resampling audio stream in chunks

```rust
use lanczos_resampler::ChunkedResampler;

let n = 1024;
let chunk = vec![0.1; n];
let mut resampler = ChunkedResampler::new(44100, 48000);
let mut output: Vec<f32> = Vec::with_capacity(resampler.max_num_output_frames(n));
let num_processed = resampler.resample(&chunk[..], &mut output);
assert_eq!(n, num_processed);
```

#### Resampling the whole audio track

```rust
use lanczos_resampler::WholeResampler;

let n = 1024;
let track = vec![0.1; n];
let output_len = lanczos_resampler::num_output_frames(n, 44100, 48000);
let mut output = vec![0.0; output_len];
let resampler = WholeResampler::new();
let mut output_slice = &mut output[..];
let num_processed = resampler.resample_into(&track[..], &mut output_slice);
assert_eq!(n, num_processed);
assert!(output_slice.is_empty());
```

### JS

#### Installation

```bash
npm install lanczos-resampler
```

#### Resampling audio stream in chunks

```javascript
import { ChunkedResampler } from 'lanczos-resampler';

const resampler = new ChunkedResampler(44100, 48000);
const input = new Float32Array(1024);
input.fill(0.1);
const output = new Float32Array(resampler.maxNumOutputFrames(input.length));
const numProcessed = resampler.resample(input, output);
assert.equal(input.length, numProcessed);
```

#### Resampling the whole audio track

```javascript
import { WholeResampler, numOutputFrames } as lanczos from 'lanczos-resampler';

const input = new Float32Array(1024);
input.fill(0.1);
const outputLen = numOutputFrames(1024, 44100, 48000);
const output = new Float32Array(outputLen);
const resampler = new WholeResampler();
const numProcessed = resampler.resampleInto(input, output);
assert.equal(input.length, numProcessed);
console.log(output)
```


## Documentation

Rust: <https://docs.rs/lanczos-resampler>

JS: <https://igankevich.github.com/lanczos-resampler>


## No-std support

This crate supports `no_std` via [`libm`](https://docs.rs/libm/latest/libm/).
When `std` feature is enabled (the default), it uses built-in mathematical functions
which are typically much faster than `libm`.
