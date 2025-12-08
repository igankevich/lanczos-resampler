# Lanczos resampler

[![Crates.io Version](https://img.shields.io/crates/v/lanczos-resampler)](https://crates.io/crates/lanczos-resampler)
[![Docs](https://docs.rs/lanczos-resampler/badge.svg)](https://docs.rs/lanczos-resampler)
[![dependency status](https://deps.rs/repo/github/igankevich/lanczos-resampler/status.svg)](https://deps.rs/repo/github/igankevich/lanczos-resampler)

An audio resampler that uses [Lanczos filter](https://en.wikipedia.org/wiki/Lanczos_resampling)
as an alternative to traditional windowed sinc filters.
The main advantage of such approach is small number of coefficients required to store the filter state,
this results in small memory footprint and high performance.


## Features

### Small memory footprint

The library doesn't use memory allocation by default,
and resampler's internal state occupies less than a hundred bytes.

### High performance

Thanks to small kernel size the processing time of a typical audio chunk on a typical laptop is below 100 μs.
This is achived without SIMD instructions.

### Predictability

When you're resampling from _N<sub>1</sub>_ Hz to _N<sub>2</sub>_ Hz,
for each _N<sub>1</sub>_ input samples you will get _exactly_ _N<sub>2</sub>_ output samples.
This results in predictable audio stream playback,
and simplifies time synchronization between different streams (e.g. video and audio)[^1].

### JS-compatible

This library can be used in web browsers and in any JS engine that supports WASM in general.
All of the abovementioned features also hold for the WASM version of the library.

[^1]: Seriously, why other libraries don't have this feature?


## Usage

### Rust

#### Chunked resampling

```rust
use lanczos_resampler::ChunkedResampler;

let n = 1024;
let chunk = vec![0.1; n];
let mut resampler = ChunkedResampler::new(44100, 48000);
let mut output: Vec<f32> = Vec::with_capacity(resampler.max_output_chunk_len(n));
let num_processed = resampler.resample_chunk(&chunk[..], &mut output);
assert_eq!(n, num_processed);
```

#### Resampling the whole audio track

```rust
use lanczos_resampler::WholeResampler;

let n = 1024;
let track = vec![0.1; n];
let output_len = lanczos_resampler::output_len(n, 44100, 48000);
let mut output = vec![0.1; output_len];
let resampler = WholeResampler::new();
let mut output_slice = &mut output[..];
let num_processed = resampler.resample_whole_into(&track[..], &mut output_slice);
assert_eq!(n, num_processed);
assert!(output_slice.is_empty());
```

### JS

#### Installation

```bash
npm install lanczos-resampler
```

#### Chunked resampling

```javascript
import { ChunkedResampler } from 'lanczos-resampler';

const resampler = new ChunkedResampler(44100, 48000);
const input = new Float32Array(1024);
input.fill(0.1);
const output = new Float32Array(resampler.maxOutputChunkLength(input.length));
const numProcessed = resampler.resampleChunk(input, output);
assert.equal(input.length, numProcessed);
```

#### Resampling the whole audio track

```javascript
import { WholeResampler, outputLength } as lanczos from 'lanczos-resampler';

const input = new Float32Array(1024);
input.fill(0.1);
const outputLen = outputLength(1024, 44100, 48000);
const output = new Float32Array(outputLen);
const resampler = new WholeResampler();
const numProcessed = resampler.resampleWholeInto(input, output);
assert.equal(input.length, numProcessed);
console.log(output)
```
