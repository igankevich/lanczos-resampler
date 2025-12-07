use js_sys::Float32Array;
use lanczos_resampler::Float32ArrayOutput;
use wasm_bindgen::prelude::*;

const N: usize = 16;
const A: usize = 3;

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[allow(non_snake_case)]
#[wasm_bindgen]
pub fn outputLength(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> usize {
    lanczos_resampler::output_len(input_len, input_sample_rate, output_sample_rate)
}

#[wasm_bindgen]
pub fn resample(
    input: Float32Array,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Float32Array {
    let output_len = lanczos_resampler::output_len(
        input.length() as usize,
        input_sample_rate,
        output_sample_rate,
    );
    let mut output = Float32Array::new_with_length(output_len as u32);
    lanczos_resampler::resample_into::<N, A>(
        &input,
        input_sample_rate,
        output_sample_rate,
        &mut Float32ArrayOutput::new(&mut output),
    );
    output
}

#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn resampleInto(
    input: Float32Array,
    input_sample_rate: usize,
    output_sample_rate: usize,
    output: Float32Array,
) -> usize {
    lanczos_resampler::resample_into::<N, A>(
        &input,
        input_sample_rate,
        output_sample_rate,
        &mut Float32ArrayOutput::new(&output),
    )
}
