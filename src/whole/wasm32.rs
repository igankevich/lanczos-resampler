use super::default as rust;
use crate::Float32ArrayOutput;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

const N: usize = crate::DEFAULT_N;
const A: usize = crate::DEFAULT_A;

#[wasm_bindgen(js_name = "outputLength")]
pub fn output_length(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> usize {
    rust::output_len(input_len, input_sample_rate, output_sample_rate)
}

#[wasm_bindgen]
pub fn resample(
    input: Float32Array,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Float32Array {
    let output_len = rust::output_len(
        input.length() as usize,
        input_sample_rate,
        output_sample_rate,
    );
    let output = Float32Array::new_with_length(output_len as u32);
    rust::resample_into::<N, A>(&input, &mut Float32ArrayOutput::new(&output));
    output
}

#[wasm_bindgen(js_name = "resampleInto")]
pub fn resample_into(input: Float32Array, output: Float32Array) -> usize {
    rust::resample_into::<N, A>(&input, &mut Float32ArrayOutput::new(&output))
}
