use crate::Float32ArrayOutput;
use crate::output_len;
use crate::resample_into;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

const N: usize = 16;
const A: usize = 3;

#[allow(non_snake_case)]
#[wasm_bindgen]
pub fn outputLength(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> usize {
    output_len(input_len, input_sample_rate, output_sample_rate)
}

#[wasm_bindgen]
pub fn resample(
    input: Float32Array,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> Float32Array {
    let output_len = output_len(
        input.length() as usize,
        input_sample_rate,
        output_sample_rate,
    );
    let mut output = Float32Array::new_with_length(output_len as u32);
    resample_into::<N, A>(
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
    resample_into::<N, A>(
        &input,
        input_sample_rate,
        output_sample_rate,
        &mut Float32ArrayOutput::new(&output),
    )
}
