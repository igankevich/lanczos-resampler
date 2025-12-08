use super::default as rust;
use crate::Float32ArrayOutput;
use core::ptr;
use core::slice;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

const WHOLE_RESAMPLER_LEN: usize = size_of::<rust::WholeResampler>();

const _: () = assert!(align_of::<WholeResampler>() == align_of::<rust::WholeResampler>());
const _: () = assert!(size_of::<WholeResampler>() == size_of::<rust::WholeResampler>());

const N: usize = crate::DEFAULT_N;
const A: usize = crate::DEFAULT_A;

#[wasm_bindgen]
#[repr(align(4))]
#[allow(unused)]
pub struct WholeResampler([u8; WHOLE_RESAMPLER_LEN]);

#[wasm_bindgen]
impl WholeResampler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut buf = [0_u8; WHOLE_RESAMPLER_LEN];
        let resampler = rust::WholeResampler::new();
        // SAFETY: Self and WholeResampler have the same size and the same aligntment.
        buf.copy_from_slice(unsafe {
            slice::from_raw_parts(ptr::from_ref(&resampler).cast(), WHOLE_RESAMPLER_LEN)
        });
        Self(buf)
    }

    #[wasm_bindgen(js_name = "resampleWhole")]
    pub fn resample_whole(
        &self,
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
        self.as_ref()
            .resample_whole_into(&input, &mut Float32ArrayOutput::new(&output));
        output
    }

    #[wasm_bindgen(js_name = "resampleWholeInto")]
    pub fn resample_into(&self, input: Float32Array, output: Float32Array) -> usize {
        self.as_ref()
            .resample_whole_into(&input, &mut Float32ArrayOutput::new(&output))
    }

    #[inline]
    fn as_ref(&self) -> &rust::WholeResampler {
        // SAFETY: Self and WholeResampler have the same size and the same aligntment.
        unsafe { core::mem::transmute(self) }
    }

    #[inline]
    fn as_mut(&mut self) -> &mut rust::WholeResampler {
        // SAFETY: Self and WholeResampler have the same size and the same aligntment.
        unsafe { core::mem::transmute(self) }
    }
}

#[wasm_bindgen(js_name = "outputLength")]
pub fn output_length(
    input_len: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
) -> usize {
    rust::output_len(input_len, input_sample_rate, output_sample_rate)
}
