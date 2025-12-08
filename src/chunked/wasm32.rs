use super::default::ChunkedResampler as RustChunkedResampler;
use crate::Float32ArrayOutput;
use core::mem::align_of;
use core::mem::size_of;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

const CHUNKED_RESAMPLER_LEN: usize = size_of::<RustChunkedResampler>();

const _: () = assert!(align_of::<ChunkedResampler>() == align_of::<RustChunkedResampler>());
const _: () = assert!(size_of::<ChunkedResampler>() == size_of::<RustChunkedResampler>());

#[wasm_bindgen]
#[repr(align(4))]
#[allow(unused)]
pub struct ChunkedResampler([u8; CHUNKED_RESAMPLER_LEN]);

#[wasm_bindgen]
impl ChunkedResampler {
    #[wasm_bindgen(constructor)]
    pub fn new(input_sample_rate: usize, output_sample_rate: usize) -> Self {
        let mut buf = [0_u8; CHUNKED_RESAMPLER_LEN];
        let resampler = RustChunkedResampler::new(input_sample_rate, output_sample_rate);
        // SAFETY: Self and ChunkedResampler have the same size and the same aligntment.
        buf.copy_from_slice(unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref(&resampler).cast(),
                CHUNKED_RESAMPLER_LEN,
            )
        });
        Self(buf)
    }

    #[wasm_bindgen(js_name = "inputSampleRate", getter)]
    pub fn input_sample_rate(&self) -> usize {
        self.as_ref().input_sample_rate()
    }

    #[wasm_bindgen(js_name = "outputSampleRate", getter)]
    pub fn output_sample_rate(&self) -> usize {
        self.as_ref().output_sample_rate()
    }

    #[wasm_bindgen(js_name = "outputSampleRate", setter)]
    pub fn set_output_sample_rate(&mut self, value: usize) {
        self.as_mut().set_output_sample_rate(value);
    }

    #[wasm_bindgen(js_name = "maxOutputChunkLength")]
    pub fn max_output_chunk_len(&self, input_chunk_len: usize) -> usize {
        self.as_ref().max_output_chunk_len(input_chunk_len)
    }

    #[wasm_bindgen(js_name = "resampleChunk")]
    pub fn resample_chunk(&mut self, chunk: Float32Array, output: Float32Array) -> usize {
        self.as_mut()
            .resample_chunk(&chunk, &mut Float32ArrayOutput::new(&output))
    }

    #[inline]
    fn as_ref(&self) -> &RustChunkedResampler {
        // SAFETY: Self and ChunkedResampler have the same size and the same aligntment.
        unsafe { core::mem::transmute(self) }
    }

    #[inline]
    fn as_mut(&mut self) -> &mut RustChunkedResampler {
        // SAFETY: Self and ChunkedResampler have the same size and the same aligntment.
        unsafe { core::mem::transmute(self) }
    }
}
