#[cfg(any(feature = "alloc", test))]
use super::default::ChunkedInterleavedResampler as RustChunkedInterleavedResampler;
use super::default::ChunkedResampler as RustChunkedResampler;
use crate::Float32ArrayOutput;
use core::mem::align_of;
use core::mem::size_of;
use core::ptr;
use core::slice;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

const CHUNKED_RESAMPLER_LEN: usize = size_of::<RustChunkedResampler>();

const _: () = assert!(align_of::<ChunkedResampler>() == align_of::<RustChunkedResampler>());
const _: () = assert!(size_of::<ChunkedResampler>() == size_of::<RustChunkedResampler>());

/// A resampler that processes audio input in chunks.
///
/// Use it to process audio streams.
///
/// ## Parameters
///
/// This resampler uses default parameters: _N = 16, A = 3_.
///
/// ## Limitations
///
/// `ChunkedResampler` produces slightly different output compared to processing the whole input at once.
/// If this is undesired, consider using {@link WholeResampler}.
#[wasm_bindgen]
#[repr(align(4))]
#[allow(unused)]
pub struct ChunkedResampler([u8; CHUNKED_RESAMPLER_LEN]);

#[wasm_bindgen]
impl ChunkedResampler {
    /// Create new resampler with the specified input and output sample rates.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(
            param_description = "input sample rate in Hz",
            js_name = "inputSampleRate"
        )]
        input_sample_rate: usize,
        #[wasm_bindgen(
            param_description = "output sample rate in Hz",
            js_name = "outputSampleRate"
        )]
        output_sample_rate: usize,
    ) -> Self {
        let mut buf = [0_u8; CHUNKED_RESAMPLER_LEN];
        let resampler = RustChunkedResampler::new(input_sample_rate, output_sample_rate);
        // SAFETY: Self and ChunkedResampler have the same size and the same alignment.
        buf.copy_from_slice(unsafe {
            slice::from_raw_parts(ptr::from_ref(&resampler).cast(), CHUNKED_RESAMPLER_LEN)
        });
        Self(buf)
    }

    /// Get input sample rate in Hz.
    #[wasm_bindgen(js_name = "inputSampleRate", getter)]
    pub fn input_sample_rate(&self) -> usize {
        self.as_ref().input_sample_rate()
    }

    /// Get/set output sample rate in Hz.
    ///
    /// After changing the sample rate you should consider updating buffer size to
    /// {@link ChunkedResampler.maxNumOutputFrames}.
    #[wasm_bindgen(js_name = "outputSampleRate", getter)]
    pub fn output_sample_rate(&self) -> usize {
        self.as_ref().output_sample_rate()
    }

    // The documentation is overwritten by the getter.
    #[allow(missing_docs)]
    #[wasm_bindgen(js_name = "outputSampleRate", setter)]
    pub fn set_output_sample_rate(
        &mut self,
        #[wasm_bindgen(param_description = "new sample rate in Hz")] value: usize,
    ) {
        self.as_mut().set_output_sample_rate(value);
    }

    /// Get maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as {@link outputLength} plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// {@link ChunkedResampler.outputSampleRate}.
    #[wasm_bindgen(js_name = "maxNumOutputFrames")]
    pub fn max_num_output_frames(
        &self,
        #[wasm_bindgen(js_name = "numInputFrames")] num_input_frames: usize,
    ) -> usize {
        self.as_ref().max_num_output_frames(num_input_frames)
    }

    /// Resets internal state.
    ///
    /// Erases any information about the previous chunk.
    ///
    /// Use this method when you want to reuse resampler for another audio stream.
    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.as_mut().reset();
    }

    /// Resamples input signal chunk from the source to the target sample rate and appends the
    /// resulting signal to the output.
    ///
    /// Returns the number of processed input samples. The output is clamped to _[-1; 1]_.
    ///
    /// For each {@link ChunkedResampler.inputSampleRate} input samples this method produces exactly
    /// {@link ChunkedResampler.outputSampleRate} output samples  even if it is called multiple times with a smaller
    /// amount of input samples; the only exception is when the output sample rate was changed in the process.
    ///
    /// #### Edge cases
    ///
    /// Returns 0 when either the input length or output length is less than _max(2, A-1)_, adjusted in
    /// accordance with sample rate ratio.
    ///
    /// #### Limitations
    ///
    /// The output depends on the chunk size, hence resampling the same audio track all at once and
    /// in chunks will produce slightly different results. This a consequence of the fact that Lanczos kernel
    /// isn't an interpolation function, but a filter. To minimize such discrepancies chunk size should
    /// be much larger than _2⋅A + 1_.
    #[wasm_bindgen(js_name = "resample")]
    pub fn resample(&mut self, chunk: &[f32], output: Float32Array) -> usize {
        self.as_mut()
            .resample(&chunk[..], &mut Float32ArrayOutput::new(&output))
    }

    #[inline]
    fn as_ref(&self) -> &RustChunkedResampler {
        // SAFETY: Self and ChunkedResampler have the same size and the same alignment.
        unsafe { core::mem::transmute(self) }
    }

    #[inline]
    fn as_mut(&mut self) -> &mut RustChunkedResampler {
        // SAFETY: Self and ChunkedResampler have the same size and the same alignment.
        unsafe { core::mem::transmute(self) }
    }
}

#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
const CHUNKED_INTERLEAVED_RESAMPLER_LEN: usize = size_of::<RustChunkedInterleavedResampler>();

#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
const _: () = assert!(
    align_of::<ChunkedInterleavedResampler>() == align_of::<RustChunkedInterleavedResampler>()
);
#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
const _: () = assert!(
    size_of::<ChunkedInterleavedResampler>() == size_of::<RustChunkedInterleavedResampler>()
);

/// A resampler that processes audio input in chunks; the channels are interleaved with each other.
///
/// Use it to process audio streams.
///
/// ## Parameters
///
/// This resampler uses default parameters: _N = 16, A = 3_.
///
/// ## Limitations
///
/// `ChunkedInterleavedResampler` produces slightly different output compared to processing the whole input at once.
/// If this is undesired, consider using {@link WholeResampler}.
#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[wasm_bindgen]
#[repr(align(4))]
#[allow(unused)]
pub struct ChunkedInterleavedResampler([u8; CHUNKED_INTERLEAVED_RESAMPLER_LEN]);

#[cfg(any(feature = "alloc", test))]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[wasm_bindgen]
impl ChunkedInterleavedResampler {
    /// Creates new instance of resampler with the specified input and output sample rates and the
    /// number of channels.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(
            param_description = "input sample rate in Hz",
            js_name = "inputSampleRate"
        )]
        input_sample_rate: usize,
        #[wasm_bindgen(
            param_description = "output sample rate in Hz",
            js_name = "outputSampleRate"
        )]
        output_sample_rate: usize,
        #[wasm_bindgen(param_description = "number of channels", js_name = "numChannels")]
        num_channels: usize,
    ) -> Self {
        let mut buf = [0_u8; CHUNKED_INTERLEAVED_RESAMPLER_LEN];
        let resampler = RustChunkedInterleavedResampler::new(
            input_sample_rate,
            output_sample_rate,
            num_channels,
        );
        // SAFETY: Self and ChunkedInterleavedResampler have the same size and the same alignment.
        buf.copy_from_slice(unsafe {
            slice::from_raw_parts(
                ptr::from_ref(&resampler).cast(),
                CHUNKED_INTERLEAVED_RESAMPLER_LEN,
            )
        });
        Self(buf)
    }

    /// Get input sample rate in Hz.
    #[wasm_bindgen(js_name = "inputSampleRate", getter)]
    pub fn input_sample_rate(&self) -> usize {
        self.as_ref().input_sample_rate()
    }

    /// Get/set output sample rate in Hz.
    ///
    /// After changing the sample rate you should consider updating buffer size to
    /// {@link ChunkedInterleavedResampler.maxNumOutputFrames}.
    #[wasm_bindgen(js_name = "outputSampleRate", getter)]
    pub fn output_sample_rate(&self) -> usize {
        self.as_ref().output_sample_rate()
    }

    // The documentation is overwritten by the getter.
    #[allow(missing_docs)]
    #[wasm_bindgen(js_name = "outputSampleRate", setter)]
    pub fn set_output_sample_rate(
        &mut self,
        #[wasm_bindgen(param_description = "new sample rate in Hz")] value: usize,
    ) {
        self.as_mut().set_output_sample_rate(value);
    }

    /// Get the number of channels.
    #[wasm_bindgen(js_name = "numChannels", getter)]
    pub fn num_channels(&self) -> usize {
        self.as_ref().num_channels()
    }

    /// Get maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as {@link outputLength} plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// {@link ChunkedInterleavedResampler.outputSampleRate}.
    #[wasm_bindgen(js_name = "maxNumOutputFrames")]
    pub fn max_num_output_frames(
        &self,
        #[wasm_bindgen(js_name = "numInputFrames")] num_input_frames: usize,
    ) -> usize {
        self.as_ref().max_num_output_frames(num_input_frames)
    }

    /// Resets internal state.
    ///
    /// Erases any information about the previous chunk.
    ///
    /// Use this method when you want to reuse resampler for another audio stream.
    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.as_mut().reset();
    }

    /// Resamples input signal chunk from the source to the target sample rate and appends the
    /// resulting signal to the output.
    ///
    /// Returns the number of processed input samples. The output is clamped to _[-1; 1]_.
    ///
    /// For each {@link ChunkedInterleavedResampler.inputSampleRate} input samples this method produces exactly
    /// {@link ChunkedInterleavedResampler.outputSampleRate} output samples  even if it is called multiple times with a smaller
    /// amount of input samples; the only exception is when the output sample rate was changed in the process.
    ///
    /// #### Edge cases
    ///
    /// Returns 0 when either the number of input or output frames is less than _max(2, A-1)_, adjusted in
    /// accordance with sample rate ratio.
    ///
    /// #### Limitations
    ///
    /// The output depends on the chunk size, hence resampling the same audio track all at once and
    /// in chunks will produce slightly different results. This a consequence of the fact that Lanczos kernel
    /// isn't an interpolation function, but a filter. To minimize such discrepancies chunk size should
    /// be much larger than _2⋅A + 1_.
    #[wasm_bindgen(js_name = "resample")]
    pub fn resample(&mut self, chunk: &[f32], output: Float32Array) -> usize {
        self.as_mut()
            .resample(&chunk[..], &mut Float32ArrayOutput::new(&output))
    }

    #[inline]
    fn as_ref(&self) -> &RustChunkedInterleavedResampler {
        // SAFETY: Self and ChunkedInterleavedResampler have the same size and the same alignment.
        unsafe { core::mem::transmute(self) }
    }

    #[inline]
    fn as_mut(&mut self) -> &mut RustChunkedInterleavedResampler {
        // SAFETY: Self and ChunkedInterleavedResampler have the same size and the same alignment.
        unsafe { core::mem::transmute(self) }
    }
}
