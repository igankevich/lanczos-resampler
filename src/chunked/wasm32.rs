#[cfg(any(feature = "alloc", test))]
use super::default::ChunkedInterleavedResampler as RustChunkedInterleavedResampler;
use super::default::ChunkedResampler as RustChunkedResampler;
use crate::Float32ArrayOutput;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

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
pub struct ChunkedResampler(RustChunkedResampler);

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
        Self(RustChunkedResampler::new(
            input_sample_rate,
            output_sample_rate,
        ))
    }

    /// Get input sample rate in Hz.
    #[wasm_bindgen(js_name = "inputSampleRate", getter)]
    pub fn input_sample_rate(&self) -> usize {
        self.0.input_sample_rate()
    }

    /// Get/set output sample rate in Hz.
    ///
    /// After changing the sample rate you should consider updating buffer size to
    /// {@link ChunkedResampler.maxNumOutputFrames}.
    #[wasm_bindgen(js_name = "outputSampleRate", getter)]
    pub fn output_sample_rate(&self) -> usize {
        self.0.output_sample_rate()
    }

    // The documentation is overwritten by the getter.
    #[allow(missing_docs)]
    #[wasm_bindgen(js_name = "outputSampleRate", setter)]
    pub fn set_output_sample_rate(
        &mut self,
        #[wasm_bindgen(param_description = "new sample rate in Hz")] value: usize,
    ) {
        self.0.set_output_sample_rate(value);
    }

    /// Get maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as {@link numOutputFrames} plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// {@link ChunkedResampler.outputSampleRate}.
    #[wasm_bindgen(js_name = "maxNumOutputFrames")]
    pub fn max_num_output_frames(
        &self,
        #[wasm_bindgen(js_name = "numInputFrames")] num_input_frames: usize,
    ) -> usize {
        self.0.max_num_output_frames(num_input_frames)
    }

    /// Resets internal state.
    ///
    /// Erases any information about the previous chunk.
    ///
    /// Use this method when you want to reuse resampler for another audio stream.
    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.0.reset();
    }

    /// Resamples input signal chunk from the source to the target sample rate and appends the
    /// resulting signal to the output.
    ///
    /// Returns the number of processed input samples and the number of produced output samples.
    /// The output is clamped to _[-1; 1]_.
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
    pub fn resample(&mut self, chunk: &[f32], output: Float32Array) -> ResampleOutcome {
        let mut output = Float32ArrayOutput::new(&output);
        let num_read = self.0.resample(&chunk[..], &mut output);
        let num_written = output.position() as usize;
        ResampleOutcome {
            num_read,
            num_written,
        }
    }

    /// Resamples input signal chunk to fill the output array.
    ///
    /// Returns the number of processed input samples.
    /// Currently this is either 0 or the input length.
    /// The output is clamped to _[-1; 1]_.
    ///
    /// This method uses _number of input samples / number of output samples_ as the input/output sample rate ratio.
    /// It's up to the caller to ensure that this ratio is close to the original one to minimize
    /// artifacts.
    ///
    /// Use this method to resample the last chunk of the input that is either too small to fill
    /// the output array or too large to fully fit into the output array.
    /// One way of doing so is to resample the last chunk together with the previous one.
    ///
    /// #### Edge cases
    ///
    /// Returns 0 when either the input length is less than _max(2, A-1)_ or output length is less than 2.
    #[wasm_bindgen(js_name = "resampleExact")]
    pub fn resample_exact(&mut self, chunk: &[f32], output: Float32Array) -> ResampleOutcome {
        let mut output = Float32ArrayOutput::new(&output);
        let num_read = self.0.resample_exact(&chunk[..], &mut output);
        let num_written = output.position() as usize;
        ResampleOutcome {
            num_read,
            num_written,
        }
    }
}

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
pub struct ChunkedInterleavedResampler(RustChunkedInterleavedResampler);

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
        Self(RustChunkedInterleavedResampler::new(
            input_sample_rate,
            output_sample_rate,
            num_channels,
        ))
    }

    /// Get input sample rate in Hz.
    #[wasm_bindgen(js_name = "inputSampleRate", getter)]
    pub fn input_sample_rate(&self) -> usize {
        self.0.input_sample_rate()
    }

    /// Get/set output sample rate in Hz.
    ///
    /// After changing the sample rate you should consider updating buffer size to
    /// {@link ChunkedInterleavedResampler.maxNumOutputFrames}.
    #[wasm_bindgen(js_name = "outputSampleRate", getter)]
    pub fn output_sample_rate(&self) -> usize {
        self.0.output_sample_rate()
    }

    // The documentation is overwritten by the getter.
    #[allow(missing_docs)]
    #[wasm_bindgen(js_name = "outputSampleRate", setter)]
    pub fn set_output_sample_rate(
        &mut self,
        #[wasm_bindgen(param_description = "new sample rate in Hz")] value: usize,
    ) {
        self.0.set_output_sample_rate(value);
    }

    /// Get the number of channels.
    #[wasm_bindgen(js_name = "numChannels", getter)]
    pub fn num_channels(&self) -> usize {
        self.0.num_channels()
    }

    /// Get maximum output chunk length given the input chunk length.
    ///
    /// Returns the same value as {@link numOutputFrames} plus one.
    /// This additional sample is used to compensate for unevenly divisible sample rates.
    ///
    /// You should consider updating buffer size every time you change output sample rate via
    /// {@link ChunkedInterleavedResampler.outputSampleRate}.
    #[wasm_bindgen(js_name = "maxNumOutputFrames")]
    pub fn max_num_output_frames(
        &self,
        #[wasm_bindgen(js_name = "numInputFrames")] num_input_frames: usize,
    ) -> usize {
        self.0.max_num_output_frames(num_input_frames)
    }

    /// Resets internal state.
    ///
    /// Erases any information about the previous chunk.
    ///
    /// Use this method when you want to reuse resampler for another audio stream.
    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.0.reset();
    }

    /// Resamples input signal chunk from the source to the target sample rate and appends the
    /// resulting signal to the output.
    ///
    /// Returns the number of processed input samples and the number of produced output samples.
    /// The output is clamped to _[-1; 1]_.
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
    pub fn resample(&mut self, chunk: &[f32], output: Float32Array) -> ResampleOutcome {
        let mut output = Float32ArrayOutput::new(&output);
        let num_read = self.0.resample(&chunk[..], &mut output);
        let num_written = output.position() as usize;
        ResampleOutcome {
            num_read,
            num_written,
        }
    }

    /// Resamples input signal chunk to fill the output array.
    ///
    /// Returns the number of processed input samples.
    /// Currently this is either 0 or the input length.
    /// The output is clamped to _[-1; 1]_.
    ///
    /// This method uses _number of input frames / number of output frames_ as the input/output sample rate ratio.
    /// It's up to the caller to ensure that this ratio is close to the original one to minimize
    /// artifacts.
    ///
    /// Use this method to resample the last chunk of the input that is either too small to fill
    /// the output array or too large to fully fit into the output array.
    /// One way of doing so is to resample the last chunk together with the previous one.
    ///
    /// #### Edge cases
    ///
    /// Returns 0 when either the number of input frames is less than _max(2, A-1)_ or output length is less than 2.
    #[wasm_bindgen(js_name = "resampleExact")]
    pub fn resample_exact(&mut self, chunk: &[f32], output: Float32Array) -> ResampleOutcome {
        let mut output = Float32ArrayOutput::new(&output);
        let num_read = self.0.resample_exact(&chunk[..], &mut output);
        let num_written = output.position() as usize;
        ResampleOutcome {
            num_read,
            num_written,
        }
    }
}

/// Resampling outcome.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct ResampleOutcome {
    /// How many samples were read from the input.
    #[wasm_bindgen(js_name = "numRead")]
    pub num_read: usize,
    /// How many samples were wrtten to the output.
    #[wasm_bindgen(js_name = "numWritten")]
    pub num_written: usize,
}
