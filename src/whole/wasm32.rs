use super::default as rust;
use crate::Float32ArrayOutput;
use js_sys::Float32Array;
use js_sys::Number;
use wasm_bindgen::prelude::*;

/// A resampler that processes audio input as a whole.
///
/// Use it to process audio streams.
///
/// ## Parameters
///
/// This resampler uses default parameters: _N = 16, A = 3_.
///
/// ## Limitations
///
/// `WholeResampler` shouldn't be used to process audio track in chunks; use {@link ChunkedResampler} instead.
#[wasm_bindgen]
#[repr(align(4))]
#[allow(unused)]
pub struct WholeResampler(rust::WholeResampler);

#[wasm_bindgen]
impl WholeResampler {
    /// Create new resampler.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let resampler = rust::WholeResampler::new();
        Self(resampler)
    }

    /// Resample input signal from the source to the target sample rate and
    /// returns the resulting output signal as a vector.
    ///
    /// #### Edge cases
    ///
    /// - Returns an empty array when either the input length or calculated output length is less than 2.
    /// - Returns an empty array when either the input length or the output sample rate is too large.
    #[wasm_bindgen(js_name = "resample")]
    pub fn resample(
        &self,
        #[wasm_bindgen(param_description = "input samples")] input: &Float32Array,
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
    ) -> Float32Array {
        let input = input.to_vec();
        let Some(output_len) =
            rust::checked_num_output_frames(input.len(), input_sample_rate, output_sample_rate)
        else {
            return Float32Array::new_with_length(0);
        };
        let output = Float32Array::new_with_length(output_len as u32);
        self.0
            .resample_into(&input, &mut Float32ArrayOutput::new(&output));
        output
    }

    /// This is a variant of {@link WholeResampler.resample} that doesn't use memory allocation.
    ///
    /// Returns the number of samples read from the input. Currently this is either 0 (see "Panics") or
    /// the input length.
    ///
    /// #### Edge cases
    ///
    /// Returns 0 when either the input length or remaining output length is less than 2.
    ///
    /// #### Panics
    ///
    /// Panics when the output isn't large enough to hold all the resampled points.
    /// Use {@link numOutputFrames} to ensure that the buffer size is sufficient.
    #[wasm_bindgen(js_name = "resampleInto")]
    pub fn resample_into(&self, input: &Float32Array, output: &Float32Array) -> usize {
        let input = input.to_vec();
        // Having &Float32Array as the output is faster than &mut [f32]...
        self.0
            .resample_into(&input[..], &mut Float32ArrayOutput::new(output))
    }

    /// This is a variant of {@link resampleInto} that processes several audio channels (one audio frame) at a time.
    ///
    /// #### Edge cases
    ///
    /// Returns 0 when either the number of input frames or the number of remaining output frames is less than 2.
    ///
    /// #### Panics
    ///
    /// - Panics when the output isn't large enough to hold all the resampled points.
    ///   Use {@link numOutputFrames} to ensure that the buffer size is sufficient.
    /// - Panics when either the input or the output length isn't evenly divisible by the number of
    ///   channels.
    #[wasm_bindgen(js_name = "resampleInterleavedInto")]
    pub fn resample_interleaved_into(
        &self,
        #[wasm_bindgen(param_description = "input frames")] input: &Float32Array,
        #[wasm_bindgen(js_name = "numChannels", param_description = "number of channels")]
        num_channels: usize,
        output: &Float32Array,
    ) -> usize {
        let input = input.to_vec();
        self.0.resample_interleaved_into(
            &input[..],
            num_channels,
            &mut Float32ArrayOutput::new(output),
        )
    }
}

/// Calculates resampled length of the input for the given input/output sample
/// rates.
///
/// #### Edge cases
///
/// Returns `Number.NAN` when the input length or the output sample rate is too large.
///
/// #### Limitations
///
/// This function shouldn't be used when processing audio track in chunks;
/// use {@link ChunkedResampler.maxNumOutputFrames} instead.
#[wasm_bindgen(js_name = "numOutputFrames")]
pub fn num_output_frames(
    #[wasm_bindgen(param_description = "input length", js_name = "inputLength")] input_len: usize,
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
) -> Number {
    match rust::checked_num_output_frames(input_len, input_sample_rate, output_sample_rate) {
        Some(len) => (len as u32).into(),
        None => Number::NAN.into(),
    }
}
