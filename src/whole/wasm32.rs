use super::default as rust;
use crate::Float32ArrayOutput;
use core::ptr;
use core::slice;
use js_sys::Float32Array;
use js_sys::Number;
use wasm_bindgen::prelude::*;

const WHOLE_RESAMPLER_LEN: usize = size_of::<rust::WholeResampler>();

const _: () = assert!(align_of::<WholeResampler>() == align_of::<rust::WholeResampler>());
const _: () = assert!(size_of::<WholeResampler>() == size_of::<rust::WholeResampler>());

/// A resampler that processes audio input as a whole.
///
/// This struct uses [Lanczos kernel](https://en.wikipedia.org/wiki/Lanczos_resampling)
/// approximated by _2⋅N - 1_ points and defined on interval _[-A; A]_. The kernel is interpolated
/// using cubic Hermite splines with second-order finite differences at spline endpoints. The
/// output is clamped to _[-1; 1]_.
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
pub struct WholeResampler([u8; WHOLE_RESAMPLER_LEN]);

#[wasm_bindgen]
impl WholeResampler {
    /// Create new resampler.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut buf = [0_u8; WHOLE_RESAMPLER_LEN];
        let resampler = rust::WholeResampler::new();
        // SAFETY: Self and WholeResampler have the same size and the same alignment.
        buf.copy_from_slice(unsafe {
            slice::from_raw_parts(ptr::from_ref(&resampler).cast(), WHOLE_RESAMPLER_LEN)
        });
        Self(buf)
    }

    /// Resample input signal from the source to the target sample rate and
    /// returns the resulting output signal as a vector.
    ///
    /// #### Edge cases
    ///
    /// - Returns an empty array when either the input length or calculated output length is less than 2.
    /// - Returns an empty array when either the input length or the output sample rate is too large.
    ///
    /// #### Limitations
    ///
    /// This function shouldn't be used when processing audio track in chunks;
    /// use {@link ChunkedResampler.resample} instead.
    #[wasm_bindgen(js_name = "resample")]
    pub fn resample(
        &self,
        #[wasm_bindgen(param_description = "input samples")] input: &[f32],
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
        let Some(output_len) =
            rust::checked_output_len(input.len(), input_sample_rate, output_sample_rate)
        else {
            return Float32Array::new_with_length(0);
        };
        let output = Float32Array::new_with_length(output_len as u32);
        self.as_ref()
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
    /// Use {@link outputLength} to ensure that the buffer size is sufficient.
    ///
    /// #### Limitations
    ///
    /// This function shouldn't be used when processing audio track in chunks;
    /// use {@link ChunkedResampler.resample} instead.
    #[wasm_bindgen(js_name = "resampleInto")]
    pub fn resample_into(&self, input: &[f32], output: &Float32Array) -> usize {
        // Having &Float32Array as the output is faster than &mut [f32]...
        self.as_ref()
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
    ///   Use {@link outputLength} to ensure that the buffer size is sufficient.
    /// - Panics when either the input or the output length isn't evenly divisible by the number of
    ///   channels.
    ///
    /// #### Limitations
    ///
    /// This function shouldn't be used when processing audio track in chunks;
    /// use {@link ChunkedResampler.resample} instead.
    #[wasm_bindgen(js_name = "resampleInterleavedInto")]
    pub fn resample_interleaved_into(
        &self,
        #[wasm_bindgen(param_description = "input frames")] input: &[f32],
        #[wasm_bindgen(js_name = "numChannels", param_description = "number of channels")]
        num_channels: usize,
        output: &Float32Array,
    ) -> usize {
        self.as_ref().resample_interleaved_into(
            &input[..],
            num_channels,
            &mut Float32ArrayOutput::new(output),
        )
    }

    #[inline]
    fn as_ref(&self) -> &rust::WholeResampler {
        // SAFETY: Self and WholeResampler have the same size and the same alignment.
        unsafe { core::mem::transmute(self) }
    }
}

/// Calculates resampled length of the input for given input/output sample
/// rates.
///
/// #### Edge cases
///
/// Returns `Number.NAN` when the input length or the output sample rate is too large.
///
/// #### Limitations
///
/// This function shouldn't be used when processing audio track in chunks;
/// use {@link ChunkedResampler} instead.
#[wasm_bindgen(js_name = "outputLength")]
pub fn output_length(
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
    match rust::checked_output_len(input_len, input_sample_rate, output_sample_rate) {
        Some(len) => (len as u32).into(),
        None => Number::NAN.into(),
    }
}
