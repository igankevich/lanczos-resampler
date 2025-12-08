use core::ops::Range;

/// Resampler's input, a collection of samples.
///
/// All samples are [`f32`], little-endian.
pub trait Input {
    /// Returns the sample at index `i`.
    ///
    /// Panics if the index is out of bounds.
    fn get(&self, i: usize) -> f32;

    /// Returns the number of samples.
    fn len(&self) -> usize;

    /// Returns `true` if the input is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice of the input containing the samples in the specified range.
    ///
    /// Panics if the range crosses the input bounds.
    fn slice(&self, range: Range<usize>) -> impl Input;
}

impl Input for &[f32] {
    fn get(&self, i: usize) -> f32 {
        self[i]
    }

    fn len(&self) -> usize {
        (*self as &[f32]).len()
    }

    fn is_empty(&self) -> bool {
        (*self as &[f32]).is_empty()
    }

    fn slice(&self, range: Range<usize>) -> impl Input {
        &self[range]
    }
}

impl Input for [f32] {
    fn get(&self, i: usize) -> f32 {
        self[i]
    }

    fn len(&self) -> usize {
        (self as &[f32]).len()
    }

    fn is_empty(&self) -> bool {
        (self as &[f32]).is_empty()
    }

    fn slice(&self, range: Range<usize>) -> impl Input {
        &self[range]
    }
}

impl<const N: usize> Input for [f32; N] {
    fn get(&self, i: usize) -> f32 {
        self[i]
    }

    fn len(&self) -> usize {
        N
    }

    fn is_empty(&self) -> bool {
        N == 0
    }

    fn slice(&self, range: Range<usize>) -> impl Input {
        &self[range]
    }
}

#[cfg(target_arch = "wasm32")]
impl Input for js_sys::Float32Array {
    fn get(&self, i: usize) -> f32 {
        self.get_index(i as u32)
    }

    fn len(&self) -> usize {
        self.length() as usize
    }

    fn is_empty(&self) -> bool {
        self.length() == 0
    }

    fn slice(&self, range: Range<usize>) -> impl Input {
        self.slice(range.start as u32, range.end as u32)
    }
}
