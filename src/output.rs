use core::mem::MaybeUninit;

/// A writable [`f32`] audio sample.
pub trait WriteF32 {
    /// Sets sample's value to `value`.
    fn write(&mut self, value: f32);
}

impl WriteF32 for f32 {
    fn write(&mut self, value: f32) {
        *self = value;
    }
}

impl WriteF32 for MaybeUninit<f32> {
    fn write(&mut self, value: f32) {
        MaybeUninit::<f32>::write(self, value);
    }
}

/// Resampler's output, a buffer of samples.
pub trait Output {
    /// Returns the remaining length of the buffer or `None` if it's not known.
    fn remaining(&self) -> Option<usize>;

    /// Appends `sample` to the buffer.
    fn write(&mut self, sample: f32);

    /// Appends `samples` to the buffer.
    fn write_slice(&mut self, samples: &[f32]);
}

impl<W: WriteF32> Output for &mut [W] {
    fn remaining(&self) -> Option<usize> {
        Some(self.len())
    }

    fn write(&mut self, sample: f32) {
        let (out, rest) = core::mem::take(self).split_at_mut(1);
        out[0].write(sample);
        *self = rest;
    }

    fn write_slice(&mut self, samples: &[f32]) {
        let n = samples.len();
        let (chunk, rest) = core::mem::take(self).split_at_mut(n);
        for (out, sample) in chunk.iter_mut().zip(samples.iter().copied()) {
            out.write(sample);
        }
        *self = rest;
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl Output for alloc::vec::Vec<f32> {
    fn remaining(&self) -> Option<usize> {
        None
    }

    fn write(&mut self, sample: f32) {
        self.push(sample);
    }

    fn write_slice(&mut self, samples: &[f32]) {
        self.extend_from_slice(samples);
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl Output for alloc::collections::VecDeque<f32> {
    fn remaining(&self) -> Option<usize> {
        None
    }

    fn write(&mut self, sample: f32) {
        self.push_back(sample);
    }

    fn write_slice(&mut self, samples: &[f32]) {
        self.extend(samples);
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
mod wasm32;

#[cfg(target_arch = "wasm32")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
pub use self::wasm32::*;
