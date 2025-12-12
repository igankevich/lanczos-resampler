use super::*;
use js_sys::Float32Array;

const MAX_FRAME_LEN: usize = 16;

/// [`Output`] implementation for [`Float32Array`].
///
/// Alleviates unnecessary copying between JS and WASM.
///
/// Supports up to 16 interleaved channels.
pub struct Float32ArrayOutput<'a> {
    inner: &'a Float32Array,
    offset: u32,
    frame: [f32; MAX_FRAME_LEN],
}

impl<'a> Float32ArrayOutput<'a> {
    /// Create new output.
    #[inline]
    pub const fn new(inner: &'a Float32Array) -> Self {
        Self {
            inner,
            offset: 0,
            frame: [0.0; MAX_FRAME_LEN],
        }
    }
}

impl Output for Float32ArrayOutput<'_> {
    fn remaining(&self) -> Option<usize> {
        Some((self.inner.length() - self.offset) as usize)
    }

    fn write(&mut self, sample: f32) {
        self.inner.set_index(self.offset, sample);
        self.offset += 1;
    }

    fn write_slice(&mut self, samples: &[f32]) {
        for sample in samples {
            self.write(*sample);
        }
    }

    /// Panics when `num_channels` is greater than 16.
    fn write_frame(&mut self, num_channels: usize, write: impl FnOnce(&mut [f32])) {
        assert!(num_channels <= MAX_FRAME_LEN);
        let frame = &mut self.frame[..num_channels];
        frame.fill(0.0);
        write(frame);
        for sample in frame {
            self.inner.set_index(self.offset, *sample);
            self.offset += 1;
        }
    }
}
