use super::*;
use js_sys::Float32Array;

pub struct Float32ArrayOutput<'a> {
    inner: &'a Float32Array,
    offset: u32,
}

impl<'a> Float32ArrayOutput<'a> {
    #[inline]
    pub const fn new(inner: &'a Float32Array) -> Self {
        Self { inner, offset: 0 }
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
}
