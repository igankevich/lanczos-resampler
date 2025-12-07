use core::mem::MaybeUninit;
use std::collections::VecDeque;

pub trait WriteF32 {
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

pub trait Output {
    fn remaining(&self) -> usize;

    fn write(&mut self, sample: f32);

    fn write_slice(&mut self, samples: &[f32]);
}

impl<W: WriteF32> Output for &mut [W] {
    fn remaining(&self) -> usize {
        self.len()
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

impl Output for Vec<f32> {
    fn remaining(&self) -> usize {
        isize::MAX as usize - self.len()
    }

    fn write(&mut self, sample: f32) {
        self.push(sample);
    }

    fn write_slice(&mut self, samples: &[f32]) {
        self.extend_from_slice(samples);
    }
}

impl Output for VecDeque<f32> {
    fn remaining(&self) -> usize {
        isize::MAX as usize - self.len()
    }

    fn write(&mut self, sample: f32) {
        self.push_back(sample);
    }

    fn write_slice(&mut self, samples: &[f32]) {
        self.extend(samples);
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Float32ArrayOutput<'a> {
    inner: &'a js_sys::Float32Array,
    offset: u32,
}

#[cfg(target_arch = "wasm32")]
impl<'a> Float32ArrayOutput<'a> {
    pub fn new(inner: &'a js_sys::Float32Array) -> Self {
        Self { inner, offset: 0 }
    }
}

#[cfg(target_arch = "wasm32")]
impl Output for Float32ArrayOutput<'_> {
    fn remaining(&self) -> usize {
        (self.inner.length() - self.offset) as usize
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
