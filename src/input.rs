use core::ops::Index;

pub trait Input: Index<usize, Output = f32> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn take(&self, n: usize) -> &(impl Input + ?Sized);
}

impl Input for [f32] {
    fn len(&self) -> usize {
        (self as &[f32]).len()
    }

    fn is_empty(&self) -> bool {
        (self as &[f32]).is_empty()
    }

    fn take(&self, n: usize) -> &(impl Input + ?Sized) {
        &self[..n]
    }
}

impl<const N: usize> Input for [f32; N] {
    fn len(&self) -> usize {
        N
    }

    fn is_empty(&self) -> bool {
        N == 0
    }

    fn take(&self, n: usize) -> &(impl Input + ?Sized) {
        &self[..n]
    }
}
