pub trait Input {
    fn get(&self, i: usize) -> f32;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn take(&self, n: usize) -> impl Input;
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

    fn take(&self, n: usize) -> impl Input {
        &self[..n]
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

    fn take(&self, n: usize) -> impl Input {
        &self[..n]
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

    fn take(&self, n: usize) -> impl Input {
        &self[..n]
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

    fn take(&self, n: usize) -> impl Input {
        self.slice(0, n as u32)
    }
}
