use core::arch::x86_64::*;

#[repr(align(32))]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct I256(pub [i32; 8]);

impl I256 {
    #[allow(unused)]
    pub const fn from_scalar(value: i32) -> Self {
        Self([value; 8])
    }

    pub const fn zero() -> Self {
        Self([0; 8])
    }

    #[allow(unused)]
    pub const fn as_ptr(&self) -> *const i32 {
        self.0.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut i32 {
        self.0.as_mut_ptr()
    }
}

impl core::ops::Deref for I256 {
    type Target = [i32; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for I256 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<I256> for __m256i {
    fn from(other: I256) -> Self {
        unsafe { _mm256_load_si256(other.as_ptr() as *const __m256i) }
    }
}

impl From<__m256i> for I256 {
    fn from(other: __m256i) -> Self {
        // TODO MaybeUninit?
        let mut value = Self::zero();
        unsafe { _mm256_store_si256(value.as_mut_ptr() as *mut __m256i, other) };
        value
    }
}

#[repr(align(32))]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct F256(pub [f32; 8]);

impl F256 {
    pub const fn from_scalar(value: f32) -> Self {
        Self([value; 8])
    }

    pub const fn zero() -> Self {
        Self([0.0; 8])
    }

    pub const fn as_ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut f32 {
        self.0.as_mut_ptr()
    }
}

impl core::ops::Deref for F256 {
    type Target = [f32; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for F256 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<F256> for __m256 {
    fn from(other: F256) -> Self {
        unsafe { _mm256_load_ps(other.as_ptr()) }
    }
}

impl From<__m256> for F256 {
    fn from(other: __m256) -> Self {
        // TODO MaybeUninit?
        let mut value = Self::zero();
        unsafe { _mm256_store_ps(value.as_mut_ptr(), other) };
        value
    }
}

pub trait M256Ext {
    /// Pairwise sum.
    fn sum(self) -> f32;

    fn as_f32_array(&self) -> &[f32; 8];
}

impl M256Ext for __m256 {
    fn sum(self) -> f32 {
        unsafe {
            // Sum left and right halves.
            let y128 = _mm_add_ps(
                _mm256_extractf128_ps(self, 0),
                _mm256_extractf128_ps(self, 1),
            );
            // Pairwise sum.
            let sum1 = f32::from_bits(_mm_extract_ps(y128, 0) as u32)
                + f32::from_bits(_mm_extract_ps(y128, 2) as u32);
            let sum2 = f32::from_bits(_mm_extract_ps(y128, 1) as u32)
                + f32::from_bits(_mm_extract_ps(y128, 3) as u32);
            sum1 + sum2
        }
    }

    fn as_f32_array(&self) -> &[f32; 8] {
        unsafe { &*(self as *const __m256 as *const [f32; 8]) }
    }
}
