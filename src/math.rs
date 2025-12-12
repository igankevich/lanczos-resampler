#[cfg(not(feature = "std"))]
#[inline]
pub fn floor(x: f32) -> f32 {
    libm::floorf(x)
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn floor(x: f32) -> f32 {
    f32::floor(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn sin(x: f32) -> f32 {
    f32::sin(x)
}
