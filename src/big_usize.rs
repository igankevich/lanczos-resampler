// Unsigned integer that is twice as large as `usize`.
#[cfg(target_pointer_width = "64")]
pub type BigUsize = u128;
#[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
pub type BigUsize = u64;

// Sanity check.
const _: () = assert!(core::mem::size_of::<usize>() * 2 == core::mem::size_of::<BigUsize>());
