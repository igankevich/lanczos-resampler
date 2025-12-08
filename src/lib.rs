#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod chunked_resampler;
mod filter;
mod input;
mod kernel;
mod math;
mod output;
mod resample_full;
#[cfg(target_arch = "x86_64")]
mod simd;

#[cfg(test)]
mod tests;

use self::filter::*;
use self::kernel::*;
use self::math::*;
#[cfg(target_arch = "x86_64")]
use self::simd::*;

pub use self::chunked_resampler::*;
pub use self::input::*;
pub use self::output::*;
pub use self::resample_full::*;

#[cfg(target_arch = "wasm32")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
mod wasm;
#[cfg(all(target_arch = "wasm32", not(test)))]
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
pub use self::wasm::*;
