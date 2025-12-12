#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod chunked;
mod filter;
mod kernel;
mod math;
mod output;
#[cfg(test)]
mod tests;
mod whole;

use self::filter::*;
use self::kernel::*;
use self::math::*;

pub use self::chunked::*;
pub use self::output::*;
pub use self::whole::*;

pub(crate) const DEFAULT_N: usize = 16;
pub(crate) const DEFAULT_A: usize = 3;
