pub(crate) mod default;
#[cfg(target_arch = "wasm32")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
mod wasm32;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(not(target_arch = "wasm32"))]
pub use self::default::*;

#[cfg(target_arch = "wasm32")]
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
pub use self::wasm32::*;
