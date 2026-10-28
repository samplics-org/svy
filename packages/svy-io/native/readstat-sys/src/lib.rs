// native/readstat-sys/src/lib.rs
// bindgen's output: its names and lints are not ours to fix.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::all
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub use bindings::*;
