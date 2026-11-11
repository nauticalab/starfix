//! Legacy hash formats, kept only to reproduce hashes produced by older starfix releases.
//!
//! **Never hash new data with a legacy module.** New data uses the main API
//! (`starfix::ArrowDigester`, hash format 2).
//!
//! - [`v0_4`]: starfix (Rust) 0.4.0, hash format 1 (`00 00 01`).

pub mod v0_4;
