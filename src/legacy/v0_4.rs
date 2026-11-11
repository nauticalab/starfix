//! **starfix 0.4 LEGACY mode** — hash format 1 (`00 00 01`), frozen.
//!
//! This module reproduces the output of the **Rust** crate starfix **0.4.0** byte for byte. It
//! exists only to read or match data that was hashed with starfix 0.4.x (for example to find
//! stored data under its old hash). **Never hash new data with it**: use the main API
//! (`starfix::ArrowDigester`, hash format 2).
//!
//! It is **frozen**: never fixed, never extended (no order options, no `hash_array_with_config`).
//! Its source (`src/legacy/v0_4.rs`, `src/legacy/v0_4/`) is the v0.4.0 source verbatim, except module paths; a test
//! (`tests/legacy_v0_4_freeze.rs`) fails if the files change, and the golden and differential
//! tests (`tests/legacy_v0_4_*.rs`) fail if any output byte changes.
//!
//! Known defects of format 1, reproduced on purpose (byte-layout spec, Section 10):
//! - Booleans are bit-packed per call without a count: a one-column non-nullable `[true]` hashes
//!   like `[true, false, false]`, and the hash of a Boolean column depends on batch boundaries.
//! - A null struct hashes like a struct whose fields are all null; with non-nullable fields,
//!   null struct rows vanish.
//! - A list whose element is a nullable list loses its own lengths; in a list of nullable
//!   structs a null element hashes like one with all-null fields.
//! - The row count of zero-column tables and empty structs is not hashed.
//! - Duplicate or `/`-aliased field names are accepted with ill-defined hashes.
//! - A nullable Boolean under a nullable struct at a byte-aligned array offset >= 8 panics.
//!
//! This is the Rust 0.4.0 behaviour. starfix-python's legacy module (`starfix.legacy.v0_4`)
//! reproduces starfix-**python** 0.4.0, which differed from Rust 0.4.0 on a few inputs (map
//! schemas, non-ASCII names in `hash_array`, map metadata paths, streaming schema checks); see
//! the divergence table in starfix-python's documentation.

use arrow::array::{Array, RecordBatch};
use arrow_schema::Schema;
use sha2::Sha256;

use crate::legacy::v0_4::arrow_digester_core::ArrowDigesterCore;

pub(crate) mod arrow_digester_core;

const VERSION_BYTES: [u8; 3] = [0_u8, 0_u8, 1_u8]; // Hash format version 0.0.1 — independent of crate version

/// Configuration for the Arrow hasher.
#[derive(Clone, Copy, Debug, Default)]
pub struct HasherConfig {
    /// When `true`, schema-level and per-field Arrow metadata are included in the hash.
    /// Default is `false`, preserving pre-v0.1.0 hash stability.
    pub include_metadata: bool,
}

/// Maps `ArrowDigesterCore` to a SHA-256 digester with version prefix.
#[derive(Clone)]
pub struct ArrowDigester {
    digester: ArrowDigesterCore<Sha256>,
}

impl ArrowDigester {
    /// Create a new instance of `ArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    pub fn new(schema: &Schema, config: HasherConfig) -> Self {
        Self {
            digester: ArrowDigesterCore::<Sha256>::new(schema, config.include_metadata),
        }
    }

    /// Update the digester with a new `RecordBatch`.
    pub fn update(&mut self, record_batch: &RecordBatch) {
        self.digester.update(record_batch);
    }

    /// Consume the digester and finalize the hash computation.
    pub fn finalize(self) -> Vec<u8> {
        Self::prepend_version_bytes(self.digester.finalize())
    }

    /// Hash an array in one go.
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_array(array))
    }

    /// Hash a complete `RecordBatch` in one go.
    pub fn hash_record_batch(record_batch: &RecordBatch, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_record_batch(
            record_batch,
            config.include_metadata,
        ))
    }

    /// Hash a schema only.
    pub fn hash_schema(schema: &Schema, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_schema(
            schema,
            config.include_metadata,
        ))
    }

    fn prepend_version_bytes(digest: Vec<u8>) -> Vec<u8> {
        let mut complete_hash = VERSION_BYTES.clone().to_vec();
        complete_hash.extend(digest);
        complete_hash
    }
}
