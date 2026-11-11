/// Crate for starfix.
extern crate uniffi as uniffi_external;
uniffi_external::setup_scaffolding!();

use arrow::array::{Array, RecordBatch};
use arrow_schema::Schema;
use sha2::Sha256;

use crate::arrow_digester_core::ArrowDigesterCore;

const VERSION_BYTES: [u8; 3] = [0_u8, 0_u8, 1_u8]; // Hash format version 0.0.1 — independent of crate version

/// Configuration for the Arrow hasher.
///
/// Every option defaults to `false`, which reproduces the v0.4.0 hashes byte for byte. Options are
/// per call (or per streaming digester), never global, so ordered and unordered hashing can be
/// used side by side. Construct with struct-update syntax to stay forward compatible:
///
/// ```
/// use starfix::HasherConfig;
/// let config = HasherConfig { ordered_columns: true, ..HasherConfig::default() };
/// ```
///
/// When any of `ordered_columns`, `ordered_struct_fields` or `nested_validity` is `true`, the
/// hash is domain separated from every other combination of those three options (see
/// `docs/byte-layout-spec.md`, Section 9).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent on/off hashing options, mirrored one-to-one by Python keyword arguments"
)]
pub struct HasherConfig {
    /// When `true`, schema-level and per-field Arrow metadata are included in the hash.
    /// Default is `false`, preserving pre-v0.1.0 hash stability.
    pub include_metadata: bool,
    /// When `true`, the order of the top-level columns is part of the hash.
    /// Default is `false`: reordering columns does not change the hash.
    pub ordered_columns: bool,
    /// When `true`, the order of struct fields is part of the hash, at every nesting depth
    /// (struct in struct, list, large list, fixed-size list, map, dictionary value type).
    /// Default is `false`: reordering struct fields does not change the hash.
    pub ordered_struct_fields: bool,
    /// When `true`, nullable nested values record their own validity: a null struct is
    /// distinguished from a struct whose fields are all null, null struct rows are counted even
    /// when the struct's fields are non-nullable, and a list whose elements are nullable lists
    /// keeps its own lengths. Default is `false` (v0.4.0 layout).
    pub nested_validity: bool,
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
            digester: ArrowDigesterCore::<Sha256>::new_with_config(schema, config),
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

    /// Hash an array in one go with the default configuration.
    ///
    /// Equivalent to `hash_array_with_config(array, HasherConfig::default())`.
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        Self::hash_array_with_config(array, HasherConfig::default())
    }

    /// Hash an array in one go with the given configuration.
    ///
    /// `include_metadata` has no effect here (there is no schema). `ordered_columns` has no
    /// effect on the layout, but like every order/validity option it is part of the domain
    /// separation tag.
    pub fn hash_array_with_config(array: &dyn Array, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_array_with_config(
            array, config,
        ))
    }

    /// Hash a complete `RecordBatch` in one go.
    pub fn hash_record_batch(record_batch: &RecordBatch, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_record_batch(
            record_batch,
            config,
        ))
    }

    /// Hash a schema only.
    pub fn hash_schema(schema: &Schema, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_schema_with_config(
            schema, config,
        ))
    }

    fn prepend_version_bytes(digest: Vec<u8>) -> Vec<u8> {
        let mut complete_hash = VERSION_BYTES.clone().to_vec();
        complete_hash.extend(digest);
        complete_hash
    }
}

pub(crate) mod arrow_digester_core;
pub mod pyarrow;

// Write a test to check that int32 digest is consistent
