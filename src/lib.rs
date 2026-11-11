/// Crate for starfix.
extern crate uniffi as uniffi_external;
uniffi_external::setup_scaffolding!();

use arrow::array::{Array, RecordBatch};
use arrow_schema::Schema;
use sha2::Sha256;

use crate::arrow_digester_core::ArrowDigesterCore;

/// Hash format version: the first 3 bytes of every hash, a 24-bit big-endian format number,
/// independent of the crate version. Format 1 (`00 00 01`): starfix 0.4 and earlier. Format 2
/// (`00 00 02`): this crate (byte-layout spec Section 1; format history in Section 10).
pub(crate) const VERSION_BYTES: [u8; 3] = [0_u8, 0_u8, 2_u8];

/// Configuration for the Arrow hasher.
///
/// Every option defaults to `false`. Options are per call (or per streaming digester), never
/// global, so ordered and unordered hashing can be used side by side. Construct with struct-update syntax to stay forward compatible:
///
/// ```
/// use starfix::HasherConfig;
/// let config = HasherConfig { ordered_columns: true, ..HasherConfig::default() };
/// ```
///
/// The order options are recorded in the header of every digest, so hashes taken with different
/// order options never coincide (see `docs/byte-layout-spec.md`, Sections 1 and 6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HasherConfig {
    /// When `true`, schema-level and per-field Arrow metadata are included in the hash.
    /// Default is `false`.
    pub include_metadata: bool,
    /// When `true`, the order of the top-level columns is part of the hash.
    /// Default is `false`: reordering columns does not change the hash.
    pub ordered_columns: bool,
    /// When `true`, the order of struct fields is part of the hash, at every nesting depth
    /// (struct in struct, list, large list, fixed-size list, map, dictionary value type).
    /// Default is `false`: reordering struct fields does not change the hash.
    pub ordered_struct_fields: bool,
}

/// SHA-256 hasher of Arrow data, producing hash format 2 (`00 00 02 || SHA-256`, 35 bytes).
///
/// # Panics
///
/// Every entry point panics on data of an unsupported type, on a schema (or array type) with two
/// fields at the same path (duplicate names, or a name containing `/` that aliases a nested
/// field), and on `ListView`, `LargeListView`, `Union` or `RunEndEncoded` types; `update` panics on
/// a batch whose schema does not match the digester's under its options (e.g. a different column
/// order with `ordered_columns`).
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
pub mod legacy;
pub mod pyarrow;

// Write a test to check that int32 digest is consistent
