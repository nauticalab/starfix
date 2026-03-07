/// Crate for starfix.
extern crate uniffi as uniffi_external;
uniffi_external::setup_scaffolding!();

use arrow::array::{Array, RecordBatch};
use arrow_schema::Schema;
use sha2::Sha256;

use crate::arrow_digester_core::ArrowDigesterCore;

const VERSION_BYTES: [u8; 3] = [0_u8, 0_u8, 1_u8]; // Version 0.0.1

/// Maps `ArrowDigesterCore` to a SHA-256 digester with version prefix.
#[derive(Clone)]
pub struct ArrowDigester {
    digester: ArrowDigesterCore<Sha256>,
}

impl ArrowDigester {
    /// Create a new instance of `ArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    pub fn new(schema: &Schema) -> Self {
        Self {
            digester: ArrowDigesterCore::<Sha256>::new(schema),
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
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_record_batch(record_batch))
    }

    /// Hash a schema only.
    pub fn hash_schema(schema: &Schema) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_schema(schema))
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
