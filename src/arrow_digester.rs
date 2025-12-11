use arrow::array::{Array, RecordBatch};
use arrow_schema::Schema;
use sha2::Sha256;

use crate::arrow_digester_core::ArrowDigesterCore;

/// Maps `arrow_digester_core` function to a `sha_256` digester + versioning
pub struct ArrowDigester {
    digester: ArrowDigesterCore<Sha256>,
}

impl ArrowDigester {
    /// Create a new instance of `ArrowDigester` with SHA256 as the digester with the schema which will be enforce through each update
    pub fn new(schema: Schema) -> Self {
        Self {
            digester: ArrowDigesterCore::<Sha256>::new(schema),
        }
    }

    /// Update the digester with a new `RecordBatch`
    pub fn update(&mut self, record_batch: &RecordBatch) {
        self.digester.update(record_batch);
    }

    /// Consume the digester and finalize the hash computation
    pub fn finalize(self) -> Vec<u8> {
        self.digester.finalize()
    }

    /// Function to hash an Array in one go
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        ArrowDigesterCore::<Sha256>::hash_array(array)
    }

    /// Function to hash a complete `RecordBatch` in one go
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        ArrowDigesterCore::<Sha256>::hash_record_batch(record_batch)
    }
}
