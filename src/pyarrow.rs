#![expect(
    unsafe_code,
    clippy::expect_used,
    reason = "Converting raw pointers to Arrow structures"
)]
use std::sync::{Arc, Mutex};

use crate::ArrowDigester;
use arrow::array::{RecordBatch, StructArray};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_schema::Schema;

/// Process an Arrow table via C Data Interface
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow

#[uniffi::export]
pub fn hash_record_batch(array_ptr: u64, schema_ptr: u64) -> Vec<u8> {
    #[expect(
        unsafe_code,
        reason = "Need to convert raw pointers to Arrow data structures"
    )]
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        clippy::expect_used,
        reason = "Okay since we are doing the same operation of dereferencing pointers, Will add proper errors later"
    )]
    // SAFETY:
    // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
    let array_data = unsafe {
        // Construct ArrayData from FFI structures
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut FFI_ArrowArray);
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
        from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data")
    };

    // Hash the table
    ArrowDigester::hash_record_batch(&RecordBatch::from(StructArray::from(array_data)))
}

/// Process an Arrow schema via C Data Interface
///
/// # Panics
/// The pointer must be a valid Arrow schema from Python's pyarrow
#[uniffi::export]
pub fn hash_schema(schema_ptr: u64) -> Vec<u8> {
    #[expect(
        unsafe_code,
        reason = "Need to convert raw pointers to Arrow data structures"
    )]
    // SAFETY:
    // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
    let schema = unsafe {
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
        Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema")
    };

    // Hash the schema
    ArrowDigester::hash_schema(&schema)
}

#[derive(uniffi::Object)]
pub struct InternalPyArrowDigester {
    digester: Arc<Mutex<ArrowDigester>>,
}

#[uniffi::export]
impl InternalPyArrowDigester {
    /// Create a new instance of `PyArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    ///
    /// # Panics
    /// The pointer must be a valid Arrow schema from Python's pyarrow. Panics if conversion fails.

    #[uniffi::constructor]
    pub fn new(schema_ptr: u64) -> Self {
        // SAFETY:
        // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
        let schema = unsafe {
            let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema")
        };
        Self {
            digester: Arc::new(Mutex::new(ArrowDigester::new(&schema))),
        }
    }

    /// Update the digester with a new `RecordBatch`
    ///
    /// # Panics
    /// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow
    pub fn update(&self, array_ptr: u64, schema_ptr: u64) {
        #[expect(
            unsafe_code,
            reason = "Need to convert raw pointers to Arrow data structures"
        )]
        #[expect(
            clippy::multiple_unsafe_ops_per_block,
            clippy::expect_used,
            reason = "Okay since we are doing the same operation of dereferencing pointers, Will add proper errors later"
        )]
        // SAFETY:
        // Need to conduct unsafe operations to convert raw pointers to Arrow data structures
        let array_data = unsafe {
            // Construct ArrayData from FFI structures
            let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut FFI_ArrowArray);
            let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut FFI_ArrowSchema);
            from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data")
        };

        self.digester
            .lock()
            .expect("Failed to acquire lock on digester")
            .update(&RecordBatch::from(StructArray::from(array_data)));
    }

    /// Consume the digester and finalize the hash computation
    ///
    /// # Panics
    /// Panics if it fails to acquire the lock on the digester.
    pub fn finalize(&self) -> Vec<u8> {
        self.digester
            .lock()
            .expect("Failed to acquire lock on digester")
            .clone()
            .finalize()
    }
}
