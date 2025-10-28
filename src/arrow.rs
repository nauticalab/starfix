use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow::array::{RecordBatch, StructArray};
use arrow::datatypes::Schema;
use arrow_digest::{RecordDigest, RecordDigestV0};
use sha2::Sha256;
use std::sync::Arc;


/// Process an Arrow table via C Data Interface
/// 
/// # Safety
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow
#[uniffi::export]
pub fn process_arrow_table(array_ptr: u64, schema_ptr: u64) -> String {
    unsafe {
        // Convert raw pointers to Arrow FFI structs
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut _);
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut _);
        
        // Convert FFI schema to Arrow Schema
        let schema = Arc::new(Schema::try_from(&ffi_schema)
            .expect("Failed to convert FFI schema to Arrow schema"));
        
        // Import array data from FFI
        let array_data = from_ffi(ffi_array, &ffi_schema)
            .expect("Failed to import Arrow array data");
        
        // Create StructArray from the array data
        let struct_array = StructArray::from(array_data);
        
        // Create RecordBatch from StructArray
        let record_batch = RecordBatch::from(struct_array);
        
        // Hash the table
        let mut digest = RecordDigestV0::<Sha256>::new(&schema);
        digest.update(&record_batch);
        
        let hash = digest.finalize();
        format!("{:x}", hash)
    }
}