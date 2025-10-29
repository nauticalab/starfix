use arrow::array::{RecordBatch, StructArray};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow_digest::{RecordDigest, RecordDigestV0};
use sha2::Sha256;

/// Process an Arrow table via C Data Interface
///
/// # Safety
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow
#[uniffi::export]
pub fn process_arrow_table(array_ptr: u64, schema_ptr: u64) -> Vec<u8> {
    let array_data = unsafe {
        // Construct ArrayData from FFI structures
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr as *mut _);
        let ffi_schema = FFI_ArrowSchema::from_raw(schema_ptr as *mut _);
        from_ffi(ffi_array, &ffi_schema).expect("Failed to import Arrow array data")
    };

    // // Convert FFI schema to Arrow Schema
    // let schema =
    //     Schema::try_from(&ffi_schema).expect("Failed to convert FFI schema to Arrow schema");

    // Import array data from FFI

    // Create StructArray from the array data
    let struct_array = StructArray::from(array_data);

    // Create RecordBatch from StructArray
    let record_batch = RecordBatch::from(struct_array);

    // Hash the table
    let hash = RecordDigestV0::<Sha256>::digest(&record_batch);
    // format!("{:x}", hash)
    hash.to_vec()
}
