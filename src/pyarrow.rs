use crate::ArrowDigester;
use arrow::array::{RecordBatch, StructArray};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};

/// Process an Arrow table via C Data Interface
///
/// # Panics
/// The pointers must be valid Arrow C Data Interface structs from Python's pyarrow

#[uniffi::export]
pub fn process_arrow_table(array_ptr: u64, schema_ptr: u64) -> Vec<u8> {
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
