#![expect(
    clippy::expect_used,
    clippy::todo,
    reason = "First iteration of code, will add proper error handling later. Allow for unsupported data types for now"
)]
use std::{collections::BTreeMap, iter::repeat_n};

use arrow::{
    array::{
        Array, BinaryArray, BooleanArray, GenericBinaryArray, GenericListArray, GenericStringArray,
        LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, OffsetSizeTrait,
        RecordBatch, StringArray, StructArray,
    },
    datatypes::{DataType, Schema},
};
use arrow_schema::Field;
use bitvec::prelude::*;
use digest::Digest;

const NULL_BYTES: &[u8] = b"NULL";

const DELIMITER_FOR_NESTED_FIELD: &str = "/";

enum DigestBufferType<D: Digest> {
    NonNullable(D),
    Nullable(BitVec, D), // Where first digest is for the bull bits, while the second is for the actual data
}

pub struct ArrowDigester<D: Digest> {
    schema: Schema,
    schema_digest: Vec<u8>,
    fields_digest_buffer: BTreeMap<String, DigestBufferType<D>>,
}

impl<D: Digest> ArrowDigester<D> {
    pub fn new(schema: Schema) -> Self {
        // Hash the schema first
        let schema_digest = Self::hash_schema(&schema);

        // Flatten all nested fields into a single map, this allows us to hash each field individually and efficiently
        let mut fields_digest_buffer = BTreeMap::new();
        schema.fields.into_iter().for_each(|field| {
            Self::extract_fields_name(field, "", &mut fields_digest_buffer);
        });

        // Store it in the new struct for now
        Self {
            schema,
            schema_digest,
            fields_digest_buffer,
        }
    }

    /// Hash an array directly without needing to create an `ArrowDigester` instance on the user side
    /// For hash array, we don't have a schema to hash, however we do have field data type.
    /// So similar to schema, we will hash based on datatype to encode the metadata information into the digest
    ///
    /// # Panics
    ///
    /// This function will panic if JSON serialization of the data type fails.
    ///
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        let mut final_digest = D::new();

        let data_type_serialized = serde_json::to_string(&array.data_type())
            .expect("Failed to serialize data type to string");

        // Update the digest buffer with the array metadata and field data
        final_digest.update(data_type_serialized);

        // Now we update it with the actual array data
        let mut digest_buffer = if array.is_nullable() {
            DigestBufferType::Nullable(BitVec::new(), D::new())
        } else {
            DigestBufferType::NonNullable(D::new())
        };
        Self::array_digest_update(array.data_type(), array, &mut digest_buffer);
        Self::finalize_digest(&mut final_digest, digest_buffer);

        // Finalize and return the digest
        final_digest.finalize().to_vec()
    }

    /// Hash record batch directly without needing to create an `ArrowDigester` instance on the user side
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        let mut digester = Self::new(record_batch.schema().as_ref().clone());
        digester.update(record_batch);
        digester.finalize()
    }

    /// This will consume the `ArrowDigester` and produce the final combined digest where the schema
    /// digest is fed in first, followed by each field digest in alphabetical order of field names
    pub fn finalize(self) -> Vec<u8> {
        // Finalize all the sub digest and combine them into a single digest
        let mut final_digest = D::new();

        // digest the schema first
        final_digest.update(&self.schema_digest);

        // Then digest each field digest in order
        self.fields_digest_buffer
            .into_iter()
            .for_each(|(_, digest)| Self::finalize_digest(&mut final_digest, digest));

        final_digest.finalize().to_vec()
    }

    #[expect(
        clippy::big_endian_bytes,
        reason = "Use for bit packing the null_bit_values"
    )]
    /// Finalize a single field digest into the final digest
    /// Helpers to reduce code duplication
    fn finalize_digest(final_digest: &mut D, digest: DigestBufferType<D>) {
        match digest {
            DigestBufferType::NonNullable(data_digest) => {
                final_digest.update(data_digest.finalize());
            }
            DigestBufferType::Nullable(null_bit_digest, data_digest) => {
                final_digest.update(null_bit_digest.len().to_le_bytes());
                for &word in null_bit_digest.as_raw_slice() {
                    final_digest.update(word.to_be_bytes());
                }
                final_digest.update(data_digest.finalize());
            }
        }
    }

    /// Serialize the schema into a `BTreeMap` for field name and its digest
    ///
    /// # Panics
    /// This function will panic if JSON serialization of the schema fails.
    fn serialized_schema(schema: &Schema) -> String {
        let fields_digest = schema
            .fields
            .iter()
            .map(|field| (field.name(), (field.to_string(), field.data_type())))
            .collect::<BTreeMap<_, _>>();

        serde_json::to_string(&fields_digest).expect("Failed to serialize field_digest to bytes")
    }

    /// Serialize the schema into a `BTreeMap` for field name and its digest
    pub fn hash_schema(schema: &Schema) -> Vec<u8> {
        // Hash the entire thing to the digest
        D::digest(Self::serialized_schema(schema)).to_vec()
    }

    /// Hash a record batch and update the internal digests
    fn update(&mut self, record_batch: &RecordBatch) {
        // Verify schema matches
        assert!(
            *record_batch.schema() == self.schema,
            "Record batch schema does not match ArrowDigester schema"
        );

        // Iterate through each field and update its digest
        self.fields_digest_buffer
            .iter_mut()
            .for_each(|(field_name, digest)| {
                // Determine if field name is nested
                let field_name_hierarchy = field_name
                    .split(DELIMITER_FOR_NESTED_FIELD)
                    .collect::<Vec<_>>();

                if field_name_hierarchy.len() == 1 {
                    Self::array_digest_update(
                        record_batch
                            .schema()
                            .field_with_name(field_name)
                            .expect("Failed to get field with name")
                            .data_type(),
                        record_batch
                            .column_by_name(field_name)
                            .expect("Failed to get column by name"),
                        digest,
                    );
                } else {
                    Self::update_nested_field(
                        &field_name_hierarchy,
                        0,
                        record_batch
                            .column_by_name(
                                field_name_hierarchy
                                    .first()
                                    .expect("Failed to get field name at idx 0, list is empty!"),
                            )
                            .expect("Failed to get column by name")
                            .as_any()
                            .downcast_ref::<StructArray>()
                            .expect("Failed to downcast to StructArray"),
                        digest,
                    );
                }
            });
    }

    /// Recursive function to update nested field digests (structs within structs)
    fn update_nested_field(
        field_name_hierarchy: &[&str],
        current_level: usize,
        array: &StructArray,
        digest: &mut DigestBufferType<D>,
    ) {
        let current_level_plus_one = current_level
            .checked_add(1)
            .expect("Field nesting level overflow");

        if field_name_hierarchy
            .len()
            .checked_sub(1)
            .expect("field_name_hierarchy underflow")
            == current_level_plus_one
        {
            let array_data = array
                .column_by_name(
                    field_name_hierarchy
                        .last()
                        .expect("Failed to get field name at idx 0, list is empty!"),
                )
                .expect("Failed to get column by name");
            // Base case, it should be a non-struct field
            Self::array_digest_update(array_data.data_type(), array_data.as_ref(), digest);
        } else {
            // Recursive case, it should be a struct field
            let next_array = array
                .column_by_name(
                    field_name_hierarchy
                        .get(current_level_plus_one)
                        .expect("Failed to get field name at current level"),
                )
                .expect("Failed to get column by name")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("Failed to downcast to StructArray");

            Self::update_nested_field(
                field_name_hierarchy,
                current_level_plus_one,
                next_array,
                digest,
            );
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Comprehensive match on all data types"
    )]
    fn array_digest_update(
        data_type: &DataType,
        array: &dyn Array,
        digest: &mut DigestBufferType<D>,
    ) {
        match data_type {
            DataType::Null => todo!(),
            DataType::Boolean => {
                // Bool Array is stored a bit differently, so we can't use the standard fixed buffer approach
                let bool_array = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("Failed to downcast to BooleanArray");

                match digest {
                    DigestBufferType::NonNullable(data_digest) => {
                        // We want to bit pack the boolean values into bytes for hashing
                        let mut bit_vec = BitVec::<u8, Msb0>::with_capacity(bool_array.len());
                        for i in 0..bool_array.len() {
                            bit_vec.push(bool_array.value(i));
                        }

                        data_digest.update(bit_vec.as_raw_slice());
                    }
                    DigestBufferType::Nullable(null_bit_vec, data_digest) => {
                        // Handle null bits first
                        Self::handle_null_bits(bool_array, null_bit_vec);

                        // Handle the data
                        let mut bit_vec = BitVec::<u8, Msb0>::with_capacity(bool_array.len());
                        for i in 0..bool_array.len() {
                            // We only want the valid bits, for null we will discard from the hash since that is already capture by null_bits
                            if bool_array.is_valid(i) {
                                bit_vec.push(bool_array.value(i));
                            }
                        }
                        data_digest.update(bit_vec.as_raw_slice());
                    }
                }
            }
            DataType::Int8 | DataType::UInt8 => Self::hash_fixed_size_array(array, digest, 1),
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => {
                Self::hash_fixed_size_array(array, digest, 2);
            }
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Decimal32(_, _) => {
                Self::hash_fixed_size_array(array, digest, 4);
            }
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Decimal64(_, _) => {
                Self::hash_fixed_size_array(array, digest, 8);
            }
            DataType::Timestamp(_, _) => todo!(),
            DataType::Time32(_) => Self::hash_fixed_size_array(array, digest, 4),
            DataType::Time64(_) => Self::hash_fixed_size_array(array, digest, 8),
            DataType::Duration(_) => todo!(),
            DataType::Interval(_) => todo!(),
            DataType::Binary => Self::hash_binary_array(
                array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .expect("Failed to downcast to BinaryArray"),
                digest,
            ),
            DataType::FixedSizeBinary(element_size) => {
                Self::hash_fixed_size_array(array, digest, *element_size);
            }
            DataType::LargeBinary => Self::hash_binary_array(
                array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .expect("Failed to downcast to LargeBinaryArray"),
                digest,
            ),
            DataType::BinaryView => todo!(),
            DataType::Utf8 => Self::hash_string_array(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Failed to downcast to StringArray"),
                digest,
            ),
            DataType::LargeUtf8 => Self::hash_string_array(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Failed to downcast to LargeStringArray"),
                digest,
            ),
            DataType::Utf8View => todo!(),
            DataType::List(field) => {
                Self::hash_list_array(
                    array
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .expect("Failed to downcast to ListArray"),
                    field.data_type(),
                    digest,
                );
            }
            DataType::ListView(_) => todo!(),
            DataType::FixedSizeList(_, _) => todo!(),
            DataType::LargeList(field) => {
                Self::hash_list_array(
                    array
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .expect("Failed to downcast to LargeListArray"),
                    field.data_type(),
                    digest,
                );
            }
            DataType::LargeListView(_) => todo!(),
            DataType::Struct(_) => todo!(),
            DataType::Union(_, _) => todo!(),
            DataType::Dictionary(_, _) => todo!(),
            DataType::Decimal128(_, _) => {
                Self::hash_fixed_size_array(array, digest, 16);
            }
            DataType::Decimal256(_, _) => {
                Self::hash_fixed_size_array(array, digest, 32);
            }
            DataType::Map(_, _) => todo!(),
            DataType::RunEndEncoded(_, _) => todo!(),
        }
    }

    #[expect(clippy::cast_sign_loss, reason = "element_size is always positive")]
    fn hash_fixed_size_array(
        array: &dyn Array,
        digest_buffer: &mut DigestBufferType<D>,
        element_size: i32,
    ) {
        let array_data = array.to_data();
        let element_size_usize = element_size as usize;

        // Get the slice with offset accounted for if there is any
        let slice = array_data
            .buffers()
            .first()
            .expect("Unable to get first buffer to determine offset")
            .as_slice()
            .get(
                array_data
                    .offset()
                    .checked_mul(element_size_usize)
                    .expect("Offset multiplication overflow")..,
            )
            .expect("Failed to get buffer slice for FixedSizeBinaryArray");

        match digest_buffer {
            DigestBufferType::NonNullable(data_digest) => {
                // No nulls, we can hash the entire buffer directly
                data_digest.update(slice);
            }
            DigestBufferType::Nullable(null_bits, data_digest) => {
                // Handle null bits first
                Self::handle_null_bits(array, null_bits);

                match array_data.nulls() {
                    Some(null_buffer) => {
                        // There are nulls, so we need to incrementally hash each value
                        for i in 0..array_data.len() {
                            if null_buffer.is_valid(i) {
                                let data_pos = i
                                    .checked_mul(element_size_usize)
                                    .expect("Data position multiplication overflow");
                                let end_pos = data_pos
                                    .checked_add(element_size_usize)
                                    .expect("End position addition overflow");

                                data_digest.update(
                                    slice
                                        .get(data_pos..end_pos)
                                        .expect("Failed to get data_slice"),
                                );
                            }
                        }
                    }
                    None => {
                        // No nulls, we can hash the entire buffer directly
                        data_digest.update(slice);
                    }
                }
            }
        }
    }

    fn hash_binary_array(
        array: &GenericBinaryArray<impl OffsetSizeTrait>,
        digest: &mut DigestBufferType<D>,
    ) {
        match digest {
            DigestBufferType::NonNullable(data_digest) => {
                for i in 0..array.len() {
                    let value = array.value(i);
                    data_digest.update(value.len().to_le_bytes());
                    data_digest.update(value);
                }
            }
            DigestBufferType::Nullable(null_bit_vec, data_digest) => {
                // Deal with the null bits first
                if let Some(null_buf) = array.nulls() {
                    // We would need to iterate through the null buffer and push it into the null_bit_vec
                    for i in 0..array.len() {
                        null_bit_vec.push(null_buf.is_valid(i));
                    }

                    for i in 0..array.len() {
                        if null_buf.is_valid(i) {
                            let value = array.value(i);
                            data_digest.update(value.len().to_le_bytes());
                            data_digest.update(value);
                        } else {
                            data_digest.update(NULL_BYTES);
                        }
                    }
                } else {
                    // All valid, therefore we can extend the bit vector with all true values
                    let len = array.len().checked_sub(1).expect("Array length underflow");
                    null_bit_vec.extend(repeat_n(true, len));

                    // Deal with the data
                    for i in 0..array.len() {
                        let value = array.value(i);
                        data_digest.update(value.len().to_le_bytes());
                        data_digest.update(value);
                    }
                }
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "String lengths from Arrow offsets are bounded"
    )]
    fn hash_string_array(
        array: &GenericStringArray<impl OffsetSizeTrait>,
        digest: &mut DigestBufferType<D>,
    ) {
        match digest {
            DigestBufferType::NonNullable(data_digest) => {
                for i in 0..array.len() {
                    let value = array.value(i);
                    data_digest.update((value.len() as u64).to_le_bytes());
                    data_digest.update(value.as_bytes());
                }
            }
            DigestBufferType::Nullable(null_bit_vec, data_digest) => {
                // Deal with the null bits first
                Self::handle_null_bits(array, null_bit_vec);

                match array.nulls() {
                    Some(null_buf) => {
                        for i in 0..array.len() {
                            if null_buf.is_valid(i) {
                                let value = array.value(i);
                                data_digest.update((value.len() as u32).to_le_bytes());
                                data_digest.update(value.as_bytes());
                            } else {
                                data_digest.update(NULL_BYTES);
                            }
                        }
                    }
                    None => {
                        for i in 0..array.len() {
                            let value = array.value(i);
                            data_digest.update((value.len() as u32).to_le_bytes());
                            data_digest.update(value.as_bytes());
                        }
                    }
                }
            }
        }
    }

    fn hash_list_array(
        array: &GenericListArray<impl OffsetSizeTrait>,
        field_data_type: &DataType,
        digest: &mut DigestBufferType<D>,
    ) {
        match digest {
            DigestBufferType::NonNullable(_) => {
                for i in 0..array.len() {
                    Self::array_digest_update(field_data_type, array.value(i).as_ref(), digest);
                }
            }
            DigestBufferType::Nullable(bit_vec, _) => {
                // Deal with null bits first
                Self::handle_null_bits(array, bit_vec);

                match array.nulls() {
                    Some(null_buf) => {
                        for i in 0..array.len() {
                            if null_buf.is_valid(i) {
                                Self::array_digest_update(
                                    field_data_type,
                                    array.value(i).as_ref(),
                                    digest,
                                );
                            }
                        }
                    }
                    None => {
                        for i in 0..array.len() {
                            Self::array_digest_update(
                                field_data_type,
                                array.value(i).as_ref(),
                                digest,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Internal recursive function to extract field names from nested structs effectively flattening the schema
    /// The format is `parent__child__grandchild__etc`... for nested fields and will be stored in `fields_digest_buffer`
    fn extract_fields_name(
        field: &Field,
        parent_field_name: &str,
        fields_digest_buffer: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        // Check if field is a nested type of struct
        if let DataType::Struct(fields) = field.data_type() {
            // We will add fields in alphabetical order
            fields.into_iter().for_each(|field_inner| {
                Self::extract_fields_name(
                    field_inner,
                    Self::construct_field_name_hierarchy(parent_field_name, field.name()).as_str(),
                    fields_digest_buffer,
                );
            });
        } else {
            // Base case, just add the the combine field name to the map
            fields_digest_buffer.insert(
                Self::construct_field_name_hierarchy(parent_field_name, field.name()),
                if field.is_nullable() {
                    DigestBufferType::Nullable(BitVec::new(), D::new())
                } else {
                    DigestBufferType::NonNullable(D::new())
                },
            );
        }
    }

    fn construct_field_name_hierarchy(parent_field_name: &str, field_name: &str) -> String {
        if parent_field_name.is_empty() {
            field_name.to_owned()
        } else {
            format!("{parent_field_name}{DELIMITER_FOR_NESTED_FIELD}{field_name}")
        }
    }

    fn handle_null_bits(array: &dyn Array, null_bit_vec: &mut BitVec) {
        match array.nulls() {
            Some(null_buf) => {
                // We would need to iterate through the null buffer and push it into the null_bit_vec
                for i in 0..array.len() {
                    null_bit_vec.push(null_buf.is_valid(i));
                }
            }
            None => {
                // All valid, therefore we can extend the bit vector with all true values
                null_bit_vec.extend(repeat_n(
                    true,
                    array.len().checked_sub(1).expect("Array length underflow"),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    use std::sync::Arc;

    use arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal32Array,
            Decimal64Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
            Int8Array, LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, RecordBatch,
            StringArray, StructArray, Time32MillisecondArray, Time32SecondArray,
            Time64MicrosecondArray, Time64NanosecondArray, UInt16Array, UInt32Array, UInt64Array,
            UInt8Array,
        },
        datatypes::Int32Type,
    };
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use hex::encode;
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use sha2::Sha256;

    use crate::arrow_digester::ArrowDigester;
    use arrow::array::Decimal128Array;

    #[expect(clippy::too_many_lines, reason = "Comprehensive schema test")]
    #[test]
    fn schema() {
        let schema = Schema::new(vec![
            Field::new("bool", DataType::Boolean, true),
            Field::new("int8", DataType::Int8, false),
            Field::new("uint8", DataType::UInt8, false),
            Field::new("int16", DataType::Int16, false),
            Field::new("uint16", DataType::UInt16, false),
            Field::new("int32", DataType::Int32, false),
            Field::new("uint32", DataType::UInt32, false),
            Field::new("int64", DataType::Int64, false),
            Field::new("uint64", DataType::UInt64, false),
            Field::new("float32", DataType::Float32, false),
            Field::new("float64", DataType::Float64, false),
            Field::new("date32", DataType::Date32, false),
            Field::new("date64", DataType::Date64, false),
            Field::new("time32_second", DataType::Time32(TimeUnit::Second), false),
            Field::new(
                "time32_millis",
                DataType::Time32(TimeUnit::Millisecond),
                false,
            ),
            Field::new(
                "time64_micro",
                DataType::Time64(TimeUnit::Microsecond),
                false,
            ),
            Field::new("time64_nano", DataType::Time64(TimeUnit::Nanosecond), false),
            Field::new("binary", DataType::Binary, true),
            Field::new("large_binary", DataType::LargeBinary, true),
            Field::new("utf8", DataType::Utf8, true),
            Field::new("large_utf8", DataType::LargeUtf8, true),
            Field::new(
                "list",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                true,
            ),
            Field::new(
                "large_list",
                DataType::LargeList(Box::new(Field::new("item", DataType::Int32, true)).into()),
                true,
            ),
            Field::new("decimal32", DataType::Decimal32(9, 2), true),
            Field::new("decimal64", DataType::Decimal64(18, 3), true),
            Field::new("decimal128", DataType::Decimal128(38, 5), true),
        ]);

        // Serialize the schema and covert it over to pretty json for comparison
        let compact_json: serde_json::Value =
            serde_json::from_str(&ArrowDigester::<Sha256>::serialized_schema(&schema)).unwrap();
        let pretty_json = serde_json::to_string_pretty(&compact_json).unwrap();

        assert_eq!(
            pretty_json,
            indoc! {r#"
{
  "binary": [
    "Field { \"binary\": nullable Binary }",
    "Binary"
  ],
  "bool": [
    "Field { \"bool\": nullable Boolean }",
    "Boolean"
  ],
  "date32": [
    "Field { \"date32\": Date32 }",
    "Date32"
  ],
  "date64": [
    "Field { \"date64\": Date64 }",
    "Date64"
  ],
  "decimal128": [
    "Field { \"decimal128\": nullable Decimal128(38, 5) }",
    {
      "Decimal128": [
        38,
        5
      ]
    }
  ],
  "decimal32": [
    "Field { \"decimal32\": nullable Decimal32(9, 2) }",
    {
      "Decimal32": [
        9,
        2
      ]
    }
  ],
  "decimal64": [
    "Field { \"decimal64\": nullable Decimal64(18, 3) }",
    {
      "Decimal64": [
        18,
        3
      ]
    }
  ],
  "float32": [
    "Field { \"float32\": Float32 }",
    "Float32"
  ],
  "float64": [
    "Field { \"float64\": Float64 }",
    "Float64"
  ],
  "int16": [
    "Field { \"int16\": Int16 }",
    "Int16"
  ],
  "int32": [
    "Field { \"int32\": Int32 }",
    "Int32"
  ],
  "int64": [
    "Field { \"int64\": Int64 }",
    "Int64"
  ],
  "int8": [
    "Field { \"int8\": Int8 }",
    "Int8"
  ],
  "large_binary": [
    "Field { \"large_binary\": nullable LargeBinary }",
    "LargeBinary"
  ],
  "large_list": [
    "Field { \"large_list\": nullable LargeList(nullable Int32) }",
    {
      "LargeList": {
        "data_type": "Int32",
        "dict_id": 0,
        "dict_is_ordered": false,
        "metadata": {},
        "name": "item",
        "nullable": true
      }
    }
  ],
  "large_utf8": [
    "Field { \"large_utf8\": nullable LargeUtf8 }",
    "LargeUtf8"
  ],
  "list": [
    "Field { \"list\": nullable List(nullable Int32) }",
    {
      "List": {
        "data_type": "Int32",
        "dict_id": 0,
        "dict_is_ordered": false,
        "metadata": {},
        "name": "item",
        "nullable": true
      }
    }
  ],
  "time32_millis": [
    "Field { \"time32_millis\": Time32(ms) }",
    {
      "Time32": "Millisecond"
    }
  ],
  "time32_second": [
    "Field { \"time32_second\": Time32(s) }",
    {
      "Time32": "Second"
    }
  ],
  "time64_micro": [
    "Field { \"time64_micro\": Time64(µs) }",
    {
      "Time64": "Microsecond"
    }
  ],
  "time64_nano": [
    "Field { \"time64_nano\": Time64(ns) }",
    {
      "Time64": "Nanosecond"
    }
  ],
  "uint16": [
    "Field { \"uint16\": UInt16 }",
    "UInt16"
  ],
  "uint32": [
    "Field { \"uint32\": UInt32 }",
    "UInt32"
  ],
  "uint64": [
    "Field { \"uint64\": UInt64 }",
    "UInt64"
  ],
  "uint8": [
    "Field { \"uint8\": UInt8 }",
    "UInt8"
  ],
  "utf8": [
    "Field { \"utf8\": nullable Utf8 }",
    "Utf8"
  ]
}"#}
        );

        // Empty Table Hashing Check

        assert_eq!(
            encode(ArrowDigester::<Sha256>::new(schema.clone()).finalize()),
            "a42e35f6623d86b72350bf0bb74b97781946df45423f192c397d435c254bc71e"
        );

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(BooleanArray::from(vec![Some(true)])),
                Arc::new(Int8Array::from(vec![1_i8])),
                Arc::new(UInt8Array::from(vec![1_u8])),
                Arc::new(Int16Array::from(vec![100_i16])),
                Arc::new(UInt16Array::from(vec![100_u16])),
                Arc::new(Int32Array::from(vec![1000_i32])),
                Arc::new(UInt32Array::from(vec![1000_u32])),
                Arc::new(Int64Array::from(vec![100_000_i64])),
                Arc::new(UInt64Array::from(vec![100_000_u64])),
                Arc::new(Float32Array::from(vec![1.5_f32])),
                Arc::new(Float64Array::from(vec![1.5_f64])),
                Arc::new(Date32Array::from(vec![18993_i32])),
                Arc::new(Date64Array::from(vec![1_640_995_200_000_i64])),
                Arc::new(Time32SecondArray::from(vec![3600_i32])),
                Arc::new(Time32MillisecondArray::from(vec![3_600_000_i32])),
                Arc::new(Time64MicrosecondArray::from(vec![3_600_000_000_i64])),
                Arc::new(Time64NanosecondArray::from(vec![3_600_000_000_000_i64])),
                Arc::new(BinaryArray::from(vec![Some(b"data1".as_ref())])),
                Arc::new(LargeBinaryArray::from(vec![Some(b"large1".as_ref())])),
                Arc::new(StringArray::from(vec![Some("text1")])),
                Arc::new(LargeStringArray::from(vec![Some("large_text1")])),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(1), Some(2)]),
                ])),
                Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
                    vec![Some(vec![Some(5), Some(6)])],
                )),
                Arc::new(
                    Decimal32Array::from_iter(vec![Some(12345)])
                        .with_precision_and_scale(9, 2)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal64Array::from_iter(vec![Some(123_456_789_012)])
                        .with_precision_and_scale(18, 3)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal128Array::from_iter(vec![Some(
                        123_456_789_012_345_678_901_234_567_890_i128,
                    )])
                    .with_precision_and_scale(38, 5)
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        // Hash the record batch
        assert_eq!(
            encode(ArrowDigester::<Sha256>::hash_record_batch(&batch)),
            "da0d7d3d76a47e88648e3a1160a5d2432647f0769e08b42315533163c36b3eb0"
        );
    }

    #[test]
    fn boolean_array_hashing() {
        let bool_array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&bool_array));
        assert_eq!(
            hash,
            "f9abeb37d9395f359b48a379f0a8467c572b19ecc6cae9fa85e1bf627a52a8f3"
        );
    }

    /// Test int32 array hashing which is really meant to test fixed size element array hashing
    #[test]
    fn int32_array_hashing() {
        let int_array = Int32Array::from(vec![Some(42), None, Some(-7), Some(0)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&int_array));
        assert_eq!(
            hash,
            "27f2411e6839eb1e3fe706ac3f01e704c7b46357360fb2ddb8a08ec98e8ba4fa"
        );
    }

    /// Test time array hashing
    #[test]
    fn time32_array_hashing() {
        let time_array = Time32SecondArray::from(vec![Some(1000), None, Some(5000), Some(0)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&time_array));
        assert_eq!(
            hash,
            "9000b74aa80f685103a8cafc7e113aa8f33ccc0c94ea3713318d2cc2f3436baa"
        );
    }

    #[test]
    fn time64_array_hashing() {
        let time_array =
            Time64MicrosecondArray::from(vec![Some(1_000_000), None, Some(5_000_000), Some(0)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&time_array));
        assert_eq!(
            hash,
            "95f12143d789f364a3ed52f7300f8f91dc21fbe00c34aed798ca8fd54182dea3"
        );
    }

    #[test]
    fn time_array_different_units_produce_different_hashes() {
        let time32_second = Time32SecondArray::from(vec![Some(1000), Some(2000)]);
        let time32_millis = Time32MillisecondArray::from(vec![Some(1000), Some(2000)]);

        let hash_second = hex::encode(ArrowDigester::<Sha256>::hash_array(&time32_second));
        let hash_millis = hex::encode(ArrowDigester::<Sha256>::hash_array(&time32_millis));

        assert_ne!(hash_second, hash_millis);
    }

    /// Test binary array hashing
    #[test]
    fn binary_array_hashing() {
        let binary_array = BinaryArray::from(vec![
            Some(b"hello".as_ref()),
            None,
            Some(b"world".as_ref()),
            Some(b"".as_ref()),
        ]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&binary_array));
        assert_eq!(
            hash,
            "466801efd880d2acecd6c78915b5c2a51476870f9116912834d79de43a000071"
        );

        // Test large binary array with same data to ensure consistency
        let large_binary_array = LargeBinaryArray::from(vec![
            Some(b"hello".as_ref()),
            None,
            Some(b"world".as_ref()),
            Some(b"".as_ref()),
        ]);

        assert_ne!(
            hex::encode(ArrowDigester::<Sha256>::hash_array(&large_binary_array)),
            hash
        );
    }

    // Test binary array collision vulnerability - different partitions should produce different hashes
    #[test]
    fn binary_array_length_prefix_prevents_collisions() {
        // Array 1: [[0x01, 0x02], [0x03]]
        let array1 = BinaryArray::from(vec![Some(&[0x01_u8, 0x02_u8][..]), Some(&[0x03_u8][..])]);

        // Array 2: [[0x01], [0x02, 0x03]]
        let array2 = BinaryArray::from(vec![Some(&[0x01_u8][..]), Some(&[0x02_u8, 0x03_u8][..])]);

        let hash1 = hex::encode(ArrowDigester::<Sha256>::hash_array(&array1));
        let hash2 = hex::encode(ArrowDigester::<Sha256>::hash_array(&array2));

        // Without length prefix, these would collide (both hash to 0x01 0x02 0x03)
        // With length prefix, they should produce different hashes
        assert_ne!(
            hash1, hash2,
            "Binary arrays with different partitions should produce different hashes"
        );
    }

    // Test string array collision vulnerability - different partitions should produce different hashes
    #[test]
    fn string_array_length_prefix_prevents_collisions() {
        // Array 1: ["ab", "c"]
        let array1 = StringArray::from(vec![Some("ab"), Some("c")]);

        // Array 2: ["a", "bc"]
        let array2 = StringArray::from(vec![Some("a"), Some("bc")]);

        let hash1 = hex::encode(ArrowDigester::<Sha256>::hash_array(&array1));
        let hash2 = hex::encode(ArrowDigester::<Sha256>::hash_array(&array2));

        // Without length prefix, these would collide (both hash to "abc")
        // With length prefix, they should produce different hashes
        assert_ne!(
            hash1, hash2,
            "String arrays with different partitions should produce different hashes"
        );
    }

    // Test String hashing
    #[test]
    fn string_array_hashing() {
        let string_array = StringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&string_array));
        assert_eq!(
            hash,
            "14a2d2eaf535b6e78fbf1d58ae93accce424eafd20fa449eff8acefc47903d3d"
        );

        // Test large string array with same data to ensure consistency
        let large_string_array =
            LargeStringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);

        assert_ne!(
            hex::encode(ArrowDigester::<Sha256>::hash_array(&large_string_array)),
            hash
        );
    }

    // List array hashing test
    #[test]
    fn list_array_hashing() {
        let list_array = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            None,
            Some(vec![Some(4), Some(5)]),
            Some(vec![Some(6)]),
        ]);

        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&list_array));
        assert_eq!(
            hash,
            "1a8d06635dec40079b979ce439f662c1fb6456bb7e02bbf7d8e8048c61498faf"
        );
    }

    // Test all types of decimal hashing
    #[test]
    fn decimal_array_hashing() {
        // Test Decimal32 (precision 1-9)
        let decimal32_array =
            Decimal128Array::from_iter(vec![Some(123), None, Some(-456), Some(0)])
                .with_precision_and_scale(9, 2)
                .unwrap();

        assert_eq!(
            encode(ArrowDigester::<Sha256>::hash_array(&decimal32_array)),
            "ef29250615f9d6ab34672c3b11dfa2dcda6e8e6164bc55899c13887f17705f5d"
        );

        // Test Decimal64 (precision 10-18)
        let decimal64_array = Decimal128Array::from_iter(vec![
            Some(1_234_567_890_123),
            None,
            Some(-9_876_543_210),
            Some(0),
        ])
        .with_precision_and_scale(15, 3)
        .unwrap();
        assert_eq!(
            encode(ArrowDigester::<Sha256>::hash_array(&decimal64_array)),
            "efa4ed72641051233889c07775366cbf2e56eb4b0fcfd46653f5741e81786f08"
        );

        // Test Decimal128 (precision 19-38)
        let decimal128_array = Decimal128Array::from_iter(vec![
            Some(123_456_789_012_345_678_901_234_567),
            None,
            Some(-987_654_321_098_765_432_109_876_543),
            Some(0),
        ])
        .with_precision_and_scale(38, 5)
        .unwrap();
        assert_eq!(
            hex::encode(ArrowDigester::<Sha256>::hash_array(&decimal128_array)),
            "55cc4d81a048dbca001ca8581673a5a6c93efd870d358df211a545c2af9b658d"
        );
    }

    #[test]
    fn commutative_tables() {
        let uids = Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4)])) as ArrayRef;
        let fake_data = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
        ])) as ArrayRef;

        // Create two record batches with same data but different order
        let batch1 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("uids", DataType::Int32, false),
                Field::new("flags", DataType::Boolean, true),
            ])),
            vec![Arc::clone(&uids), Arc::clone(&fake_data)],
        );

        let batch2 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("flags", DataType::Boolean, true),
                Field::new("uids", DataType::Int32, false),
            ])),
            vec![fake_data, uids],
        );

        // Hash both record batches
        assert_eq!(
            encode(ArrowDigester::<Sha256>::hash_record_batch(
                batch1.as_ref().unwrap()
            )),
            encode(ArrowDigester::<Sha256>::hash_record_batch(
                batch2.as_ref().unwrap()
            ))
        );
    }

    #[test]
    fn record_batch_hashing() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("uids", DataType::Int32, false),
            Field::new("flags", DataType::Boolean, true),
        ]));

        // Create two record batches with different data to simulate loading at different times
        let uids = Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4)])) as ArrayRef;
        let fake_data = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
        ]));

        let batch1 = RecordBatch::try_new(Arc::clone(&schema), vec![uids, fake_data]).unwrap();

        let uids2 =
            Arc::new(Int32Array::from(vec![Some(5), Some(6), Some(7), Some(8)])) as ArrayRef;
        let fake_data2 = Arc::new(BooleanArray::from(vec![
            Some(false),
            Some(true),
            Some(true),
            None,
        ]));

        let batch2 = RecordBatch::try_new(Arc::clone(&schema), vec![uids2, fake_data2]).unwrap();
        // Hash both record batches
        let mut digester = ArrowDigester::<Sha256>::new((*schema).clone());
        digester.update(&batch1);
        digester.update(&batch2);
        assert_eq!(
            encode(digester.finalize()),
            "37954b3edd169c7a9e65604c191caf6a307940357305d182a5d2168047e9cc51"
        );
    }

    #[test]
    fn nested_fields() {
        // Test nested struct field name extraction
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "nested",
                DataType::Struct(
                    vec![
                        Field::new("name", DataType::Utf8, true),
                        Field::new(
                            "deep",
                            DataType::Struct(
                                vec![Field::new("value", DataType::Int64, false)].into(),
                            ),
                            false,
                        ),
                    ]
                    .into(),
                ),
                false,
            ),
        ]);

        let mut digester = ArrowDigester::<Sha256>::new(schema.clone());
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        assert_eq!(field_names.len(), 3);
        assert!(field_names.contains(&&"id".to_owned()));
        assert!(field_names.contains(&&"nested/name".to_owned()));
        assert!(field_names.contains(&&"nested/deep/value".to_owned()));

        // Test the nested field update by creating record_batch and using the update method
        let id_array = Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef;
        let name_array = Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])) as ArrayRef;
        let value_array = Arc::new(Int64Array::from(vec![Some(100), Some(200)])) as ArrayRef;

        let schema_ref = Arc::new(schema);

        let nested_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, true)),
                name_array,
            ),
            (
                Arc::new(Field::new(
                    "deep",
                    DataType::Struct(vec![Field::new("value", DataType::Int64, false)].into()),
                    false,
                )),
                Arc::new(StructArray::from(vec![(
                    Arc::new(Field::new("value", DataType::Int64, false)),
                    value_array,
                )])) as ArrayRef,
            ),
        ]);

        let record_batch = RecordBatch::try_new(
            Arc::clone(&schema_ref),
            vec![id_array, Arc::new(nested_struct)],
        )
        .unwrap();

        digester.update(&record_batch);

        // Check the digest
        assert_eq!(
            encode(digester.finalize()),
            "b7faf50f1328ec80b575e018c121eed9d0e7e84ad72645499ebc8667e64199a7"
        );
    }
}
