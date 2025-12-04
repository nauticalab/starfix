#![expect(
    clippy::expect_used,
    clippy::todo,
    clippy::panic,
    reason = "First iteration of code, will add proper error handling later. Allow for unsupported data types for now"
)]
use std::collections::BTreeMap;

use arrow::{
    array::{
        Array, BinaryArray, BooleanArray, GenericBinaryArray, GenericListArray, GenericStringArray,
        LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, OffsetSizeTrait,
        RecordBatch, StringArray, StructArray,
    },
    datatypes::{DataType, Schema},
};
use arrow_schema::{Field, TimeUnit};
use digest::Digest;
use postcard::to_vec;

const NULL_BYTES: &[u8] = b"NULL";

pub struct ArrowDigester<D: Digest> {
    schema: Schema,
    schema_digest: Vec<u8>,
    fields_digest_buffer: BTreeMap<String, D>,
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

    /// Hash a array directly without needing to create an `ArrowDigester` instance on the user side
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        let mut digest = D::new();
        Self::array_digest_update(array.data_type(), array, &mut digest);
        digest.finalize().to_vec()
    }

    /// Hash record batch directly without needing to create an `ArrowDigester` instance on the user side
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        let mut digester = Self::new(record_batch.schema().as_ref().clone());
        digester.update(&record_batch.clone());
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
            .for_each(|(_, digest)| {
                let field_hash = digest.finalize();
                final_digest.update(&field_hash);
            });

        final_digest.finalize().to_vec()
    }

    /// Serialize the schema into a `BTreeMap` for field name and its digest
    fn hash_schema(schema: &Schema) -> Vec<u8> {
        let fields_digest = schema
            .fields
            .into_iter()
            .map(|field| {
                (
                    field.name(),
                    to_vec::<_, 256>(field).expect("Failed to serialize field of schema"),
                )
            })
            .collect::<BTreeMap<_, _>>();

        // Hash the entire thing to the digest
        D::digest(
            to_vec::<_, 1024>(&fields_digest).expect("Failed to serialize field_digest to bytes"),
        )
        .to_vec()
    }

    /// Hash a record batch and update the internal digests
    fn update(&mut self, record_batch: &RecordBatch) {
        // Verify schema matches
        assert!(
            !(*record_batch.schema() != self.schema),
            "Record batch schema does not match ArrowDigester schema"
        );

        // Iterate through each field and update its digest
        self.fields_digest_buffer
            .iter_mut()
            .for_each(|(field_name, digest)| {
                // Determine if field name is nested
                let field_name_hierarchy = field_name.split('_').collect::<Vec<_>>();

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
        field_name_hierarchy: &Vec<&str>,
        current_level: usize,
        array: &StructArray,
        digest: &mut D,
    ) {
        if field_name_hierarchy.len() == current_level {
            let array_data = array
                .column_by_name(
                    field_name_hierarchy
                        .first()
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
                        .get(current_level)
                        .expect("Failed to get field name at current level"),
                )
                .expect("Failed to get column by name")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("Failed to downcast to StructArray");

            Self::update_nested_field(
                field_name_hierarchy,
                current_level
                    .checked_add(1)
                    .expect("Field nesting level overflow"),
                next_array,
                digest,
            );
        }
    }

    fn array_digest_update(data_type: &DataType, array: &dyn Array, digest: &mut D) {
        match data_type {
            DataType::Null => todo!(),
            DataType::Boolean => {
                // Bool Array is stored a bit differently, so we can't use the standard fixed buffer approach
                let bool_array = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("Failed to downcast to BooleanArray");

                bool_array.into_iter().for_each(|value| match value {
                    Some(b) => digest.update([u8::from(b)]),
                    None => digest.update(NULL_BYTES),
                });
            }
            DataType::Int8 | DataType::UInt8 => Self::hash_fixed_size_array(array, digest, 1),
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => {
                Self::hash_fixed_size_array(array, digest, 2);
            }
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => {
                Self::hash_fixed_size_array(array, digest, 4);
            }
            DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Date64 => {
                Self::hash_fixed_size_array(array, digest, 8);
            }
            DataType::Timestamp(_, _) => todo!(),
            DataType::Time32(time_unit) => Self::hash_time_array(array, *time_unit, digest, 4),
            DataType::Time64(time_unit) => Self::hash_time_array(array, *time_unit, digest, 8),
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
            DataType::Decimal32(precision, scale)
            | DataType::Decimal64(precision, scale)
            | DataType::Decimal128(precision, scale)
            | DataType::Decimal256(precision, scale) => {
                Self::hash_decimal(*precision, *scale, array, digest);
            }
            DataType::Map(_, _) => todo!(),
            DataType::RunEndEncoded(_, _) => todo!(),
        }
    }

    #[expect(clippy::cast_sign_loss, reason = "element_size is always positive")]
    fn hash_fixed_size_array(array: &dyn Array, digest: &mut D, element_size: i32) {
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

        // Deal with null
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
                        if let Some(data_slice) = slice.get(data_pos..end_pos) {
                            digest.update(data_slice);
                        } else {
                            digest.update(NULL_BYTES);
                        }
                    } else {
                        digest.update(NULL_BYTES);
                    }
                }
            }
            None => {
                // No nulls, we can hash the entire buffer directly
                digest.update(slice);
            }
        }
    }

    fn hash_binary_array(array: &GenericBinaryArray<impl OffsetSizeTrait>, digest: &mut D) {
        match array.nulls() {
            Some(null_buf) => {
                for i in 0..array.len() {
                    if null_buf.is_valid(i) {
                        let value = array.value(i);
                        digest.update(value);
                    } else {
                        digest.update(NULL_BYTES);
                    }
                }
            }
            None => {
                for i in 0..array.len() {
                    let value = array.value(i);
                    digest.update(value);
                }
            }
        }
    }

    fn hash_time_array(array: &dyn Array, time_unit: TimeUnit, digest: &mut D, element_size: i32) {
        // We need to update the digest with the time unit first to ensure different time units produce different hashes
        digest.update([match time_unit {
            TimeUnit::Second => 0_u8,
            TimeUnit::Millisecond => 1_u8,
            TimeUnit::Microsecond => 2_u8,
            TimeUnit::Nanosecond => 3_u8,
        }]);

        // Now hash the underlying fixed size array based on time unit
        Self::hash_fixed_size_array(array, digest, element_size);
    }

    fn hash_string_array(array: &GenericStringArray<impl OffsetSizeTrait>, digest: &mut D) {
        match array.nulls() {
            Some(null_buf) => {
                for i in 0..array.len() {
                    if null_buf.is_valid(i) {
                        let value = array.value(i);
                        digest.update(value.as_bytes());
                    } else {
                        digest.update(NULL_BYTES);
                    }
                }
            }
            None => {
                for i in 0..array.len() {
                    let value = array.value(i);
                    digest.update(value.as_bytes());
                }
            }
        }
    }

    fn hash_list_array(
        array: &GenericListArray<impl OffsetSizeTrait>,
        field_data_type: &DataType,
        digest: &mut D,
    ) {
        match array.nulls() {
            Some(null_buf) => {
                for i in 0..array.len() {
                    if null_buf.is_valid(i) {
                        Self::array_digest_update(field_data_type, array.value(i).as_ref(), digest);
                    } else {
                        digest.update(NULL_BYTES);
                    }
                }
            }
            None => {
                for i in 0..array.len() {
                    Self::array_digest_update(field_data_type, array.value(i).as_ref(), digest);
                }
            }
        }
    }

    #[expect(clippy::cast_sign_loss, reason = "Scale should always be non-negative")]
    fn hash_decimal(precision: u8, scale: i8, array: &dyn Array, digest: &mut D) {
        // Include the precision and scale in the hash
        digest.update([precision]);
        digest.update([scale as u8]);

        // Hash the underlying fixed size array based on precision
        match precision {
            1..=9 => Self::hash_fixed_size_array(array, digest, 4),
            10..=18 => Self::hash_fixed_size_array(array, digest, 8),
            19..=38 => Self::hash_fixed_size_array(array, digest, 16),
            39..=76 => Self::hash_fixed_size_array(array, digest, 32),
            _ => panic!("Unsupported decimal precision: {precision}"),
        }
    }

    /// Internal recursive function to extract field names from nested structs effectively flattening the schema
    /// The format is `parent__child__grandchild__etc`... for nested fields and will be stored in `fields_digest_buffer`
    fn extract_fields_name(
        field: &Field,
        parent_field_name: &str,
        fields_digest_buffer: &mut BTreeMap<String, D>,
    ) {
        // Check if field is a nested type of struct
        if let DataType::Struct(fields) = field.data_type() {
            println!(
                "Extracting nested struct field: {} with parent: {}",
                field.name(),
                parent_field_name
            );
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
                D::new(),
            );
        }
    }

    fn construct_field_name_hierarchy(parent_field_name: &str, field_name: &str) -> String {
        if parent_field_name.is_empty() {
            field_name.to_owned()
        } else {
            format!("{parent_field_name}__{field_name}")
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    use std::sync::Arc;

    use arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Int32Array, LargeBinaryArray, LargeStringArray,
            ListArray, RecordBatch, StringArray, Time32MillisecondArray, Time32SecondArray,
            Time64MicrosecondArray,
        },
        datatypes::Int32Type,
    };
    use arrow_schema::{DataType, Field, Schema};
    use pretty_assertions::assert_eq;
    use sha2::Sha256;

    use crate::arrow_digester::ArrowDigester;
    use arrow::array::Decimal128Array;

    #[test]
    fn boolean_array_hashing() {
        let bool_array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&bool_array));
        assert_eq!(
            hash,
            "d7b7a73916d3f0c693ebcfa94fe2eee163d31a38ba8fe44ef81c5ffbff50c9be"
        );
    }

    /// Test int32 array hashing which is really meant to test fixed size element array hashing
    #[test]
    fn int32_array_hashing() {
        let int_array = Int32Array::from(vec![Some(42), None, Some(-7), Some(0)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&int_array));
        assert_eq!(
            hash,
            "bb36e54f5e2d937a05bb716a8d595f1c8da67fda48feeb7ab5b071a69e63d648"
        );
    }

    /// Test time array hashing
    #[test]
    fn time32_array_hashing() {
        let time_array = Time32SecondArray::from(vec![Some(1000), None, Some(5000), Some(0)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&time_array));
        assert_eq!(
            hash,
            "b5d70eca0650399a9b00440e3cd9985e58b0f033d446bdd5947f96a62397002a"
        );
    }

    #[test]
    fn time64_array_hashing() {
        let time_array =
            Time64MicrosecondArray::from(vec![Some(1_000_000), None, Some(5_000_000), Some(0)]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&time_array));
        assert_eq!(
            hash,
            "1f0847660ea421c266f226293d2f0c54ea5de0c168ac7e4bebfabf6d348a6d18"
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
            "078347d3063fb5bbe0bdbd3315cf8e5e140733ea34e6b73cbc0838b60a9c8012"
        );

        // Test large binary array with same data to ensure consistency
        let large_binary_array = LargeBinaryArray::from(vec![
            Some(b"hello".as_ref()),
            None,
            Some(b"world".as_ref()),
            Some(b"".as_ref()),
        ]);

        assert_eq!(
            hex::encode(ArrowDigester::<Sha256>::hash_array(&large_binary_array)),
            hash
        );
    }

    // Test String hashing
    #[test]
    fn string_array_hashing() {
        let string_array = StringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);
        let hash = hex::encode(ArrowDigester::<Sha256>::hash_array(&string_array));
        assert_eq!(
            hash,
            "078347d3063fb5bbe0bdbd3315cf8e5e140733ea34e6b73cbc0838b60a9c8012"
        );

        // Test large string array with same data to ensure consistency
        let large_string_array =
            LargeStringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);

        assert_eq!(
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
            "d30c8845c58f71bcec4910c65a91328af2cc86d26001662270da3a3d5222dd36"
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
            hex::encode(ArrowDigester::<Sha256>::hash_array(&decimal32_array)),
            "bd639e8df756f0bd194f18572e89ea180307e6d46e88d96ade52b61e196c3268"
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
            hex::encode(ArrowDigester::<Sha256>::hash_array(&decimal64_array)),
            "ca1f8a6fb179ddafad1e02738ad2d869da187c72a9b815d8e12a85692525d231"
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
            "d2a1a2d8c87193032d46a541405e1bf60124d08a7c431ce3fe55f26508b400f3"
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
            ArrowDigester::<Sha256>::hash_record_batch(batch1.as_ref().unwrap()),
            ArrowDigester::<Sha256>::hash_record_batch(batch2.as_ref().unwrap())
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
            hex::encode(digester.finalize()),
            "9ba289655f0c7dd359ababc5a6f6188b352e45483623fbbf8b967723e2b798f8"
        );
    }

    #[test]
    fn field_names() {
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

        let digester = ArrowDigester::<Sha256>::new(schema);
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        assert_eq!(field_names.len(), 3);
        assert!(field_names.contains(&&"id".to_owned()));
        assert!(field_names.contains(&&"nested__name".to_owned()));
        assert!(field_names.contains(&&"nested__deep__value".to_owned()));
    }
}
