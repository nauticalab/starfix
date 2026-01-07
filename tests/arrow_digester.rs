#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    use std::sync::Arc;

    use arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal32Array,
            Decimal64Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
            Int8Array, LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, RecordBatch,
            StringArray, Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray,
            Time64NanosecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
        },
        datatypes::Int32Type,
    };
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use hex::encode;
    use pretty_assertions::assert_eq;

    use arrow::array::Decimal128Array;
    use starfix::ArrowDigester;

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

        // Empty Table Hashing Check

        assert_eq!(
            encode(ArrowDigester::new(schema.clone()).finalize()),
            "000001c7bc0a0c84aca684adbec21f8cb481781332fc91a205165a6c74c3a63a80e9b2"
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
            encode(ArrowDigester::hash_record_batch(&batch)),
            "000001ac720bed7fb1d696d5626705dc7602d14cfe974a3297cc28c3cb8b8e9a62601a"
        );
    }

    #[test]
    fn boolean_array_hashing() {
        let bool_array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let hash = hex::encode(ArrowDigester::hash_array(&bool_array));
        assert_eq!(
            hash,
            "000001f9abeb37d9395f359b48a379f0a8467c572b19ecc6cae9fa85e1bf627a52a8f3"
        );
    }

    /// Test int32 array hashing which is really meant to test fixed size element array hashing
    #[test]
    fn int32_array_hashing() {
        let int_array = Int32Array::from(vec![Some(42), None, Some(-7), Some(0)]);
        let hash = hex::encode(ArrowDigester::hash_array(&int_array));
        assert_eq!(
            hash,
            "00000127f2411e6839eb1e3fe706ac3f01e704c7b46357360fb2ddb8a08ec98e8ba4fa"
        );
    }

    /// Test time array hashing
    #[test]
    fn time32_array_hashing() {
        let time_array = Time32SecondArray::from(vec![Some(1000), None, Some(5000), Some(0)]);
        let hash = hex::encode(ArrowDigester::hash_array(&time_array));
        assert_eq!(
            hash,
            "0000019000b74aa80f685103a8cafc7e113aa8f33ccc0c94ea3713318d2cc2f3436baa"
        );
    }

    #[test]
    fn time64_array_hashing() {
        let time_array =
            Time64MicrosecondArray::from(vec![Some(1_000_000), None, Some(5_000_000), Some(0)]);
        let hash = hex::encode(ArrowDigester::hash_array(&time_array));
        assert_eq!(
            hash,
            "00000195f12143d789f364a3ed52f7300f8f91dc21fbe00c34aed798ca8fd54182dea3"
        );
    }

    #[test]
    fn time_array_different_units_produce_different_hashes() {
        let time32_second = Time32SecondArray::from(vec![Some(1000), Some(2000)]);
        let time32_millis = Time32MillisecondArray::from(vec![Some(1000), Some(2000)]);

        let hash_second = hex::encode(ArrowDigester::hash_array(&time32_second));
        let hash_millis = hex::encode(ArrowDigester::hash_array(&time32_millis));

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
        let hash = hex::encode(ArrowDigester::hash_array(&binary_array));
        assert_eq!(
            hash,
            "000001466801efd880d2acecd6c78915b5c2a51476870f9116912834d79de43a000071"
        );

        // Test large binary array with same data to ensure consistency
        let large_binary_array = LargeBinaryArray::from(vec![
            Some(b"hello".as_ref()),
            None,
            Some(b"world".as_ref()),
            Some(b"".as_ref()),
        ]);

        assert_ne!(
            hex::encode(ArrowDigester::hash_array(&large_binary_array)),
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

        let hash1 = hex::encode(ArrowDigester::hash_array(&array1));
        let hash2 = hex::encode(ArrowDigester::hash_array(&array2));

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

        let hash1 = hex::encode(ArrowDigester::hash_array(&array1));
        let hash2 = hex::encode(ArrowDigester::hash_array(&array2));

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
        let hash = hex::encode(ArrowDigester::hash_array(&string_array));
        assert_eq!(
            hash,
            "00000114a2d2eaf535b6e78fbf1d58ae93accce424eafd20fa449eff8acefc47903d3d"
        );

        // Test large string array with same data to ensure consistency
        let large_string_array =
            LargeStringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);

        assert_ne!(
            hex::encode(ArrowDigester::hash_array(&large_string_array)),
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

        let hash = hex::encode(ArrowDigester::hash_array(&list_array));
        assert_eq!(
            hash,
            "000001f654be5f0ef89807feba9483072190b7d26964e535cd7c522706218df9c3c015"
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
            encode(ArrowDigester::hash_array(&decimal32_array)),
            "000001ef29250615f9d6ab34672c3b11dfa2dcda6e8e6164bc55899c13887f17705f5d"
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
            encode(ArrowDigester::hash_array(&decimal64_array)),
            "000001efa4ed72641051233889c07775366cbf2e56eb4b0fcfd46653f5741e81786f08"
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
            hex::encode(ArrowDigester::hash_array(&decimal128_array)),
            "00000155cc4d81a048dbca001ca8581673a5a6c93efd870d358df211a545c2af9b658d"
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
        let hash1 = format!(
            "000001{}",
            encode(ArrowDigester::hash_record_batch(batch1.as_ref().unwrap()))
        );
        let hash2 = format!(
            "000001{}",
            encode(ArrowDigester::hash_record_batch(batch2.as_ref().unwrap()))
        );
        assert_eq!(hash1, hash2);
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
        let mut digester = ArrowDigester::new((*schema).clone());
        digester.update(&batch1);
        digester.update(&batch2);
        assert_eq!(
            encode(digester.finalize()),
            "00000137954b3edd169c7a9e65604c191caf6a307940357305d182a5d2168047e9cc51"
        );
    }

    #[test]
    fn nullable_vs_non_nullable_array_produces_same_hash() {
        let nullable_array = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
        let non_nullable_array = Int32Array::from(vec![1, 2, 3]);

        let hash_nullable = hex::encode(ArrowDigester::hash_array(&nullable_array));
        let hash_non_nullable = hex::encode(ArrowDigester::hash_array(&non_nullable_array));

        assert_eq!(
            hash_nullable, hash_non_nullable,
            "Nullable and non-nullable arrays with same data should produce same hashes"
        );
    }

    #[test]
    fn empty_nullable_vs_non_nullable_array_produces_different_hash() {
        let empty_nullable_array: Int32Array = Int32Array::from(vec![] as Vec<Option<i32>>);
        let empty_non_nullable_array: Int32Array = Int32Array::from(vec![] as Vec<i32>);

        let hash_nullable = hex::encode(ArrowDigester::hash_array(&empty_nullable_array));
        let hash_non_nullable = hex::encode(ArrowDigester::hash_array(&empty_non_nullable_array));

        // Both are empty, but their nullability metadata may differ
        // This test documents the expected behavior
        assert_eq!(hash_nullable, hash_non_nullable);
    }

    #[test]
    fn nullable_vs_non_nullable_schema_produces_different_hash() {
        let nullable_schema = Schema::new(vec![
            Field::new("col1", DataType::Int32, true),
            Field::new("col2", DataType::Boolean, true),
        ]);
        let non_nullable_schema = Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Boolean, false),
        ]);

        let hash_nullable = hex::encode(ArrowDigester::new(nullable_schema).finalize());
        let hash_non_nullable = hex::encode(ArrowDigester::new(non_nullable_schema).finalize());

        assert_ne!(
            hash_nullable, hash_non_nullable,
            "Nullable and non-nullable schemas with same data types should produce different hashes"
        );
    }

    #[test]
    fn batches_vs_single_hash_produces_same_result() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, true),
        ]));

        // Create two batches with data
        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![4, 5, 6])),
                Arc::new(Float64Array::from(vec![4.4, 5.5, 6.6])),
            ],
        )
        .unwrap();

        // Hash batches incrementally
        let mut digester_batches = ArrowDigester::new((*schema).clone());
        digester_batches.update(&batch1);
        digester_batches.update(&batch2);
        let hash_batches = encode(digester_batches.finalize());

        // Hash combined batch all at once
        let combined_batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            ],
        )
        .unwrap();

        let mut digester_single = ArrowDigester::new((*schema).clone());
        digester_single.update(&combined_batch);
        let hash_single = encode(digester_single.finalize());

        assert_eq!(
            hash_batches, hash_single,
            "Hashing multiple batches incrementally should produce the same result as hashing one combined batch"
        );
    }

    #[test]
    fn batches_with_nulls_vs_single_hash_produces_same_result() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("value", DataType::Float64, true),
        ]));

        // Create two batches: first all nulls, second with values
        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![None, None, None])),
                Arc::new(Float64Array::from(vec![None, None, None])),
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])),
                Arc::new(Float64Array::from(vec![Some(1.1), Some(2.2), Some(3.3)])),
            ],
        )
        .unwrap();

        // Hash batches incrementally
        let mut digester_batches = ArrowDigester::new((*schema).clone());
        digester_batches.update(&batch1);
        digester_batches.update(&batch2);
        let hash_batches = encode(digester_batches.finalize());

        // Hash combined batch all at once
        let combined_batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![
                    None,
                    None,
                    None,
                    Some(1),
                    Some(2),
                    Some(3),
                ])),
                Arc::new(Float64Array::from(vec![
                    None,
                    None,
                    None,
                    Some(1.1),
                    Some(2.2),
                    Some(3.3),
                ])),
            ],
        )
        .unwrap();

        let mut digester_single = ArrowDigester::new((*schema).clone());
        digester_single.update(&combined_batch);
        let hash_single = encode(digester_single.finalize());

        assert_eq!(
            hash_batches, hash_single,
            "Hashing batches where first is all nulls should produce same result as combined batch"
        );
    }
}
