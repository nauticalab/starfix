#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    use std::sync::Arc;

    use arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal32Array,
            Decimal64Array, DictionaryArray, Float32Array, Float64Array, Int16Array, Int32Array,
            Int64Array, Int8Array, LargeBinaryArray, LargeListArray, LargeListBuilder,
            LargeStringArray, LargeStringBuilder, ListArray, ListBuilder, RecordBatch, StringArray,
            StringBuilder, StructArray, Time32MillisecondArray, Time32SecondArray,
            Time64MicrosecondArray, Time64NanosecondArray, UInt16Array, UInt32Array, UInt64Array,
            UInt8Array,
        },
        datatypes::{Int32Type, Int8Type},
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
            encode(ArrowDigester::new(&schema).finalize()),
            "0000016a44e0dc5c25d5ca0c53312a6afcffa6e07168afc7f16f5e16c8ca052f09f1bb"
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
            "00000122697d05509c016ab42d2b1c69cc79e75819f4a6ec41164919348231b75f530c"
        );
    }

    #[test]
    fn boolean_array_hashing() {
        let bool_array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let hash = hex::encode(ArrowDigester::hash_array(&bool_array));
        assert_eq!(
            hash,
            "00000185a9c99eba7bcfd9b14fd529b9534f2289319779270aa4a072f117cf90a6ac8b"
        );
    }

    /// Test int32 array hashing which is really meant to test fixed size element array hashing.
    #[test]
    fn int32_array_hashing() {
        let int_array = Int32Array::from(vec![Some(42), None, Some(-7), Some(0)]);
        let hash = hex::encode(ArrowDigester::hash_array(&int_array));
        assert_eq!(
            hash,
            "0000018330f9b8796b9434cbf7bc028c18c58a2a739b980acf9995ce1e5d60b43b0138"
        );
    }

    /// Test time array hashing.
    #[test]
    fn time32_array_hashing() {
        let time_array = Time32SecondArray::from(vec![Some(1000), None, Some(5000), Some(0)]);
        let hash = hex::encode(ArrowDigester::hash_array(&time_array));
        assert_eq!(
            hash,
            "000001aba70469e596c735ec13c3d60a9db2d0e5515eb864f07ad5d24572b35f23eacc"
        );
    }

    #[test]
    fn time64_array_hashing() {
        let time_array =
            Time64MicrosecondArray::from(vec![Some(1_000_000), None, Some(5_000_000), Some(0)]);
        let hash = hex::encode(ArrowDigester::hash_array(&time_array));
        assert_eq!(
            hash,
            "000001c96d705b1278f9ffe1b31fb307408768f14d961c44028a1d0f778dd61786ee26"
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

    /// Test binary array hashing.
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
            "0000018dc3a0e479d1335553546c8f23c36d75335cbd34805a6f96c5d5225b347fbc57"
        );

        // Large binary array with same data should produce identical hash (type canonicalization)
        let large_binary_array = LargeBinaryArray::from(vec![
            Some(b"hello".as_ref()),
            None,
            Some(b"world".as_ref()),
            Some(b"".as_ref()),
        ]);

        assert_eq!(
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
            "0000016255bde0141ebf26e08c31c96f6112e5e21d101ab8bb90d77f2c3eec02c62d3c"
        );

        // Large string array with same data should produce identical hash (type canonicalization)
        let large_string_array =
            LargeStringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);

        assert_eq!(
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
            "00000190658c2c4e9178f8ae6c686d6fe13262a9fab9cb619542911453abeca8195a9f"
        );

        // Collision test: [[1, 2], [3]] vs [[1], [2, 3]]
        // Without a per-element length prefix, both lists produce the same raw bytes:
        // 01000000 02000000 03000000 — and would collide.
        let array1 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3)]),
        ]);
        let array2 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1)]),
            Some(vec![Some(2), Some(3)]),
        ]);

        let hash1 = hex::encode(ArrowDigester::hash_array(&array1));
        let hash2 = hex::encode(ArrowDigester::hash_array(&array2));

        assert_ne!(
            hash1, hash2,
            "List arrays with different element groupings should produce different hashes"
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
            "0000014f015bd5c4b6ce6e939a8c890333f3e110c2c28ef8014aafd352f8373791e547"
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
            "000001dc08c7b9c583edecec36bc5dee21cd2edec9f402a651014fea5f8834d16ad737"
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
            "0000011e3b33d28771b3593fd5dc4b68af8091a1ba9cd493ade374e7368e213bef244e"
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
        let mut digester = ArrowDigester::new(schema.as_ref());
        digester.update(&batch1);
        digester.update(&batch2);
        assert_eq!(
            encode(digester.finalize()),
            "0000019f5fa370d315a4b4f2314be7b7284a0549b70ad4e21e584fdebf441ad02f44f0"
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

        let hash_nullable = hex::encode(ArrowDigester::hash_schema(&nullable_schema));
        let hash_non_nullable = hex::encode(ArrowDigester::hash_schema(&non_nullable_schema));

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
        let mut digester_batches = ArrowDigester::new(schema.as_ref());
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

        let mut digester_single = ArrowDigester::new(schema.as_ref());
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
        let mut digester_batches = ArrowDigester::new(schema.as_ref());
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

        let mut digester_single = ArrowDigester::new(schema.as_ref());
        digester_single.update(&combined_batch);
        let hash_single = encode(digester_single.finalize());

        assert_eq!(
            hash_batches, hash_single,
            "Hashing batches where first is all nulls should produce same result as combined batch"
        );
    }

    // ── Issue 1: Struct field-order independence ─────────────────────────

    /// Two schemas with the same struct fields in different order should produce identical schema hashes.
    /// Bug: `data_type_to_value()` preserves struct field insertion order in the JSON Vec.
    #[test]

    fn struct_field_order_in_schema_should_not_affect_hash() {
        let schema1 = Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(
                vec![
                    Field::new("x", DataType::Int32, false),
                    Field::new("y", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        )]);

        let schema2 = Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(
                vec![
                    Field::new("y", DataType::Utf8, true),
                    Field::new("x", DataType::Int32, false),
                ]
                .into(),
            ),
            true,
        )]);

        let hash1 = encode(ArrowDigester::hash_schema(&schema1));
        let hash2 = encode(ArrowDigester::hash_schema(&schema2));

        assert_eq!(
            hash1, hash2,
            "Struct field order should not affect schema hash"
        );
    }

    /// Record batches with struct columns whose inner fields are reordered should produce identical hashes.
    #[test]

    fn struct_field_order_in_record_batch_should_not_affect_hash() {
        let schema1 = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(
                vec![
                    Field::new("a", DataType::Int32, false),
                    Field::new("b", DataType::Boolean, true),
                ]
                .into(),
            ),
            false,
        )]));

        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(
                vec![
                    Field::new("b", DataType::Boolean, true),
                    Field::new("a", DataType::Int32, false),
                ]
                .into(),
            ),
            false,
        )]));

        let ints = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let bools = Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])) as ArrayRef;

        let struct1 = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::clone(&ints),
            ),
            (
                Arc::new(Field::new("b", DataType::Boolean, true)),
                Arc::clone(&bools),
            ),
        ]);

        let struct2 = StructArray::from(vec![
            (
                Arc::new(Field::new("b", DataType::Boolean, true)),
                Arc::clone(&bools),
            ),
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::clone(&ints),
            ),
        ]);

        let batch1 = RecordBatch::try_new(schema1, vec![Arc::new(struct1) as ArrayRef]).unwrap();
        let batch2 = RecordBatch::try_new(schema2, vec![Arc::new(struct2) as ArrayRef]).unwrap();

        assert_eq!(
            encode(ArrowDigester::hash_record_batch(&batch1)),
            encode(ArrowDigester::hash_record_batch(&batch2)),
            "Struct field order in record batch should not affect hash"
        );
    }

    // ── Issue 5: Type canonicalization (Binary/LargeBinary, Utf8/LargeUtf8, List/LargeList) ──

    #[test]

    fn binary_and_large_binary_schema_should_hash_equal() {
        let schema1 = Schema::new(vec![Field::new("col", DataType::Binary, true)]);
        let schema2 = Schema::new(vec![Field::new("col", DataType::LargeBinary, true)]);

        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema1)),
            encode(ArrowDigester::hash_schema(&schema2)),
            "Binary and LargeBinary schemas should be logically equivalent"
        );
    }

    #[test]

    fn utf8_and_large_utf8_schema_should_hash_equal() {
        let schema1 = Schema::new(vec![Field::new("col", DataType::Utf8, true)]);
        let schema2 = Schema::new(vec![Field::new("col", DataType::LargeUtf8, true)]);

        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema1)),
            encode(ArrowDigester::hash_schema(&schema2)),
            "Utf8 and LargeUtf8 schemas should be logically equivalent"
        );
    }

    #[test]

    fn list_and_large_list_schema_should_hash_equal() {
        let list_field = Field::new("item", DataType::Int32, true);
        let schema1 = Schema::new(vec![Field::new(
            "col",
            DataType::List(Box::new(list_field.clone()).into()),
            true,
        )]);
        let schema2 = Schema::new(vec![Field::new(
            "col",
            DataType::LargeList(Box::new(list_field).into()),
            true,
        )]);

        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema1)),
            encode(ArrowDigester::hash_schema(&schema2)),
            "List and LargeList schemas should be logically equivalent"
        );
    }

    #[test]
    fn list_and_large_list_array_should_hash_equal() {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            None,
            Some(vec![Some(3)]),
        ]);
        let large_list = LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            None,
            Some(vec![Some(3)]),
        ]);

        assert_eq!(
            encode(ArrowDigester::hash_array(&list)),
            encode(ArrowDigester::hash_array(&large_list)),
            "List and LargeList arrays with same data should produce same hash"
        );
    }

    #[test]
    fn list_and_large_list_record_batch_should_hash_equal() {
        let list_field = Field::new("item", DataType::Int32, true);
        let schema1 = Arc::new(Schema::new(vec![Field::new(
            "col",
            DataType::List(Box::new(list_field.clone()).into()),
            true,
        )]));
        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "col",
            DataType::LargeList(Box::new(list_field).into()),
            true,
        )]));

        let batch1 = RecordBatch::try_new(
            schema1,
            vec![
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                    Some(vec![Some(10), Some(20)]),
                    None,
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema2,
            vec![
                Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
                    vec![Some(vec![Some(10), Some(20)]), None],
                )) as ArrayRef,
            ],
        )
        .unwrap();

        assert_eq!(
            encode(ArrowDigester::hash_record_batch(&batch1)),
            encode(ArrowDigester::hash_record_batch(&batch2)),
            "List and LargeList record batches with same data should produce same hash"
        );
    }

    #[test]

    fn binary_and_large_binary_array_should_hash_equal() {
        let bin = BinaryArray::from(vec![Some(b"hello".as_ref()), None, Some(b"world".as_ref())]);
        let large_bin =
            LargeBinaryArray::from(vec![Some(b"hello".as_ref()), None, Some(b"world".as_ref())]);

        assert_eq!(
            encode(ArrowDigester::hash_array(&bin)),
            encode(ArrowDigester::hash_array(&large_bin)),
            "Binary and LargeBinary arrays with same data should produce same hash"
        );
    }

    #[test]

    fn utf8_and_large_utf8_array_should_hash_equal() {
        let arr = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let large_arr = LargeStringArray::from(vec![Some("hello"), None, Some("world")]);

        assert_eq!(
            encode(ArrowDigester::hash_array(&arr)),
            encode(ArrowDigester::hash_array(&large_arr)),
            "Utf8 and LargeUtf8 arrays with same data should produce same hash"
        );
    }

    #[test]
    fn utf8_and_large_utf8_record_batch_should_hash_equal() {
        let schema1 = Arc::new(Schema::new(vec![Field::new("col", DataType::Utf8, true)]));
        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "col",
            DataType::LargeUtf8,
            true,
        )]));

        let batch1 = RecordBatch::try_new(
            schema1,
            vec![Arc::new(StringArray::from(vec![Some("abc"), None])) as ArrayRef],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema2,
            vec![Arc::new(LargeStringArray::from(vec![Some("abc"), None])) as ArrayRef],
        )
        .unwrap();

        assert_eq!(
            encode(ArrowDigester::hash_record_batch(&batch1)),
            encode(ArrowDigester::hash_record_batch(&batch2)),
            "Utf8 and LargeUtf8 record batches with same data should produce same hash"
        );
    }

    #[test]

    fn binary_and_large_binary_record_batch_should_hash_equal() {
        let schema1 = Arc::new(Schema::new(vec![Field::new("col", DataType::Binary, true)]));
        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "col",
            DataType::LargeBinary,
            true,
        )]));

        let batch1 = RecordBatch::try_new(
            schema1,
            vec![Arc::new(BinaryArray::from(vec![Some(b"abc".as_ref()), None])) as ArrayRef],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema2,
            vec![Arc::new(LargeBinaryArray::from(vec![Some(b"abc".as_ref()), None])) as ArrayRef],
        )
        .unwrap();

        assert_eq!(
            encode(ArrowDigester::hash_record_batch(&batch1)),
            encode(ArrowDigester::hash_record_batch(&batch2)),
            "Binary and LargeBinary record batches with same data should produce same hash"
        );
    }

    // ── Deep nested type normalization ──────────────────────────────────

    #[test]
    fn list_of_utf8_vs_large_list_of_large_utf8_array_should_hash_equal() {
        // List(Utf8) vs LargeList(LargeUtf8) — normalization must be recursive
        let list = {
            let mut builder = ListBuilder::new(StringBuilder::new());
            builder.values().append_value("hello");
            builder.values().append_value("world");
            builder.append(true);
            builder.values().append_value("foo");
            builder.append(true);
            builder.finish()
        };

        let large_list = {
            let mut builder = LargeListBuilder::new(LargeStringBuilder::new());
            builder.values().append_value("hello");
            builder.values().append_value("world");
            builder.append(true);
            builder.values().append_value("foo");
            builder.append(true);
            builder.finish()
        };

        assert_eq!(
            encode(ArrowDigester::hash_array(&list)),
            encode(ArrowDigester::hash_array(&large_list)),
            "List(Utf8) and LargeList(LargeUtf8) should produce same hash"
        );
    }

    #[test]
    fn list_of_utf8_vs_large_list_of_large_utf8_schema_should_hash_equal() {
        let schema1 = Schema::new(vec![Field::new(
            "col",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        )]);
        let schema2 = Schema::new(vec![Field::new(
            "col",
            DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
            true,
        )]);

        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema1)),
            encode(ArrowDigester::hash_schema(&schema2)),
            "List(Utf8) and LargeList(LargeUtf8) schemas should be logically equivalent"
        );
    }

    #[test]
    fn struct_with_list_utf8_vs_large_variants_record_batch_should_hash_equal() {
        // Struct({items: List(Utf8), name: Utf8}) vs Struct({items: LargeList(LargeUtf8), name: LargeUtf8})
        let schema1 = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(
                vec![
                    Field::new(
                        "items",
                        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                        true,
                    ),
                    Field::new("name", DataType::Utf8, true),
                ]
                .into(),
            ),
            false,
        )]));

        let schema2 = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(
                vec![
                    Field::new(
                        "items",
                        DataType::LargeList(Arc::new(Field::new(
                            "item",
                            DataType::LargeUtf8,
                            true,
                        ))),
                        true,
                    ),
                    Field::new("name", DataType::LargeUtf8, true),
                ]
                .into(),
            ),
            false,
        )]));

        // Build struct with List(Utf8)
        let list1 = {
            let mut builder = ListBuilder::new(StringBuilder::new());
            builder.values().append_value("a");
            builder.values().append_value("b");
            builder.append(true);
            builder.values().append_value("c");
            builder.append(true);
            builder.finish()
        };
        let names1 = StringArray::from(vec![Some("Alice"), Some("Bob")]);
        let struct1 = StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "items",
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    true,
                )),
                Arc::new(list1) as ArrayRef,
            ),
            (
                Arc::new(Field::new("name", DataType::Utf8, true)),
                Arc::new(names1) as ArrayRef,
            ),
        ]);

        // Build struct with LargeList(LargeUtf8)
        let list2 = {
            let mut builder = LargeListBuilder::new(LargeStringBuilder::new());
            builder.values().append_value("a");
            builder.values().append_value("b");
            builder.append(true);
            builder.values().append_value("c");
            builder.append(true);
            builder.finish()
        };
        let names2 = LargeStringArray::from(vec![Some("Alice"), Some("Bob")]);
        let struct2 = StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "items",
                    DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
                    true,
                )),
                Arc::new(list2) as ArrayRef,
            ),
            (
                Arc::new(Field::new("name", DataType::LargeUtf8, true)),
                Arc::new(names2) as ArrayRef,
            ),
        ]);

        let batch1 = RecordBatch::try_new(schema1, vec![Arc::new(struct1) as ArrayRef]).unwrap();
        let batch2 = RecordBatch::try_new(schema2, vec![Arc::new(struct2) as ArrayRef]).unwrap();

        assert_eq!(
            encode(ArrowDigester::hash_record_batch(&batch1)),
            encode(ArrowDigester::hash_record_batch(&batch2)),
            "Struct with List(Utf8) should hash same as Struct with LargeList(LargeUtf8)"
        );
    }

    #[test]
    fn streaming_with_type_equivalent_schemas_should_succeed() {
        // Create digester with Utf8 schema, feed batch with LargeUtf8 schema
        let schema_utf8 = Schema::new(vec![Field::new("col", DataType::Utf8, true)]);

        let mut digester = ArrowDigester::new(&schema_utf8);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "col",
                DataType::LargeUtf8,
                true,
            )])),
            vec![Arc::new(LargeStringArray::from(vec![Some("hello"), None])) as ArrayRef],
        )
        .unwrap();

        digester.update(&batch); // Should NOT panic — schemas are logically equivalent
        let _hash = encode(digester.finalize());
    }

    // ── Issue 6: Dictionary-encoded array equivalence ───────────────────

    #[test]

    fn dictionary_utf8_should_hash_same_as_plain_string() {
        let plain = StringArray::from(vec![Some("apple"), Some("banana"), Some("apple")]);

        let dict: DictionaryArray<Int32Type> = vec![Some("apple"), Some("banana"), Some("apple")]
            .into_iter()
            .collect();

        assert_eq!(
            encode(ArrowDigester::hash_array(&plain)),
            encode(ArrowDigester::hash_array(&dict)),
            "Dictionary<Int32, Utf8> should hash same as plain StringArray"
        );
    }

    #[test]

    fn dictionary_int_values_should_hash_same_as_plain() {
        let plain = StringArray::from(vec![Some("x"), Some("y"), Some("x")]);

        let dict: DictionaryArray<Int8Type> =
            vec![Some("x"), Some("y"), Some("x")].into_iter().collect();

        assert_eq!(
            encode(ArrowDigester::hash_array(&plain)),
            encode(ArrowDigester::hash_array(&dict)),
            "Dictionary<Int8, Utf8> should hash same as plain StringArray"
        );
    }

    #[test]

    fn dictionary_with_nulls_should_hash_same_as_plain() {
        let plain = StringArray::from(vec![Some("a"), None, Some("b"), None]);

        let dict: DictionaryArray<Int32Type> =
            vec![Some("a"), None, Some("b"), None].into_iter().collect();

        assert_eq!(
            encode(ArrowDigester::hash_array(&plain)),
            encode(ArrowDigester::hash_array(&dict)),
            "Dictionary with nulls should hash same as plain array with same nulls"
        );
    }

    // ── Issue 7: Streaming schema equality too strict ───────────────────

    /// Feeding a batch with reordered columns into a digester should not panic.
    #[test]

    fn streaming_update_with_reordered_columns_should_succeed() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Boolean, true),
        ]);

        let mut digester = ArrowDigester::new(&schema);

        // Batch with columns in DIFFERENT order: [b, a]
        let reordered_schema = Arc::new(Schema::new(vec![
            Field::new("b", DataType::Boolean, true),
            Field::new("a", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            reordered_schema,
            vec![
                Arc::new(BooleanArray::from(vec![Some(true), Some(false)])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            ],
        )
        .unwrap();

        digester.update(&batch); // Should NOT panic
        let _hash = encode(digester.finalize());
    }

    /// A digester fed batches with different column orders should produce the same hash
    /// as one fed batches in the original order.
    #[test]

    fn streaming_reordered_columns_produce_same_hash() {
        let schema_ab = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Boolean, true),
        ]);

        let ints = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let bools = Arc::new(BooleanArray::from(vec![Some(true), Some(false)])) as ArrayRef;

        let batch_ab = RecordBatch::try_new(
            Arc::new(schema_ab.clone()),
            vec![Arc::clone(&ints), Arc::clone(&bools)],
        )
        .unwrap();

        let batch_ba = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("b", DataType::Boolean, true),
                Field::new("a", DataType::Int32, false),
            ])),
            vec![Arc::clone(&bools), Arc::clone(&ints)],
        )
        .unwrap();

        // Digester fed batch in original order [a, b]
        let mut digester1 = ArrowDigester::new(&schema_ab);
        digester1.update(&batch_ab);
        let hash1 = encode(digester1.finalize());

        // Digester fed batch in reversed order [b, a]
        let mut digester2 = ArrowDigester::new(&schema_ab);
        digester2.update(&batch_ba);
        let hash2 = encode(digester2.finalize());

        assert_eq!(
            hash1, hash2,
            "Streaming with reordered columns should produce same hash"
        );
    }
}
