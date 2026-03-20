/// Manual byte-level verification tests for the Starfix hashing specification.
///
/// Each test in this module manually computes the expected SHA-256 hash by
/// feeding the exact bytes described in `docs/byte-layout-spec.md` into a
/// fresh SHA-256 hasher, then asserts that the library produces the identical
/// result. This serves as both a conformance check and a reference
/// implementation for anyone porting Starfix to another language.
#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::redundant_clone, reason = "Clones for clarity in test setup")]
    #![expect(clippy::absolute_paths, reason = "One-off use in test")]

    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BinaryArray, BooleanArray, Int32Array, LargeListArray, LargeStringArray,
        RecordBatch, StringArray, StructArray,
    };
    use arrow::buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Schema};
    use sha2::{Digest as _, Sha256};
    use starfix::ArrowDigester;

    const VERSION: [u8; 3] = [0x00, 0x00, 0x01];

    // ── Helper ───────────────────────────────────────────────────────────

    /// Prepend the 3-byte version prefix to a 32-byte SHA-256 digest,
    /// returning the full 35-byte Starfix hash.
    fn with_version(digest: Vec<u8>) -> Vec<u8> {
        let mut out = VERSION.to_vec();
        out.extend(digest);
        out
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example A: Simple Two-Column Table (record batch)
    //   Schema: {age: Int32 non-nullable, name: LargeUtf8 nullable}
    //   Row 0:  age=25, name="Alice"
    //   Row 1:  age=30, name=NULL
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_a_two_column_table() {
        // ── Build the table ──────────────────────────────────────────────
        let schema = Schema::new(vec![
            Field::new("age", DataType::Int32, false),
            Field::new("name", DataType::LargeUtf8, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![25_i32, 30])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec![Some("Alice"), None])) as ArrayRef,
            ],
        )
        .unwrap();

        // ── Step 1: Schema digest ────────────────────────────────────────
        let schema_json = r#"{"age":{"data_type":"Int32","nullable":false},"name":{"data_type":"LargeUtf8","nullable":true}}"#;
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        // Verify the library agrees on schema hash
        assert_eq!(
            ArrowDigester::hash_schema(&schema),
            with_version(schema_digest.to_vec()),
            "Schema hash mismatch — canonical JSON may differ"
        );

        // ── Step 2: Field "age" (Int32, non-nullable) ────────────────────
        // Values: [25, 30]  →  little-endian bytes
        let mut age_data = Sha256::new();
        age_data.update(25_i32.to_le_bytes()); // 19 00 00 00
        age_data.update(30_i32.to_le_bytes()); // 1e 00 00 00
        let age_data_finalized = age_data.finalize();

        // ── Step 3: Field "name" (LargeUtf8, nullable) ───────────────────
        // Values: ["Alice", NULL]
        //
        // Validity BitVec (Lsb0, u8 storage):
        //   bit 0 = 1 (valid), bit 1 = 0 (null)
        //   → u8 word = 0b01 = 1
        //   bit_count = 2
        let bit_count: u64 = 2;
        let validity_word: u8 = 1; // bits: [1, 0] in Lsb0

        // Data bytes (only valid elements):
        //   "Alice" → len=5 as u64 LE, then UTF-8 bytes
        //   NULL → skipped
        let mut name_data = Sha256::new();
        name_data.update(5_u64.to_le_bytes()); // length prefix
        name_data.update(b"Alice"); // raw UTF-8 bytes
                                    // NULL element: nothing fed
        let name_data_finalized = name_data.finalize();

        // ── Step 4: Final combination ────────────────────────────────────
        // Fields in alphabetical order: "age", "name"
        let mut final_digest = Sha256::new();

        // Schema
        final_digest.update(schema_digest);

        // Field "age" (non-nullable → just the data digest)
        final_digest.update(age_data_finalized);

        // Field "name" (nullable → bit_count + validity words + data digest)
        final_digest.update(bit_count.to_le_bytes()); // 02 00 00 00 00 00 00 00
        final_digest.update(validity_word.to_le_bytes()); // 01
        final_digest.update(name_data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        // ── Verify ───────────────────────────────────────────────────────
        assert_eq!(
            ArrowDigester::hash_record_batch(&batch),
            expected,
            "Example A: two-column table hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example B: Boolean Array with Nulls (hash_array API)
    //   BooleanArray [true, NULL, false, true]  (nullable)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_b_boolean_array_with_nulls() {
        let array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);

        // ── Type metadata ────────────────────────────────────────────────
        // data_type_to_value(Boolean) → JSON value "Boolean"
        // serde_json::to_string(json!("Boolean")) → "\"Boolean\""
        let type_json = b"\"Boolean\"";

        // ── Validity bits (Lsb0, u8 storage) ──────────────────────────
        // [valid, null, valid, valid] → bits [1, 0, 1, 1]
        // Lsb0 in u8: bit0=1, bit1=0, bit2=1, bit3=1 → 0b1101 = 13
        let bit_count: u64 = 4;
        let validity_word: u8 = 0b1101; // = 13

        // ── Data bits (Lsb0 packed, valid values only) ───────────────────
        // Valid values: [true, false, true] → 3 bits
        // Lsb0: bit0=1(true), bit1=0(false), bit2=1(true) → 0b101 = 0x05
        let mut data_digest = Sha256::new();
        data_digest.update([0x05_u8]);
        let data_finalized = data_digest.finalize();

        // ── Final combination ────────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        // Nullable finalization
        final_digest.update(bit_count.to_le_bytes());
        final_digest.update(validity_word.to_le_bytes());
        final_digest.update(data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            expected,
            "Example B: boolean array hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example C: Non-Nullable Int32 Array (hash_array API)
    //   Int32Array [1, 2, 3]  (non-nullable)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_c_non_nullable_int32_array() {
        let array = Int32Array::from(vec![1_i32, 2, 3]);

        // ── Type metadata ────────────────────────────────────────────────
        let type_json = b"\"Int32\"";

        // ── Data (contiguous LE buffer) ──────────────────────────────────
        // [1, 2, 3] as i32 LE:
        //   01 00 00 00  02 00 00 00  03 00 00 00
        let mut data_digest = Sha256::new();
        data_digest.update(1_i32.to_le_bytes());
        data_digest.update(2_i32.to_le_bytes());
        data_digest.update(3_i32.to_le_bytes());
        let data_finalized = data_digest.finalize();

        // ── Final (non-nullable) ─────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        final_digest.update(data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            expected,
            "Example C: non-nullable int32 array hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example D: Non-Nullable Binary Array (hash_array API)
    //   BinaryArray [b"hi", b""]  (non-nullable)
    //   Tests type canonicalization: Binary → LargeBinary
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_d_non_nullable_binary_array() {
        let array = BinaryArray::from(vec![b"hi".as_ref(), b"".as_ref()]);

        // ── Type metadata (canonicalized) ────────────────────────────────
        // Binary → LargeBinary in canonical form
        let type_json = b"\"LargeBinary\"";

        // ── Data ─────────────────────────────────────────────────────────
        // b"hi": len=2 as u64 LE + raw bytes
        // b"":   len=0 as u64 LE + (no bytes)
        let mut data_digest = Sha256::new();
        data_digest.update(2_u64.to_le_bytes()); // 02 00 00 00 00 00 00 00
        data_digest.update(b"hi"); // 68 69
        data_digest.update(0_u64.to_le_bytes()); // 00 00 00 00 00 00 00 00
        let data_finalized = data_digest.finalize();

        // ── Final (non-nullable) ─────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        final_digest.update(data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            expected,
            "Example D: non-nullable binary array hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example E: Column-Order Independence
    //   Batch 1: columns [x: Int32, y: Boolean nullable] → x=10, y=true
    //   Batch 2: columns [y: Boolean nullable, x: Int32] → y=true, x=10
    //   Both must produce the same hash.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_e_column_order_independence() {
        let ints = Arc::new(Int32Array::from(vec![10_i32])) as ArrayRef;
        let bools = Arc::new(BooleanArray::from(vec![Some(true)])) as ArrayRef;

        let batch_xy = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int32, false),
                Field::new("y", DataType::Boolean, true),
            ])),
            vec![Arc::clone(&ints), Arc::clone(&bools)],
        )
        .unwrap();

        let batch_yx = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("y", DataType::Boolean, true),
                Field::new("x", DataType::Int32, false),
            ])),
            vec![Arc::clone(&bools), Arc::clone(&ints)],
        )
        .unwrap();

        // ── Manual computation ───────────────────────────────────────────
        let schema_json = r#"{"x":{"data_type":"Int32","nullable":false},"y":{"data_type":"Boolean","nullable":true}}"#;
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        // Field "x" (Int32, non-nullable): value 10
        let mut x_data = Sha256::new();
        x_data.update(10_i32.to_le_bytes()); // 0a 00 00 00
        let x_finalized = x_data.finalize();

        // Field "y" (Boolean, nullable): value true (valid)
        // Validity: [1] → bit_count=1, word=1 (Lsb0)
        // Data: [true] Lsb0 → bit0=1 → 0x01
        let bit_count: u64 = 1;
        let validity_word: u8 = 1;

        let mut y_data = Sha256::new();
        y_data.update([0x01_u8]); // true in Lsb0 = 0000_0001
        let y_finalized = y_data.finalize();

        // Final combination: schema, then fields alphabetically (x, y)
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        // x (non-nullable)
        final_digest.update(x_finalized);
        // y (nullable)
        final_digest.update(bit_count.to_le_bytes());
        final_digest.update(validity_word.to_le_bytes());
        final_digest.update(y_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        // ── Verify both column orderings produce the same hash ───────────
        let hash_xy = ArrowDigester::hash_record_batch(&batch_xy);
        let hash_yx = ArrowDigester::hash_record_batch(&batch_yx);

        assert_eq!(hash_xy, hash_yx, "Column order should not affect hash");
        assert_eq!(
            hash_xy, expected,
            "Example E: column-order independence hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example F: Type Equivalence (Utf8 vs LargeUtf8, hash_array API)
    //   StringArray ["ab"]  (Utf8, non-nullable)
    //   LargeStringArray ["ab"]  (LargeUtf8, non-nullable)
    //   Both must produce the same hash.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_f_utf8_large_utf8_equivalence() {
        let small = StringArray::from(vec!["ab"]);
        let large = LargeStringArray::from(vec!["ab"]);

        // ── Manual computation ───────────────────────────────────────────
        // Type metadata: both canonicalize to "LargeUtf8"
        let type_json = b"\"LargeUtf8\"";

        // Data: "ab" → len=2 as u64 LE + UTF-8 bytes
        let mut data_digest = Sha256::new();
        data_digest.update(2_u64.to_le_bytes());
        data_digest.update(b"ab");
        let data_finalized = data_digest.finalize();

        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        final_digest.update(data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&small),
            expected,
            "Example F: Utf8 hash mismatch"
        );
        assert_eq!(
            ArrowDigester::hash_array(&large),
            expected,
            "Example F: LargeUtf8 hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example G: Nullable Int32 Array with Nulls (hash_array API)
    //   Int32Array [Some(42), None, Some(-7), Some(0)]
    //   Tests nullable fixed-size path with actual nulls.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_g_nullable_int32_with_nulls() {
        let array = Int32Array::from(vec![Some(42), None, Some(-7), Some(0)]);

        // ── Type metadata ────────────────────────────────────────────────
        let type_json = b"\"Int32\"";

        // ── Validity bits (Lsb0, u8) ──────────────────────────────────
        // [valid, null, valid, valid] → bits [1, 0, 1, 1] → 0b1101 = 13
        let bit_count: u64 = 4;
        let validity_word: u8 = 0b1101; // 13

        // ── Data (only valid elements, in order) ─────────────────────────
        // 42 as i32 LE:  2a 00 00 00
        // -7 as i32 LE:  f9 ff ff ff
        //  0 as i32 LE:  00 00 00 00
        let mut data_digest = Sha256::new();
        data_digest.update(42_i32.to_le_bytes());
        data_digest.update((-7_i32).to_le_bytes());
        data_digest.update(0_i32.to_le_bytes());
        let data_finalized = data_digest.finalize();

        // ── Final (nullable) ─────────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        final_digest.update(bit_count.to_le_bytes());
        final_digest.update(validity_word.to_le_bytes());
        final_digest.update(data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            expected,
            "Example G: nullable int32 array hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example H: Nullable String Array with Nulls (hash_array API)
    //   StringArray [Some("hello"), None, Some("world"), Some("")]
    //   Tests nullable variable-length path with type canonicalization.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_h_nullable_string_array_with_nulls() {
        let array = StringArray::from(vec![Some("hello"), None, Some("world"), Some("")]);

        // ── Type metadata (canonicalized) ────────────────────────────────
        // Utf8 → LargeUtf8
        let type_json = b"\"LargeUtf8\"";

        // ── Validity bits (Lsb0, u8) ──────────────────────────────────
        // [valid, null, valid, valid] → bits [1, 0, 1, 1] → 0b1101 = 13
        let bit_count: u64 = 4;
        let validity_word: u8 = 0b1101;

        // ── Data (only valid elements) ───────────────────────────────────
        // "hello" → len=5 u64 LE + "hello"
        // "world" → len=5 u64 LE + "world"
        // ""      → len=0 u64 LE
        let mut data_digest = Sha256::new();
        data_digest.update(5_u64.to_le_bytes());
        data_digest.update(b"hello");
        // NULL: skipped
        data_digest.update(5_u64.to_le_bytes());
        data_digest.update(b"world");
        data_digest.update(0_u64.to_le_bytes());
        let data_finalized = data_digest.finalize();

        // ── Final (nullable) ─────────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        final_digest.update(bit_count.to_le_bytes());
        final_digest.update(validity_word.to_le_bytes());
        final_digest.update(data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            expected,
            "Example H: nullable string array hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example I: Empty Table (schema only, no data)
    //   Tests that finalize() on a fresh digester with no update() calls
    //   produces schema_digest + empty field digests.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_i_empty_table() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Boolean, true),
        ]);

        // ── Schema digest ────────────────────────────────────────────────
        let schema_json = r#"{"a":{"data_type":"Int32","nullable":false},"b":{"data_type":"Boolean","nullable":true}}"#;
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        // ── Field "a" (Int32, non-nullable): no data fed ─────────────────
        // data_digest = SHA-256() with no updates → SHA-256 of empty input
        let a_data_finalized = Sha256::digest(b"");

        // ── Field "b" (Boolean, nullable): no data fed ───────────────────
        // bit_count = 0 (no elements)
        // as_raw_slice() = [] (no words)
        // data_digest = SHA-256 of empty input
        let bit_count: u64 = 0;
        let b_data_finalized = Sha256::digest(b"");

        // ── Final ────────────────────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        // Field "a" (non-nullable)
        final_digest.update(a_data_finalized);
        // Field "b" (nullable) — bit_count=0, no words, empty data digest
        final_digest.update(bit_count.to_le_bytes());
        // no validity words (raw_slice is empty for 0-length BitVec)
        final_digest.update(b_data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        let digester = ArrowDigester::new(&schema);
        assert_eq!(
            digester.finalize(),
            expected,
            "Example I: empty table hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example J: Multi-Batch Streaming
    //   Feeding two small batches must produce the same hash as feeding
    //   one combined batch (batch-split independence).
    //   Schema: {v: Int32 non-nullable}
    //   Batch 1: [1, 2]
    //   Batch 2: [3]
    //   Combined: [1, 2, 3]
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_j_multi_batch_streaming() {
        let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

        // ── Two-batch path ───────────────────────────────────────────────
        let batch1 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![1_i32, 2])) as ArrayRef],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![3_i32])) as ArrayRef],
        )
        .unwrap();

        let mut digester_stream = ArrowDigester::new(&schema);
        digester_stream.update(&batch1);
        digester_stream.update(&batch2);
        let hash_stream = digester_stream.finalize();

        // ── Single-batch path ────────────────────────────────────────────
        let combined = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int32Array::from(vec![1_i32, 2, 3])) as ArrayRef],
        )
        .unwrap();
        let hash_combined = ArrowDigester::hash_record_batch(&combined);

        assert_eq!(
            hash_stream, hash_combined,
            "Streaming two batches should equal single combined batch"
        );

        // ── Manual computation ───────────────────────────────────────────
        let schema_json = r#"{"v":{"data_type":"Int32","nullable":false}}"#;
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        // Field "v": data is [1, 2, 3] as i32 LE — accumulated across batches
        // The digester is streaming, so it updates the same SHA-256 state:
        //   update(01 00 00 00  02 00 00 00)  from batch 1
        //   update(03 00 00 00)               from batch 2
        // SHA-256 is incremental, so this is identical to hashing all 12 bytes at once.
        let mut v_data = Sha256::new();
        v_data.update(1_i32.to_le_bytes());
        v_data.update(2_i32.to_le_bytes());
        v_data.update(3_i32.to_le_bytes());
        let v_finalized = v_data.finalize();

        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        final_digest.update(v_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            hash_stream, expected,
            "Example J: multi-batch streaming hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example K: Struct Column in a Record Batch
    //   Schema: {person: Struct<age: Int32 non-null, name: LargeUtf8 non-null> non-nullable}
    //   Row 0: {age: 25, name: "Alice"}
    //   Row 1: {age: 30, name: "Bob"}
    //
    //   In the record-batch path, struct fields are decomposed into leaf
    //   fields: "person/age" and "person/name", each hashed independently.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_k_struct_column_in_record_batch() {
        // ── Build the table ──────────────────────────────────────────────
        let age = Arc::new(Int32Array::from(vec![25_i32, 30])) as ArrayRef;
        let name = Arc::new(LargeStringArray::from(vec!["Alice", "Bob"])) as ArrayRef;
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("age", DataType::Int32, false)),
                Arc::clone(&age),
            ),
            (
                Arc::new(Field::new("name", DataType::LargeUtf8, false)),
                Arc::clone(&name),
            ),
        ]);

        let schema = Schema::new(vec![Field::new(
            "person",
            DataType::Struct(
                vec![
                    Field::new("age", DataType::Int32, false),
                    Field::new("name", DataType::LargeUtf8, false),
                ]
                .into(),
            ),
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(struct_array) as ArrayRef],
        )
        .unwrap();

        // ── Step 1: Schema digest ────────────────────────────────────────
        // Canonical JSON: struct fields sorted by name, keys sorted recursively
        // "person" has data_type: {"Struct": [{"data_type": "Int32", "name": "age", "nullable": false},
        //                                     {"data_type": "LargeUtf8", "name": "name", "nullable": false}]}
        let schema_json = r#"{"person":{"data_type":{"Struct":[{"data_type":"Int32","name":"age","nullable":false},{"data_type":"LargeUtf8","name":"name","nullable":false}]},"nullable":false}}"#;
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        assert_eq!(
            ArrowDigester::hash_schema(&schema),
            with_version(schema_digest.to_vec()),
            "Example K: schema hash mismatch"
        );

        // ── Step 2: Leaf field "person/age" (Int32, non-nullable) ────────
        // Values: [25, 30] as i32 LE
        let mut age_data = Sha256::new();
        age_data.update(25_i32.to_le_bytes());
        age_data.update(30_i32.to_le_bytes());
        let age_data_finalized = age_data.finalize();

        // ── Step 3: Leaf field "person/name" (LargeUtf8, non-nullable) ───
        // Values: ["Alice", "Bob"]
        let mut name_data = Sha256::new();
        name_data.update(5_u64.to_le_bytes()); // "Alice" length
        name_data.update(b"Alice");
        name_data.update(3_u64.to_le_bytes()); // "Bob" length
        name_data.update(b"Bob");
        let name_data_finalized = name_data.finalize();

        // ── Step 4: Final combination ────────────────────────────────────
        // Fields alphabetically: "person/age", "person/name"
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        // "person/age" (non-nullable): just data digest
        final_digest.update(age_data_finalized);
        // "person/name" (non-nullable): just data digest
        final_digest.update(name_data_finalized);

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_record_batch(&batch),
            expected,
            "Example K: struct column record batch hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example L: Struct Array via hash_array (non-nullable struct)
    //   StructArray [{a: 1, b: true}, {a: 2, b: false}]
    //   Children: a: Int32 non-null, b: Boolean non-null
    //
    //   In hash_array, the struct is hashed compositely:
    //   type_json + data where data = finalized(child_a) || finalized(child_b)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_l_struct_array_hash_array() {
        let a = Arc::new(Int32Array::from(vec![1_i32, 2])) as ArrayRef;
        let b = Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef;
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::clone(&a),
            ),
            (
                Arc::new(Field::new("b", DataType::Boolean, false)),
                Arc::clone(&b),
            ),
        ]);

        // ── Type metadata ────────────────────────────────────────────────
        // Canonical: {"Struct":[{"data_type":"Int32","name":"a","nullable":false},
        //                       {"data_type":"Boolean","name":"b","nullable":false}]}
        let type_json = r#"{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"Boolean","name":"b","nullable":false}]}"#;

        // ── Decomposition ────────────────────────────────────────────────
        // Struct is transparent: no BTreeMap entry for the struct itself.
        // Children become separate entries, finalized directly into the
        // final digest (no parent_data wrapper).
        //
        // BTreeMap entries (sorted by key): "a", "b"

        // ── Entry "a" (Int32, non-nullable) ──────────────────────────────
        // data = SHA256(1_i32_le, 2_i32_le)
        let mut data_a = Sha256::new();
        data_a.update(1_i32.to_le_bytes());
        data_a.update(2_i32.to_le_bytes());

        // ── Entry "b" (Boolean, non-nullable) ────────────────────────────
        // Values: [true, false] → Lsb0: bit0=1(true), bit1=0(false) → 0x01
        let mut data_b = Sha256::new();
        data_b.update([0x01_u8]);

        // ── Final combination ────────────────────────────────────────────
        // type_json → finalize_digest("a") → finalize_digest("b")
        // Each entry: non-nullable → no null_bits, no structural, just data.finalize()
        let mut final_digest = Sha256::new();
        final_digest.update(type_json.as_bytes());
        final_digest.update(data_a.finalize());
        final_digest.update(data_b.finalize());

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&struct_array),
            expected,
            "Example L: struct array hash_array mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example M: Nullable Struct Array via hash_array (struct-level nulls)
    //   StructArray [Some({a: 10, b: "x"}), None, Some({a: 30, b: "z"})]
    //   Struct is nullable. Children: a: Int32 non-null, b: LargeUtf8 non-null
    //
    //   Struct-level nulls propagate to children: at row 1 (null struct),
    //   children's data is undefined and must be skipped.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_m_nullable_struct_array_hash_array() {
        // Build a nullable struct array with a null at row 1
        let a = Int32Array::from(vec![10_i32, 0, 30]); // row 1 value is undefined (0 placeholder)
        let b = LargeStringArray::from(vec!["x", "", "z"]); // row 1 value is undefined
        let struct_array = StructArray::from((
            vec![
                (
                    Arc::new(Field::new("a", DataType::Int32, false)),
                    Arc::new(a) as ArrayRef,
                ),
                (
                    Arc::new(Field::new("b", DataType::LargeUtf8, false)),
                    Arc::new(b) as ArrayRef,
                ),
            ],
            // Struct-level validity: [valid, null, valid]
            NullBuffer::from(vec![true, false, true])
                .into_inner()
                .into_inner(),
        ));

        // ── Type metadata ────────────────────────────────────────────────
        let type_json = r#"{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"LargeUtf8","name":"b","nullable":false}]}"#;

        // ── Decomposition ────────────────────────────────────────────────
        // Struct is transparent: no BTreeMap entry. Struct-level nulls
        // [1, 0, 1] are AND-propagated to children for data hashing.
        // Children "a" and "b" are non-nullable per their Field definitions,
        // so their entries have no null_bits — but null rows are skipped
        // in the data stream.
        //
        // BTreeMap entries (sorted by key): "a", "b"

        // ── Entry "a" (Int32, non-nullable) ──────────────────────────────
        // Struct nulls propagated: rows 0,2 valid → data = [10, 30]
        let mut data_a = Sha256::new();
        data_a.update(10_i32.to_le_bytes());
        // row 1: skipped (struct null)
        data_a.update(30_i32.to_le_bytes());

        // ── Entry "b" (LargeUtf8, non-nullable) ─────────────────────────
        // Struct nulls propagated: rows 0,2 valid → data = ["x", "z"]
        let mut data_b = Sha256::new();
        data_b.update(1_u64.to_le_bytes()); // "x" len
        data_b.update(b"x");
        // row 1: skipped (struct null)
        data_b.update(1_u64.to_le_bytes()); // "z" len
        data_b.update(b"z");

        // ── Final combination ────────────────────────────────────────────
        // type_json → finalize_digest("a") → finalize_digest("b")
        // Each entry: non-nullable → no null_bits, no structural, just data.finalize()
        let mut final_digest = Sha256::new();
        final_digest.update(type_json.as_bytes());
        final_digest.update(data_a.finalize());
        final_digest.update(data_b.finalize());

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&struct_array),
            expected,
            "Example M: nullable struct array hash_array mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example N: List-of-Struct in a Record Batch
    //   Schema: {items: LargeList<Struct<id: Int32 non-null, label: LargeUtf8 non-null>> nullable}
    //   Row 0: [{id: 1, label: "a"}, {id: 2, label: "b"}]   (2 elements)
    //   Row 1: [{id: 3, label: "c"}]                          (1 element)
    //
    //   Recursively decomposed into separate BTreeMap entries:
    //   "items"       → validity-only (null_bits: [V, V])
    //   "items/"      → structural-only (list lengths: [2, 1])
    //   "items//id"   → data-only ([1, 2, 3] as i32 LE)
    //   "items//label"→ data-only (["a", "b", "c"] as LargeUtf8)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_n_list_of_struct_record_batch() {
        // ── Build the table ──────────────────────────────────────────────
        let struct_fields = vec![
            Field::new("id", DataType::Int32, false),
            Field::new("label", DataType::LargeUtf8, false),
        ];
        let inner_struct_field = Field::new(
            "item",
            DataType::Struct(struct_fields.clone().into()),
            false,
        );
        let list_field = Field::new(
            "items",
            DataType::LargeList(Arc::new(inner_struct_field.clone())),
            true,
        );
        let schema = Schema::new(vec![list_field.clone()]);

        // Build struct sub-arrays
        // Row 0: [{id:1, label:"a"}, {id:2, label:"b"}], Row 1: [{id:3, label:"c"}]
        // Total struct rows: 3 (ids: [1,2,3], labels: ["a","b","c"])
        let ids = Int32Array::from(vec![1_i32, 2, 3]);
        let labels = LargeStringArray::from(vec!["a", "b", "c"]);
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("id", DataType::Int32, false)),
                Arc::new(ids) as ArrayRef,
            ),
            (
                Arc::new(Field::new("label", DataType::LargeUtf8, false)),
                Arc::new(labels) as ArrayRef,
            ),
        ]);

        // Build large list array with offsets [0, 2, 3]
        let list_array = LargeListArray::new(
            Arc::new(inner_struct_field),
            arrow::buffer::OffsetBuffer::new(vec![0_i64, 2, 3].into()),
            Arc::new(struct_array) as ArrayRef,
            None, // all list elements valid
        );

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(list_array) as ArrayRef],
        )
        .unwrap();

        // ── Step 1: Schema digest ────────────────────────────────────────
        // Canonical: element type has no name (element_type_to_value drops "item")
        // The inner struct's data_type is {"Struct": [sorted children]}
        let schema_json = r#"{"items":{"data_type":{"LargeList":{"data_type":{"Struct":[{"data_type":"Int32","name":"id","nullable":false},{"data_type":"LargeUtf8","name":"label","nullable":false}]},"nullable":false}},"nullable":true}}"#;
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        assert_eq!(
            ArrowDigester::hash_schema(&schema),
            with_version(schema_digest.to_vec()),
            "Example N: schema hash mismatch"
        );

        // ── Step 2: Recursive decomposition ──────────────────────────────
        //
        // With recursive list/struct decomposition, entries are (sorted):
        //   "items"     → validity-only: null_bits [V, V] (2 bits, both valid)
        //   "items/"    → structural-only: list lengths [2, 1]
        //   "items//id" → data-only: [1, 2, 3] as i32 LE
        //   "items//label" → data-only: ["a", "b", "c"] as LargeUtf8

        // ── Step 3: Final combination ────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);

        // Entry "items": null_bits V,V → bit_count=2, validity=0b11=3
        final_digest.update(2_u64.to_le_bytes());
        final_digest.update(3_u8.to_le_bytes());

        // Entry "items/": structural [2, 1]
        let mut items_structural = Sha256::new();
        items_structural.update(2_u64.to_le_bytes());
        items_structural.update(1_u64.to_le_bytes());
        final_digest.update(items_structural.finalize());

        // Entry "items//id": data [1, 2, 3] as i32 LE
        let mut id_data = Sha256::new();
        id_data.update(1_i32.to_le_bytes());
        id_data.update(2_i32.to_le_bytes());
        id_data.update(3_i32.to_le_bytes());
        final_digest.update(id_data.finalize());

        // Entry "items//label": data ["a", "b", "c"] as LargeUtf8
        let mut label_data = Sha256::new();
        label_data.update(1_u64.to_le_bytes());
        label_data.update(b"a");
        label_data.update(1_u64.to_le_bytes());
        label_data.update(b"b");
        label_data.update(1_u64.to_le_bytes());
        label_data.update(b"c");
        final_digest.update(label_data.finalize());

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_record_batch(&batch),
            expected,
            "Example N: list-of-struct record batch hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example O: Nested Struct in a Record Batch (two levels of struct)
    //   Schema: {s: Struct<a: Int32 non-null,
    //                      nested: Struct<p: Int32 non-null,
    //                                    q: Int32 non-null> non-null>
    //                non-null}
    //   Row 0: s.a=10, s.nested.p=20, s.nested.q=30
    //
    //   Struct is transparent at both levels. BTreeMap entries (sorted):
    //     "s/a"         → data-only  (Int32: [10])
    //     "s/nested/p"  → data-only  (Int32: [20])
    //     "s/nested/q"  → data-only  (Int32: [30])
    //
    //   This test verifies that the recursive struct decomposition produces
    //   the correct leaf paths and that each leaf is hashed with the right bytes.
    // ══════════════════════════════════════════════════════════════════════

    #[expect(
        clippy::similar_names,
        reason = "variable names mirror spec path notation (snp/snq)"
    )]
    #[test]
    fn example_o_nested_struct_record_batch() {
        // ── Build the table ──────────────────────────────────────────────
        let a_arr = Arc::new(Int32Array::from(vec![10_i32])) as ArrayRef;
        let p_arr = Arc::new(Int32Array::from(vec![20_i32])) as ArrayRef;
        let q_arr = Arc::new(Int32Array::from(vec![30_i32])) as ArrayRef;

        let nested = StructArray::from(vec![
            (
                Arc::new(Field::new("p", DataType::Int32, false)),
                Arc::clone(&p_arr),
            ),
            (
                Arc::new(Field::new("q", DataType::Int32, false)),
                Arc::clone(&q_arr),
            ),
        ]);
        let nested_type = DataType::Struct(nested.fields().clone());

        let outer = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::clone(&a_arr),
            ),
            (
                Arc::new(Field::new("nested", nested_type.clone(), false)),
                Arc::new(nested) as ArrayRef,
            ),
        ]);
        let outer_type = DataType::Struct(outer.fields().clone());

        let schema = Schema::new(vec![Field::new("s", outer_type, false)]);
        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(outer) as ArrayRef])
                .unwrap();

        // ── Step 1: Schema digest ────────────────────────────────────────
        // Outer struct children sorted: [a, nested]
        // Inner struct (nested) children sorted: [p, q]
        // Canonical JSON (all object keys sorted alphabetically):
        let schema_json = concat!(
            r#"{"s":{"data_type":{"Struct":["#,
            r#"{"data_type":"Int32","name":"a","nullable":false},"#,
            r#"{"data_type":{"Struct":["#,
            r#"{"data_type":"Int32","name":"p","nullable":false},"#,
            r#"{"data_type":"Int32","name":"q","nullable":false}"#,
            r#"]},"name":"nested","nullable":false}"#,
            r#"]},"nullable":false}}"#
        );
        let schema_digest = Sha256::digest(schema_json.as_bytes());

        assert_eq!(
            ArrowDigester::hash_schema(&schema),
            with_version(schema_digest.to_vec()),
            "Example O: schema hash mismatch — check canonical JSON"
        );

        // ── Step 2: Leaf entries (alphabetical order of full path) ────────
        //
        // "s/a"  (Int32, non-nullable) → data = SHA-256(10_i32_le)
        let mut data_sa = Sha256::new();
        data_sa.update(10_i32.to_le_bytes()); // 0a 00 00 00
        let data_sa_finalized = data_sa.finalize();

        // "s/nested/p"  (Int32, non-nullable) → data = SHA-256(20_i32_le)
        let mut data_snp = Sha256::new();
        data_snp.update(20_i32.to_le_bytes()); // 14 00 00 00
        let data_snp_finalized = data_snp.finalize();

        // "s/nested/q"  (Int32, non-nullable) → data = SHA-256(30_i32_le)
        let mut data_snq = Sha256::new();
        data_snq.update(30_i32.to_le_bytes()); // 1e 00 00 00
        let data_snq_finalized = data_snq.finalize();

        // ── Step 3: Final combination ────────────────────────────────────
        // schema_digest → "s/a" → "s/nested/p" → "s/nested/q"
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        final_digest.update(data_sa_finalized); // "s/a" (non-nullable, data-only)
        final_digest.update(data_snp_finalized); // "s/nested/p" (non-nullable, data-only)
        final_digest.update(data_snq_finalized); // "s/nested/q" (non-nullable, data-only)

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_record_batch(&batch),
            expected,
            "Example O: nested struct record batch hash mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example P: Nested Struct Field-Order Independence (schema hash)
    //
    //   Schema 1: {s: Struct<nested: Struct<b: Int32, a: Int32>, z: Int32>}
    //   Schema 2: {s: Struct<z: Int32, nested: Struct<a: Int32, b: Int32>>}
    //
    //   Both have the same logical structure. After recursive alphabetical
    //   sorting the canonical JSON is identical, so hash_schema must agree.
    //
    //   This test pins the expected canonical JSON string so that any
    //   regression in the recursive-ordering logic immediately fails here
    //   with a clear diff.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn example_p_nested_struct_field_order_independence() {
        // Schema 1: outer fields [nested, z], inner fields [b, a]  (neither sorted)
        let schema1 = Schema::new(vec![Field::new(
            "s",
            DataType::Struct(
                vec![
                    Field::new(
                        "nested",
                        DataType::Struct(
                            vec![
                                Field::new("b", DataType::Int32, false),
                                Field::new("a", DataType::Int32, false),
                            ]
                            .into(),
                        ),
                        false,
                    ),
                    Field::new("z", DataType::Int32, false),
                ]
                .into(),
            ),
            false,
        )]);

        // Schema 2: outer fields [z, nested], inner fields [a, b]  (both sorted)
        let schema2 = Schema::new(vec![Field::new(
            "s",
            DataType::Struct(
                vec![
                    Field::new("z", DataType::Int32, false),
                    Field::new(
                        "nested",
                        DataType::Struct(
                            vec![
                                Field::new("a", DataType::Int32, false),
                                Field::new("b", DataType::Int32, false),
                            ]
                            .into(),
                        ),
                        false,
                    ),
                ]
                .into(),
            ),
            false,
        )]);

        // ── Expected canonical JSON ──────────────────────────────────────
        // After recursive alphabetical sorting:
        //   outer children sorted: [nested, z]   (n < z)
        //   inner children sorted: [a, b]         (a < b)
        let canonical_json = concat!(
            r#"{"s":{"data_type":{"Struct":["#,
            r#"{"data_type":{"Struct":["#,
            r#"{"data_type":"Int32","name":"a","nullable":false},"#,
            r#"{"data_type":"Int32","name":"b","nullable":false}"#,
            r#"]},"name":"nested","nullable":false},"#,
            r#"{"data_type":"Int32","name":"z","nullable":false}"#,
            r#"]},"nullable":false}}"#
        );
        let expected_schema_digest = Sha256::digest(canonical_json.as_bytes());
        let expected = with_version(expected_schema_digest.to_vec());

        // ── Both schemas must produce that same hash ──────────────────────
        let hash1 = ArrowDigester::hash_schema(&schema1);
        let hash2 = ArrowDigester::hash_schema(&schema2);

        assert_eq!(
            hash1, expected,
            "Example P: schema1 (fields out-of-order) hash mismatch — recursive ordering broken"
        );
        assert_eq!(
            hash2, expected,
            "Example P: schema2 (fields in-order) hash mismatch"
        );
        assert_eq!(hash1, hash2, "Example P: schema1 and schema2 must be equal");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example Q: Nested Struct via hash_array
    //   StructArray [{inner: {x: 5, y: 7}}, {inner: {x: 9, y: 11}}]
    //   Outer struct: non-nullable, one child "inner" (non-nullable Struct)
    //   Inner struct: non-nullable, children x: Int32, y: Int32
    //
    //   hash_array entry point:
    //     1. type_json = canonical data_type_to_value of the outer struct
    //     2. BTreeMap entries (sorted by path): "inner/x", "inner/y"
    //     3. Finalize each entry into the final digest
    //
    //   This is the first byte-exact test of hash_array with a nested struct,
    //   complementing Example O (which exercises hash_record_batch).
    // ══════════════════════════════════════════════════════════════════════

    #[expect(
        clippy::similar_names,
        reason = "variable names mirror spec field notation (x/y)"
    )]
    #[test]
    fn example_q_nested_struct_hash_array() {
        // ── Build the array ──────────────────────────────────────────────
        let x_arr = Arc::new(Int32Array::from(vec![5_i32, 9])) as ArrayRef;
        let y_arr = Arc::new(Int32Array::from(vec![7_i32, 11])) as ArrayRef;

        let inner = StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::Int32, false)),
                Arc::clone(&x_arr),
            ),
            (
                Arc::new(Field::new("y", DataType::Int32, false)),
                Arc::clone(&y_arr),
            ),
        ]);
        let inner_type = DataType::Struct(inner.fields().clone());

        let outer = StructArray::from(vec![(
            Arc::new(Field::new("inner", inner_type.clone(), false)),
            Arc::new(inner) as ArrayRef,
        )]);

        // ── Step 1: Type metadata ────────────────────────────────────────
        // Outer struct has one child "inner" whose data_type is
        // Struct([x: Int32, y: Int32]) — inner children already alphabetical.
        //
        // Canonical JSON (outer struct children sorted: ["inner"]; inner sorted: ["x","y"]):
        let type_json = concat!(
            r#"{"Struct":["#,
            r#"{"data_type":{"Struct":["#,
            r#"{"data_type":"Int32","name":"x","nullable":false},"#,
            r#"{"data_type":"Int32","name":"y","nullable":false}"#,
            r#"]},"name":"inner","nullable":false}"#,
            r#"]}"#
        );

        // ── Step 2: Decomposed entries (sorted by path) ──────────────────
        //
        // "inner/x" (Int32, non-nullable) → data = SHA-256([5, 9] as i32 LE)
        let mut data_x = Sha256::new();
        data_x.update(5_i32.to_le_bytes()); // 05 00 00 00
        data_x.update(9_i32.to_le_bytes()); // 09 00 00 00
        let data_x_finalized = data_x.finalize();

        // "inner/y" (Int32, non-nullable) → data = SHA-256([7, 11] as i32 LE)
        let mut data_y = Sha256::new();
        data_y.update(7_i32.to_le_bytes()); // 07 00 00 00
        data_y.update(11_i32.to_le_bytes()); // 0b 00 00 00
        let data_y_finalized = data_y.finalize();

        // ── Step 3: Final combination ────────────────────────────────────
        // type_json → "inner/x" (non-nullable) → "inner/y" (non-nullable)
        let mut final_digest = Sha256::new();
        final_digest.update(type_json.as_bytes());
        final_digest.update(data_x_finalized); // "inner/x"
        final_digest.update(data_y_finalized); // "inner/y"

        let expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&outer),
            expected,
            "Example Q: nested struct hash_array mismatch"
        );
    }
}
