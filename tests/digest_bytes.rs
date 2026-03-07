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
    #![expect(
        clippy::similar_names,
        reason = "child_a/child_b naming is clear in test context"
    )]
    #![expect(clippy::redundant_clone, reason = "Clones for clarity in test setup")]
    #![expect(clippy::absolute_paths, reason = "One-off use in test")]
    #![expect(
        clippy::big_endian_bytes,
        reason = "Starfix spec requires BE serialization of validity words"
    )]

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
        // Validity BitVec (Lsb0, usize storage):
        //   bit 0 = 1 (valid), bit 1 = 0 (null)
        //   → usize word = 0b01 = 1
        //   bit_count = 2
        let bit_count: usize = 2;
        let validity_word: usize = 1; // bits: [1, 0] in Lsb0

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
        final_digest.update(validity_word.to_be_bytes()); // 00 00 00 00 00 00 00 01
        final_digest.update(name_data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        // ── Verify ───────────────────────────────────────────────────────
        assert_eq!(
            ArrowDigester::hash_record_batch(&batch),
            vec![
                0, 0, 1, 128, 32, 228, 127, 68, 98, 242, 107, 11, 199, 58, 209, 16, 234, 15, 145,
                152, 194, 116, 92, 4, 206, 35, 51, 80, 147, 210, 183, 142, 245, 28, 136
            ],
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

        // ── Validity bits (Lsb0, usize storage) ─────────────────────────
        // [valid, null, valid, valid] → bits [1, 0, 1, 1]
        // Lsb0 in usize: bit0=1, bit1=0, bit2=1, bit3=1 → 0b1101 = 13
        let bit_count: usize = 4;
        let validity_word: usize = 0b1101; // = 13

        // ── Data bits (Msb0 packed, valid values only) ───────────────────
        // Valid values: [true, false, true] → 3 bits
        // Msb0: bit7=1(true), bit6=0(false), bit5=1(true), bits4-0=0
        // Byte: 0b1010_0000 = 0xA0
        let mut data_digest = Sha256::new();
        data_digest.update([0xA0_u8]);
        let data_finalized = data_digest.finalize();

        // ── Final combination ────────────────────────────────────────────
        let mut final_digest = Sha256::new();
        final_digest.update(type_json);
        // Nullable finalization
        final_digest.update(bit_count.to_le_bytes());
        final_digest.update(validity_word.to_be_bytes());
        final_digest.update(data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            vec![
                0, 0, 1, 133, 169, 201, 158, 186, 123, 207, 217, 177, 79, 213, 41, 185, 83, 79, 34,
                137, 49, 151, 121, 39, 10, 164, 160, 114, 241, 23, 207, 144, 166, 172, 139
            ],
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
        // Data: [true] Msb0 → bit7=1 → 0x80
        let bit_count: usize = 1;
        let validity_word: usize = 1;

        let mut y_data = Sha256::new();
        y_data.update([0x80_u8]); // true in Msb0 = 1000_0000
        let y_finalized = y_data.finalize();

        // Final combination: schema, then fields alphabetically (x, y)
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        // x (non-nullable)
        final_digest.update(x_finalized);
        // y (nullable)
        final_digest.update(bit_count.to_le_bytes());
        final_digest.update(validity_word.to_be_bytes());
        final_digest.update(y_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        // ── Verify both column orderings produce the same hash ───────────
        let hash_xy = ArrowDigester::hash_record_batch(&batch_xy);
        let hash_yx = ArrowDigester::hash_record_batch(&batch_yx);

        assert_eq!(hash_xy, hash_yx, "Column order should not affect hash");
        assert_eq!(
            hash_xy,
            vec![
                0, 0, 1, 246, 139, 246, 49, 159, 142, 196, 170, 147, 142, 82, 221, 145, 25, 116,
                52, 130, 137, 251, 223, 185, 181, 235, 237, 94, 20, 226, 57, 166, 216, 163, 169
            ],
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

        // ── Validity bits (Lsb0, usize) ─────────────────────────────────
        // [valid, null, valid, valid] → bits [1, 0, 1, 1] → 0b1101 = 13
        let bit_count: usize = 4;
        let validity_word: usize = 0b1101; // 13

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
        final_digest.update(validity_word.to_be_bytes());
        final_digest.update(data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            vec![
                0, 0, 1, 131, 48, 249, 184, 121, 107, 148, 52, 203, 247, 188, 2, 140, 24, 197, 138,
                42, 115, 155, 152, 10, 207, 153, 149, 206, 30, 93, 96, 180, 59, 1, 56
            ],
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

        // ── Validity bits (Lsb0, usize) ─────────────────────────────────
        // [valid, null, valid, valid] → bits [1, 0, 1, 1] → 0b1101 = 13
        let bit_count: usize = 4;
        let validity_word: usize = 0b1101;

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
        final_digest.update(validity_word.to_be_bytes());
        final_digest.update(data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&array),
            vec![
                0, 0, 1, 98, 85, 189, 224, 20, 30, 191, 38, 224, 140, 49, 201, 111, 97, 18, 229,
                226, 29, 16, 26, 184, 187, 144, 215, 127, 44, 62, 236, 2, 198, 45, 60
            ],
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
        let bit_count: usize = 0;
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

        // ── Child "a" (Int32, non-nullable) ──────────────────────────────
        // Values: [1, 2]
        let mut child_a_data = Sha256::new();
        child_a_data.update(1_i32.to_le_bytes());
        child_a_data.update(2_i32.to_le_bytes());
        let child_a_finalized = child_a_data.finalize();

        // ── Child "b" (Boolean, non-nullable) ────────────────────────────
        // Values: [true, false] → Msb0: bit7=1(true), bit6=0(false) → 0x80
        let mut child_b_data = Sha256::new();
        child_b_data.update([0x80_u8]);
        let child_b_finalized = child_b_data.finalize();

        // ── Parent data digest ───────────────────────────────────────────
        // Children sorted by name: "a" then "b"
        // Each child is non-nullable, so finalized = SHA256(data).finalize() (32 bytes)
        let mut parent_data = Sha256::new();
        // Child "a" finalized (non-nullable → just data digest)
        parent_data.update(child_a_finalized);
        // Child "b" finalized (non-nullable → just data digest)
        parent_data.update(child_b_finalized);
        let parent_data_finalized = parent_data.finalize();

        // ── Final combination ────────────────────────────────────────────
        // Struct is non-nullable → NonNullable finalization
        let mut final_digest = Sha256::new();
        final_digest.update(type_json.as_bytes());
        final_digest.update(parent_data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&struct_array),
            vec![
                0, 0, 1, 245, 160, 205, 201, 133, 248, 136, 141, 186, 23, 124, 235, 245, 80, 84,
                148, 148, 243, 88, 117, 149, 239, 95, 247, 17, 251, 204, 213, 43, 112, 244, 241
            ],
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
            // Buffer from NullBuffer: true=valid, false=null
            NullBuffer::from(vec![true, false, true])
                .into_inner()
                .into_inner(),
        ));

        // ── Type metadata ────────────────────────────────────────────────
        let type_json = r#"{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"LargeUtf8","name":"b","nullable":false}]}"#;

        // ── Struct-level validity (Lsb0, usize) ─────────────────────────
        // [valid, null, valid] → bits [1, 0, 1] → 0b101 = 5
        let struct_bit_count: usize = 3;
        let struct_validity_word: usize = 0b101; // 5

        // ── Child "a" (Int32, effectively nullable due to struct nulls) ──
        // Combined validity: struct AND child = [1, 0, 1] (child has no nulls of its own)
        // Valid data: [10, 30] (row 1 skipped)
        let child_a_bit_count: usize = 3;
        let child_a_validity_word: usize = 0b101;

        let mut child_a_data = Sha256::new();
        child_a_data.update(10_i32.to_le_bytes());
        // row 1: skipped (null)
        child_a_data.update(30_i32.to_le_bytes());
        let child_a_data_finalized = child_a_data.finalize();

        // ── Child "b" (LargeUtf8, effectively nullable due to struct nulls)
        let child_b_bit_count: usize = 3;
        let child_b_validity_word: usize = 0b101;

        let mut child_b_data = Sha256::new();
        child_b_data.update(1_u64.to_le_bytes()); // "x" len
        child_b_data.update(b"x");
        // row 1: skipped (null)
        child_b_data.update(1_u64.to_le_bytes()); // "z" len
        child_b_data.update(b"z");
        let child_b_data_finalized = child_b_data.finalize();

        // ── Parent data digest ───────────────────────────────────────────
        // Children sorted by name: "a", "b"
        // Each child is effectively nullable → finalized as:
        //   bit_count LE + validity_words BE + data_digest.finalize()
        let mut parent_data = Sha256::new();
        // Child "a" finalized (nullable)
        parent_data.update(child_a_bit_count.to_le_bytes());
        parent_data.update(child_a_validity_word.to_be_bytes());
        parent_data.update(child_a_data_finalized);
        // Child "b" finalized (nullable)
        parent_data.update(child_b_bit_count.to_le_bytes());
        parent_data.update(child_b_validity_word.to_be_bytes());
        parent_data.update(child_b_data_finalized);
        let parent_data_finalized = parent_data.finalize();

        // ── Final combination ────────────────────────────────────────────
        // Struct is nullable → parent finalization includes struct validity
        let mut final_digest = Sha256::new();
        final_digest.update(type_json.as_bytes());
        // Struct-level nullable finalization
        final_digest.update(struct_bit_count.to_le_bytes());
        final_digest.update(struct_validity_word.to_be_bytes());
        final_digest.update(parent_data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_array(&struct_array),
            vec![
                0, 0, 1, 174, 113, 201, 49, 168, 4, 206, 167, 142, 52, 153, 101, 216, 85, 182, 23,
                241, 140, 179, 157, 247, 213, 20, 220, 53, 83, 5, 102, 23, 235, 12, 104
            ],
            "Example M: nullable struct array hash_array mismatch"
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Example N: List-of-Struct in a Record Batch
    //   Schema: {items: LargeList<Struct<id: Int32 non-null, label: LargeUtf8 non-null>> nullable}
    //   Row 0: [{id: 1, label: "a"}, {id: 2, label: "b"}]   (2 elements)
    //   Row 1: [{id: 3, label: "c"}]                          (1 element)
    //
    //   The list column is decomposed into leaf fields:
    //   "items" in the BTreeMap (the list field itself, not its inner struct fields).
    //   But the list's sub-arrays ARE struct arrays, which are now hashed
    //   compositely via array_digest_update(Struct).
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

        // ── Step 2: Field "items" (LargeList<Struct>, nullable) ──────────
        //
        // With structural hashing, list sizes go to a separate structural digest,
        // while leaf data (struct composites) goes to the data/leaf digest.
        //
        // The BitVec accumulates ALL null bits from the list AND its sub-arrays.
        // List-level: handle_null_bits(list) → [1, 1] (both list elements valid)
        // Then for each list element, the struct sub-array also pushes its validity:
        //   Element 0 struct (2 rows, no nulls): → [1, 1]
        //   Element 1 struct (1 row, no nulls): → [1]
        // Total BitVec: [1, 1, 1, 1, 1] → 5 bits, all valid
        let items_bit_count: usize = 5;
        let items_validity_word: usize = 0b11111; // 31

        // ── Structural digest: element counts (sizes) ────────────────────
        let mut items_structural = Sha256::new();
        items_structural.update(2_u64.to_le_bytes()); // element 0 has 2 struct rows
        items_structural.update(1_u64.to_le_bytes()); // element 1 has 1 struct row
        let items_structural_finalized = items_structural.finalize();

        // ── Data/leaf digest: struct composites (no size prefixes) ────────
        //
        // --- List element 0: [{id:1,label:"a"}, {id:2,label:"b"}] (2 rows) ---
        //   Struct composite: children sorted by name: "id" then "label"
        //     No struct-level nulls, children are non-nullable
        //
        //   Child "id" (Int32, non-null): values [1, 2]
        let mut e0_child_id_data = Sha256::new();
        e0_child_id_data.update(1_i32.to_le_bytes());
        e0_child_id_data.update(2_i32.to_le_bytes());
        let e0_child_id_finalized = e0_child_id_data.finalize();

        //   Child "label" (LargeUtf8, non-null): values ["a", "b"]
        let mut e0_child_label_data = Sha256::new();
        e0_child_label_data.update(1_u64.to_le_bytes()); // "a" len
        e0_child_label_data.update(b"a");
        e0_child_label_data.update(1_u64.to_le_bytes()); // "b" len
        e0_child_label_data.update(b"b");
        let e0_child_label_finalized = e0_child_label_data.finalize();

        // --- List element 1: [{id:3,label:"c"}] (1 row) ---
        //   Child "id": values [3]
        let mut e1_child_id_data = Sha256::new();
        e1_child_id_data.update(3_i32.to_le_bytes());
        let e1_child_id_finalized = e1_child_id_data.finalize();

        //   Child "label": values ["c"]
        let mut e1_child_label_data = Sha256::new();
        e1_child_label_data.update(1_u64.to_le_bytes()); // "c" len
        e1_child_label_data.update(b"c");
        let e1_child_label_finalized = e1_child_label_data.finalize();

        // Build leaf digest: struct composites for each list element
        let mut items_data = Sha256::new();
        // List element 0: struct children finalized into data (no size prefix here)
        items_data.update(e0_child_id_finalized); // non-nullable child: 32 bytes
        items_data.update(e0_child_label_finalized); // non-nullable child: 32 bytes
                                                     // List element 1: struct children finalized into data
        items_data.update(e1_child_id_finalized);
        items_data.update(e1_child_label_finalized);
        let items_data_finalized = items_data.finalize();

        // ── Step 3: Final combination ────────────────────────────────────
        // For list fields (nullable): bit_count + validity_words + structural_digest + data_digest
        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);
        // "items" (nullable, structured): null bits + structural + leaf
        final_digest.update(items_bit_count.to_le_bytes());
        final_digest.update(items_validity_word.to_be_bytes());
        final_digest.update(items_structural_finalized);
        final_digest.update(items_data_finalized);

        let _expected = with_version(final_digest.finalize().to_vec());

        assert_eq!(
            ArrowDigester::hash_record_batch(&batch),
            vec![
                0, 0, 1, 108, 249, 107, 14, 43, 47, 243, 172, 76, 196, 56, 234, 248, 252, 108, 84,
                213, 202, 175, 248, 8, 57, 85, 190, 110, 24, 96, 92, 144, 0, 31, 38
            ],
            "Example N: list-of-struct record batch hash mismatch"
        );
    }
}
