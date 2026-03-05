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
        clippy::big_endian_bytes,
        reason = "Starfix spec requires BE serialization of validity words"
    )]

    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BinaryArray, BooleanArray, Int32Array, LargeStringArray, RecordBatch,
        StringArray,
    };
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
        let schema_json =
            r#"{"age":{"data_type":"Int32","nullable":false},"name":{"data_type":"LargeUtf8","nullable":true}}"#;
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
        let schema_json =
            r#"{"x":{"data_type":"Int32","nullable":false},"y":{"data_type":"Boolean","nullable":true}}"#;
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
        let schema_json =
            r#"{"a":{"data_type":"Int32","nullable":false},"b":{"data_type":"Boolean","nullable":true}}"#;
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

        let digester = ArrowDigester::new(schema);
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

        let mut digester_stream = ArrowDigester::new(schema.clone());
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
}
