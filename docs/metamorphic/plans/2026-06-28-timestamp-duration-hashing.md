# Timestamp and Duration Leaf Data Hashing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use sensei:subagent-driven-development (recommended) or sensei:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for `timestamp` and `duration` Arrow types in `ArrowDigester`'s leaf data hashing — replacing `todo!()` in Rust and a missing branch in Python — across both `nauticalab/starfix` and `nauticalab/starfix-python`.

**Architecture:** All `timestamp` and `duration` Arrow variants are physically stored as signed 64-bit integers (8 bytes). The fix in both repos treats them identically to `time64`: a fixed-size 8-byte element. The Rust fix is implemented first; its test output provides the authoritative golden hash values that the Python golden-parity tests assert against.

**Tech Stack:** Rust (arrow-rs, sha2, hex, pretty_assertions), Python (pyarrow ≥ 14, hashlib, pytest)

---

**Working directories:**

- Rust: `starfix/` (all `cargo` commands run here)
- Python: `starfix-python/` (all `pytest` commands run here)
- Branch in both repos: `eywalker/itl-438-add-support-for-timestamp-and-duration-types-in-leaf-data`

---

## Phase 1: Rust (`nauticalab/starfix`)

### Task 1: Add timestamp/duration imports to `tests/arrow_digester.rs`

**Files:**
- Modify: `tests/arrow_digester.rs` (import block at top of `mod tests`)

- [ ] **Step 1: Extend the existing arrow array import block**

Open `tests/arrow_digester.rs`. Find the `use arrow::{ array::{` import block (lines 6–17). Replace it with:

```rust
use arrow::{
    array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal32Array,
        Decimal64Array, DictionaryArray, DurationMicrosecondArray, DurationMillisecondArray,
        DurationNanosecondArray, DurationSecondArray, Float32Array, Float64Array, Int16Array,
        Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeListArray, LargeListBuilder,
        LargeStringArray, LargeStringBuilder, ListArray, ListBuilder, RecordBatch, StringArray,
        StringBuilder, StructArray, Time32MillisecondArray, Time32SecondArray,
        Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
        UInt32Array, UInt64Array, UInt8Array,
    },
    datatypes::{Int32Type, Int8Type},
};
```

- [ ] **Step 2: Verify the file compiles**

```bash
cargo check --tests 2>&1 | head -20
```

Expected: no errors (warnings about unused imports are fine at this stage).

---

### Task 2: Write failing unit tests for timestamp hashing

**Files:**
- Modify: `tests/arrow_digester.rs` (add 5 new test functions after the existing `time_array_different_units_produce_different_hashes` test)

- [ ] **Step 1: Add `timestamp_array_hashing` — prints hash values, asserts format only**

Insert after `time_array_different_units_produce_different_hashes` (around line 192):

```rust
#[test]
fn timestamp_array_hashing() {
    // [0 epoch, None, 1000 units] — raw i64 values; meaning depends on unit
    let values = vec![Some(0_i64), None, Some(1_000_i64)];

    // tz-aware UTC, all 4 units
    let h_s_utc = hex::encode(ArrowDigester::hash_array(
        &TimestampSecondArray::from(values.clone()).with_timezone("UTC"),
    ));
    println!("ts_s_utc:  {h_s_utc}");

    let h_ms_utc = hex::encode(ArrowDigester::hash_array(
        &TimestampMillisecondArray::from(values.clone()).with_timezone("UTC"),
    ));
    println!("ts_ms_utc: {h_ms_utc}");

    let h_us_utc = hex::encode(ArrowDigester::hash_array(
        &TimestampMicrosecondArray::from(values.clone()).with_timezone("UTC"),
    ));
    println!("ts_us_utc: {h_us_utc}");

    let h_ns_utc = hex::encode(ArrowDigester::hash_array(
        &TimestampNanosecondArray::from(values.clone()).with_timezone("UTC"),
    ));
    println!("ts_ns_utc: {h_ns_utc}");

    // tz-naive, all 4 units
    let h_s = hex::encode(ArrowDigester::hash_array(
        &TimestampSecondArray::from(values.clone()),
    ));
    println!("ts_s:      {h_s}");

    let h_ms = hex::encode(ArrowDigester::hash_array(
        &TimestampMillisecondArray::from(values.clone()),
    ));
    println!("ts_ms:     {h_ms}");

    let h_us = hex::encode(ArrowDigester::hash_array(
        &TimestampMicrosecondArray::from(values.clone()),
    ));
    println!("ts_us:     {h_us}");

    let h_ns = hex::encode(ArrowDigester::hash_array(
        &TimestampNanosecondArray::from(values.clone()),
    ));
    println!("ts_ns:     {h_ns}");

    // Weak assertions for now — exact values filled in after fix (Task 6)
    assert!(h_s_utc.starts_with("000001"), "unexpected prefix: {h_s_utc}");
    assert!(h_ms_utc.starts_with("000001"), "unexpected prefix: {h_ms_utc}");
    assert!(h_us_utc.starts_with("000001"), "unexpected prefix: {h_us_utc}");
    assert!(h_ns_utc.starts_with("000001"), "unexpected prefix: {h_ns_utc}");
    assert!(h_s.starts_with("000001"), "unexpected prefix: {h_s}");
    assert!(h_ms.starts_with("000001"), "unexpected prefix: {h_ms}");
    assert!(h_us.starts_with("000001"), "unexpected prefix: {h_us}");
    assert!(h_ns.starts_with("000001"), "unexpected prefix: {h_ns}");
}
```

- [ ] **Step 2: Add `duration_array_hashing`**

```rust
#[test]
fn duration_array_hashing() {
    let values = vec![Some(0_i64), None, Some(1_000_i64)];

    let h_s = hex::encode(ArrowDigester::hash_array(
        &DurationSecondArray::from(values.clone()),
    ));
    println!("dur_s:  {h_s}");

    let h_ms = hex::encode(ArrowDigester::hash_array(
        &DurationMillisecondArray::from(values.clone()),
    ));
    println!("dur_ms: {h_ms}");

    let h_us = hex::encode(ArrowDigester::hash_array(
        &DurationMicrosecondArray::from(values.clone()),
    ));
    println!("dur_us: {h_us}");

    let h_ns = hex::encode(ArrowDigester::hash_array(
        &DurationNanosecondArray::from(values.clone()),
    ));
    println!("dur_ns: {h_ns}");

    // Weak assertions for now — exact values filled in after fix (Task 6)
    assert!(h_s.starts_with("000001"), "unexpected prefix: {h_s}");
    assert!(h_ms.starts_with("000001"), "unexpected prefix: {h_ms}");
    assert!(h_us.starts_with("000001"), "unexpected prefix: {h_us}");
    assert!(h_ns.starts_with("000001"), "unexpected prefix: {h_ns}");
}
```

- [ ] **Step 3: Add `timestamp_units_differ`**

```rust
#[test]
fn timestamp_units_differ() {
    let values = vec![Some(1_000_i64), Some(2_000_i64)];
    let hashes = [
        hex::encode(ArrowDigester::hash_array(
            &TimestampSecondArray::from(values.clone()),
        )),
        hex::encode(ArrowDigester::hash_array(
            &TimestampMillisecondArray::from(values.clone()),
        )),
        hex::encode(ArrowDigester::hash_array(
            &TimestampMicrosecondArray::from(values.clone()),
        )),
        hex::encode(ArrowDigester::hash_array(
            &TimestampNanosecondArray::from(values.clone()),
        )),
    ];
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(
                hashes[i], hashes[j],
                "units {i} and {j} produced identical hashes"
            );
        }
    }
}
```

- [ ] **Step 4: Add `timestamp_tz_differs`**

```rust
#[test]
fn timestamp_tz_differs() {
    let values = vec![Some(1_000_i64), Some(2_000_i64)];
    let naive = hex::encode(ArrowDigester::hash_array(
        &TimestampMicrosecondArray::from(values.clone()),
    ));
    let utc = hex::encode(ArrowDigester::hash_array(
        &TimestampMicrosecondArray::from(values.clone()).with_timezone("UTC"),
    ));
    assert_ne!(naive, utc, "tz-naive and tz=UTC must produce different hashes");
}
```

- [ ] **Step 5: Add `duration_units_differ`**

```rust
#[test]
fn duration_units_differ() {
    let values = vec![Some(1_000_i64), Some(2_000_i64)];
    let hashes = [
        hex::encode(ArrowDigester::hash_array(
            &DurationSecondArray::from(values.clone()),
        )),
        hex::encode(ArrowDigester::hash_array(
            &DurationMillisecondArray::from(values.clone()),
        )),
        hex::encode(ArrowDigester::hash_array(
            &DurationMicrosecondArray::from(values.clone()),
        )),
        hex::encode(ArrowDigester::hash_array(
            &DurationNanosecondArray::from(values.clone()),
        )),
    ];
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(hashes[i], hashes[j]);
        }
    }
}
```

- [ ] **Step 6: Run to confirm all 5 new tests fail**

```bash
cargo test timestamp_array_hashing duration_array_hashing timestamp_units_differ timestamp_tz_differs duration_units_differ 2>&1 | tail -20
```

Expected: all 5 tests fail with `not yet implemented` / `called \`Option::unwrap()\` on a \`None\` value` or similar panic from `todo!()`.

---

### Task 3: Add timestamp/duration columns to the existing `schema()` test

**Files:**
- Modify: `tests/arrow_digester.rs` (the `fn schema()` test — search for `"time64_nano"`)

- [ ] **Step 1: Add 12 new fields to the schema definition**

In `fn schema()`, find the line with `Field::new("time64_nano", DataType::Time64(TimeUnit::Nanosecond), false),` and insert immediately after it:

```rust
            // timestamp — all 4 units, tz-aware (UTC)
            Field::new(
                "timestamp_s_utc",
                DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
                false,
            ),
            Field::new(
                "timestamp_ms_utc",
                DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
                false,
            ),
            Field::new(
                "timestamp_us_utc",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new(
                "timestamp_ns_utc",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                false,
            ),
            // timestamp — all 4 units, tz-naive
            Field::new("timestamp_s", DataType::Timestamp(TimeUnit::Second, None), false),
            Field::new("timestamp_ms", DataType::Timestamp(TimeUnit::Millisecond, None), false),
            Field::new("timestamp_us", DataType::Timestamp(TimeUnit::Microsecond, None), false),
            Field::new("timestamp_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            // duration — all 4 units
            Field::new("duration_s", DataType::Duration(TimeUnit::Second), false),
            Field::new("duration_ms", DataType::Duration(TimeUnit::Millisecond), false),
            Field::new("duration_us", DataType::Duration(TimeUnit::Microsecond), false),
            Field::new("duration_ns", DataType::Duration(TimeUnit::Nanosecond), false),
```

- [ ] **Step 2: Add 12 corresponding arrays to the `RecordBatch::try_new` call**

In the same `fn schema()`, find the line `Arc::new(Time64NanosecondArray::from(vec![3_600_000_000_000_i64])),` and insert immediately after it:

```rust
                // timestamp arrays (tz-aware UTC, all units; raw value = 1_000 ticks)
                Arc::new(
                    TimestampSecondArray::from(vec![1_000_i64]).with_timezone("UTC"),
                ),
                Arc::new(
                    TimestampMillisecondArray::from(vec![1_000_i64]).with_timezone("UTC"),
                ),
                Arc::new(
                    TimestampMicrosecondArray::from(vec![1_000_i64]).with_timezone("UTC"),
                ),
                Arc::new(
                    TimestampNanosecondArray::from(vec![1_000_i64]).with_timezone("UTC"),
                ),
                // timestamp arrays (tz-naive, all units)
                Arc::new(TimestampSecondArray::from(vec![1_000_i64])),
                Arc::new(TimestampMillisecondArray::from(vec![1_000_i64])),
                Arc::new(TimestampMicrosecondArray::from(vec![1_000_i64])),
                Arc::new(TimestampNanosecondArray::from(vec![1_000_i64])),
                // duration arrays (all units; 1_000 ticks each)
                Arc::new(DurationSecondArray::from(vec![1_000_i64])),
                Arc::new(DurationMillisecondArray::from(vec![1_000_i64])),
                Arc::new(DurationMicrosecondArray::from(vec![1_000_i64])),
                Arc::new(DurationNanosecondArray::from(vec![1_000_i64])),
```

- [ ] **Step 3: Run to confirm `schema()` test fails (panics from `todo!()`)**

```bash
cargo test 'tests::schema' -- --nocapture 2>&1 | tail -20
```

Expected: test panics with `not yet implemented`.

---

### Task 4: Write failing raw buffer tests in `src/arrow_digester_core.rs`

**Files:**
- Modify: `src/arrow_digester_core.rs` (internal `mod tests` — add after `digest_time64_nullable_bytes`)

- [ ] **Step 1: Add timestamp array imports to the internal test module**

In `src/arrow_digester_core.rs`, find the internal `mod tests` block (around line 1244). Add to the `use arrow::array::{ ... }` import:

```rust
use arrow::array::{
    /* existing imports ... */
    TimestampMicrosecondArray,
    DurationMicrosecondArray,
};
```

The full updated import block (replace the existing one starting at `use arrow::{ array::{`):

```rust
    use arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
            Decimal32Array, DurationMicrosecondArray, FixedSizeBinaryBuilder, Float16Array,
            Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
            LargeBinaryArray, LargeListArray, LargeListBuilder, LargeStringArray, ListBuilder,
            PrimitiveBuilder, RecordBatch, StringArray, StructArray, Time32SecondArray,
            Time64MicrosecondArray, TimestampMicrosecondArray, UInt16Array, UInt32Array,
            UInt64Array, UInt8Array,
        },
        datatypes::Int32Type,
    };
```

- [ ] **Step 2: Add `digest_timestamp_nullable_bytes` test**

Insert after `digest_time64_nullable_bytes` (around line 2156):

```rust
    // ── Timestamp / Duration ──────────────────────────────────────────────

    #[test]
    fn digest_timestamp_nullable_bytes() {
        // Microseconds since epoch: [0, None, 3_600_000_000]
        let array = TimestampMicrosecondArray::from(vec![
            Some(0_i64),
            None,
            Some(3_600_000_000_i64),
        ])
        .with_timezone("UTC");
        let mut digester = ArrowDigesterCore::<Sha256>::new(
            &Schema::new(vec![Field::new(
                "col",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            )]),
            false,
        );
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0], "index 0 should be valid");
        assert!(!null_bit_vec[1], "index 1 (None) should be null");
        assert!(null_bit_vec[2], "index 2 should be valid");

        // Physical storage: int64 LE bytes for valid elements only
        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(3_600_000_000_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_duration_nullable_bytes() {
        // Duration microseconds: [0, None, 3_600_000_000]
        let array = DurationMicrosecondArray::from(vec![
            Some(0_i64),
            None,
            Some(3_600_000_000_i64),
        ]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(
            &Schema::new(vec![Field::new(
                "col",
                DataType::Duration(TimeUnit::Microsecond),
                true,
            )]),
            false,
        );
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Duration(TimeUnit::Microsecond),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0], "index 0 should be valid");
        assert!(!null_bit_vec[1], "index 1 (None) should be null");
        assert!(null_bit_vec[2], "index 2 should be valid");

        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(3_600_000_000_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }
```

- [ ] **Step 3: Run to confirm both raw buffer tests fail**

```bash
cargo test digest_timestamp_nullable_bytes digest_duration_nullable_bytes -- --nocapture 2>&1 | tail -20
```

Expected: both tests fail with panic from `todo!()`.

---

### Task 5: Implement the Rust fix

**Files:**
- Modify: `src/arrow_digester_core.rs` (lines ~941–944 in `array_digest_update`)

- [ ] **Step 1: Replace the two `todo!()` arms**

Find:

```rust
            DataType::Timestamp(_, _) => todo!(),
            DataType::Time32(_) => Self::hash_fixed_size_array(effective_array, digest, 4),
            DataType::Time64(_) => Self::hash_fixed_size_array(effective_array, digest, 8),
            DataType::Duration(_) => todo!(),
```

Replace with:

```rust
            DataType::Timestamp(_, _) => {
                // int64 physical storage; unit and tz are schema metadata, not data bytes
                Self::hash_fixed_size_array(effective_array, digest, 8);
            }
            DataType::Time32(_) => Self::hash_fixed_size_array(effective_array, digest, 4),
            DataType::Time64(_) => Self::hash_fixed_size_array(effective_array, digest, 8),
            DataType::Duration(_) => {
                // int64 physical storage; unit is schema metadata, not data bytes
                Self::hash_fixed_size_array(effective_array, digest, 8);
            }
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo build 2>&1 | tail -10
```

Expected: `Finished` with no errors.

---

### Task 6: Run all Rust tests and capture golden hash values

- [ ] **Step 1: Run all tests and capture the printed hash values**

```bash
cargo test timestamp_array_hashing duration_array_hashing -- --nocapture 2>&1
```

Expected: both tests pass and print 12 lines like:

```
ts_s_utc:  000001...
ts_ms_utc: 000001...
ts_us_utc: 000001...
ts_ns_utc: 000001...
ts_s:      000001...
ts_ms:     000001...
ts_us:     000001...
ts_ns:     000001...
dur_s:     000001...
dur_ms:    000001...
dur_us:    000001...
dur_ns:    000001...
```

**Copy all 12 hex strings to a scratch file.** You will need them in Task 7 and in the Python golden parity tests (Task 10).

- [ ] **Step 2: Run the full test suite to confirm all other tests still pass**

```bash
cargo test 2>&1 | tail -20
```

Expected: all tests pass, including the 5 new unit tests and 2 new raw buffer tests.

---

### Task 7: Finalize unit test golden hash assertions

**Files:**
- Modify: `tests/arrow_digester.rs` (replace `println!` + `starts_with` in `timestamp_array_hashing` and `duration_array_hashing`)

- [ ] **Step 1: Replace `timestamp_array_hashing` with exact assertions**

Replace the body of `timestamp_array_hashing` with (substituting the hex values captured in Task 6):

```rust
#[test]
fn timestamp_array_hashing() {
    let values = vec![Some(0_i64), None, Some(1_000_i64)];

    // tz-aware UTC, all 4 units
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampSecondArray::from(values.clone()).with_timezone("UTC"),
        )),
        "<ts_s_utc value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampMillisecondArray::from(values.clone()).with_timezone("UTC"),
        )),
        "<ts_ms_utc value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampMicrosecondArray::from(values.clone()).with_timezone("UTC"),
        )),
        "<ts_us_utc value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampNanosecondArray::from(values.clone()).with_timezone("UTC"),
        )),
        "<ts_ns_utc value from Task 6>"
    );
    // tz-naive, all 4 units
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampSecondArray::from(values.clone()),
        )),
        "<ts_s value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampMillisecondArray::from(values.clone()),
        )),
        "<ts_ms value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampMicrosecondArray::from(values.clone()),
        )),
        "<ts_us value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &TimestampNanosecondArray::from(values.clone()),
        )),
        "<ts_ns value from Task 6>"
    );
}
```

- [ ] **Step 2: Replace `duration_array_hashing` with exact assertions**

```rust
#[test]
fn duration_array_hashing() {
    let values = vec![Some(0_i64), None, Some(1_000_i64)];

    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &DurationSecondArray::from(values.clone()),
        )),
        "<dur_s value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &DurationMillisecondArray::from(values.clone()),
        )),
        "<dur_ms value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &DurationMicrosecondArray::from(values.clone()),
        )),
        "<dur_us value from Task 6>"
    );
    assert_eq!(
        hex::encode(ArrowDigester::hash_array(
            &DurationNanosecondArray::from(values.clone()),
        )),
        "<dur_ns value from Task 6>"
    );
}
```

- [ ] **Step 3: Run to confirm exact assertions pass**

```bash
cargo test timestamp_array_hashing duration_array_hashing 2>&1 | tail -10
```

Expected: both pass.

---

### Task 8: Update the `schema()` golden hash

**Files:**
- Modify: `tests/arrow_digester.rs` (the existing `assert_eq!` for `hash_record_batch` in `fn schema()`)

- [ ] **Step 1: Run the schema test and capture the new hash**

```bash
cargo test 'tests::schema' -- --nocapture 2>&1 | tail -10
```

Expected: test fails because the golden hash `"000001487059003be1..."` no longer matches (12 new columns added). The failure message will show the actual new hash value.

- [ ] **Step 2: Update the `hash_record_batch` assertion with the new hash**

In `fn schema()`, find:

```rust
        assert_eq!(
            encode(ArrowDigester::hash_record_batch(
                &batch,
                HasherConfig::default()
            )),
            "000001487059003be1a84dbe29ba6e90ea50798a76d22e46e221b6a0c332421dc4062e"
        );
```

Replace the hash string with the value printed in Step 1.

- [ ] **Step 3: Run `schema()` to confirm it passes**

```bash
cargo test 'tests::schema' 2>&1 | tail -10
```

Expected: PASS.

---

### Task 9: `cargo fmt`, full test run, and commit

- [ ] **Step 1: Format**

```bash
cargo fmt
```

- [ ] **Step 2: Run the full test suite one final time**

```bash
cargo test 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
cd starfix
git add src/arrow_digester_core.rs tests/arrow_digester.rs
git commit -m "feat: add timestamp and duration support in leaf data hashing (ITL-438)

- Replace todo!() for DataType::Timestamp and DataType::Duration with
  hash_fixed_size_array(..., 8) — int64 physical storage
- Add timestamp_array_hashing, duration_array_hashing, timestamp_units_differ,
  timestamp_tz_differs, duration_units_differ unit tests
- Add digest_timestamp_nullable_bytes, digest_duration_nullable_bytes raw buffer tests
- Update comprehensive schema() golden hash for new columns

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 2: Python (`nauticalab/starfix-python`)

### Task 10: Write failing Python schema serialization tests

**Files:**
- Modify: `tests/test_arrow_digester.py` (add to `TestSchemaSerialization` class)

- [ ] **Step 1: Add `test_timestamp_types_in_schema`**

Append to the `TestSchemaSerialization` class:

```python
    def test_timestamp_types_in_schema(self):
        schema = pa.schema([
            pa.field("ts_utc_s",  pa.timestamp("s",  tz="UTC"), nullable=False),
            pa.field("ts_utc_ms", pa.timestamp("ms", tz="UTC"), nullable=False),
            pa.field("ts_utc_us", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("ts_utc_ns", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("ts_naive_us", pa.timestamp("us"), nullable=False),
        ])
        s = _serialized_schema(schema)
        assert '{"Timestamp":["Second","UTC"]}' in s
        assert '{"Timestamp":["Millisecond","UTC"]}' in s
        assert '{"Timestamp":["Microsecond","UTC"]}' in s
        assert '{"Timestamp":["Nanosecond","UTC"]}' in s
        assert '{"Timestamp":["Microsecond",null]}' in s

    def test_duration_types_in_schema(self):
        schema = pa.schema([
            pa.field("dur_s",  pa.duration("s"),  nullable=False),
            pa.field("dur_ms", pa.duration("ms"), nullable=False),
            pa.field("dur_us", pa.duration("us"), nullable=False),
            pa.field("dur_ns", pa.duration("ns"), nullable=False),
        ])
        s = _serialized_schema(schema)
        assert '{"Duration":"Second"}' in s
        assert '{"Duration":"Millisecond"}' in s
        assert '{"Duration":"Microsecond"}' in s
        assert '{"Duration":"Nanosecond"}' in s
```

- [ ] **Step 2: Run to confirm the new schema tests pass**

The schema serialization tests should NOT fail — `_data_type_to_value` already handles timestamp/duration correctly. Confirm:

```bash
cd starfix-python
pytest tests/test_arrow_digester.py::TestSchemaSerialization -v 2>&1 | tail -20
```

Expected: all pass (including the two new schema tests).

---

### Task 11: Write failing Python golden parity tests

**Files:**
- Modify: `tests/test_golden_parity.py` (add a new `TestTimestampDurationHashing` class)

- [ ] **Step 1: Add the `TestTimestampDurationHashing` class**

Append to `tests/test_golden_parity.py` (use the hash values captured from Task 6 in Phase 1):

```python
# ---------------------------------------------------------------------------
# Timestamp and Duration hashing — golden values match Rust (ITL-438)
# ---------------------------------------------------------------------------


class TestTimestampDurationHashing:
    """Golden-hash parity for timestamp and duration types.

    All expected values were generated by the Rust starfix test suite
    (timestamp_array_hashing / duration_array_hashing in tests/arrow_digester.rs).
    Array contents: [0, None, 1_000] as raw i64 values.
    """

    # -- Timestamp (tz-aware UTC) ------------------------------------------

    def test_timestamp_second_utc(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("s", tz="UTC"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_s_utc from Task 6>"

    def test_timestamp_millisecond_utc(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("ms", tz="UTC"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_ms_utc from Task 6>"

    def test_timestamp_microsecond_utc(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("us", tz="UTC"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_us_utc from Task 6>"

    def test_timestamp_nanosecond_utc(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("ns", tz="UTC"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_ns_utc from Task 6>"

    # -- Timestamp (tz-naive) ----------------------------------------------

    def test_timestamp_second_naive(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("s"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_s from Task 6>"

    def test_timestamp_millisecond_naive(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("ms"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_ms from Task 6>"

    def test_timestamp_microsecond_naive(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("us"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_us from Task 6>"

    def test_timestamp_nanosecond_naive(self):
        arr = pa.array([0, None, 1_000], type=pa.timestamp("ns"))
        assert ArrowDigester.hash_array(arr).hex() == "<ts_ns from Task 6>"

    # -- Timestamp: unit and tz produce different hashes -------------------

    def test_timestamp_units_differ(self):
        values = [1_000, 2_000]
        hashes = [
            ArrowDigester.hash_array(pa.array(values, type=pa.timestamp("s"))).hex(),
            ArrowDigester.hash_array(pa.array(values, type=pa.timestamp("ms"))).hex(),
            ArrowDigester.hash_array(pa.array(values, type=pa.timestamp("us"))).hex(),
            ArrowDigester.hash_array(pa.array(values, type=pa.timestamp("ns"))).hex(),
        ]
        assert len(set(hashes)) == 4, "all 4 timestamp units must produce distinct hashes"

    def test_timestamp_tz_differs(self):
        values = [1_000, 2_000]
        naive = ArrowDigester.hash_array(pa.array(values, type=pa.timestamp("us"))).hex()
        utc = ArrowDigester.hash_array(
            pa.array(values, type=pa.timestamp("us", tz="UTC"))
        ).hex()
        assert naive != utc, "tz-naive and tz=UTC must produce different hashes"

    # -- Duration ----------------------------------------------------------

    def test_duration_second(self):
        arr = pa.array([0, None, 1_000], type=pa.duration("s"))
        assert ArrowDigester.hash_array(arr).hex() == "<dur_s from Task 6>"

    def test_duration_millisecond(self):
        arr = pa.array([0, None, 1_000], type=pa.duration("ms"))
        assert ArrowDigester.hash_array(arr).hex() == "<dur_ms from Task 6>"

    def test_duration_microsecond(self):
        arr = pa.array([0, None, 1_000], type=pa.duration("us"))
        assert ArrowDigester.hash_array(arr).hex() == "<dur_us from Task 6>"

    def test_duration_nanosecond(self):
        arr = pa.array([0, None, 1_000], type=pa.duration("ns"))
        assert ArrowDigester.hash_array(arr).hex() == "<dur_ns from Task 6>"

    def test_duration_units_differ(self):
        values = [1_000, 2_000]
        hashes = [
            ArrowDigester.hash_array(pa.array(values, type=pa.duration("s"))).hex(),
            ArrowDigester.hash_array(pa.array(values, type=pa.duration("ms"))).hex(),
            ArrowDigester.hash_array(pa.array(values, type=pa.duration("us"))).hex(),
            ArrowDigester.hash_array(pa.array(values, type=pa.duration("ns"))).hex(),
        ]
        assert len(set(hashes)) == 4, "all 4 duration units must produce distinct hashes"
```

- [ ] **Step 2: Run to confirm all golden parity tests fail**

```bash
pytest tests/test_golden_parity.py::TestTimestampDurationHashing -v 2>&1 | tail -30
```

Expected: all 15 tests fail with `NotImplementedError: Unsupported leaf type: ...`.

---

### Task 12: Implement the Python fix

**Files:**
- Modify: `src/starfix/arrow_digester.py` (`_element_size_for_type` function, around line 629)

- [ ] **Step 1: Add `is_timestamp` and `is_duration` checks**

Find in `_element_size_for_type`:

```python
    if pa.types.is_time32(dt):
        return 4
    if pa.types.is_time64(dt):
        return 8
    if pa.types.is_decimal(dt):
```

Replace with:

```python
    if pa.types.is_time32(dt):
        return 4
    if pa.types.is_time64(dt):
        return 8
    if pa.types.is_timestamp(dt):
        return 8  # int64 physical storage; unit/tz are schema metadata
    if pa.types.is_duration(dt):
        return 8  # int64 physical storage; unit is schema metadata
    if pa.types.is_decimal(dt):
```

- [ ] **Step 2: Run the golden parity tests to confirm they now pass**

```bash
pytest tests/test_golden_parity.py::TestTimestampDurationHashing -v 2>&1 | tail -30
```

Expected: all 15 tests pass.

- [ ] **Step 3: Run the full Python test suite**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd starfix-python
git add src/starfix/arrow_digester.py tests/test_arrow_digester.py tests/test_golden_parity.py
git commit -m "feat: add timestamp and duration support in leaf data hashing (ITL-438)

- Add is_timestamp and is_duration checks to _element_size_for_type (8 bytes each)
- Add test_timestamp_types_in_schema and test_duration_types_in_schema
- Add TestTimestampDurationHashing golden parity tests (15 cases)
  matching Rust starfix byte-for-byte

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 3: Pull Requests

### Task 13: Create the starfix (Rust) PR

- [ ] **Step 1: Push the branch**

```bash
cd starfix
git push -u origin eywalker/itl-438-add-support-for-timestamp-and-duration-types-in-leaf-data
```

- [ ] **Step 2: Create the PR targeting `main`**

```bash
gh pr create \
  --title "feat: add timestamp and duration support in leaf data hashing (ITL-438)" \
  --base main \
  --body "$(cat <<'EOF'
## Summary

- Replaces `todo!()` for `DataType::Timestamp(_, _)` and `DataType::Duration(_)` in `array_digest_update` with `hash_fixed_size_array(..., 8)` — both types are physically stored as int64 (8 bytes), identical to `time64`
- Adds unit tests: `timestamp_array_hashing`, `duration_array_hashing`, `timestamp_units_differ`, `timestamp_tz_differs`, `duration_units_differ`
- Adds raw buffer tests: `digest_timestamp_nullable_bytes`, `digest_duration_nullable_bytes`
- Updates the comprehensive `schema()` golden hash for the 12 new columns

## Polars note

`timestamp[s]` and `duration[s]` are silently coerced to `ms` by Polars — documented in `docs/metamorphic/specs/2026-06-28-timestamp-duration-hashing-design.md`. starfix does not normalise this; the hash reflects the actual Arrow type.

## Test plan

- [ ] `cargo test` passes locally
- [ ] `cargo fmt` applied

Closes ITL-438
🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

### Task 14: Create the starfix-python PR

- [ ] **Step 1: Push the branch**

```bash
cd starfix-python
git push -u origin eywalker/itl-438-add-support-for-timestamp-and-duration-types-in-leaf-data
```

- [ ] **Step 2: Create the PR targeting `main`**

```bash
gh pr create \
  --title "feat: add timestamp and duration support in leaf data hashing (ITL-438)" \
  --base main \
  --body "$(cat <<'EOF'
## Summary

- Adds `is_timestamp` and `is_duration` checks to `_element_size_for_type` in `arrow_digester.py` (return 8 — int64 physical storage)
- Adds `test_timestamp_types_in_schema` and `test_duration_types_in_schema` to `test_arrow_digester.py`
- Adds `TestTimestampDurationHashing` class in `test_golden_parity.py` with 15 tests, golden hash values taken from the Rust `starfix` implementation

## Dependency

Golden hash values were generated from `nauticalab/starfix` after implementing the Rust fix. Python and Rust hashes are byte-for-byte identical.

## Test plan

- [ ] `pytest tests/` passes locally

Closes ITL-438
🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
