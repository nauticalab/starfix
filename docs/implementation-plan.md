# Implementation Plan: Complete Stable Logical Hashing

This plan addresses all identified gaps in the Starfix hashing implementation, organized into tiers by priority. Each item follows the project's TDD workflow: write failing tests first, then implement.

**Files primarily affected:**
- `src/arrow_digester_core.rs` — core implementation
- `tests/arrow_digester.rs` — integration tests
- `tests/digest_bytes.rs` — byte-level specification conformance tests
- `docs/byte-layout-spec.md` — specification updates

---

## Tier 1 — Blocks Production Use

### 1.1 Implement `Timestamp` data hashing

**Current state:** `todo!()` in `array_digest_update` for `DataType::Timestamp`. Schema serialization already works (falls through to Arrow serde: `{"Timestamp":["Nanosecond","UTC"]}`).

**Implementation:** Timestamp is always `i64` (8 bytes LE), regardless of unit or timezone.

```rust
DataType::Timestamp(_, _) => Self::hash_fixed_size_array(effective_array, digest, 8),
```

**Design decision — Timezone equivalence:**
Arrow's serde serializes `Timestamp(Nanosecond, Some("UTC"))` as `{"Timestamp":["Nanosecond","UTC"]}` and `Timestamp(Nanosecond, None)` as `{"Timestamp":["Nanosecond",null]}`. These naturally produce different schema hashes, which means **two columns with the same epoch values but different timezone annotations will hash differently** (because their schemas differ). This is the correct behavior — timezone is part of the logical type identity. **No special handling needed.**

However, there is a subtler question: should `Timestamp(Nanosecond, Some("UTC"))` and `Timestamp(Nanosecond, Some("Etc/UTC"))` hash the same? They refer to the same timezone but have different string representations. **Recommended decision: do NOT normalize timezone strings.** Timezone alias resolution is complex, locale-dependent, and outside Starfix's scope. Document this as a known limitation.

**Tests:**
- `Timestamp(Nanosecond, Some("UTC"))` basic hashing (hash_array)
- `Timestamp(Microsecond, None)` with nulls
- Different units with same raw value produce different schema hashes (schema difference)
- Same unit, same data, different timezone strings produce different hashes
- Byte-level test in `digest_bytes.rs`

**Spec update:** Add Section 3.7 for Timestamp, or extend Section 3.1 with a note that Timestamp/Duration are 8-byte fixed-size types.

---

### 1.2 Implement `Duration` data hashing

**Current state:** `todo!()` in `array_digest_update` for `DataType::Duration`. Schema serialization works (`{"Duration":"Millisecond"}`).

**Implementation:** Duration is always `i64` (8 bytes LE).

```rust
DataType::Duration(_) => Self::hash_fixed_size_array(effective_array, digest, 8),
```

**Design decision:** None needed. The unit is encoded in the schema JSON, so different Duration units produce different schema hashes. Data is just raw i64 bytes.

**Tests:**
- `Duration(Millisecond)` basic hashing
- Different units produce different schema hashes
- Byte-level test

---

### 1.3 Implement `Interval` data hashing

**Current state:** `todo!()` in `array_digest_update` for `DataType::Interval`.

**Implementation:** Element size depends on the IntervalUnit variant:

```rust
DataType::Interval(unit) => {
    let size = match unit {
        IntervalUnit::YearMonth => 4,   // i32
        IntervalUnit::DayTime => 8,     // i32 + i32 packed as i64
        IntervalUnit::MonthDayNano => 16, // i32 + i32 + i64
    };
    Self::hash_fixed_size_array(effective_array, digest, size);
}
```

**Design decision:** None needed. Schema serialization (`{"Interval":"MonthDayNano"}`) already differentiates variants. Each variant has a fixed physical size, so `hash_fixed_size_array` works directly.

**Tests:**
- One test per IntervalUnit variant
- `MonthDayNano` with nulls
- Different interval units produce different schema hashes
- Byte-level test for `YearMonth` (simplest, 4-byte)

---

### 1.4 Implement `FixedSizeList` data hashing

**Current state:** `todo!()` in `array_digest_update` for `DataType::FixedSizeList`. Schema normalization and serialization already work correctly (`{"FixedSizeList":[<element>, size]}`). Normalization recurses into the inner field but does **not** collapse `FixedSizeList` → `LargeList`.

**Design decision — Should `FixedSizeList(Int32, 3)` be equivalent to `LargeList(Int32)`?**
**Recommended: No.** They are semantically different types (fixed-length vs variable-length). A `FixedSizeList` guarantees every element has exactly N items; a `LargeList` does not. Keep them as distinct types in the hash. This is consistent with how FixedSizeBinary is already handled (kept separate from LargeBinary).

**Implementation:** `FixedSizeList` is conceptually a list where every element has exactly `size` items. For hashing, we can treat it like `LargeList` but without structural size prefixes (since all sizes are identical and encoded in the schema).

However, for consistency with `LargeList`, we should still use structural hashing with the fixed size. This ensures that if a user ever needs to compare a `FixedSizeList` hash against a manually reconstructed one, the logic is consistent.

**Alternative (simpler):** Treat `FixedSizeList(field, n)` as a flat buffer of `n * element_size` bytes per row. This only works for fixed-size inner types. For variable-size inner types (e.g., `FixedSizeList(Utf8, 3)`), we must recurse.

**Recommended approach:** Reuse `hash_list_array` logic by casting `FixedSizeListArray` to `LargeListArray`. Arrow's `cast` supports this. This is the simplest and most consistent approach.

```rust
DataType::FixedSizeList(field, _) => {
    let as_large_list = cast(effective_array, &DataType::LargeList(Arc::clone(field)))
        .expect("Failed to cast FixedSizeList to LargeList");
    Self::hash_list_array(
        as_large_list.as_any().downcast_ref::<LargeListArray>()
            .expect("Failed to downcast to LargeListArray"),
        field.data_type(),
        digest,
    );
}
```

**Design decision — Normalization update needed?** If we cast at hash time, we should also normalize `FixedSizeList` → `LargeList` in `normalize_data_type` to keep schema and data hashing consistent. But then `FixedSizeList` and `LargeList` with the same element type would be logically equivalent (same hash), which loses the fixed-size guarantee in the hash. **Decision needed from project owner:**
- **(A)** Normalize `FixedSizeList(f, n)` → `LargeList(f)` — treats them as equivalent (like Utf8/LargeUtf8)
- **(B)** Keep separate — `FixedSizeList` and `LargeList` always hash differently (different schema JSON)
- **(C)** Keep schema separate but use same data hashing logic (cast at data time, don't normalize schema) — this is the recommended approach

If **(C)**: schema JSON stays as `{"FixedSizeList":[..., n]}` (preserving the size), but data hashing uses LargeList logic internally. This means two arrays with identical data but different types (`FixedSizeList` vs `LargeList`) produce different hashes (because their schemas differ), which is correct.

**Tests:**
- `FixedSizeList(Int32, 2)` basic hashing
- `FixedSizeList(LargeUtf8, 3)` with variable-length inner type
- Nullable `FixedSizeList` with null elements
- Verify `FixedSizeList(Int32, 2)` ≠ `LargeList(Int32)` (if option B/C chosen)
- Byte-level test

---

### 1.5 Implement `Map` data hashing

**Current state:** `todo!()` in `array_digest_update` for `DataType::Map`. Schema normalization and serialization work (`{"Map":[<field>, sorted]}`).

**Background:** A `Map` in Arrow is physically stored as `LargeList<Struct<key, value>>`. The Arrow `MapArray` wraps a `ListArray` of `StructArray` entries.

**Design decision — Should `Map` be normalized to `LargeList<Struct<...>>`?**
**Recommended: No.** `Map` has semantic meaning (key-value pairs, optional sort guarantee) that `LargeList<Struct>` does not. The `sorted` flag is part of the schema JSON and should affect the hash. Keep `Map` as a distinct type.

**Implementation:** Treat `Map` as a list of structs. Use the same approach as `LargeList`:

```rust
DataType::Map(field, _sorted) => {
    // Map is physically stored as a list of key-value structs
    let map_array = effective_array.as_any()
        .downcast_ref::<MapArray>()
        .expect("Failed to downcast to MapArray");
    // Reinterpret as list of entries
    // MapArray provides .entries() as StructArray and offsets
    // Hash like a LargeList<Struct<key, value>>
    // ...
}
```

Concretely, `MapArray` exposes `keys()`, `values()`, and offsets. The cleanest path is to extract the underlying `ListArray` and hash it:

```rust
DataType::Map(field, _) => {
    // MapArray is backed by a ListArray of Struct entries
    let map_array = effective_array.as_any()
        .downcast_ref::<MapArray>()
        .expect("Failed to downcast to MapArray");
    Self::hash_list_array(
        // MapArray derefs to its inner ListArray representation
        // We may need to access the underlying storage
        ...,
        field.data_type(),
        digest,
    );
}
```

**Note:** The exact API depends on Arrow's `MapArray` internals. May need to construct a `LargeListArray` from the Map's offsets and entries struct. Check `arrow::array::MapArray` API.

**Tests:**
- Simple `Map<Utf8, Int32>` with 2 rows
- Nullable Map with null entries
- Verify `Map` ≠ `LargeList<Struct<key, value>>` (different schema hashes)
- Byte-level test

---

### 1.6 Add multi-word validity bitmap test

**Current state:** All existing tests use arrays with ≤ 8 elements, so validity bitmaps always fit in a single `u8` word. No test verifies correct behavior across word boundaries.

**Implementation:** No code change needed — just add tests.

**Tests:**
- Array with 9 elements (null at position 8 → triggers second u8 word)
- Array with 16 elements (nulls spanning exactly 2 full words)
- Array with 20 elements (partial third word, verifying zero-padding of unused high bits)
- All three as byte-level tests in `digest_bytes.rs` to verify exact word serialization

---

## Tier 2 — Robustness

### 2.1 Implement `Null` type

**Current state:** `todo!()` in `array_digest_update` for `DataType::Null`.

**Design decision:** A `Null` column has no data — every element is null. The only information to hash is the validity bitmap (all zeros) and the count.

**Implementation:**
```rust
DataType::Null => {
    // Null type: no data bytes. Only push null bits (all false).
    if let Some(ref mut null_bits) = digest.null_bits {
        null_bits.extend(repeat_n(false, effective_array.len()));
    }
    // No data to feed into digest.data — intentionally empty.
}
```

**Tests:**
- `NullArray` with 3 elements via hash_array
- Nullable vs non-nullable Null column in record batch
- Byte-level test: verify only validity bits (all 0s) and empty data digest

---

### 2.2 Add nullable list element tests

**Current state:** No test creates a `LargeListArray` where some list entries themselves are NULL (not list *values* being null, but entire list entries absent).

**Tests:**
- `LargeList<Int32>` with data `[[1,2], NULL, [3]]` — verify null list entry is skipped (no structural size, no data)
- Byte-level test verifying exact bytes: validity = `[1, 0, 1]`, structural receives only 2 sizes, data receives only `[1,2,3]`

---

### 2.3 Document metadata exclusion in spec

**Current state:** Arrow Field/Schema metadata (`HashMap<String, String>`) is silently ignored. `normalize_field()` drops metadata. This is correct but undocumented.

**Changes:**
- Add to `docs/byte-layout-spec.md` Section 2.1: "Arrow field metadata and schema metadata are **excluded** from the hash. Only field names, data types (recursively), and nullability are included. This means two schemas that differ only in metadata produce identical hashes."
- Add a test: two schemas identical except for metadata → same hash

---

### 2.4 Add property-based test: column reorder invariance

**Current state:** Column order independence is tested with 2 fixed examples. A property test would strengthen this.

**Design decision:** Use `proptest` or `quickcheck` crate? **Recommend `proptest`** — more flexible, better shrinking.

**Tests:**
- Generate random schemas with 2-10 fields of supported types
- Generate random data matching schema
- Shuffle column order → hash must be identical
- This would also serve as a crash test for unsupported types (should not panic for supported types)

**Note:** This is a `dev-dependency` addition. Keep it behind a feature flag if desired.

---

## Tier 3 — Completeness

### 3.1 Implement `Union` types (Dense and Sparse)

**Current state:** `todo!()` in `array_digest_update` for `DataType::Union`.

**Design decision — This is the hardest type to hash correctly:**

A Union contains multiple child arrays and a type_ids buffer that says which child each row comes from. DenseUnion also has an offsets buffer.

Options:
- **(A) Resolve to concrete values:** For each row, look up the active child + offset, extract the value, hash it. This is like dictionary resolution. Simple but loses the "which variant" information.
- **(B) Hash type_ids + child data separately:** Feed `type_ids` as a fixed-size array, then hash each child independently. This preserves variant identity.
- **(C) Hash compositely:** For each row, hash `(type_id, value_bytes)`. This is the most collision-resistant.

**Recommended: (C)** — hash `type_id` byte followed by value bytes for each row. This ensures that a union value `Int32(5)` hashes differently from `Float32(5.0)` even if they happen to have similar byte representations.

**Implementation sketch:**
```rust
DataType::Union(fields, mode) => {
    let union_array = effective_array.as_any()
        .downcast_ref::<UnionArray>()
        .expect("Failed to downcast to UnionArray");
    for i in 0..union_array.len() {
        let type_id = union_array.type_id(i);
        digest.data.update(type_id.to_le_bytes());
        let child = union_array.value(i);
        // Hash the single-element child value
        // Need a way to hash a single scalar — possibly slice the child array
        ...
    }
}
```

**Complexity:** High. Union hashing requires per-element dispatch. Defer if not needed for initial production use.

**Tests:**
- SparseUnion with Int32 and Utf8 children
- DenseUnion with nulls (if Union supports nulls — it depends on Arrow version)
- Byte-level test

---

### 3.2 Implement `RunEndEncoded`

**Current state:** `todo!()` in `array_digest_update` for `DataType::RunEndEncoded`.

**Design decision:** RunEndEncoded is a compression format. Like Dictionary, the logical values are what matter.

**Recommended:** Resolve/decode to the plain array equivalent and hash that. Arrow should support `cast()` from REE to plain arrays.

```rust
DataType::RunEndEncoded(_, values_field) => {
    let plain = cast(effective_array, values_field.data_type())
        .expect("Failed to decode RunEndEncoded");
    Self::array_digest_update(values_field.data_type(), plain.as_ref(), digest);
}
```

**Design decision:** Should REE normalize in the schema? **Recommended: Yes** — normalize `RunEndEncoded(run_ends, values)` → `normalize_data_type(values.data_type())`. This treats REE as a pure encoding optimization, like Dictionary.

**Tests:**
- REE Int32 array hashes same as plain Int32 array
- REE with runs of different lengths

---

### 3.3 Implement View types (`BinaryView`, `Utf8View`)

**Current state:** `todo!()` at lines 533, 541.

**Implementation:** View types are logically equivalent to their non-view counterparts. Normalize in both schema and data:

**Schema normalization** (add to `normalize_data_type`):
```rust
DataType::Utf8View => DataType::LargeUtf8,
DataType::BinaryView => DataType::LargeBinary,
```

**Data hashing** (add to normalization block at top of `array_digest_update`):
```rust
DataType::Utf8View => {
    normalized_type = DataType::LargeUtf8;
    cast_array = cast(array, &normalized_type).expect("Failed to cast Utf8View to LargeUtf8");
    (&normalized_type, cast_array.as_ref())
}
DataType::BinaryView => {
    normalized_type = DataType::LargeBinary;
    cast_array = cast(array, &normalized_type).expect("Failed to cast BinaryView to LargeBinary");
    (&normalized_type, cast_array.as_ref())
}
```

**Tests:**
- `Utf8View ["hello"]` hashes same as `LargeUtf8 ["hello"]`
- `BinaryView` hashes same as `LargeBinary`
- Schema equivalence test

---

### 3.4 Implement `ListView` / `LargeListView`

**Current state:** `todo!()` at lines 542, 554.

**Implementation:** Normalize to `LargeList` (same logical semantics, different physical layout):

**Schema normalization:**
```rust
DataType::ListView(field) | DataType::LargeListView(field) => {
    DataType::LargeList(Arc::new(normalize_field(field)))
}
```

**Data hashing:** Cast to `LargeList` at the normalization block in `array_digest_update`.

**Tests:**
- `ListView<Int32>` hashes same as `LargeList<Int32>`
- With nulls

---

### 3.5 Add fuzz testing for panic detection

**Implementation:** Add a fuzz target that generates random `RecordBatch` instances from random schemas (using only supported types) and ensures `hash_record_batch` never panics.

**Tool:** `cargo-fuzz` with `libfuzzer` or `afl`.

**Scope:** Generate schemas with 1-20 fields, types drawn from supported set, 0-100 rows, random null patterns.

---

## Execution Order

Recommended implementation sequence (respecting dependencies):

1. **1.1–1.3** (Timestamp, Duration, Interval) — independent, trivial implementations
2. **1.6** (multi-word validity test) — test-only, no code changes
3. **2.1** (Null type) — trivial
4. **2.2** (nullable list test) — test-only
5. **2.3** (document metadata exclusion) — docs-only
6. **3.3** (View types) — simple normalization + cast
7. **3.4** (ListView) — simple normalization + cast
8. **1.4** (FixedSizeList) — needs design decision on normalization
9. **1.5** (Map) — moderate complexity, needs Arrow API exploration
10. **3.2** (RunEndEncoded) — needs design decision on normalization
11. **3.1** (Union) — highest complexity
12. **2.4** (property tests) — after all types implemented
13. **3.5** (fuzz testing) — after all types implemented

Items 1-7 can likely be done in a single PR. Items 8-11 may warrant individual PRs due to design decisions. Items 12-13 are infrastructure additions.

---

## Python Bindings

The Python interface should be provided via **PyO3 bindings** to the Rust library (not a parallel pure-Python implementation). This lives in the separate `nauticalab/starfix-python` repository.

**TODO:**
- Configure PyO3/maturin build for the starfix crate
- Expose `ArrowDigester`, `hash_array`, `hash_record_batch`, `hash_table` to Python
- Use `arrow-rs` ↔ `pyarrow` interop via `arrow::pyarrow` feature or `pyo3-arrow`
- Publish to PyPI as `starfix`

---

## Open Design Decisions Summary

| # | Question | Recommendation | Impact |
|---|----------|---------------|--------|
| 1 | Should timezone strings be normalized (e.g., "UTC" == "Etc/UTC")? | **No** — document as known limitation | Low risk |
| 2 | Should `FixedSizeList` normalize to `LargeList`? | **No** — keep schema separate, use same data hashing logic (option C) | Affects schema equivalence |
| 3 | Should `Map` normalize to `LargeList<Struct>`? | **No** — keep as distinct type | Affects schema equivalence |
| 4 | Should `RunEndEncoded` normalize to its value type? | **Yes** — treat as encoding optimization like Dictionary | Affects schema equivalence |
| 5 | Should View types normalize to Large equivalents? | **Yes** — `Utf8View`→`LargeUtf8`, etc. | Affects schema equivalence |
| 6 | How should Union be hashed? | **(C)** — type_id + value bytes per row | Affects hash format |
| 7 | Should metadata affect the hash? | **No** — current behavior is correct, just document it | Documentation only |
