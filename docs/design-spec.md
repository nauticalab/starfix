# Starfix Arrow Logical Hashing - Design Specification

## 1. Overview

Starfix computes **stable, logical hashes** of Apache Arrow tables and record batches. Two tables that are *logically equivalent* must produce the **same hash**, regardless of:

- Column order in the schema / record batch
- Struct field order within nested struct types
- Dictionary encoding vs. plain encoding
- Whether the data is `List` or `LargeList`, `Utf8` or `LargeUtf8`, `Binary` or `LargeBinary`
- Batch partitioning (one large batch vs. many small batches)
- Presence or absence of a validity bitmap when all values are non-null

The hash algorithm is parameterized via Rust's `digest::Digest` trait. The public `ArrowDigester` type binds it to **SHA-256** and prepends a 3-byte version prefix (`0x00 0x00 0x01` for v0.0.1).

---

## 2. Terminology

| Term | Definition |
|------|-----------|
| **Logical equivalence** | Two Arrow structures represent the same data regardless of physical layout choices (encoding, column order, batch splits). |
| **Validity bitmap** | A bit vector where `1` = valid, `0` = null, tracked per nullable field. |
| **Data digest** | A running hash of the non-null data bytes for a single field. |
| **Schema digest** | A hash of the canonicalized JSON representation of the schema. |
| **Field path** | A `/`-separated path for nested struct fields (e.g., `address/city`). |

---

## 3. High-Level Architecture

```
                         ┌─────────────┐
                         │ ArrowDigester│  (SHA-256 + version prefix)
                         └──────┬──────┘
                                │
                     ┌──────────┴──────────┐
                     │  ArrowDigesterCore<D>│  (generic over Digest)
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
        Schema Hash     Field Digests        Final Combine
     (canonical JSON)  (BTreeMap<path,       (schema_digest +
                        DigestBuffer>)        sorted field digests)
```

### Lifecycle

1. **Construction** (`new(schema)`): Serialize and hash the schema; flatten all fields (including nested struct children) into a `BTreeMap<String, DigestBufferType>` keyed by field path.
2. **Streaming updates** (`update(record_batch)`): For each field, feed data bytes (skipping nulls) into the field's data digest and extend its validity bitmap.
3. **Finalization** (`finalize()`): Combine the schema digest with each field's finalized digest (in sorted field-path order) into one final hash.

---

## 4. Schema Serialization (Canonical Form)

The schema is serialized to a **canonical JSON string** that is order-independent. This JSON is then hashed to produce the schema digest.

### 4.1 Top-Level Structure

The schema is represented as a JSON object (dictionary) where:
- **Keys** are field names (strings).
- **Values** are objects containing `{"data_type": ..., "nullable": bool}`.

Because the top-level is a `BTreeMap<String, Value>`, field names are automatically sorted alphabetically.

**Example:**
```json
{
  "age": {"data_type": "Int32", "nullable": false},
  "name": {"data_type": "Utf8", "nullable": true}
}
```

### 4.2 Data Type Serialization

#### Primitive types
Serialized using Arrow's built-in serde, producing strings like `"Int32"`, `"Boolean"`, `"Float64"`, or objects like `{"Decimal128": [38, 5]}`, `{"Time32": "Second"}`.

#### Logical type equivalence classes

For fully logical hashing, certain types that differ only in physical representation are canonicalized to a single form in the schema:

| Types in equivalence class | Canonical form in schema |
|---|---|
| `Binary`, `LargeBinary` | `"LargeBinary"` |
| `Utf8`, `LargeUtf8` | `"LargeUtf8"` |
| `List(field)`, `LargeList(field)` | `{"LargeList": <inner_field>}` |

The "large" variant is always the canonical form because it is the superset representation.

#### Nested types

- **Struct**: `{"Struct": [<sorted array of inner field objects>]}` — inner fields are **sorted alphabetically by field name** before serialization.
- **List / LargeList**: `{"LargeList": <inner_field_object>}` (canonicalized to large variant).
- **FixedSizeList**: `{"FixedSizeList": [<inner_field_object>, <size>]}`.
- **Map**: `{"Map": [<inner_field_object>, <sorted>]}`.

Each inner field object has the form:
```json
{
  "data_type": <recursive data_type>,
  "name": "<field_name>",
  "nullable": <bool>
}
```

All JSON objects have their keys sorted recursively via `sort_json_value` to ensure deterministic serialization.

### 4.3 Schema Digest Computation

```
schema_digest = SHA256(canonical_json_string)
```

---

## 5. Data Serialization (Byte Layout)

Each field is hashed independently. The field's digest buffer is one of:
- `NonNullable(D)` — a single running digest for data bytes.
- `Nullable(BitVec, D)` — a validity bitmap (`BitVec`) plus a running data digest.

### 5.1 Fixed-Size Types

**Types:** `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Float16`, `Float32`, `Float64`, `Date32`, `Date64`, `Time32(*)`, `Time64(*)`, `Decimal32`, `Decimal64`, `Decimal128`, `Decimal256`, `FixedSizeBinary(n)`.

**Byte layout per element:** Little-endian bytes of the value's native representation.

| Type | Bytes per element | Byte order |
|------|-------------------|------------|
| Int8 / UInt8 | 1 | N/A |
| Int16 / UInt16 / Float16 | 2 | Little-endian |
| Int32 / UInt32 / Float32 / Date32 / Decimal32 / Time32 | 4 | Little-endian |
| Int64 / UInt64 / Float64 / Date64 / Decimal64 / Time64 | 8 | Little-endian |
| Decimal128 | 16 | Little-endian |
| Decimal256 | 32 | Little-endian |
| FixedSizeBinary(n) | n | Raw bytes |

**Non-nullable path:** The entire buffer slice (accounting for offset) is fed into the digest in one call.

**Nullable path:**
1. Extend the validity bitmap with `is_valid(i)` for each element.
2. For each valid element, feed its little-endian bytes into the data digest.
3. Null elements are **skipped** — no data bytes are fed (null information is captured solely by the validity bitmap).

### 5.2 Boolean Type

Boolean values are **bit-packed** using MSB-first (`Msb0`) ordering into bytes.

**Non-nullable path:** All values are packed sequentially.

**Nullable path:**
1. Extend the validity bitmap.
2. Only **valid** values are packed — nulls are skipped entirely.
3. The packed bytes are fed into the data digest.

**Example:** `[true, NULL, false, true]` (nullable)
- Validity bitmap: `[1, 0, 1, 1]`
- Data bits (valid only): `[true, false, true]` → Msb0 packed: `1010_0000` = `0xA0`

### 5.3 Variable-Length Types (Binary, String)

**Types:** `Binary`, `LargeBinary`, `Utf8`, `LargeUtf8`.

Each element is serialized as:
```
[length as u64 little-endian (8 bytes)] [raw bytes]
```

The length prefix is **always u64** (8 bytes, little-endian) regardless of the offset type (`i32` for `Binary`/`Utf8`, `i64` for `LargeBinary`/`LargeUtf8`). This ensures cross-platform stability and logical equivalence between small/large variants.

**Non-nullable path:** For each element, feed `len.to_le_bytes()` (u64) then the raw bytes.

**Nullable path:**
1. Extend the validity bitmap.
2. For valid elements: feed length prefix + raw bytes.
3. For null elements: **skip entirely** — no sentinel bytes. Null information is captured by the validity bitmap.

### 5.4 List Types

**Types:** `List(field)`, `LargeList(field)`.

Each list element (a sub-array) is serialized as:
```
[sub-array length as u64 little-endian (8 bytes)] [recursive serialization of sub-array elements]
```

The sub-array length prefix prevents collisions between differently-partitioned lists (e.g., `[[1,2],[3]]` vs `[[1],[2,3]]`).

**Nullable path:** Same as other types — extend validity bitmap, skip null list entries.

The sub-array elements are hashed recursively using the same `array_digest_update` dispatch, so nested lists and nested structs within lists follow the same rules.

### 5.5 Struct Types

Struct fields are **not hashed as a composite** — instead, each leaf field within the struct is extracted and hashed independently under its own field path (e.g., `address/city`, `address/zip`). The field paths are stored in a `BTreeMap`, so they are always processed in alphabetical order.

This design means:
- Struct field order in the Arrow schema does not affect the hash.
- Each leaf field maintains its own independent validity bitmap and data digest.

### 5.6 Dictionary-Encoded Arrays

Dictionary-encoded arrays are **resolved to their plain equivalent** before hashing. The dictionary is unpacked so that the resulting data stream is identical to what a non-dictionary-encoded array with the same logical values would produce.

This ensures that `DictionaryArray<Int32, Utf8>(indices=[0,1,0], dict=["a","b"])` produces the same hash as `StringArray(["a","b","a"])`.

---

## 6. Final Digest Assembly

### 6.1 Field Digest Finalization

Each field's digest buffer is finalized and fed into the combined final digest:

**Non-nullable field:**
```
feed: SHA256_finalize(data_digest)    // 32 bytes
```

**Nullable field:**
```
feed: validity_bitmap_length as u64 LE  // 8 bytes (number of bits)
feed: validity_bitmap words (BE bytes)  // ceil(length/8) bytes, each u8 word in big-endian
feed: SHA256_finalize(data_digest)      // 32 bytes
```

The validity bitmap is serialized as:
1. The bit count (number of elements seen) as `u64` little-endian.
2. The raw backing storage words, each converted to big-endian bytes.

### 6.2 Combined Final Digest

```
final_digest = SHA256(
    schema_digest                           // 32 bytes
    || finalized_field_1                    // field "aaa" (alphabetical)
    || finalized_field_2                    // field "bbb"
    || ...                                  // remaining fields in alphabetical order
)
```

Fields are iterated from the `BTreeMap` which maintains alphabetical ordering by field path.

### 6.3 Version Prefix

The public `ArrowDigester` prepends a 3-byte version prefix to the final digest:

```
output = [0x00, 0x00, 0x01] || final_digest   // 3 + 32 = 35 bytes total
```

---

## 7. Standalone `hash_array` Function

`hash_array` hashes a single array without a full schema context. Its digest is:

```
final = SHA256(
    canonical_json(data_type)     // data type metadata
    || finalized_field_digest     // nullable or non-nullable, same rules as above
)
```

The data type is serialized using the same `data_type_to_value` logic (with type canonicalization) and then `serde_json::to_string`.

---

## 8. Invariants and Guarantees

1. **Column-order independence:** Top-level fields are sorted alphabetically via `BTreeMap`.
2. **Struct field-order independence:** Struct children are sorted by name during schema serialization and field extraction.
3. **Batch-split independence:** Streaming `update()` calls produce the same hash as a single combined batch.
4. **Encoding independence:** Dictionary-encoded arrays are resolved before hashing.
5. **Physical type independence:** `Binary`/`LargeBinary`, `Utf8`/`LargeUtf8`, `List`/`LargeList` are canonicalized to their large variants in the schema and use identical data serialization.
6. **Platform independence:** All length prefixes use `u64` (8 bytes LE), all numeric values use little-endian byte order.
7. **Null handling consistency:** Null values are tracked solely via the validity bitmap. No sentinel bytes are fed into the data digest for any type.
8. **Non-null arrays with/without validity bitmap:** An array with all valid values produces the same data digest whether or not a validity bitmap is present (nulls simply mean bits are not pushed and values are not fed, and all-valid arrays feed the same bytes).

---

## 9. Known Issues and Required Fixes

The following issues have been identified in the current implementation that must be fixed to achieve the guarantees above:

### 9.1 Struct Fields Not Sorted in Schema Serialization

**File:** `arrow_digester_core.rs`, `data_type_to_value()` (line ~206)

**Issue:** Struct inner fields are collected into a `Vec` in their original order. Two schemas with the same struct fields in different order will produce different schema hashes.

**Fix:** Sort the fields iterator by field name before collecting into the Vec.

### 9.2 `inner_field_to_value` Not Recursively Sorted

**File:** `arrow_digester_core.rs`, `inner_field_to_value()` (line ~232)

**Issue:** The JSON object produced by `serde_json::json!` has non-deterministic key order. While `sort_json_value` is applied at the top level in `serialized_schema`, it is NOT applied to the output of `data_type_to_value`/`inner_field_to_value`.

**Fix:** Apply `sort_json_value` recursively in `data_type_to_value` before returning.

### 9.3 Binary Length Prefix Uses Platform-Dependent `usize`

**File:** `arrow_digester_core.rs`, `hash_binary_array()` (line ~518)

**Issue:** `value.len().to_le_bytes()` produces 4 bytes on 32-bit and 8 bytes on 64-bit platforms.

**Fix:** Cast to `u64` before calling `to_le_bytes()`: `(value.len() as u64).to_le_bytes()`.

### 9.4 `NULL_BYTES` Sentinel in Binary/String Nullable Paths

**File:** `arrow_digester_core.rs`, `hash_binary_array()` (line ~536), `hash_string_array()` (line ~579)

**Issue:** Null values feed `b"NULL"` into the data digest, but `hash_fixed_size_array` skips nulls entirely. Since null information is already captured in the validity bitmap, the sentinel is redundant and inconsistent.

**Fix:** Remove `data_digest.update(NULL_BYTES)` from the null branches. Skip null values entirely, matching the fixed-size type behavior.

### 9.5 No Type Canonicalization for Binary/Utf8/List Variants

**File:** `arrow_digester_core.rs`, `data_type_to_value()` and `serialized_schema()`

**Issue:** `Binary` and `LargeBinary` serialize to different JSON strings, causing logically equivalent schemas to hash differently.

**Fix:** In `data_type_to_value`, map `Binary` → `LargeBinary`, `Utf8` → `LargeUtf8`, `List` → `LargeList` before serialization.

### 9.6 Dictionary-Encoded Arrays Not Supported

**File:** `arrow_digester_core.rs`, `array_digest_update()` (line ~437)

**Issue:** Dictionary-encoded arrays hit `todo!()` and panic.

**Fix:** Resolve dictionary arrays to their plain value arrays using Arrow's `take` kernel or equivalent, then recursively hash the result.

### 9.7 Schema Equality Check in `update()` Too Strict

**File:** `arrow_digester_core.rs`, `update()` (line ~61)

**Issue:** `*record_batch.schema() == self.schema` uses strict Arrow schema equality which includes column order. This prevents streaming batches with different column orders.

**Fix:** Compare schemas logically (same set of fields with same types and nullability, regardless of order).

---

## 10. Comprehensive Test Plan

### 10.1 Column-Order Independence Tests

- **Top-level column reorder:** Two record batches with columns `[a, b, c]` vs `[c, a, b]` with same data produce identical hashes.
- **Schema-only column reorder:** Two schemas with same fields in different order produce identical schema hashes.
- **Streaming with reordered batches:** Feed batch1 with order `[a, b]`, batch2 with order `[b, a]` — should produce same hash as feeding both in order `[a, b]`.

### 10.2 Struct Field-Order Independence Tests

- **Flat struct reorder:** `Struct({x: Int32, y: Utf8})` vs `Struct({y: Utf8, x: Int32})` with same data produce identical hashes.
- **Nested struct reorder:** Deeply nested structs with shuffled field orders at every level.
- **Schema hash with reordered struct fields:** Verify schema digest is identical.

### 10.3 Dictionary Encoding Equivalence Tests

- **String dictionary vs plain:** `DictionaryArray<Int32, Utf8>` vs `StringArray` with same logical values.
- **Integer dictionary vs plain:** Dictionary-encoded integers vs plain integer array.
- **Dictionary with nulls:** Dictionary arrays containing null entries match plain arrays with same nulls.
- **Nested dictionary:** List of dictionary-encoded strings vs list of plain strings.

### 10.4 Binary/Utf8/List Size Variant Equivalence Tests

- **Binary vs LargeBinary:** Same byte data in both produces identical hash.
- **Utf8 vs LargeUtf8:** Same string data produces identical hash.
- **List vs LargeList:** Same list data produces identical hash.
- **Schema equivalence:** Schema with `Binary` field hashes same as schema with `LargeBinary` field (same name, same nullability).

### 10.5 Null Handling Tests

- **No sentinel bytes:** Verify that null values in binary/string arrays don't feed any extra bytes into the data digest (after fix).
- **All-null array:** Array of all nulls produces a hash that depends only on the validity bitmap.
- **All-valid nullable vs non-nullable:** Array with all valid values produces same data digest whether schema says nullable or not.
- **Mixed nulls across batches:** First batch all nulls, second batch all valid — same as single combined batch.
- **Null at different positions:** `[1, NULL, 3]` vs `[NULL, 1, 3]` produce different hashes.

### 10.6 Batch Splitting Independence Tests

- **Two batches vs one:** Already tested, but extend to more types and edge cases.
- **Many small batches:** Split into single-row batches vs one large batch.
- **Empty batches:** Inserting empty batches between data batches doesn't change the hash.

### 10.7 Edge Cases

- **Empty table:** Schema-only hash (no data).
- **Zero-length arrays:** Arrays with length 0 for each type.
- **Single element arrays:** Arrays with exactly 1 element.
- **Maximum values:** `i32::MAX`, `i64::MAX`, `u64::MAX`, etc.
- **Special float values:** `NaN`, `+Inf`, `-Inf`, `-0.0` vs `+0.0`, subnormals.
- **Empty strings/binary:** `""` and `b""` as values (length prefix 0 + no data bytes).
- **Nested empty lists:** `[[]]`, `[[], [1]]`, `[[1], []]` all distinct.
- **Unicode strings:** Strings with multi-byte UTF-8 characters.
- **Sliced arrays:** Arrays created via `array.slice(offset, length)` should hash the same as a fresh array with the same values.

### 10.8 Collision Resistance Tests

- **Binary partition collision:** `[[0x01, 0x02], [0x03]]` vs `[[0x01], [0x02, 0x03]]` (already tested).
- **String partition collision:** `["ab", "c"]` vs `["a", "bc"]` (already tested).
- **List partition collision:** `[[1,2],[3]]` vs `[[1],[2,3]]` (already tested).
- **Null vs zero:** `[NULL]` vs `[0]` produce different hashes.
- **Empty vs null:** `[Some("")]` vs `[None]` for string type.

### 10.9 Regression / Golden Value Tests

- Maintain golden hash values for a comprehensive schema with data, verified against manually computed expected bytes.
- Byte-level verification tests (already partially present) for each data type confirming exact bytes fed into the digest.

### 10.10 Cross-Type Distinction Tests

- **Float32 vs Float64:** Same numeric value (e.g., `1.5`) in different float types produces different hashes (schema distinguishes them).
- **Int32 vs Int64:** Same integer value in different integer types produces different hashes.
- **Time32(Second) vs Time32(Millisecond):** Same raw value produces different hashes (schema distinguishes time units).
- **Decimal with different precision/scale:** Same unscaled value with different precision/scale produces different hashes.
