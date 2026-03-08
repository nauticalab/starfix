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
| **Data digest** | A running hash of the non-null leaf data bytes for a single field. |
| **Structural digest** | A running hash of element counts for list-type fields, separating structure from leaf data. |
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
  "name": {"data_type": "LargeUtf8", "nullable": true}
}
```

### 4.2 Data Type Serialization (`data_type_to_value`)

All data type serialization goes through `data_type_to_value`, which produces a canonical JSON representation. The output is recursively key-sorted via `sort_json_value` before returning.

#### Primitive types
Serialized using Arrow's built-in serde, producing strings like `"Int32"`, `"Boolean"`, `"Float64"`, or objects like `{"Decimal128": [38, 5]}`, `{"Time32": "Second"}`.

#### Logical type equivalence classes

Certain types that differ only in physical representation (offset width) are canonicalized to a single form:

| Types in equivalence class | Canonical form in schema |
|---|---|
| `Binary`, `LargeBinary` | `"LargeBinary"` |
| `Utf8`, `LargeUtf8` | `"LargeUtf8"` |
| `List(field)`, `LargeList(field)` | `{"LargeList": <element_type>}` |
| `Dictionary(key_type, value_type)` | Recursive `data_type_to_value(value_type)` |

The "large" variant is always the canonical form because it is the superset representation.

#### Nested types

- **Struct**: `{"Struct": [<sorted array of inner field objects>]}` — inner fields are **sorted alphabetically by field name** before serialization.
- **List / LargeList**: `{"LargeList": <element_type_object>}` (canonicalized to large variant). The element type uses `element_type_to_value` which omits the Arrow-internal field name (e.g., `"item"`), including only `data_type` and `nullable`.
- **FixedSizeList**: `{"FixedSizeList": [<element_type_object>, <size>]}`. Also uses `element_type_to_value` (no field name).
- **Map**: `{"Map": [<inner_field_object>, <sorted>]}`.

**Inner field object** (for struct children, map entries):
```json
{
  "data_type": <recursive data_type>,
  "name": "<field_name>",
  "nullable": <bool>
}
```

**Element type object** (for list/fixed-size-list items):
```json
{
  "data_type": <recursive data_type>,
  "nullable": <bool>
}
```

All JSON objects have their keys sorted recursively via `sort_json_value` to ensure deterministic serialization.

### 4.3 Schema Digest Computation

```
schema_digest = SHA256(canonical_json_string)
```

---

## 5. DigestBufferType

Each entry in the BTreeMap has a `DigestBufferType` struct with three **optional** components:

```rust
struct DigestBufferType<D: Digest> {
    null_bits: Option<BitVec<u8, Lsb0>>,  // Present for nullable entries
    structural: Option<D>,                  // Present for list-type entries
    data: Option<D>,                        // Present for leaf and list-leaf entries
}
```

- **`null_bits`**: Validity bitmap. Present for nullable fields, absent for non-nullable.
- **`structural`**: A separate running digest for list element counts. Present for list-type entries. Separates structure (how elements are partitioned into lists) from leaf data.
- **`data`**: The running digest for actual data bytes (leaf values). Present for leaf and list-leaf entries, absent for validity-only and structural-only entries.

There are four entry types, constructed via dedicated constructors:
- **`new_data_only(nullable)`**: Leaf field (e.g., `Int32`). Has `data`, optionally `null_bits`.
- **`new_structural_only(nullable)`**: List intermediate node above a struct or nested list. Has `structural`, optionally `null_bits`.
- **`new_list_leaf(nullable)`**: List whose value type is a leaf (e.g., `List<Int32>`). Has `structural` + `data`, optionally `null_bits`.
- **`new_validity_only()`**: Nullable parent whose descendants have their own entries. Has `null_bits` only.

---

## 6. Data Serialization (Byte Layout)

### 6.1 Fixed-Size Types

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

**Non-nullable path:** The entire buffer slice (accounting for offset) is fed into the data digest in one call.

**Nullable path:**
1. Extend the validity bitmap with `is_valid(i)` for each element.
2. For each valid element, feed its little-endian bytes into the data digest.
3. Null elements are **skipped** — no data bytes are fed (null information is captured solely by the validity bitmap).

### 6.2 Boolean Type

Boolean values are **bit-packed** using LSB-first (`Lsb0`) ordering with `u8` storage words into bytes via `BitVec<u8, Lsb0>`.

**Non-nullable path:** All values are packed sequentially into a `BitVec<u8, Lsb0>`, and the raw backing bytes are fed into the data digest.

**Nullable path:**
1. Extend the validity bitmap.
2. Only **valid** values are packed — nulls are skipped entirely.
3. The packed bytes are fed into the data digest.

**Example:** `[true, NULL, false, true]` (nullable)
- Validity bitmap: `[1, 0, 1, 1]`
- Data bits (valid only): `[true, false, true]` → Lsb0 packed: bit0=1, bit1=0, bit2=1 → `0000_0101` = `0x05`

**Example:** `[true, false, true]` (non-nullable)
- Lsb0 packed: bit0=1, bit1=0, bit2=1 → `0000_0101` = `0x05`

### 6.3 Variable-Length Types (Binary, String)

**Types:** `Binary`, `LargeBinary`, `Utf8`, `LargeUtf8`.

Each element is serialized as:
```
[length as u64 little-endian (8 bytes)] [raw bytes]
```

The length prefix is **always u64** (8 bytes, little-endian) regardless of the offset type (`i32` for `Binary`/`Utf8`, `i64` for `LargeBinary`/`LargeUtf8`). This ensures cross-platform stability and logical equivalence between small/large variants.

**Non-nullable path:** For each element, feed `(value.len() as u64).to_le_bytes()` then the raw bytes.

**Nullable path:**
1. Extend the validity bitmap.
2. For valid elements: feed length prefix + raw bytes.
3. For null elements: **skip entirely** — no sentinel bytes. Null information is captured by the validity bitmap.

### 6.4 List Types (Record-Batch Path)

**Types:** `List(field)`, `LargeList(field)`.

List columns are **recursively decomposed** into separate BTreeMap entries. A list creates an intermediate entry at `path/` (path + delimiter). The value type is then recursively traversed.

**Decomposition by value type:**
- **`List<leaf>`** (e.g., `List<Int32>`): Entry at `path/` is a **list-leaf** with both structural and data digests.
- **`List<Struct<...>>`**: Entry at `path/` is **structural-only**. The struct is transparent, and each struct child creates its own entry at `path//childname`.
- **`List<List<...>>`**: Entry at `path/` is structural-only. The inner list creates another entry at `path//`.

**Nullable list columns:** A **validity-only** entry is created at `path` (without trailing `/`), recording which rows are null vs valid. Null list elements are not traversed.

**Traversal:** For each non-null list element, write the sub-array length (u64 LE) to the structural digest at `path/`, then recurse into the sub-array.

### 6.5 Struct Types (Record-Batch Path)

Struct fields are **transparent** — they do not create a BTreeMap entry. Instead:

1. **Children are traversed** in alphabetical order by field name.
2. **Struct-level nulls are AND-propagated** to all descendant entries via `combine_nulls`. If a struct row is null, none of its children's data is hashed for that row.
3. Each child is recursively decomposed (leaf → data entry, list → structural entry, nested struct → recurse further).

**Path naming:** Struct adds `/fieldname` to the path. Combined with list's trailing `/`, this produces paths like `items//id` (list `/` + struct `/id`).

### 6.6 Dictionary-Encoded Arrays

Dictionary-encoded arrays are **resolved to their plain equivalent** before hashing. The dictionary is unpacked using Arrow's `cast` kernel so that the resulting data stream is identical to what a non-dictionary-encoded array with the same logical values would produce.

This ensures that `DictionaryArray<Int32, Utf8>(indices=[0,1,0], dict=["a","b"])` produces the same hash as `StringArray(["a","b","a"])`.

---

## 7. Final Digest Assembly

### 7.1 Field Digest Finalization

Each entry's `DigestBufferType` is finalized and fed into the combined final digest via `finalize_digest`. Each component is written only if present:

```
// If nullable (null_bits is Some):
feed: validity_bitmap_length as u64 LE    // 8 bytes (number of bits)
feed: validity_bitmap raw bytes (LE)      // ceil(length/8) bytes (u8 words, to_le_bytes is identity for u8)

// If list type (structural is Some):
feed: SHA256_finalize(structural_digest)  // 32 bytes

// If leaf/list-leaf (data is Some):
feed: SHA256_finalize(data_digest)        // 32 bytes
```

The validity bitmap uses `BitVec<u8, Lsb0>` storage. Each `u8` word is serialized via `to_le_bytes()` (identity for single-byte words). The bit count (not byte count) is written as the length prefix.

### 7.2 Combined Final Digest

```
final_digest = SHA256(
    schema_digest                           // 32 bytes
    || finalized_field_1                    // field "aaa" (alphabetical)
    || finalized_field_2                    // field "bbb"
    || ...                                  // remaining fields in alphabetical order
)
```

Fields are iterated from the `BTreeMap` which maintains alphabetical ordering by field path.

### 7.3 Version Prefix

The public `ArrowDigester` prepends a 3-byte version prefix to the final digest:

```
output = [0x00, 0x00, 0x01] || final_digest   // 3 + 32 = 35 bytes total
```

---

## 8. Standalone `hash_array` Function

`hash_array` hashes a single array without a full schema context. It uses the **same recursive decomposition** as the record-batch path (`extract_type_entries` + `traverse_and_update`), ensuring consistent hashing regardless of which API is used.

```
final = SHA256(
    serde_json::to_string(data_type_to_value(effective_type))   // canonical type JSON string
    || for each BTreeMap entry: finalize_digest(entry)           // same decomposition as record-batch
)
```

If the input is a dictionary array, it is first resolved to its plain value type via `cast`. The effective type is then serialized using `data_type_to_value` (with type canonicalization and recursive key sorting), converted to a JSON string, and fed into the digest before the decomposed field entries.

---

## 9. Schema Equality in `update()`

When `update(record_batch)` is called, the record batch's schema is compared against the digester's schema **logically** — both schemas are serialized via `serialized_schema()` (which uses `data_type_to_value` with type canonicalization) and the resulting strings are compared. This means:
- Column order doesn't matter (both are sorted by `BTreeMap`).
- `Utf8` vs `LargeUtf8`, `Binary` vs `LargeBinary`, `List` vs `LargeList` are treated as equivalent.
- Dictionary types are canonicalized to their value types.

---

## 10. Invariants and Guarantees

1. **Column-order independence:** Top-level fields are sorted alphabetically via `BTreeMap`.
2. **Struct field-order independence:** Struct children are sorted by name during schema serialization and during composite hashing in `array_digest_update`.
3. **Batch-split independence:** Streaming `update()` calls produce the same hash as a single combined batch.
4. **Encoding independence:** Dictionary-encoded arrays are resolved before hashing.
5. **Physical type independence:** `Binary`/`LargeBinary`, `Utf8`/`LargeUtf8`, `List`/`LargeList` are canonicalized to their large variants in the schema and use identical data serialization.
6. **Platform independence:** All length prefixes use `u64` (8 bytes LE), all numeric values use little-endian byte order, validity bitmaps use `BitVec<u8, Lsb0>` (u8-width words, not platform-dependent `usize`).
7. **Null handling consistency:** Null values are tracked solely via the validity bitmap. No sentinel bytes are fed into the data digest for any type.
8. **Non-null arrays with/without validity bitmap:** An array with all valid values produces the same data digest whether or not a validity bitmap is present.

---

## 11. Comprehensive Test Plan

### 11.1 Column-Order Independence Tests

- **Top-level column reorder:** Two record batches with columns `[a, b, c]` vs `[c, a, b]` with same data produce identical hashes.
- **Schema-only column reorder:** Two schemas with same fields in different order produce identical schema hashes.
- **Streaming with reordered batches:** Feed batch1 with order `[a, b]`, batch2 with order `[b, a]` — should produce same hash as feeding both in order `[a, b]`.

### 11.2 Struct Field-Order Independence Tests

- **Flat struct reorder:** `Struct({x: Int32, y: Utf8})` vs `Struct({y: Utf8, x: Int32})` with same data produce identical hashes.
- **Nested struct reorder:** Deeply nested structs with shuffled field orders at every level.
- **Schema hash with reordered struct fields:** Verify schema digest is identical.

### 11.3 Dictionary Encoding Equivalence Tests

- **String dictionary vs plain:** `DictionaryArray<Int32, Utf8>` vs `StringArray` with same logical values.
- **Integer dictionary vs plain:** Dictionary-encoded integers vs plain integer array.
- **Dictionary with nulls:** Dictionary arrays containing null entries match plain arrays with same nulls.
- **Nested dictionary:** List of dictionary-encoded strings vs list of plain strings.

### 11.4 Binary/Utf8/List Size Variant Equivalence Tests

- **Binary vs LargeBinary:** Same byte data in both produces identical hash.
- **Utf8 vs LargeUtf8:** Same string data produces identical hash.
- **List vs LargeList:** Same list data produces identical hash.
- **Schema equivalence:** Schema with `Binary` field hashes same as schema with `LargeBinary` field (same name, same nullability).

### 11.5 Null Handling Tests

- **No sentinel bytes:** Verify that null values in binary/string arrays don't feed any extra bytes into the data digest.
- **All-null array:** Array of all nulls produces a hash that depends only on the validity bitmap.
- **All-valid nullable vs non-nullable:** Array with all valid values produces same data digest whether schema says nullable or not.
- **Mixed nulls across batches:** First batch all nulls, second batch all valid — same as single combined batch.
- **Null at different positions:** `[1, NULL, 3]` vs `[NULL, 1, 3]` produce different hashes.

### 11.6 Batch Splitting Independence Tests

- **Two batches vs one:** Already tested, but extend to more types and edge cases.
- **Many small batches:** Split into single-row batches vs one large batch.
- **Empty batches:** Inserting empty batches between data batches doesn't change the hash.

### 11.7 Edge Cases

- **Empty table:** Schema-only hash (no data).
- **Zero-length arrays:** Arrays with length 0 for each type.
- **Single element arrays:** Arrays with exactly 1 element.
- **Maximum values:** `i32::MAX`, `i64::MAX`, `u64::MAX`, etc.
- **Special float values:** `NaN`, `+Inf`, `-Inf`, `-0.0` vs `+0.0`, subnormals.
- **Empty strings/binary:** `""` and `b""` as values (length prefix 0 + no data bytes).
- **Nested empty lists:** `[[]]`, `[[], [1]]`, `[[1], []]` all distinct.
- **Unicode strings:** Strings with multi-byte UTF-8 characters.
- **Sliced arrays:** Arrays created via `array.slice(offset, length)` should hash the same as a fresh array with the same values.

### 11.8 Collision Resistance Tests

- **Binary partition collision:** `[[0x01, 0x02], [0x03]]` vs `[[0x01], [0x02, 0x03]]` (already tested).
- **String partition collision:** `["ab", "c"]` vs `["a", "bc"]` (already tested).
- **List partition collision:** `[[1,2],[3]]` vs `[[1],[2,3]]` (already tested).
- **Null vs zero:** `[NULL]` vs `[0]` produce different hashes.
- **Empty vs null:** `[Some("")]` vs `[None]` for string type.

### 11.9 Regression / Golden Value Tests

- Maintain golden hash values for a comprehensive schema with data, verified against manually computed expected bytes.
- Byte-level verification tests (already partially present) for each data type confirming exact bytes fed into the digest.

### 11.10 Cross-Type Distinction Tests

- **Float32 vs Float64:** Same numeric value (e.g., `1.5`) in different float types produces different hashes (schema distinguishes them).
- **Int32 vs Int64:** Same integer value in different integer types produces different hashes.
- **Time32(Second) vs Time32(Millisecond):** Same raw value produces different hashes (schema distinguishes time units).
- **Decimal with different precision/scale:** Same unscaled value with different precision/scale produces different hashes.
