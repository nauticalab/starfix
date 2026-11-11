# Starfix Byte Layout Specification

This document describes the **exact byte-level serialization** used by Starfix to compute deterministic hashes of Apache Arrow schemas and record batches. Every byte fed into SHA-256 is specified here, making it possible to implement a compatible hasher in any language.

All multi-byte integers use **little-endian** byte order unless explicitly stated otherwise.

**Ordering principle.** By default starfix ignores exactly two orders: the order of the
top-level columns and the order of struct fields (at every depth). It sorts or reorders nothing
else: row order, list element order (also for `list<struct<key, value>>`), and the logical values
of dictionary-encoded arrays are hashed as given. The two ignored orders can be made part of the
hash with `ordered_columns` and `ordered_struct_fields` (Section 9). The only other
canonicalizations are not orderings of data: metadata keys (unordered maps) are sorted when
`include_metadata` is on; the canonical JSON's own object keys are sorted; physically different
but logically equal types are unified (Section 2.1); the Arrow-internal list element field name
is omitted.

Sections 1–8 specify the **default** layout (as corrected in v0.5; Section 10 records exactly
what changed from v0.4.0 and which inputs are affected). The opt-in options `ordered_columns`
and `ordered_struct_fields` (v0.5) are specified in **Section 9**; with both off, Section 9 does
not apply.

---

## 1. Output Format

Every Starfix hash is **35 bytes**:

```
[version: 3 bytes] [SHA-256 digest: 32 bytes]
```

The version prefix is currently `0x00 0x00 0x01` (version 0.0.1), in every mode (Section 9.2).

When displayed as hex, a hash looks like:

```
000001 <64 hex chars of SHA-256>
```

---

## 2. Schema Serialization

### 2.1 Canonical JSON String

The schema is serialized as a **compact JSON string** (no whitespace) of an object where:

- **Keys** are field names, sorted alphabetically (via `BTreeMap`). (With `ordered_columns` the top level is an array instead; Section 9.3.)
- **Values** are objects with keys `"data_type"` and `"nullable"`, with JSON keys sorted alphabetically within every nested object (recursively).

Because all JSON object keys are sorted recursively, the key order is always `"data_type"` before `"nullable"` (and `"data_type"` before `"name"` before `"nullable"` for struct children).

#### Type Canonicalization

Before serialization, these logical equivalence classes are collapsed:

| Arrow type(s)              | Canonical JSON form           |
|----------------------------|-------------------------------|
| `Binary`, `LargeBinary`   | `"LargeBinary"`               |
| `Utf8`, `LargeUtf8`       | `"LargeUtf8"`                 |
| `List(f)`, `LargeList(f)` | `{"LargeList": <element>}`    |
| `Dictionary(k, v)`        | canonical form of `v`         |

#### Nested Type Serialization

**Struct fields** are serialized as:
```json
{"Struct": [<array of child objects sorted by "name">]}
```
(With `ordered_struct_fields` the children keep their declaration order; Section 9.3.)
Each child object: `{"data_type": ..., "name": "<field_name>", "nullable": <bool>}`.

**List / LargeList elements** are serialized as:
```json
{"LargeList": {"data_type": ..., "nullable": <bool>}}
```
Note: the Arrow-internal field name (typically `"item"`) is **omitted** — only `data_type` and `nullable` are included.

**Fixed-size list elements** omit the element field name, like lists, and carry the size:
```json
{"FixedSizeList": [{"data_type": ..., "nullable": <bool>}, <size as JSON integer>]}
```

**Map types** keep the entries field **with** its name, and the map's `sorted` flag as given:
```json
{"Map": [{"data_type": <entries struct type>, "name": "<entries field name>", "nullable": <bool>}, <sorted: true|false>]}
```
The entries struct is serialized like any struct (its children, normally `key` and `value`, keep
their names; by default sorted by name). Example, `m: Map<Utf8, Struct<z: Int32, a: Int32>>`
nullable (arrow-rs default names `entries`, `key`, `value`; `sorted = false`):
```
{"m":{"data_type":{"Map":[{"data_type":{"Struct":[{"data_type":"LargeUtf8","name":"key","nullable":false},{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"Int32","name":"z","nullable":false}]},"name":"value","nullable":true}]},"name":"entries","nullable":false},false]},"nullable":true}}
hash_schema = 000001edad0efdc8086afa34e6633ee4213d2ed803cbd8849e7d062f882f698bdcf897
```
`f: FixedSizeList<Int32 nullable, 2>` non-nullable:
```
{"f":{"data_type":{"FixedSizeList":[{"data_type":"Int32","nullable":true},2]},"nullable":false}}
hash_schema = 000001faa6e75b09b2788440c79a1e297c73c75a5cb4d36c9f9cad6e3062a33c91c696
```
(Map and fixed-size list *data* hashing is not supported; only their schemas hash.)

#### JSON text rules

All JSON fed to a digest (Phase 1 schema JSON, `hash_array` type JSON, Phase 2 metadata JSON) is
produced by `serde_json::to_string`:

* compact: no whitespace; `,` and `:` without spaces;
* object keys sorted by their UTF-8 bytes (equivalently by Unicode code point);
* strings in UTF-8 with **non-ASCII characters written raw** (never `\uXXXX`); escaped are only
  `"` → `\"`, `\` → `\\`, U+0008 → `\b`, U+0009 → `\t`, U+000A → `\n`, U+000C → `\f`, U+000D → `\r`,
  and every other U+0000–U+001F → `\u00xx` with **lowercase** hex; `/` and U+007F are not escaped;
* integers in decimal, booleans `true` / `false`.

For example a column named `é` serializes its key as the two bytes `c3 a9`, and a column named
`a<LF>b<U+001B>` as `"a\nb\u001b"`. The same rules apply to `hash_array`:
`hash_array` of `StructArray [{é: 1}]` (child `é: Int32` non-nullable) feeds the type JSON
`{"Struct":[{"data_type":"Int32","name":"é","nullable":false}]}` with `é` raw, giving
`000001e445fa3cdc5fe37df7cb3de55cf94dcee2a8990bb8dd46fb5c3b00398f8cfb2c`.

**Primitive types** use these exact forms (they are arrow-rs serde's, listed here so that no
implementation depends on arrow-rs). `Utf8` and `Binary` never appear: they are canonicalized
to `LargeUtf8` / `LargeBinary` first.

| Arrow type | Canonical JSON |
|---|---|
| Null, Boolean | `"Null"`, `"Boolean"` |
| Int8, Int16, Int32, Int64 | `"Int8"`, `"Int16"`, `"Int32"`, `"Int64"` |
| UInt8, UInt16, UInt32, UInt64 | `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"` |
| Float16, Float32, Float64 | `"Float16"`, `"Float32"`, `"Float64"` |
| Date32, Date64 | `"Date32"`, `"Date64"` |
| Time32(unit), unit ∈ Second, Millisecond | `{"Time32":"Second"}` |
| Time64(unit), unit ∈ Microsecond, Nanosecond | `{"Time64":"Nanosecond"}` |
| Timestamp(unit, tz) | `{"Timestamp":["Microsecond","UTC"]}`; no time zone: `{"Timestamp":["Second",null]}`; the tz string is copied verbatim (`"+05:30"`, `"America/New_York"`) |
| Duration(unit) | `{"Duration":"Millisecond"}` |
| Interval(unit), unit ∈ YearMonth, DayTime, MonthDayNano | `{"Interval":"DayTime"}` |
| Decimal32/64/128/256(precision, scale) | `{"Decimal128":[38,10]}`; scale may be negative: `{"Decimal64":[18,-3]}` |
| FixedSizeBinary(n) | `{"FixedSizeBinary":16}` |
| Utf8 / LargeUtf8 | `"LargeUtf8"` |
| Binary / LargeBinary | `"LargeBinary"` |
| Utf8View, BinaryView | `"Utf8View"`, `"BinaryView"` (not canonicalized; data unsupported) |

Time units are spelled `Second`, `Millisecond`, `Microsecond`, `Nanosecond`. The golden file has
one `hash_schema` vector per form (`type_*_schema_f0`).

`ListView`, `LargeListView`, `Union` and `RunEndEncoded` types are serialized with arrow-rs's
full internal `Field` representation; their schema hashes are not part of the portable
specification, and other implementations may reject them.

### 2.2 Schema Digest (include_metadata = false)

```
schema_digest = SHA-256(canonical_json_string_bytes)
```

The UTF-8 bytes of the JSON string are fed directly into SHA-256. The result is 32 bytes.

This is the default behaviour (`HasherConfig { include_metadata: false }`). Only the canonical JSON produced in Section 2.1 contributes to the digest.

### 2.2.1 Schema Digest with Metadata (include_metadata = true)

When `HasherConfig { include_metadata: true }` is set, the schema digest is computed by a **two-phase** process that feeds into a **single SHA-256 hasher instance**.

**Phase 1 (always — identical to Section 2.2):**

```
hasher.update(canonical_json_string_bytes)
```

**Phase 2 (only when `include_metadata = true`):**

```
meta_doc = BTreeMap{}
if any field has metadata: meta_doc["fields"] = { path: BTreeMap{k:v}, ... }
if schema.metadata() non-empty: meta_doc["schema"] = BTreeMap{k:v}
if meta_doc non-empty: hasher.update(serde_json::to_string(meta_doc).as_bytes())
```

The final `schema_digest` is `hasher.finalize()` — 32 bytes, as before.

**Field paths in `"fields"`.** Every field of the schema tree that has non-empty metadata is
recorded under its **metadata path** (these equal the node paths of Section 9.6 except below a
column named `""`):

| Field | Path |
|---|---|
| top-level column `c` | `c` |
| child `n` of a struct field at path `p` | `p + "/" + n` (e.g. `s/n`; always with the `/`, so below a column named `""` the path starts with `/`) |
| element field of a list / large list / fixed-size list at path `p` (its name, e.g. `item`, is not used) | `p + "/"` (e.g. `l/`) |
| entries field of a map at path `p` | `p + "/"` (e.g. `m/`) |
| key and value fields of that map (children of the entries struct) | `p + "//key"`, `p + "//value"` (struct-child rule; their actual names are used) |

Dictionary types are replaced by their value type before the walk. Paths are sorted by their
UTF-8 bytes (the map is a `BTreeMap`), each field's metadata map is sorted by key, and metadata
values are JSON strings. Example: `m: Map<Utf8, Int32>` where the value field has metadata
`{"unit": "kg"}` and the entries field has `{"role": "pairs"}`:

```
{"fields":{"m/":{"role":"pairs"},"m//value":{"unit":"kg"}}}
```

**Empty-metadata invariant:** When a schema has no metadata at any level, Phase 2 adds nothing to the hasher. A schema with `include_metadata = true` therefore produces the **same hash** as `include_metadata = false` when no metadata is present.

### 2.3 Concrete Example

Schema: `{name: LargeUtf8 nullable, age: Int32 non-nullable}`

Canonical JSON string (compact, keys sorted):
```
{"age":{"data_type":"Int32","nullable":false},"name":{"data_type":"LargeUtf8","nullable":true}}
```

Note: `"age"` comes before `"name"` alphabetically, and `"data_type"` comes before `"nullable"`.

```
schema_digest = SHA-256(b'{"age":{"data_type":"Int32","nullable":false},"name":{"data_type":"LargeUtf8","nullable":true}}')
```

---

## 3. Field Data Serialization

The schema is recursively decomposed into a `BTreeMap` of entries. **Leaf fields** and **list intermediate nodes** get their own entries. A **struct** creates no data entry of its own: its children are recursively traversed and its null validity is AND-propagated to descendant entries; a **nullable** struct additionally records its own validity bits (Section 3.5).

> **v0.5 fix.** Sections 3.2, 3.4 and 3.5 describe the corrected layout of v0.5. The v0.4.0
> layout lost information for Booleans, nullable structs and lists whose element is a nullable
> list or struct; Section 10 lists exactly what changed and which inputs are affected.

Each entry has a **digest buffer** containing up to three **optional** components:

| Component | Present when | Purpose |
|-----------|-------------|---------|
| `null_bits` (BitVec) | field is nullable | Tracks which elements are valid vs null |
| `structural` (SHA-256) | entry is a list type (`List` or `LargeList`) | Accumulates element counts (structure) |
| `data` (SHA-256) | leaf fields and list-leaf entries | Accumulates leaf data bytes |

There are four entry types:

| Entry type | `null_bits` | `structural` | `data` | Example |
|------------|:-----------:|:------------:|:------:|---------|
| **data-only** | — | — | yes | Non-nullable leaf field (e.g., `Int32`) |
| **validity + data** | yes | — | yes | Nullable leaf field |
| **validity-only** | yes | — | — | Nullable list or nullable struct (not a list element) whose descendants have their own entries |
| **structural-only** | optional | yes | — | List element entry when the element is a struct or a nested list (`null_bits` present iff the element is nullable) |
| **list_leaf** | optional | yes | yes | List whose value type is a leaf (e.g., `List<Int32>`) |

**Naming convention**: Struct adds `/fieldname` to the path. List adds a trailing `/`. Nested lists add `//`, etc.

This separation of structural information from leaf data ensures that list element boundaries are hashed independently from the values they contain. For example, `[[1,2],[3]]` and `[[1],[2,3]]` differ in their structural digest (element counts `[2,1]` vs `[1,2]`) even though their leaf data digest is identical (`[1,2,3]`).

### 3.1 Fixed-Size Types

**Types**: `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Float16`, `Float32`, `Float64`, `Date32`, `Date64`, `Time32(*)`, `Time64(*)`, `Decimal32`, `Decimal64`, `Decimal128`, `Decimal256`, `FixedSizeBinary(n)`.

| Type | Bytes per element |
|------|-------------------|
| Int8 / UInt8 | 1 |
| Int16 / UInt16 / Float16 | 2 |
| Int32 / UInt32 / Float32 / Date32 / Decimal32 / Time32 | 4 |
| Int64 / UInt64 / Float64 / Date64 / Decimal64 / Time64 | 8 |
| Decimal128 | 16 |
| Decimal256 | 32 |
| FixedSizeBinary(n) | n |

**Non-nullable path**: The entire contiguous byte buffer (all elements concatenated, little-endian) is fed into the data digest in a single update.

**Nullable path**:
1. For each element `i`, push `is_valid(i)` (true=1, false=0) into the validity `BitVec`.
2. For each **valid** element, feed its little-endian bytes into the data digest.
3. **Null elements are skipped entirely** — no data bytes are fed.

If a nullable field has no actual nulls (null buffer absent), all elements are marked valid and the entire buffer is fed in one update (same as non-nullable data path).

### 3.2 Boolean Type

Each **valid** Boolean value is fed to the data digest as **one byte**: `0x01` for `true`, `0x00`
for `false`. Null values are skipped (their validity bit records them).

**Non-nullable**: one byte per value.

**Nullable**:
1. Extend the validity `BitVec` as usual.
2. Feed one byte per **valid** value; nulls are skipped.

**Example**: `[true, NULL, false, true]` (nullable, 4 elements)
- Validity bits: `[1, 0, 1, 1]`
- Bytes fed to data digest: `01 00 01`

(v0.4.0 bit-packed the valid values per call into whole bytes, `0x05` here; that lost trailing
`false` values and depended on batch boundaries. Fixed in v0.5, Section 10.)

### 3.3 Variable-Length Types (Binary, String)

**Types**: `Binary`, `LargeBinary`, `Utf8`, `LargeUtf8`.

Each element is serialized as:
```
[length as u64 little-endian: 8 bytes] [raw bytes: length bytes]
```

The length prefix is **always `u64`** (8 bytes, little-endian) regardless of the Arrow offset type.

**Non-nullable**: For each element, feed `(len as u64).to_le_bytes()` then the raw bytes.

**Nullable**:
1. Extend the validity `BitVec`.
2. For valid elements: feed length prefix + raw bytes.
3. For null elements: **skip entirely** — no bytes fed to data digest.

### 3.4 List Types (Record-Batch Path)

**Types**: `List(field)`, `LargeList(field)`.

List columns are **recursively decomposed** into separate BTreeMap entries. A list creates an intermediate entry at `path/` (path + delimiter). The value type is then recursively traversed to create further entries.

**Decomposition by value type:**

- **`List<leaf>`** (e.g., `List<Int32>`): The entry at `path/` is a **list-leaf** with both structural and data digests. List lengths go to structural; leaf values go to data.
- **`List<Struct<...>>`**: The entry at `path/` is **structural-only** (list lengths). Each struct child creates its own entry at `path//childname`.
- **`List<List<...>>`**: The entry at `path/` is structural-only. The inner list creates another entry at `path//`, and so on recursively.

The element entry at `path/` has `null_bits` iff the element field is nullable. For a nullable
element that is a **struct or a list**, those `null_bits` record the element validity (one bit per
element), and the entry keeps its structural digest. (A nullable leaf element's validity is
recorded there too, as a list-leaf entry.)

**Nullable list columns** (and nullable lists that are struct children): the entry at `path`
(without trailing `/`) is **validity-only**, recording which rows are null vs valid. A list that
is itself a list element has no such entry: its validity goes into the outer list's element
entry, as above. Null list elements are not traversed — no structural or data bytes are written
for them.

**Traversal**: For each non-null list element, write the sub-array length (u64 LE) to the structural digest at `path/`, then recurse into the sub-array using the value type.

(v0.4.0 replaced the outer element entry with a validity-only entry when the element was a
nullable list, losing the outer lengths, and recorded nothing for a nullable struct element.
Fixed in v0.5, Section 10.)

#### Concrete Example: Structural vs Leaf Separation

For `LargeList<Int32>` (non-nullable) with data `[[1,2],[3]]`:

The single entry at `col/` is a list-leaf:

```
structural digest receives:
    02 00 00 00 00 00 00 00     (element 0: 2 items, u64 LE)
    01 00 00 00 00 00 00 00     (element 1: 1 item, u64 LE)

data digest receives:
    01 00 00 00                  (1 as i32 LE)
    02 00 00 00                  (2 as i32 LE)
    03 00 00 00                  (3 as i32 LE)
```

Compare with `[[1],[2,3]]`: same data digest but different structural digest — so the final hashes differ.

### 3.5 Struct Types (Record-Batch Path)

A struct creates no data entry. Instead:

1. **Children are traversed** (the order does not matter: every child has its own entries, finalized in path order).
2. **Struct-level nulls are AND-propagated** to all descendant entries. If a struct row is null, none of its children's data is hashed for that row, and the null is reflected in each descendant's effective validity.
3. Each child is recursively decomposed (leaf → data entry, list → structural entry, nested struct → recurse further).
4. A **nullable** struct records its own validity: a **validity-only entry at the struct's path**
   (`s` for a column `s`, `parent/child` for a struct child, `""` for `hash_array` of a struct
   array with nulls), or, when the struct is a list element, the list's element entry `path/`
   (Section 3.4). When a nullable struct array of length `n` is traversed, `n` bits are pushed:
   bit `i` = struct's own validity AND the ancestor struct validity (the same combined validity
   that is propagated to its children); all `1` when neither has a null buffer. Non-nullable
   structs record nothing (like non-nullable leaves).

**Example**: A non-nullable struct field `address` with children `city` (LargeUtf8) and `zip` (Int32) creates two leaf entries: `address/city` and `address/zip`. If `address` is nullable there is also a validity-only entry `address` (sorted before its children).

(v0.4.0 recorded no struct validity, so a null struct hashed like a struct with all-null fields,
and null struct rows vanished when the fields were non-nullable. Fixed in v0.5, Section 10.)

### 3.6 Dictionary-Encoded Arrays

Dictionary arrays are **resolved to their plain equivalent** before hashing. The dictionary is unpacked so that the data stream is identical to a non-dictionary array with the same logical values.

---

## 4. Field Digest Finalization

After all record batches have been fed, each entry's digest buffer is finalized and fed into the **final combining digest**. Each entry may have up to three optional components, written in this fixed order (skipping absent components):

```
1. null_bits    (if present — nullable entries only)
2. structural   (if present — list entries only)
3. data         (if present — leaf and list-leaf entries only)
```

### 4.1 Data-Only Entry

```
final_digest.update( SHA-256(data_bytes).finalize() )    // 32 bytes
```

### 4.2 Validity + Data Entry (Nullable Leaf)

```
final_digest.update( bit_count.to_le_bytes() )           // 8 bytes (u64 LE)
for each word in validity_bitvec.as_raw_slice():          // each word is u8 (1 byte)
    final_digest.update( word.to_le_bytes() )             // 1 byte per word (u8, LE is trivial)
final_digest.update( SHA-256(data_bytes).finalize() )     // 32 bytes
```

### 4.3 Validity-Only Entry

```
final_digest.update( bit_count.to_le_bytes() )           // 8 bytes (u64 LE)
for each word in validity_bitvec.as_raw_slice():
    final_digest.update( word.to_le_bytes() )             // 1 byte per word (u8)
```

No structural or data digest is written.

### 4.4 Structural-Only Entry

```
final_digest.update( SHA-256(structural_bytes).finalize() )   // 32 bytes (element counts)
```

### 4.5 List-Leaf Entry (Structural + Data)

```
final_digest.update( SHA-256(structural_bytes).finalize() )   // 32 bytes (element counts)
final_digest.update( SHA-256(data_bytes).finalize() )          // 32 bytes (leaf values)
```

If nullable, prepend null_bits before structural:

```
final_digest.update( bit_count.to_le_bytes() )                // 8 bytes (u64 LE)
for each word in validity_bitvec.as_raw_slice():
    final_digest.update( word.to_le_bytes() )                  // 1 byte per word (u8)
final_digest.update( SHA-256(structural_bytes).finalize() )    // 32 bytes
final_digest.update( SHA-256(data_bytes).finalize() )          // 32 bytes
```

**Validity BitVec details** (applies to all entries with `null_bits`):
- Storage type: `u8` (1 byte per word).
- Bit order: `Lsb0` (least significant bit first within each word).
- `bit_count` = total number of elements (valid + null), serialized as `u64` little-endian (8 bytes).
- Each storage word is serialized as `u8` little-endian (trivially 1 byte).
- The last word may have unused high bits (zero-padded).

---

## 5. Final Combining Digest

The final hash is computed by feeding into a fresh SHA-256:

```
final_digest = SHA-256()

// 1. Schema digest (32 bytes)
final_digest.update( schema_digest )

// 2. Field digests in alphabetical order of field path
for field_path in sorted(field_paths):
    finalize field's DigestBufferType into final_digest (see Section 4)

raw_hash = final_digest.finalize()    // 32 bytes
output = [0x00, 0x00, 0x01] ++ raw_hash   // 35 bytes
```

---

## 6. `hash_array` API

The `hash_array` function hashes a single array (without a schema context). It uses the **same recursive decomposition** as the record-batch path, ensuring consistent hashing regardless of which API is used:

```
final_digest = SHA-256()

// 1. Type metadata (canonical JSON string)
canonical_type = data_type_to_value(effective_data_type)
json_string = JSON.serialize(canonical_type)     // compact, keys sorted
final_digest.update( json_string.as_bytes() )

// 2. Build BTreeMap entries from the type tree (same as record-batch path)
fields = extract_type_entries(effective_data_type, nullable, root_path="")

// 3. Traverse and populate entries
traverse_and_update(effective_data_type, nullable, effective_array, "", fields)

// 4. Finalize all entries into the digest (same order as record-batch finalize)
for (_, entry) in fields:
    finalize_digest(final_digest, entry)   // see Section 4

raw_hash = final_digest.finalize()    // 32 bytes
output = [0x00, 0x00, 0x01] ++ raw_hash   // 35 bytes
```

Dictionary arrays are resolved to their value type before hashing.

---

## 7. Worked Examples

### Example A: Simple Two-Column Table

**Schema**: `{age: Int32 non-nullable, name: LargeUtf8 nullable}`

**Data** (1 record batch, 2 rows):

| age | name    |
|-----|---------|
| 25  | "Alice" |
| 30  | NULL    |

#### Step 1: Schema Digest

Canonical JSON (compact):
```
{"age":{"data_type":"Int32","nullable":false},"name":{"data_type":"LargeUtf8","nullable":true}}
```

```
schema_digest = SHA-256("{"age":{"data_type":"Int32","nullable":false},"name":{"data_type":"LargeUtf8","nullable":true}}")
```

#### Step 2: Field "age" (Int32, non-nullable)

Values: `[25, 30]`

Little-endian bytes:
- 25 as i32 LE: `19 00 00 00`
- 30 as i32 LE: `1e 00 00 00`

Data fed to digest: `19 00 00 00 1e 00 00 00` (8 bytes, one contiguous slice)

```
age_data_digest = SHA-256(0x19000000_1e000000)
```

Finalization into final_digest (non-nullable):
```
final_digest.update( age_data_digest.finalize() )   // 32 bytes
```

#### Step 3: Field "name" (LargeUtf8, nullable)

Values: `["Alice", NULL]`

**Validity bits** (Lsb0 in u8 words):
- Element 0 ("Alice"): valid → bit = 1
- Element 1 (NULL): null → bit = 0
- BitVec contents: bits `[1, 0]`, bit_count = 2
- As u8 (Lsb0): bit 0 = 1, bit 1 = 0 → binary `0000_0001` = 1
- `as_raw_slice()` = `[1_u8]`

Validity serialization:
```
bit_count LE:  02 00 00 00 00 00 00 00     (2 as u64 little-endian)
word 0 LE:     01                           (1 as u8)
```

**Data bytes** (only valid elements):
- "Alice": length 5 as u64 LE = `05 00 00 00 00 00 00 00`, then UTF-8 bytes `41 6c 69 63 65`
- NULL: skipped entirely

```
name_data_digest = SHA-256(0x0500000000000000_416c696365)
```

Finalization into final_digest (nullable):
```
final_digest.update( 0x0200000000000000 )                   // bit count (u64 LE)
final_digest.update( 0x01 )                                  // word 0 (u8)
final_digest.update( name_data_digest.finalize() )           // 32 bytes
```

#### Step 4: Final Combination

Fields in alphabetical order: `age`, then `name`.

```
final_digest = SHA-256()
final_digest.update( schema_digest )                          // 32 bytes
final_digest.update( age_data_digest.finalize() )             // 32 bytes (non-nullable)
final_digest.update( 0x0200000000000000 )                     // name bit count (u64 LE)
final_digest.update( 0x01 )                                   // name validity word (u8)
final_digest.update( name_data_digest.finalize() )            // 32 bytes
raw_hash = final_digest.finalize()
output = 0x000001 ++ raw_hash
```

---

### Example B: Boolean Array with Nulls (hash_array API)

**Array**: `BooleanArray [true, NULL, false, true]` (nullable)

#### Step 1: Type Metadata

Canonical type JSON: `"Boolean"` (7 bytes as UTF-8)

```
final_digest.update(b'"Boolean"')
```

Note: `serde_json::to_string` of a JSON string value includes the surrounding quotes.

#### Step 2: Data

**Validity bits** (Lsb0 in u8):
- `[1, 0, 1, 1]` → bits: b0=1, b1=0, b2=1, b3=1
- As u8 (Lsb0): binary `0000_1101` = 13
- `as_raw_slice()` = `[13_u8]`

**Data** (one byte per valid value, Section 3.2):
- Valid values: `[true, false, true]` → `01 00 01`

```
data_digest = SHA-256(0x010001)
```

(v0.4.0 fed the bit-packed byte `0x05`; Section 10.)

#### Step 3: Finalization

```
final_digest = SHA-256()
final_digest.update(b'"Boolean"')                             // type metadata
final_digest.update( 0x0400000000000000 )                     // 4 bits (bit count as u64 LE)
final_digest.update( 0x0D )                                   // 13 as u8
final_digest.update( data_digest.finalize() )                 // 32 bytes
raw_hash = final_digest.finalize()
output = 0x000001 ++ raw_hash
```

---

### Example C: Non-Nullable Int32 Array (hash_array API)

**Array**: `Int32Array [1, 2, 3]` (non-nullable)

#### Step 1: Type Metadata

Canonical type JSON: `"Int32"` (6 bytes: `22 49 6e 74 33 32 22`... wait, `"Int32"` is the JSON string `"Int32"` including quotes)

Actually: `serde_json::to_string(&json!("Int32"))` produces `"\"Int32\""`, but `data_type_to_value` for Int32 produces the JSON value `"Int32"` (a JSON string). Then `serde_json::to_string` of that JSON string value produces `"\"Int32\""` — the 7-byte string `"Int32"` with quotes.

```
final_digest.update(b'"Int32"')     // 7 bytes: 22 49 6e 74 33 32 22
```

#### Step 2: Data

Values as i32 LE bytes:
- 1: `01 00 00 00`
- 2: `02 00 00 00`
- 3: `03 00 00 00`

Entire buffer fed as one slice: `01 00 00 00 02 00 00 00 03 00 00 00` (12 bytes)

```
data_digest = SHA-256(0x010000000200000003000000)
```

#### Step 3: Finalization (non-nullable)

```
final_digest = SHA-256()
final_digest.update(b'"Int32"')                               // 7 bytes
final_digest.update( data_digest.finalize() )                 // 32 bytes
raw_hash = final_digest.finalize()
output = 0x000001 ++ raw_hash
```

---

### Example D: Binary Array (hash_array API)

**Array**: `BinaryArray [b"hi", b""]` (non-nullable)

#### Step 1: Type Metadata

`Binary` is canonicalized to `LargeBinary`.

```
final_digest.update(b'"LargeBinary"')      // 13 bytes
```

#### Step 2: Data

Each element: `[u64 LE length] [raw bytes]`

- `b"hi"`: length 2 → `02 00 00 00 00 00 00 00` + `68 69`
- `b""`: length 0 → `00 00 00 00 00 00 00 00` (no raw bytes)

```
data_digest = SHA-256(0x0200000000000000_6869_0000000000000000)
```

#### Step 3: Finalization (non-nullable)

```
final_digest = SHA-256()
final_digest.update(b'"LargeBinary"')
final_digest.update( data_digest.finalize() )
raw_hash = final_digest.finalize()
output = 0x000001 ++ raw_hash
```

---

### Example E: Column-Order Independence

Two record batches with the same logical data but different column orders must produce identical hashes.

**Batch 1** (columns: x, y):
```
Schema: {x: Int32 non-nullable, y: Boolean nullable}
x: [10]
y: [true]
```

**Batch 2** (columns: y, x):
```
Schema: {y: Boolean nullable, x: Int32 non-nullable}
y: [true]
x: [10]
```

Both produce the same canonical schema JSON:
```
{"x":{"data_type":"Int32","nullable":false},"y":{"data_type":"Boolean","nullable":true}}
```

Both produce the same field digests (fields processed alphabetically: `x` then `y`):
- Field `x`: `SHA-256(0x0a000000)` (10 as i32 LE)
- Field `y`: validity `[1]` (1 bit, 1 word), data `0x01` (one byte for `true`)

Therefore `hash_record_batch(batch1) == hash_record_batch(batch2)`.

---

### Example F: Type Equivalence (Utf8 vs LargeUtf8)

**Array 1**: `StringArray ["ab"]` (non-nullable, Arrow type `Utf8`)
**Array 2**: `LargeStringArray ["ab"]` (non-nullable, Arrow type `LargeUtf8`)

Both produce the same type metadata: `"LargeUtf8"` (after canonicalization).

Both produce the same data bytes:
```
02 00 00 00 00 00 00 00   (length 2 as u64 LE)
61 62                      ("ab" as UTF-8)
```

Therefore `hash_array(array1) == hash_array(array2)`.

---

### Example G: Nullable Int32 Array with Nulls (hash_array API)

**Array**: `Int32Array [Some(42), None, Some(-7), Some(0)]` (nullable)

#### Step 1: Type Metadata

```
final_digest.update(b'"Int32"')     // 7 bytes
```

#### Step 2: Data

**Validity bits** (Lsb0 in u8):
- `[1, 0, 1, 1]` → bits: b0=1, b1=0, b2=1, b3=1
- As u8 (Lsb0): binary `0000_1101` = 13
- bit_count = 4

**Data bytes** (only valid elements):
- 42 as i32 LE: `2a 00 00 00`
- -7 as i32 LE: `f9 ff ff ff`
-  0 as i32 LE: `00 00 00 00`

```
data_digest = SHA-256(0x2a000000_f9ffffff_00000000)
```

#### Step 3: Finalization (nullable)

```
final_digest = SHA-256()
final_digest.update(b'"Int32"')                                 // type metadata
final_digest.update( 0x0400000000000000 )                       // 4 bits (bit count as u64 LE)
final_digest.update( 0x0D )                                     // 13 as u8
final_digest.update( data_digest.finalize() )                   // 32 bytes
raw_hash = final_digest.finalize()
output = 0x000001 ++ raw_hash
```

---

### Example H: Nullable String Array with Nulls (hash_array API)

**Array**: `StringArray [Some("hello"), None, Some("world"), Some("")]` (nullable, Arrow type `Utf8`)

#### Step 1: Type Metadata

`Utf8` is canonicalized to `LargeUtf8`.

```
final_digest.update(b'"LargeUtf8"')     // 12 bytes
```

#### Step 2: Data

**Validity bits** (Lsb0 in u8):
- `[1, 0, 1, 1]` → 0b1101 = 13
- bit_count = 4

**Data bytes** (only valid elements, null skipped entirely):
- `"hello"`: `05 00 00 00 00 00 00 00` (len=5 as u64 LE) + `68 65 6c 6c 6f`
- `"world"`: `05 00 00 00 00 00 00 00` (len=5 as u64 LE) + `77 6f 72 6c 64`
- `""`: `00 00 00 00 00 00 00 00` (len=0 as u64 LE, no raw bytes)

```
data_digest = SHA-256(len+"hello" + len+"world" + len+"")
```

#### Step 3: Finalization (nullable)

```
final_digest = SHA-256()
final_digest.update(b'"LargeUtf8"')
final_digest.update( 0x0400000000000000 )                       // bit_count=4 as u64 LE
final_digest.update( 0x0D )                                     // validity=13 as u8
final_digest.update( data_digest.finalize() )                   // 32 bytes
raw_hash = final_digest.finalize()
output = 0x000001 ++ raw_hash
```

---

### Example I: Empty Table (no data, schema only)

**Schema**: `{a: Int32 non-nullable, b: Boolean nullable}`

When no record batches are fed (i.e., `finalize()` is called immediately after construction), the field digests still exist — they just contain no data.

#### Schema Digest

```
schema_json = '{"a":{"data_type":"Int32","nullable":false},"b":{"data_type":"Boolean","nullable":true}}'
schema_digest = SHA-256(schema_json)
```

#### Field "a" (Int32, non-nullable)

No data was fed, so:
```
a_data_digest = SHA-256("")     // SHA-256 of empty input
```

#### Field "b" (Boolean, nullable)

No data was fed:
- `bit_count` = 0 (no elements, BitVec is empty)
- `as_raw_slice()` = `[]` (no words)
- Data digest = SHA-256 of empty input

#### Final Combination

```
final_digest = SHA-256()
final_digest.update( schema_digest )                             // 32 bytes
final_digest.update( SHA-256("").finalize() )                    // field "a" (non-nullable, 32 bytes)
final_digest.update( 0x0000000000000000 )                        // field "b" bit_count=0 (u64 LE)
// no validity words (raw_slice is empty for 0-length BitVec)
final_digest.update( SHA-256("").finalize() )                    // field "b" data (32 bytes)
output = 0x000001 ++ final_digest.finalize()
```

---

### Example J: Multi-Batch Streaming (batch-split independence)

**Schema**: `{v: Int32 non-nullable}`

Feeding two batches must produce the same hash as feeding one combined batch:

- **Batch 1**: `v = [1, 2]`
- **Batch 2**: `v = [3]`
- **Combined**: `v = [1, 2, 3]`

Because the internal SHA-256 state is incremental:
```
update(01 00 00 00  02 00 00 00)   // from batch 1
update(03 00 00 00)                // from batch 2
```
is identical to:
```
update(01 00 00 00  02 00 00 00  03 00 00 00)   // single combined batch
```

#### Manual Computation

```
schema_json = '{"v":{"data_type":"Int32","nullable":false}}'
schema_digest = SHA-256(schema_json)

v_data_digest = SHA-256(0x010000000200000003000000)

final_digest = SHA-256()
final_digest.update( schema_digest )
final_digest.update( v_data_digest.finalize() )
output = 0x000001 ++ final_digest.finalize()
```

Therefore `hash(batch1 + batch2) == hash(combined)`.

---

### Example K: Struct Column in a Record Batch

**Schema**: `{person: Struct<age: Int32 non-null, name: LargeUtf8 non-null> non-nullable}`

**Data** (2 rows):

| person.age | person.name |
|------------|-------------|
| 25         | "Alice"     |
| 30         | "Bob"       |

In the record-batch path, the struct is **decomposed into leaf fields**: `person/age` and `person/name`. Each is hashed independently.

#### Step 1: Schema Digest

Canonical JSON:
```
{"person":{"data_type":{"Struct":[{"data_type":"Int32","name":"age","nullable":false},{"data_type":"LargeUtf8","name":"name","nullable":false}]},"nullable":false}}
```

#### Step 2: Leaf field "person/age" (Int32, non-nullable)

```
age_data_digest = SHA-256(0x19000000_1e000000)    // [25, 30] as i32 LE
```

#### Step 3: Leaf field "person/name" (LargeUtf8, non-nullable)

```
name_data_digest = SHA-256(
    0x0500000000000000 "Alice"    // len=5 u64 LE + UTF-8
    0x0300000000000000 "Bob"      // len=3 u64 LE + UTF-8
)
```

#### Step 4: Final Combination

Fields alphabetically: `person/age`, `person/name`.

```
final_digest = SHA-256()
final_digest.update( schema_digest )                     // 32 bytes
final_digest.update( age_data_digest.finalize() )        // 32 bytes (non-nullable)
final_digest.update( name_data_digest.finalize() )       // 32 bytes (non-nullable)
output = 0x000001 ++ final_digest.finalize()
```

---

### Example L: Struct Array via hash_array (non-nullable, decomposed)

**Array**: `StructArray [{a: 1, b: true}, {a: 2, b: false}]`

Children: `a: Int32 non-null`, `b: Boolean non-null`. Struct is non-nullable.

`hash_array` uses the same recursive decomposition as the record-batch path. The struct has no nulls, so it is treated as non-nullable and gets no entry of its own. Children become separate entries.

#### Step 1: Type Metadata

Canonical type JSON (struct fields sorted alphabetically, keys sorted):
```
{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"Boolean","name":"b","nullable":false}]}
```

#### Step 2: Decomposed Entries

BTreeMap entries (sorted by key): `"a"`, `"b"`

**Entry "a"** (Int32, non-nullable → data-only):
```
data_a = SHA-256(0x01000000_02000000)    // [1, 2] as i32 LE
```

**Entry "b"** (Boolean, non-nullable → data-only):
```
// [true, false] → one byte per value: 01 00   (v0.4.0: bit-packed 0x01)
data_b = SHA-256(0x0100)
```

#### Step 3: Finalization

Each entry is non-nullable → no null_bits, no structural, just data.finalize().

```
final_digest = SHA-256()
final_digest.update( type_json_bytes )       // type metadata
final_digest.update( data_a.finalize() )     // entry "a": 32 bytes
final_digest.update( data_b.finalize() )     // entry "b": 32 bytes
output = 0x000001 ++ final_digest.finalize()
```

---

### Example M: Nullable Struct Array via hash_array (struct-level nulls, decomposed)

**Array**: `StructArray [Some({a: 10, b: "x"}), None, Some({a: 30, b: "z"})]`

Children: `a: Int32 non-null`, `b: LargeUtf8 non-null`. Struct is **nullable**.

Row 1 is a null struct. The array has a null, so it is nullable: the struct gets a validity-only entry at its path `""` (Section 3.5). Its null is also AND-propagated to the children for data hashing; since the children are non-nullable per their Field definitions, their entries have no null_bits, and null rows are skipped in their data streams. (v0.4.0 had no `""` entry, so the null row vanished; Section 10.)

#### Step 1: Type Metadata

```
{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"LargeUtf8","name":"b","nullable":false}]}
```

#### Step 2: Decomposed Entries (with struct-null propagation)

BTreeMap entries (sorted by key): `""`, `"a"`, `"b"`

**Entry ""** (struct validity, validity-only): bits `[1, 0, 1]`, bit count 3, word `0b101` = `0x05`.

**Entry "a"** (Int32, non-nullable → data-only):
- Struct nulls propagated: rows 0, 2 valid → data: `[10, 30]`

```
data_a = SHA-256(0x0a000000_1e000000)     // [10, 30] as i32 LE
```

**Entry "b"** (LargeUtf8, non-nullable → data-only):
- Struct nulls propagated: rows 0, 2 valid → data: `"x"`, `"z"`

```
data_b = SHA-256(
    0x0100000000000000 "x"     // len=1 + "x"
    0x0100000000000000 "z"     // len=1 + "z"
)
```

#### Step 3: Finalization

```
final_digest = SHA-256()
final_digest.update( type_json_bytes )       // type metadata
final_digest.update( 0x0300000000000000 )    // entry "": bit count 3 (u64 LE)
final_digest.update( 0x05 )                  // entry "": validity word 0b101
final_digest.update( data_a.finalize() )     // entry "a": 32 bytes
final_digest.update( data_b.finalize() )     // entry "b": 32 bytes
output = 0x000001 ++ final_digest.finalize()
```

---

### Example N: List-of-Struct in a Record Batch (Recursive Decomposition)

**Schema**: `{items: LargeList<Struct<id: Int32 non-null, label: LargeUtf8 non-null>> nullable}`

**Data** (2 rows):

| items |
|-------|
| `[{id: 1, label: "a"}, {id: 2, label: "b"}]` |
| `[{id: 3, label: "c"}]` |

The list-of-struct column is **recursively decomposed** into four BTreeMap entries:

| Path | Entry type | Components |
|------|-----------|------------|
| `items` | validity-only | null_bits: `[V, V]` (2 bits) |
| `items/` | structural-only | list lengths: `[2, 1]` |
| `items//id` | data-only | leaf values: `[1, 2, 3]` as i32 LE |
| `items//label` | data-only | leaf values: `len+"a"`, `len+"b"`, `len+"c"` |

Note the path naming: `items` (column) → `items/` (list adds `/`) → `items//id` (struct adds `/id`, producing `//` because parent ends in `/`).

#### Step 1: Schema Digest

Canonical JSON (element type omits Arrow-internal field name "item"):
```
{"items":{"data_type":{"LargeList":{"data_type":{"Struct":[{"data_type":"Int32","name":"id","nullable":false},{"data_type":"LargeUtf8","name":"label","nullable":false}]},"nullable":false}},"nullable":true}}
```

#### Step 2: Traversal

The top-down recursive traversal processes each row:

**Row 0** (valid list, 2 elements):
- `items` entry: push `valid` to null_bits
- `items/` entry: write `2_u64.to_le_bytes()` to structural
- Recurse into sub-array `[{id:1, label:"a"}, {id:2, label:"b"}]`:
  - The struct elements are non-nullable, so no struct validity is recorded — recurse into children ("id", "label"):
    - `items//id` entry: write `1_i32.to_le_bytes()`, `2_i32.to_le_bytes()` to data
    - `items//label` entry: write `len+"a"`, `len+"b"` to data

**Row 1** (valid list, 1 element):
- `items` entry: push `valid` to null_bits
- `items/` entry: write `1_u64.to_le_bytes()` to structural
- Recurse into sub-array `[{id:3, label:"c"}]`:
  - `items//id` entry: write `3_i32.to_le_bytes()` to data
  - `items//label` entry: write `len+"c"` to data

#### Step 3: Final Combination

Entries are finalized in BTreeMap (alphabetical) order:

```
final_digest = SHA-256()
final_digest.update( schema_digest )                              // 32 bytes

// Entry "items" (validity-only)
final_digest.update( 0x0200000000000000 )                         // bit_count=2 (u64 LE)
final_digest.update( 0x03 )                                       // validity word: 0b11 = 3 (u8)

// Entry "items/" (structural-only)
items_structural = SHA-256(
    0x0200000000000000                                            // row 0: 2 elements
    0x0100000000000000                                            // row 1: 1 element
)
final_digest.update( items_structural.finalize() )                // 32 bytes

// Entry "items//id" (data-only)
id_data = SHA-256(
    0x01000000                                                    // 1 as i32 LE
    0x02000000                                                    // 2 as i32 LE
    0x03000000                                                    // 3 as i32 LE
)
final_digest.update( id_data.finalize() )                         // 32 bytes

// Entry "items//label" (data-only)
label_data = SHA-256(
    0x0100000000000000 0x61                                       // len=1 + "a"
    0x0100000000000000 0x62                                       // len=1 + "b"
    0x0100000000000000 0x63                                       // len=1 + "c"
)
final_digest.update( label_data.finalize() )                      // 32 bytes

output = 0x000001 ++ final_digest.finalize()
```

---

### Example O: Nested Struct in a Record Batch (Two Levels of Struct)

**Schema**: `{s: Struct<a: Int32 non-null, nested: Struct<p: Int32 non-null, q: Int32 non-null> non-null> non-null}`

**Data** (1 row):

| s.a | s.nested.p | s.nested.q |
|-----|------------|------------|
| 10  | 20         | 30         |

Both struct levels are transparent. The recursive decomposition produces three leaf entries:

| Path | Entry type | Data |
|------|-----------|------|
| `s/a` | data-only | `[10]` as i32 LE |
| `s/nested/p` | data-only | `[20]` as i32 LE |
| `s/nested/q` | data-only | `[30]` as i32 LE |

Note: `s/nested/p` sorts after `s/a` but before `s/nested/q` — the full path string is compared alphabetically.

#### Step 1: Schema Digest

Outer struct children sorted: `[a, nested]`. Inner struct children sorted: `[p, q]`.

Canonical JSON:
```
{"s":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":{"Struct":[{"data_type":"Int32","name":"p","nullable":false},{"data_type":"Int32","name":"q","nullable":false}]},"name":"nested","nullable":false}]},"nullable":false}}
```

```
schema_digest = SHA-256(canonical_json_bytes)
```

#### Step 2: Leaf Entries

**Entry `s/a`** (Int32, non-nullable → data-only):
```
data_sa = SHA-256(0x0a000000)     // 10 as i32 LE
```

**Entry `s/nested/p`** (Int32, non-nullable → data-only):
```
data_snp = SHA-256(0x14000000)    // 20 as i32 LE
```

**Entry `s/nested/q`** (Int32, non-nullable → data-only):
```
data_snq = SHA-256(0x1e000000)    // 30 as i32 LE
```

#### Step 3: Final Combination

Entries in alphabetical path order: `s/a`, `s/nested/p`, `s/nested/q`.

```
final_digest = SHA-256()
final_digest.update( schema_digest )              // 32 bytes
final_digest.update( data_sa.finalize() )         // "s/a" (non-nullable): 32 bytes
final_digest.update( data_snp.finalize() )        // "s/nested/p" (non-nullable): 32 bytes
final_digest.update( data_snq.finalize() )        // "s/nested/q" (non-nullable): 32 bytes
output = 0x000001 ++ final_digest.finalize()
```

---

### Example P: Nested Struct Field-Order Independence (Schema Hash)

Two schemas with the same logical structure but different field declaration orders must produce identical hashes after recursive alphabetical sorting.

**Schema 1** (outer: `[nested, z]`, inner: `[b, a]` — neither sorted):
```
{s: Struct<nested: Struct<b: Int32, a: Int32> non-null, z: Int32 non-null> non-null}
```

**Schema 2** (outer: `[z, nested]`, inner: `[a, b]` — both sorted):
```
{s: Struct<z: Int32 non-null, nested: Struct<a: Int32, b: Int32> non-null> non-null}
```

#### Canonical JSON (same for both schemas)

After recursive alphabetical sorting, outer children are `[nested, z]` (`n < z`) and inner children are `[a, b]` (`a < b`):

```
{"s":{"data_type":{"Struct":[{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"Int32","name":"b","nullable":false}]},"name":"nested","nullable":false},{"data_type":"Int32","name":"z","nullable":false}]},"nullable":false}}
```

```
schema_digest = SHA-256(canonical_json_bytes)
output = 0x000001 ++ schema_digest
```

Both schemas produce this same output. This confirms that recursive sorting normalizes field declaration order at every nesting level.

---

### Example Q: Nested Struct via hash_array

**Array**: `StructArray [{inner: {x: 5, y: 7}}, {inner: {x: 9, y: 11}}]`

Outer struct: non-nullable, one child `inner` (non-nullable Struct). Inner struct: non-nullable, children `x: Int32`, `y: Int32`.

This example uses the `hash_array` API (Section 6). The layout mirrors the record-batch path — struct levels are transparent, leaves become BTreeMap entries.

BTreeMap entries (sorted by path): `inner/x`, `inner/y`.

#### Step 1: Type Metadata

Canonical type JSON (outer struct → one child `inner`; inner struct sorted: `[x, y]`):
```
{"Struct":[{"data_type":{"Struct":[{"data_type":"Int32","name":"x","nullable":false},{"data_type":"Int32","name":"y","nullable":false}]},"name":"inner","nullable":false}]}
```

```
final_digest.update( type_json_bytes )     // type metadata (not schema_digest)
```

#### Step 2: Leaf Entries

**Entry `inner/x`** (Int32, non-nullable → data-only):
```
data_x = SHA-256(
    0x05000000    // 5 as i32 LE
    0x09000000    // 9 as i32 LE
)
```

**Entry `inner/y`** (Int32, non-nullable → data-only):
```
data_y = SHA-256(
    0x07000000    // 7 as i32 LE
    0x0b000000    // 11 as i32 LE
)
```

#### Step 3: Finalization

```
final_digest = SHA-256()
final_digest.update( type_json_bytes )       // type metadata (canonical type JSON above)
final_digest.update( data_x.finalize() )     // "inner/x": 32 bytes
final_digest.update( data_y.finalize() )     // "inner/y": 32 bytes
output = 0x000001 ++ final_digest.finalize()
```

Note: unlike `hash_record_batch`, `hash_array` feeds the **type JSON** (not a schema digest) directly into the final digest. See Section 6 for the full `hash_array` layout.

---

### Example R: hash_schema with Per-Field Metadata (include_metadata = true)

**Schema**: `{v: Int32 non-nullable}`, field `v` metadata: `{"unit": "meters"}`, `include_metadata = true`

#### Step 1 (Phase 1): Canonical JSON

The canonical JSON string for this schema is:
```
{"v":{"data_type":"Int32","nullable":false}}
```

UTF-8 bytes (44 bytes):
```
7B 22 76 22 3A 7B 22 64 61 74 61 5F 74 79 70 65
22 3A 22 49 6E 74 33 32 22 2C 22 6E 75 6C 6C 61
62 6C 65 22 3A 66 61 6C 73 65 7D 7D
```

```
hasher.update(canonical_json_string_bytes)    // 44 bytes
```

#### Step 2 (Phase 2): Per-Field Metadata

```
── Phase 2: single JSON metadata update ─────────────────────────────
// meta_doc = {"fields": {"v": {"unit":"meters"}}}
// ("schema" key absent — no schema-level metadata)
hasher.update( {"fields":{"v":{"unit":"meters"}}} )   ← 34 bytes
```

Total bytes fed to hasher: 44 (Phase 1) + 34 (Phase 2) = 78 bytes.

#### Step 3: Schema Digest

```
schema_digest = hasher.finalize()    // 32 bytes
```

#### Step 4: Final Output

```
output = 00 00 01 || schema_digest   // 35 bytes
```

Note: with `include_metadata = false` for this same schema, only the 44 Phase 1 bytes would be fed into SHA-256, producing a different digest.

---

### Example S: hash_schema with Schema-Level Metadata (include_metadata = true)

**Schema**: `{v: Int32 non-nullable}`, no field metadata, schema-level metadata: `{"source": "sensor-1"}`, `include_metadata = true`

#### Step 1 (Phase 1): Canonical JSON

Same schema structure as Example R:
```
{"v":{"data_type":"Int32","nullable":false}}
```

```
hasher.update(canonical_json_string_bytes)    // 44 bytes (same as Example R)
```

#### Step 2 (Phase 2): Metadata

```
── Phase 2: single JSON metadata update ─────────────────────────────
// meta_doc = {"schema": {"source":"sensor-1"}}
// ("fields" key absent — no field has metadata)
hasher.update( {"schema":{"source":"sensor-1"}} )   ← 32 bytes
```

Total bytes fed to hasher: 44 (Phase 1) + 32 (Phase 2) = 76 bytes.

#### Step 4: Schema Digest and Final Output

```
schema_digest = hasher.finalize()           // 32 bytes
output = 00 00 01 || schema_digest          // 35 bytes
```

---

## 8. Platform Considerations

- **Integer sizes**: All length prefixes use `u64` (8 bytes, LE). Validity bitmaps use `BitVec<u8, Lsb0>` (1 byte per word). Bit counts use `u64` (8 bytes, LE). Hashes are **platform-independent**.
- **Byte order**: All values use little-endian. Validity words are `u8` (1 byte, so endianness is trivial). Bit counts use little-endian.
- **Floating point**: IEEE 754 representation is hashed directly. `NaN` values with different bit patterns produce different hashes. `+0.0` and `-0.0` produce different hashes.

---

## 9. Hashing Options: Order Sensitivity (v0.5)

Sections 1–8 describe the default layout. This section specifies two opt-in options. With both
off, nothing in this section applies. Rust is the reference implementation; the golden vectors
in `tests/golden/ordered_v0.5.json` pin the values (Section 9.11).

### 9.1 Options and the flags byte

| Option | Rust (`HasherConfig` field) / Python keyword | Flag bit | Effect |
|---|---|---|---|
| order of top-level columns is part of the hash | `ordered_columns` | `0x01` | Section 9.3 |
| order of struct fields is part of the hash, at every depth | `ordered_struct_fields` | `0x02` | Sections 9.3, 9.4 |

```
flags = (ordered_columns ? 0x01 : 0) | (ordered_struct_fields ? 0x02 : 0)
```

Bits 2–7 are reserved and always 0. **Bit 2 (`0x04`) must never be reused:** during the v0.5
work it briefly selected a provisional third option (the layout fixes of Section 10), which was
never released; those fixes are now the only layout. `include_metadata` is not a flag bit: it
keeps its Section 2.2.1 behaviour in every mode. Options are per call (per streaming digester);
they are never global.

Every entry point takes both options. `ordered_columns` has no effect on the `hash_array` layout
but still enters its flags byte. There is no per-entry-point masking.

### 9.2 Options tag (domain separation)

```
options_tag = 0x00 || "starfix-options-v1" || flags        // 1 + 18 + 1 = 20 bytes
            = 00 73 74 61 72 66 69 78 2d 6f 70 74 69 6f 6e 73 2d 76 31 <flags>
```

When `flags == 0x00` the tag is **not** written. When `flags != 0x00`:

**Schema digest** (`hash_schema`, and the schema digest at the start of every record-batch /
streaming hash, Section 5):

```
hasher = SHA-256()
hasher.update(options_tag)                                  // 20 bytes
hasher.update(phase1_json(schema, flags))                   // Section 9.3
if include_metadata: Phase 2 exactly as Section 2.2.1       // unchanged
schema_digest = hasher.finalize()
```

`hash_schema` returns `<version prefix> || schema_digest`. The record-batch final digest
(Section 5) starts with this `schema_digest`, then the entries, exactly as in Section 5.

**`hash_array`** (Section 6): the tag is the first input of the final digest:

```
final_digest = SHA-256()
final_digest.update(options_tag)                            // 20 bytes
final_digest.update(type_json(effective_type, flags))       // Section 9.4
... entries exactly as Section 6 ...
output = <version prefix> || final_digest.finalize()
```

The options do not change the version prefix.

**Why this separates the modes.** Untagged inputs to these SHA-256 instances always start with
canonical JSON (`{`, `[` or `"`), never with `0x00`. Tagged inputs start with `0x00` and carry the
whole `flags` byte at offset 19. So the SHA-256 inputs for two different `flags` values always
differ, and equal hashes would require a SHA-256 collision. In particular a table hashed with
an order option on never has the same hash as with the options off, even when order cannot
matter (a one-column table without structs). For record batches the separation is carried by
the schema digest, the first 32 bytes of the final input.

Within one `flags` value the empty-metadata invariant (Section 2.2.1) still holds:
`include_metadata = true` on a schema without metadata gives the same hash as `false`.

### 9.3 Phase 1 canonical schema JSON with options

Same canonicalization as Section 2.1 (type equivalence classes, dictionary replaced by its value
type, list elements without their field name, compact JSON, object keys sorted recursively),
with two changes:

**`ordered_columns`.** The top level is a **JSON array** with one element per column, in schema
order. Each element has the struct-child form, keys sorted:

```
{"data_type": <canonical type>, "name": "<column name>", "nullable": <bool>}
```

Without `ordered_columns` the top level is the Section 2.1 object keyed by column name.

**`ordered_struct_fields`.** In every `{"Struct": [...]}` array the children appear in their
**declaration order** instead of sorted by name. This applies wherever a struct type occurs: a
column, a struct child, a list / large list element (`{"LargeList": {...}}`), a fixed-size list
element (`{"FixedSizeList": [{...}, size]}`), map entries (`{"Map": [{...}, sorted]}`, whose
entries struct lists `key` then `value` as declared), a dictionary value type, and any nesting
of these. Object keys inside each child (`data_type`, `name`, `nullable`) stay sorted.

Without `ordered_struct_fields` struct children are sorted by name (Section 2.1).

Examples. Schema `[b: Int32 non-null, a: Utf8 nullable]`:

```
flags 0x00 (no tag):  {"a":{"data_type":"LargeUtf8","nullable":true},"b":{"data_type":"Int32","nullable":false}}
flags 0x01:           [{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}]
```

Schema `[s: Struct<y: Boolean non-null, x: Int32 non-null> non-null]`:

```
flags 0x00, 0x01:  ..."Struct":[{"data_type":"Int32","name":"x",...},{"data_type":"Boolean","name":"y",...}]...
flags 0x02, 0x03:  ..."Struct":[{"data_type":"Boolean","name":"y",...},{"data_type":"Int32","name":"x",...}]...
```

(With `ordered_columns` the whole thing is additionally wrapped as a top-level array element.)

**Data entries are unchanged by the order options.** Entry paths, entry contents and the
finalization order (sorted by path, Section 5) are exactly as in Sections 3–5. Position enters
the hash only through the canonical JSON: given the ordered JSON, every entry path names exactly
one field position (guaranteed by Section 9.6), so two inputs that differ only in column or
struct-field order differ in the schema digest (or, for `hash_array`, in the type JSON).

### 9.4 `hash_array` type JSON with options

`type_json` is the Section 6 canonical type JSON of the effective type, with the
`ordered_struct_fields` rule of Section 9.3. `ordered_columns` does not change it (it only enters
the flags byte). The top-level nullability used by `hash_array` is unchanged: the array is
treated as nullable iff it contains at least one null.

### 9.5 (removed)

This section described the provisional third option. Its layout is now the default
(Sections 3.2–3.5); see Section 10.

### 9.6 Rejected schemas (any option on)

When `flags != 0`, a schema (or, for `hash_array`, the array's type) is **rejected** (Rust
panics, documented under `# Panics`; Python raises `ValueError`) when

1. two of its nodes have the same **node path**, or
2. two of its fields have the same **metadata path** (Section 2.2.1), or
3. it contains a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type (their canonical
   JSON is not portable, Section 2.1).

With `flags == 0` nothing is checked (default behaviour unchanged).

```
seen_node = {}; seen_meta = {}
claim(np, mp): if np in seen_node or mp in seen_meta: reject; add np, mp
schema:     for column in schema: claim(column.name, column.name); walk(column.type, column.name, column.name)
hash_array: claim("", ""); walk(array_type, "", "")
walk(type, np, mp):
    t = normalize(type)
    if t is ListView, LargeListView, Union or RunEndEncoded: reject
    if t is Struct(children):
        for c in children:
            cnp = join(np, c.name); cmp = mp + "/" + c.name
            claim(cnp, cmp); walk(c.type, cnp, cmp)
    elif t is LargeList(e) or FixedSizeList(e, _) or Map(e, _):
        claim(np + "/", mp + "/"); walk(e.type, np + "/", mp + "/")
```

`join(p, n)` is `n` when `p` is empty, else `p + "/" + n`; the metadata path always appends
`"/" + n`, so the two differ only below a column named `""`. For `Map(e, _)`, `e` is the entries
field (a struct), so a map column `m` claims `m`, `m/`, `m//key` and `m//value`.

This rejects duplicate column names, duplicate field names within a struct (at any depth), names
containing `/` that alias another node (a column `"a/b"` next to a struct column `a` with child
`b`; a column `"x/"` next to a list column `x`), and a struct column `""` with child `a` next to a
column `"/a"` (equal metadata paths). A `/` that aliases nothing (`"km/h"`) is accepted. The check
runs in `hash_schema`, when a streaming digester is created, on every batch passed to `update`,
and in `hash_record_batch` and `hash_array`.

Reason: entries (and metadata) are keyed by path, so two nodes with one path would feed one
entry, making the hash depend on how rows are split into batches and letting different tables
collide.

### 9.7 Streaming

Every batch given to `update` is checked against the schema the digester was created with. The
batch is accepted iff

1. its Phase 1 JSON under the digester's `flags` (Section 9.3; with `flags == 0`, Section 2.1)
   equals the digester's, and
2. when `include_metadata` is on: its Phase 2 metadata document (Section 2.2.1) equals the
   digester's, or both have none; and
3. when `flags != 0`: it passes Section 9.6.

Otherwise it is rejected (Rust panics with "Record batch schema does not match ArrowDigester
schema"; Python raises `ValueError`). With `flags == 0` and `include_metadata` off this is exactly
the v0.4.0 check (Phase 1 JSON equality only). Because the Phase 1 JSON is canonical, a batch
whose types differ only physically (`Utf8` vs `LargeUtf8`, dictionary vs plain, `List` vs
`LargeList`) is accepted. Hence:

* `ordered_columns`: a batch with the same columns in a different order is rejected. Without it,
  such a batch is accepted (as in the default).
* `ordered_struct_fields`: a batch whose struct fields (any depth) are in a different order is
  rejected. Without it, it is accepted.

Batch-split independence (Example J) holds in every mode, for every supported type (Section 10).

A Python `hash_table(table)` is `new(table.schema)`, `update` for each batch of
`table.to_batches()`, `finalize`.

### 9.8 Invariants

| # | Invariant |
|---|---|
| I1 | `flags = 0`: no options tag; the default layout of Sections 1–8. |
| I2 | Hashes taken with different `flags` never coincide (modulo SHA-256 collisions). |
| I3 | `ordered_columns`: permuting columns changes the hash. Without it, it does not (whatever the other option). |
| I4 | `ordered_struct_fields`: permuting struct fields at any depth changes the hash. Without it, it does not. |
| I5 | In every mode: `Utf8`≡`LargeUtf8`, `Binary`≡`LargeBinary`, `List`≡`LargeList`, dictionary≡plain. |
| I6 | In every mode, for every supported type: splitting rows into batches does not change the hash; slicing (non-zero offsets) does not; a zero-row batch equals no batch; adding rows changes it (Section 10; `tests/lossless_properties.rs`, `tests/nested_family.rs`). |
| I7 | Within one `flags` value: `include_metadata = true` without any metadata equals `false`. |
| I8 | Any option on: ambiguous schemas are rejected (Section 9.6). |

Exceptions to I6: a table with zero columns does not hash its row count (in every mode).

### 9.9 Worked examples

**Example T1: `hash_schema`, `ordered_columns` (flags `0x01`).** Schema `[b: Int32 non-null, a: Utf8 nullable]`.

```
schema_digest = SHA-256(
    00 73 74 61 72 66 69 78 2d 6f 70 74 69 6f 6e 73 2d 76 31 01
    [{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}]
)
output = 00 00 01 || schema_digest
```

**Example T2: record batch, `ordered_columns`.** Columns `b: Int32 non-null [5]`, `a: Int32 non-null [6]`.

```
schema_digest = SHA-256(options_tag(0x01) ||
    [{"data_type":"Int32","name":"b","nullable":false},{"data_type":"Int32","name":"a","nullable":false}])
final = SHA-256(
    schema_digest
    SHA-256(06 00 00 00)        // entry "a" (entries stay sorted by path)
    SHA-256(05 00 00 00)        // entry "b"
)
output = 00 00 01 || final
```

**Example T3: `hash_array`, `ordered_struct_fields` (flags `0x02`).** `StructArray [{z: 1, a: 2}]`,
children declared `z: Int32 non-null`, `a: Int32 non-null`, no nulls.

```
final = SHA-256(
    options_tag(0x02)
    {"Struct":[{"data_type":"Int32","name":"z","nullable":false},{"data_type":"Int32","name":"a","nullable":false}]}
    SHA-256(02 00 00 00)        // entry "a"
    SHA-256(01 00 00 00)        // entry "z"
)
```

Examples of the default layout's v0.5 fixes are in Section 10.4.

### 9.10 Type support

The options change no type support. Data hashing supports `Boolean`, all integer and float types,
`Date32/64`, `Time32/64`, `Timestamp`, `Duration`, `Decimal32/64/128/256`, `FixedSizeBinary`,
`Binary`/`LargeBinary`, `Utf8`/`LargeUtf8`, `List`/`LargeList`, `Struct`, and `Dictionary` of a
supported value type. `Null`, `FixedSizeList`, `Map`, `Interval`, the view types, `Union` and
`RunEndEncoded` data are not supported (Rust panics with `not yet implemented`). Schema hashing
accepts all types, and the Section 9.3 struct rule applies inside each of them.

### 9.11 Golden vectors

`tests/golden/ordered_v0.5.json` is generated by Rust (`cargo run --bin emit_golden_ordered`,
append-only: the generator aborts if an existing vector's hash would change) and must be matched
exactly by every other implementation. Format:

```json
{
  "version": "0.5",
  "generated_by": "cargo run --bin emit_golden_ordered",
  "rust_commit": "<git sha>",
  "vectors": [
    { "id": "...", "description": "...",
      "method": "hash_schema | hash_record_batch | hash_stream | hash_array",
      "include_metadata": false, "ordered_columns": true,
      "ordered_struct_fields": false,
      "ipc_b64": "<base64 Arrow IPC stream>", "expected_hash": "000001..." }
  ],
  "relations": [ { "kind": "equal | different", "ids": ["...", "..."], "why": "..." } ]
}
```

| method | what to hash (IPC decoded with an Arrow stream reader) |
|---|---|
| `hash_schema` | the stream's schema |
| `hash_record_batch` | the single batch |
| `hash_stream` | `new(schema)`, `update` with each batch in order (possibly none), `finalize` |
| `hash_array` | column 0 of the single batch, as an array |

A relation `equal` requires all listed vectors to have the same `expected_hash`; `different`
requires them to be pairwise different. Vectors whose `id` ends in `_f0` use no options; a
`hash_array` vector with `_f0` also equals `hash_array` called without options. The file is
append-only: the generator aborts when a committed value would change, except for one reviewed
exception list (the v0.5 fix, Section 10.5).


---

## 10. v0.5 Fix: Information Losses in the v0.4.0 Layout

v0.4.0's data layout lost information for some types: different data hashed the same, and some
hashes depended on how rows were split into batches. These are bugs; v0.5 fixes them in the
default layout (developer ruling, ITL-898, 2026-11-11). Inputs whose type is affected (Section
10.2) get new hashes. **All other inputs keep their v0.4.0 hash byte for byte.**

> **Version prefix — open decision.** (a) Fix in place: the prefix stays `00 00 01`; unaffected
> inputs keep their v0.4.0 hash and only affected inputs move. (b) Bump the prefix to `00 00 02`:
> every hash moves (in its first three bytes; the 32-byte digest of unaffected inputs is still
> identical), so a v0.4.0 hash can always be told from a v0.5 hash. The implementation is one
> line either way (`VERSION_BYTES` in `src/lib.rs`). This document is written for (a).

### 10.1 The defects

| Label | v0.4.0 behaviour | Example of a v0.4.0 collision |
|---|---|---|
| `boolean_packing` | Boolean values were bit-packed per call into whole bytes; a non-nullable Boolean entry had no count, and each batch was padded separately | one-column non-nullable `[true]` == `[true, false, false]`; `hash_array([true]) == hash_array([true, false])`; one batch `[true, false, true]` != the same rows as two batches |
| `nullable_struct` | a struct had no entry; its nulls were only ANDed into its descendants | `[null]` == `[{a: null}]`; with non-nullable fields, `[{a:1}, null, {a:3}]` == `[{a:1}, {a:3}]` |
| `list_of_nullable_list` | in `List<nullable List<..>>` the inner list's validity entry *replaced* the outer list's element entry, so the outer per-row lengths were lost | rows `[[1]], [[2],[3]]` == rows `[[1],[2]], [[3]]`; in three levels even a single row: `[[[1]],[[2],[3]]]` == `[[[1],[2]],[[3]]]` |
| `list_of_nullable_struct` | in `List<nullable Struct<..>>` the element entry had `null_bits` that were never filled | element `null` == element `{a: null}` |

Found by a search over 420 nested column types (list / large list / struct / struct with a second
field, nullable or not, around integer and dictionary leaves, up to three levels): 268 of them
lost information in v0.4.0 (`tests/nested_family.rs`), and by a sweep of every supported leaf
type x nullability x position (top level, struct, list, nullable struct, nullable list) x
{row count, batch split, slice, empty vs absent}: 68 of 320 cases failed in v0.4.0
(`tests/lossless_properties.rs`). Both find nothing in v0.5, in every mode.

### 10.2 Which inputs are affected

A `hash_record_batch` / streaming / `hash_array` input is **affected** iff its type contains

* a `Boolean` (also as a dictionary value type), or
* a nullable struct (for `hash_array`, the array itself counts as nullable iff it has a null), or
* a `List` / `LargeList` whose element field is nullable and whose element type is a list or a
  struct.

`hash_schema` is never affected. Unaffected inputs hash byte-identically to v0.4.0 (checked
permanently by `tests/v0_5_fix.rs`). Affected inputs change for all but degenerate data (zero
rows; all-null Booleans; a single `true` Boolean, whose packed byte happens to equal the new
one).

### 10.3 The corrected layout

Sections 3.2, 3.4 and 3.5 are the corrected layout. In one place, entry extraction:

```
extract(type, nullable, path, list_element = false):
    t = normalize(type)
    if t is Struct(children):
        if nullable and not list_element:
            entries[path] = validity_only()
        for child in children:
            extract(child.type, child.nullable, join(path, child.name))
    elif t is LargeList(element):
        if nullable and not list_element:
            entries[path] = validity_only()
        lp = path + "/"
        if normalize(element.type) is Struct or LargeList:
            entries[lp] = structural_only(nullable = element.nullable)
            extract(element.type, element.nullable, lp, list_element = true)
        else:
            entries[lp] = list_leaf(nullable = element.nullable)
    else:
        entries[path] = data_only(nullable)
```

and, during traversal, a nullable struct array of length `n` at `path` pushes `n` combined
validity bits into `entries[path].null_bits` (Section 3.5); a nullable list already pushes its
validity into `entries[path].null_bits` (Section 3.4), which for a list element is the outer
element entry. Booleans: one byte per valid value (Section 3.2).

### 10.4 Worked examples

**Example V1: nullable struct.** Column `s: Struct<a: Int32 nullable> nullable`, rows
`[null, {a: null}, {a: 1}]` (child array `[5, null, 1]`, struct validity `[0, 1, 1]`).

```
schema_digest = SHA-256({"s":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":true}]},"nullable":true}})
final = SHA-256(
    schema_digest
    03 00 00 00 00 00 00 00  06          // entry "s": validity-only, bits [0,1,1] -> 0b110
    03 00 00 00 00 00 00 00  04          // entry "s/a": [1,0,1] AND [0,1,1] = [0,0,1] -> 0b100
    SHA-256(01 00 00 00)                 //   data: only row 2
)
```

**Example V2: list of nullable list.** Column `c: List<List<Int32 nullable> nullable>` non-null,
one row `[[1], [2]]`.

```
schema_digest = SHA-256({"c":{"data_type":{"LargeList":{"data_type":{"LargeList":{"data_type":"Int32","nullable":true}},"nullable":true}},"nullable":false}})
final = SHA-256(
    schema_digest
    // entry "c/": structural entry KEPT, null_bits = inner-list validity [1,1]
    02 00 00 00 00 00 00 00  03
    SHA-256(02 00 00 00 00 00 00 00)                          // outer lengths [2]
    // entry "c//": list-leaf, item validity [1,1], inner lengths [1,1], data [1,2]
    02 00 00 00 00 00 00 00  03
    SHA-256(01 00 00 00 00 00 00 00  01 00 00 00 00 00 00 00)
    SHA-256(01 00 00 00  02 00 00 00)
)
```

**Example V3: Booleans.** Column `b: Boolean nullable`, rows `[true, null, false]`:

```
final = SHA-256(
    SHA-256({"b":{"data_type":"Boolean","nullable":true}})
    03 00 00 00 00 00 00 00  05          // validity [1,0,1] -> 0b101
    SHA-256(01 00)                       // one byte per VALID value: true, false
)
```

`hash_array` of the non-nullable `BooleanArray [true, false, true]`:
`SHA-256( "Boolean" (with quotes) || SHA-256(01 00 01) )`.

Examples B, L and M of Section 7 show the corrected layout too.

### 10.5 Record of changed values

`tests/golden/fixed_in_v0.5.json` (checked by `tests/v0_5_fix.rs`) records:

* `existing_tests`: every value pinned by a v0.4.0 test that changed (7: two hash constants in
  `tests/arrow_digester.rs`, two Boolean byte-layout tests in `src/arrow_digester_core.rs`, and
  Examples B, L and M in `tests/digest_bytes.rs`), each with its v0.4.0 and v0.5 value and the
  defect;
* `reference_vectors`: every option-free vector of `tests/golden/ordered_v0.5.json` (99), with
  its v0.4.0 hash (computed by the released v0.4.0 crate) and its v0.5 hash, its status
  (`fixed` / `unchanged`), whether its type is affected (Section 10.2) and the defects.

`tests/golden/include_metadata_v0.3.json` (schema hashes only) is unchanged.
