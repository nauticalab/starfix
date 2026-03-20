# Starfix Byte Layout Specification

This document describes the **exact byte-level serialization** used by Starfix to compute deterministic hashes of Apache Arrow schemas and record batches. Every byte fed into SHA-256 is specified here, making it possible to implement a compatible hasher in any language.

All multi-byte integers use **little-endian** byte order unless explicitly stated otherwise.

---

## 1. Output Format

Every Starfix hash is **35 bytes**:

```
[version: 3 bytes] [SHA-256 digest: 32 bytes]
```

The version prefix is currently `0x00 0x00 0x01` (version 0.0.1).

When displayed as hex, a hash looks like:

```
000001 <64 hex chars of SHA-256>
```

---

## 2. Schema Serialization

### 2.1 Canonical JSON String

The schema is serialized as a **compact JSON string** (no whitespace) of an object where:

- **Keys** are field names, sorted alphabetically (via `BTreeMap`).
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
Each child object: `{"data_type": ..., "name": "<field_name>", "nullable": <bool>}`.

**List / LargeList elements** are serialized as:
```json
{"LargeList": {"data_type": ..., "nullable": <bool>}}
```
Note: the Arrow-internal field name (typically `"item"`) is **omitted** — only `data_type` and `nullable` are included.

**Primitive types** use Arrow's built-in serde:
- `"Int32"`, `"Boolean"`, `"Float64"`, `"LargeBinary"`, `"LargeUtf8"`, etc.
- `{"Decimal128": [38, 5]}`, `{"Time32": "Second"}`, etc.

### 2.2 Schema Digest

```
schema_digest = SHA-256(canonical_json_string_bytes)
```

The UTF-8 bytes of the JSON string are fed directly into SHA-256. The result is 32 bytes.

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

The schema is recursively decomposed into a `BTreeMap` of entries. **Leaf fields** and **list intermediate nodes** get their own entries. **Struct fields are transparent** — they do not create entries themselves; instead, their null validity is AND-propagated to descendant entries, and their children are recursively traversed.

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
| **validity-only** | yes | — | — | Nullable parent whose descendants have their own entries |
| **structural-only** | — | yes | — | Non-nullable list whose value type is a struct or nested list |
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

Boolean values are **bit-packed** using **LSB-first** (`Lsb0`) ordering into bytes.

**Non-nullable**: All values are packed sequentially into a `BitVec<u8, Lsb0>`, then the raw bytes are fed into the data digest.

**Nullable**:
1. Extend the validity `BitVec` as usual.
2. Only **valid** values are packed (nulls are skipped).
3. The packed bytes are fed into the data digest.

**Example**: `[true, NULL, false, true]` (nullable, 4 elements)
- Validity bits: `[1, 0, 1, 1]`
- Data bits (valid only): `[true, false, true]` → Lsb0 packed: `00000_1_0_1` = `0x05`
- Bytes fed to data digest: `[0x05]`

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
- **`List<Struct<...>>`**: The entry at `path/` is **structural-only** (list lengths). The struct is transparent, and each struct child creates its own entry at `path//childname`.
- **`List<List<...>>`**: The entry at `path/` is structural-only. The inner list creates another entry at `path//`, and so on recursively.

**Nullable list columns**: The column-level entry at `path` (without trailing `/`) is **validity-only**, recording which rows are null vs valid. Null list elements are not traversed — no structural or data bytes are written for them.

**Traversal**: For each non-null list element, write the sub-array length (u64 LE) to the structural digest at `path/`, then recurse into the sub-array using the value type.

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

Struct fields are **transparent** in the record-batch path — they do not create a BTreeMap entry. Instead:

1. **Children are traversed** in alphabetical order by field name.
2. **Struct-level nulls are AND-propagated** to all descendant entries. If a struct row is null, none of its children's data is hashed for that row, and the null is reflected in each descendant's effective validity.
3. Each child is recursively decomposed (leaf → data entry, list → structural entry, nested struct → recurse further).

**Example**: A struct field `address` with children `city` (LargeUtf8) and `zip` (Int32) creates two leaf entries: `address/city` and `address/zip`. No entry exists for `address` itself.

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

**Data bits** (Lsb0 packed, valid values only):
- Valid values: `[true, false, true]` (3 values)
- Lsb0 packing: bit0=true(1), bit1=false(0), bit2=true(1), bits3-7=0
- Byte: `00000101` = `0x05`

```
data_digest = SHA-256(0x05)
```

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
- Field `y`: validity `[1]` (1 bit, 1 word), data `0x01` (true packed Lsb0)

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

`hash_array` uses the same recursive decomposition as the record-batch path. Struct is transparent — no BTreeMap entry for the struct itself. Children become separate entries.

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
// [true, false] → Lsb0: bit0=1, bit1=0 → 0x01
data_b = SHA-256(0x01)
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

Row 1 is a null struct. Struct is transparent — its null is AND-propagated to children for data hashing. Since children are non-nullable per their Field definitions, their entries have no null_bits — but null rows are skipped in the data stream.

#### Step 1: Type Metadata

```
{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"LargeUtf8","name":"b","nullable":false}]}
```

#### Step 2: Decomposed Entries (with struct-null propagation)

BTreeMap entries (sorted by key): `"a"`, `"b"`

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

Each entry is non-nullable → no null_bits, no structural, just data.finalize().

```
final_digest = SHA-256()
final_digest.update( type_json_bytes )       // type metadata
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
  - Struct is transparent — recurse into children (sorted: "id", "label"):
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

## 8. Platform Considerations

- **Integer sizes**: All length prefixes use `u64` (8 bytes, LE). Validity bitmaps use `BitVec<u8, Lsb0>` (1 byte per word). Bit counts use `u64` (8 bytes, LE). Hashes are **platform-independent**.
- **Byte order**: All values use little-endian. Validity words are `u8` (1 byte, so endianness is trivial). Bit counts use little-endian.
- **Floating point**: IEEE 754 representation is hashed directly. `NaN` values with different bit patterns produce different hashes. `+0.0` and `-0.0` produce different hashes.
