# StarFix: Arrow Data Deterministic Hashing

## Overview

StarFix is a cryptographic hashing library for Apache Arrow data tables. It provides a deterministic way to compute unique digests for Arrow data structures, enabling efficient identification and comparison of data regardless of storage order or location.

The hashing system is built on top of SHA-256 (configurable to other digest algorithms via the `Digest` trait) and uses a hierarchical approach to hash different components of an Arrow table: schema metadata and field values.

## Core Architecture

### Main Components

The hashing system consists of three main hashing levels:

1. **Schema Digest** - Hash of the table schema (field names, types, and nullability)
2. **Field Digests** - Individual hashes for each field's data
3. **Final Digest** - Combined hash from schema + all field digests

### DigestBufferType Enum

The codebase uses a `DigestBufferType` enum to differentiate between nullable and non-nullable fields:

```rust
enum DigestBufferType<D: Digest> {
    NonNullable(D),                    // Just the data digest
    Nullable(BitVec, D),               // Null bits vector + data digest
}
```

This separation is crucial because nullable and non-nullable fields must be hashed differently to ensure data integrity and distinguish between actual nulls and missing data.

## Hashing Flow

### Record Batch Hashing

When hashing a complete `RecordBatch`, the process follows these steps:

```
1. Create ArrowDigester with schema
   ├─ Hash the schema (JSON serialized)
   └─ Initialize field digest buffers
        └─ Flatten nested struct fields with "/" delimiter
        └─ Mark each field as Nullable or NonNullable

2. Update with record batch data
   ├─ For each field:
   │  └─ Match on data type and call appropriate hashing function
   │  └─ Update both null bits (if nullable) and data digest
   └─ Accumulate all digests

3. Finalize
   ├─ Combine schema digest
   ├─ Process each field digest in alphabetical order
   │  ├─ If nullable: hash (null_bits.len + raw_null_bits + data)
   │  └─ If non-nullable: hash data only
   └─ Return final digest
```

### Direct Array Hashing

Arrays can also be hashed independently without schema context:

```
1. Hash the data type metadata (JSON serialized)
2. Initialize digest buffer based on array nullability
3. Call array_digest_update with appropriate handler
4. Finalize and combine digests
```

## Null Bits Handling

### Why Null Bits Matter

Null bits are essential to the hashing algorithm because:
- They distinguish between actual null values and valid data
- They enable reliable hashing of nullable vs non-nullable fields
- They preserve data integrity across different representations

### Null Bits Processing

For nullable fields, the system maintains a `BitVec` (bitvector) where each bit represents whether a value at that index is valid (`true`) or null (`false`).

#### Processing Steps:

1. **If null buffer exists:**
   ```
   - Iterate through each element
   - Set bit to true if value is valid
   - Set bit to false if value is null
   - For data digest: only hash valid values
   - For null values: hash the NULL_BYTES constant (b"NULL")
   ```

2. **If no null buffer (all values valid):**
   ```
   - Extend bitvector with all true values (one per element)
   - Hash all data normally
   ```

### Finalization of Nullable Fields

When finalizing a nullable field digest:

```rust
final_digest.update(null_bits.len().to_le_bytes());      // Size of bitvector
for &word in null_bits.as_raw_slice() {
    final_digest.update(word.to_be_bytes());              // Actual null bits
}
final_digest.update(data_digest.finalize());              // Data values
```

This ensures the null bit pattern is part of the final hash, making nullable arrays with actual nulls hash differently from arrays without nulls.


### Nullable Array with No Null Values

As demonstrated in the `nullable_vs_non_nullable_array_produces_same_hash` test in `/tests/arrow_digester.rs`:

When an Arrow array is created with a nullable type but contains no actual null values, Arrow optimizes the internal representation by removing the null buffer. This means the **hasher treats the array identically to a non-nullable array, producing the same hash result.**


## Supported Data Types

### Fixed-Size Types

These types have consistent byte widths and can be hashed directly:

| Data Type | Size | Handling |
|-----------|------|----------|
| Boolean | Variable | Bit-packed into bytes |
| Int8, UInt8 | 1 byte | Direct buffer hashing |
| Int16, UInt16, Float16 | 2 bytes | Direct buffer hashing |
| Int32, UInt32, Float32, Date32 | 4 bytes | Direct buffer hashing |
| Int64, UInt64, Float64, Date64 | 8 bytes | Direct buffer hashing |
| Decimal32 | 4 bytes | Direct buffer hashing |
| Decimal64 | 8 bytes | Direct buffer hashing |
| Decimal128 | 16 bytes | Direct buffer hashing |
| Decimal256 | 32 bytes | Direct buffer hashing |
| Time32 | 4 bytes | Direct buffer hashing |
| Time64 | 8 bytes | Direct buffer hashing |

**Hashing Strategy:**
- Get the data buffer from Arrow array
- Account for array offset
- For non-nullable: hash the entire slice directly
- For nullable: iterate element by element, skipping null values

### Boolean Type

Booleans receive special handling because Arrow stores them as bit-packed values (1 bit per value):

```rust
// For non-nullable:
- Extract each boolean value
- Pack into BitVec using MSB0 ordering
- Hash the raw bytes

// For nullable:
- Handle null bits (as described above)
- Pack only valid boolean values
- Hash the packed bytes
```

### Variable-Length Types

#### Binary Arrays

Binary data (raw byte sequences) must include length prefixes to prevent collisions:

```
For each element:
  - Hash: value.len().to_le_bytes()  // Length prefix
  - Hash: value.as_slice()           // Actual data
```

**Example collision prevention:**
- Without prefix: `[0x01, 0x02]` + `[0x03]` = `[0x01, 0x02, 0x03]`
- With prefix: `len=2, 0x01, 0x02, len=1, 0x03` (different!)

#### String Arrays

Strings are similar to binary but UTF-8 encoded:

```
For each element:
  - Hash: (value.len() as u32).to_le_bytes()  // Length as u32
  - Hash: value.as_bytes()                     // UTF-8 data
```

#### List Arrays

Lists/Array types prefix each sub-array's element count before recursively hashing the nested values, preventing cross-boundary collisions:

```
For each list element:
  - Hash: (sub.len() as u64).to_le_bytes()  // Element count prefix
  - Recursively call array_digest_update
  - Use the inner field's data type
  - Skip null list entries
```

**Example collision prevention:**
- Without prefix: `[[1,2],[3]]` and `[[1],[2,3]]` both flatten to `01000000 02000000 03000000` — identical!
- With prefix: `len=2, 1, 2, len=1, 3` vs `len=1, 1, len=2, 2, 3` → different hashes ✓

## Schema Handling

### Schema Flattening

Nested struct fields are flattened into a single-level map using the `/` delimiter:

```
Original schema:
  person (struct)
    ├─ name (string)
    └─ address (struct)
        ├─ street (string)
        └─ zip (int32)

Flattened:
  person/name
  person/address/street
  person/address/zip
```

### Schema Serialization

The schema is serialized as a JSON string containing:
- Field names
- Field types (as DataType serialization)
- Nullability flags

```rust
{
  "address/street": ("string", Utf8),
  "address/zip": ("int32", Int32),
  "name": ("string", Utf8)
}
```

Fields are stored in a `BTreeMap` to ensure **consistent alphabetical ordering**, which is critical for deterministic hashing.

### Schema Hash Inclusion

The schema digest is always the first component hashed into the final digest. This ensures that changes to schema structure produce different hashes, preventing false collisions.

## Collision Prevention

The hashing algorithm includes multiple safeguards against collisions:

### 1. Length Prefixes (Variable-Length Types)

Binary, string, and list arrays include length prefixes to prevent merging boundaries:

```
Array1: ["ab", "c"]        → len=2, "ab", len=1, "c"
Array2: ["a", "bc"]        → len=1, "a", len=2, "bc"
Result: Different hashes! ✓

List1: [[1,2],[3]]          → len=2, 1, 2, len=1, 3
List2: [[1],[2,3]]          → len=1, 1, len=2, 2, 3
Result: Different hashes! ✓
```

### 2. Null Bit Vectors (Nullable Fields)

Distinguishes between actual nulls and non-nullable fields:

```
NonNullable [1, 2, 3]   → Only data hash
Nullable [1, 2, 3]      → Null bits [true, true, true] + data hash
Result: Different hashes! ✓
```

### 3. Schema Digests

Encodes all metadata (type information, field names, nullability) into the hash:

```
Field "col1" Int32 (non-nullable) ≠ Field "col1" Int32 (nullable)
Result: Different hashes! ✓
```

### 4. Recursive Data Type Hashing

Complex types like lists recursively hash their components using the full schema information.

## Data Type Conversion Details

### Fixed-Size Array Processing

When hashing fixed-size types, the algorithm:

1. **Gets the data buffer** - Contains raw bytes for all elements
2. **Accounts for offset** - Arrow arrays can have offsets; these are applied
3. **Handles nullability:**
   - **NonNullable**: Hash entire buffer slice directly
   - **Nullable with nulls**: Iterate element-by-element, only hashing valid entries
   - **Nullable without nulls**: Hash entire buffer slice (simpler path)

**Example: Int32Array([1, 2, 3])**
```
Size per element: 4 bytes
Buffer: [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00]
Hash entire 12 bytes
```

**Example: Int32Array([1, null, 3])**
```
Size per element: 4 bytes
Buffer: [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00]
Null bits: [true, false, true]

Process:
  1. Hash null bits [true, false, true]
  2. Hash bytes 0-3 (index 0, valid)
  3. Skip bytes 4-7 (index 1, null)
  4. Hash bytes 8-11 (index 2, valid)
```

## Determinism Guarantees

The hashing algorithm ensures deterministic output because:

1. **Schema fields are sorted** - BTreeMap maintains alphabetical order
2. **Field order is deterministic** - Always process in alphabetical field name order
3. **Data types are consistent** - Each type uses the same hashing strategy
4. **Byte order is consistent** - Uses little-endian for length prefixes and big-endian for bitvectors
5. **Null handling is predictable** - Same rules applied consistently

**Implication:** The same data in different storage order or location will always produce the same hash.

## Performance Considerations

### Efficient Schema Hashing

- Schema is hashed only once during initialization
- Uses JSON serialization (fast) rather than alternative formats
- Schema digest is reused for all record batches

### Incremental Updates

- Each record batch update accumulates into the same digest buffers
- No need to re-hash previous batches
- Final digest combines all incremental updates

### Memory Efficiency

- Null bits use bit-packing (1 bit per value, not 1 byte)
- Streaming approach avoids loading entire dataset into memory
- Field flattening enables hierarchical processing

### Buffer Slicing

- Fixed-size arrays hash the raw buffer directly when possible
- Avoids element-by-element iteration for non-nullable arrays
- Significant speedup for large datasets

## Known Limitations

The current implementation marks the following data types as `todo!()`:

- `Null` - Null data type itself
- `Timestamp` - Timestamp variants
- `Duration` - Duration types
- `Interval` - Interval types
- `BinaryView` - Binary view type
- `Utf8View` - UTF-8 view type
- `ListView` - List view type
- `FixedSizeList` - Fixed-size lists
- `LargeListView` - Large list view type
- `Struct` - Struct types (partial support for nested fields)
- `Union` - Union types
- `Dictionary` - Dictionary-encoded types
- `Map` - Map types
- `RunEndEncoded` - Run-end encoded types

These types will panic if encountered during hashing and should be implemented in future versions.
## SHA-256 Hashing Implementation

### Overview

ArrowDigester uses SHA-256 as its default cryptographic hash function, providing a 256-bit (32-byte) digest. The digest algorithm is configurable through the `Digest` trait, allowing alternative implementations, but SHA-256 is the standard choice for production use.

### Versioning Header

Every hash produced by ArrowDigester is prefixed with a 3-byte version identifier:

```
[Version Byte 0] [Version Byte 1] [Version Byte 2] [SHA-256 Digest (32 bytes)]
```

This 3-byte header ensures forward compatibility and enables detection of incompatible hash formats across different library versions. If the hashing algorithm or data format changes in future versions, the version bytes allow consumers to:
- Reject hashes from incompatible versions
- Implement migration or conversion logic
- Maintain a stable hash contract with external systems

### SHA-256 Digest Process

The hashing workflow follows this structure:

```
1. Initialize SHA-256 digester with version header
    └─ Write 3 version bytes

2. Hash schema component
    └─ Update digester with schema JSON

3. Hash field digests (alphabetical order)
    ├─ For each field:
    │  ├─ Hash null bits (if nullable)
    │  └─ Hash data digest
    └─ Accumulate into SHA-256 state

4. Finalize
    └─ Return 35-byte result: [3 version bytes] + [32-byte SHA-256 hash]
```

### Implementation Details

- **Hash Algorithm**: SHA-256 (256-bit output)
- **Version Prefix**: 3 bytes (allows 16.7 million versions)
- **Total Output**: 35 bytes (3 version + 32 digest)
- **State Management**: SHA-256 maintains running state across multiple `update()` calls
- **Finalization**: Single call to `finalize()` produces immutable digest

## Example Usage

### Hashing a Single Array

```rust
use arrow::array::Int32Array;
use starfix::ArrowDigester;

let array = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
let hash = ArrowDigester::hash_array(&array);
println!("Hash: {}", hex::encode(hash));
```

### Hashing a Record Batch

```rust
use arrow::record_batch::RecordBatch;
use starfix::ArrowDigester;

let batch = RecordBatch::try_new(...)?;
let hash = ArrowDigester::hash_record_batch(&batch);
println!("Hash: {}", hex::encode(hash));
```

### Streaming Multiple Batches

```rust
use starfix::ArrowDigester;

let mut digester = ArrowDigester::new(schema);
digester.update(&batch1);
digester.update(&batch2);
digester.update(&batch3);

let final_hash = digester.finalize();
println!("Combined hash: {}", hex::encode(final_hash));
```

## Testing Strategy

The codebase includes comprehensive tests covering:

- **Data type coverage** - Tests for each supported data type
- **Nullable handling** - Arrays with and without null values
- **Collision prevention** - Length prefix verification
- **Determinism** - Same data produces same hash
- **Schema metadata** - Different schemas produce different hashes
- **Field ordering** - Different field orders produce same hash (commutative)

## Implementation Notes

### About the Delimiter

The code uses `/` as the delimiter for nested field hierarchies. This was chosen to be URL-safe and visually clear while avoiding common naming conflicts.

### About Byte Order

- **Length prefixes**: Little-endian (`to_le_bytes()`) - standard for Arrow
- **Bitvector words**: Big-endian (`to_be_bytes()`) - matches bitvector convention
- **Size fields**: Little-endian - consistent with Arrow buffers

### About Bitpacking

Boolean values and null indicators use `BitVec<u8, Msb0>` (Most Significant Bit ordering):
- Compresses 8 boolean values into 1 byte
- Reduces hash input size by 8x for boolean arrays
- Uses MSB0 for consistent bit ordering

---

**For more information, see the main README.md and examine test cases in `tests/arrow_digester.rs`**
