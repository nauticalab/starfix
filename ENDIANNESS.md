# Endianness in Arrow and StarFix

## Overview

Endianness refers to the byte order in which multi-byte values are stored in memory. This document explains how Arrow and StarFix handle endianness and why it matters for data hashing.

## Endianness Basics

### Little-Endian vs Big-Endian

**Little-Endian** (LE):
- Least significant byte first
- Example: `0x12345678` stored as `[0x78, 0x56, 0x34, 0x12]`
- Used by: x86, x64, ARM (most common modern systems)

**Big-Endian** (BE):
- Most significant byte first  
- Example: `0x12345678` stored as `[0x12, 0x34, 0x56, 0x78]`
- Used by: PowerPC, SPARC, network protocols (legacy systems)

### Rust Byte Order Methods

```rust
// Native byte order (system-dependent)
value.to_ne_bytes()    // native-endian
i32::from_ne_bytes()   // native-endian

// Explicit little-endian
value.to_le_bytes()    // always little-endian
i32::from_le_bytes()   // always little-endian

// Explicit big-endian
value.to_be_bytes()    // always big-endian
i32::from_be_bytes()   // always big-endian
```

### Compile-time Endianness Detection

```rust
#[cfg(target_endian = "little")]
const IS_LITTLE_ENDIAN: bool = true;

#[cfg(target_endian = "big")]
const IS_LITTLE_ENDIAN: bool = false;

// Or use at runtime:
#[inline]
fn is_little_endian() -> bool {
    u32::from_ne_bytes([1, 0, 0, 0]) == 1
}
```

## Arrow's Approach to Endianness

### Arrow's Design Philosophy

**Arrow stores all data in the system's native byte order.**

- On little-endian systems (x86, ARM, modern CPUs): data is little-endian
- On big-endian systems (legacy): data is big-endian
- No conversion needed for local operations
- Minimal performance overhead

### Arrow Buffers

Arrow stores data in columnar buffers with the following structure:

```
Arrow Array (e.g., Int32Array with [1, 2, 3])
├─ Metadata
│  ├─ Data type
│  ├─ Length (3)
│  ├─ Null count
│  └─ Byte order (from Arrow metadata)
├─ Data Buffer
│  └─ Raw bytes in NATIVE order
│     [01 00 00 00 | 02 00 00 00 | 03 00 00 00]  (on little-endian)
└─ Null Buffer (optional)
   └─ Bitmap [1, 1, 1] (all valid)
```

### Inter-Process Communication (IPC)

Arrow's IPC format (used for serialization) includes endianness metadata:

```
Arrow IPC Message
├─ Version
├─ Body size
├─ Endianness flag ← Specifies byte order for interpretation
└─ Actual data (may need conversion on deserialization)
```

**If you receive an Arrow message from a big-endian system and your system is little-endian, Arrow handles the conversion automatically.**

## StarFix's Use of Byte Order

### Current Implementation

StarFix uses **mixed byte orders intentionally** for different purposes:

#### 1. Little-Endian for Length Prefixes

```rust
// In hash_binary_array and hash_string_array
data_digest.update(value.len().to_le_bytes());  // Little-endian
data_digest.update(value);
```

**Why little-endian?**
- Arrow uses little-endian natively on most systems
- Consistent with Arrow's buffer layout
- Deterministic across platforms when considering IPC

**Example:**
```
String "hello" (length 5):
Hash input: [05 00 00 00] + "hello"
          ↑↑↑↑
          little-endian length
```

#### 2. Big-Endian for Null Bits

```rust
// In finalize_digest
for &word in null_bit_digest.as_raw_slice() {
    final_digest.update(word.to_be_bytes());  // Big-endian!
}
```

**Why big-endian?**
- BitVec convention for consistency
- Ensures null bits are interpreted canonically
- Makes null bit patterns deterministic regardless of system endianness

**Example:**
```
Null bits: [true, true, false, true, ...] (packed into bytes)
Raw word: 0x0D (1101 in binary)
Hashed as: [0x0D] in big-endian representation
```

### Determinism Guarantee

The mixed approach ensures **deterministic hashing**:

1. **Data buffers**: Hashed in native byte order (as Arrow stores them)
2. **Length prefixes**: Converted to little-endian (Arrow standard)
3. **Null bits**: Converted to big-endian (BitVec standard)
4. **Result**: Same data always produces the same hash, regardless of which system ran the code

## Platform Considerations

### x86/x64 and ARM (Little-Endian)

```
System Endianness: Little-Endian (native)
↓
Arrow Buffers: Little-Endian (native, no conversion)
↓
StarFix Hashing:
  - Data: Little-Endian (as-is from Arrow)
  - Lengths: Little-Endian (explicit)
  - Null bits: Big-Endian (converted)
↓
Hash produced deterministically
```

### PowerPC/SPARC (Big-Endian) - Hypothetical

```
System Endianness: Big-Endian (native)
↓
Arrow Buffers: Big-Endian (native, no conversion)
↓
StarFix Hashing:
  - Data: Big-Endian (as-is from Arrow)
  - Lengths: Little-Endian (converted from native)
  - Null bits: Big-Endian (as-is, no conversion needed)
↓
Hash produced deterministically
```

**Important:** Even though intermediate representations differ, the final hash should be identical because both use the same explicit byte order for lengths and null bits.

## Cross-Platform Hashing

### Challenge

Two systems with different native endianness processing the same data could produce different hashes if not handled carefully.

### StarFix's Solution

1. **Data buffers**: Use Arrow's native representation (system-dependent but consistent)
2. **Explicit conversions**: All metadata uses explicit byte orders
3. **Schema metadata**: Hashed separately, includes nullability info
4. **Result**: Deterministic hashing within a system; comparable across systems

### Recommendation for Cross-Platform Use

If you need hashes to match across little-endian and big-endian systems:

```rust
// Current: May differ between systems
let hash = ArrowDigester::hash_array(&array);

// Better: Use record batch with explicit schema
let batch = RecordBatch::try_new(schema, arrays)?;
let hash = ArrowDigester::hash_record_batch(&batch);  // Schema-aware hashing
```

The schema digest is computed from serialized field information, which includes nullability flags and can be made platform-agnostic.

## Code Examples

### Checking System Endianness

```rust
// Compile-time check (preferred)
#[cfg(target_endian = "little")]
fn byte_order_name() -> &'static str {
    "little-endian"
}

#[cfg(target_endian = "big")]
fn byte_order_name() -> &'static str {
    "big-endian"
}

// Runtime check
fn is_little_endian() -> bool {
    u32::from_ne_bytes([1, 0, 0, 0]) == 1
}

// More explicit
fn is_little_endian_v2() -> bool {
    (1u16).to_le_bytes()[0] == 1
}
```

### Getting Arrow's Byte Order

```rust
use arrow::array::Array;

fn check_arrow_native_order(array: &dyn Array) {
    // Arrow stores in native byte order
    // No explicit API to check - it's always native
    
    #[cfg(target_endian = "little")]
    println!("Arrow on this system: little-endian buffers");
    
    #[cfg(target_endian = "big")]
    println!("Arrow on this system: big-endian buffers");
}
```

### Safe Cross-Platform Hashing

```rust
use arrow::record_batch::RecordBatch;
use starfix::ArrowDigester;
use std::sync::Arc;

fn hash_with_platform_info(batch: &RecordBatch) -> (Vec<u8>, &'static str) {
    let hash = ArrowDigester::hash_record_batch(batch);
    
    #[cfg(target_endian = "little")]
    return (hash, "little-endian");
    
    #[cfg(target_endian = "big")]
    return (hash, "big-endian");
}
```

## Testing Considerations

When testing StarFix hashing:

1. **Same-system tests**: Will pass regardless of implementation details
2. **Cross-platform tests**: Require explicit endianness handling
3. **Integration tests**: Should verify determinism on target platform

```rust
#[test]
fn deterministic_hashing() {
    // Same data → same hash (guaranteed)
    let array1 = Int32Array::from(vec![1, 2, 3]);
    let array2 = Int32Array::from(vec![1, 2, 3]);
    
    assert_eq!(
        ArrowDigester::hash_array(&array1),
        ArrowDigester::hash_array(&array2)
    );
}

#[test]
fn endianness_consistency() {
    // Different byte orders of same value should hash differently
    let value_a = 0x12345678u32;
    let array_a = UInt32Array::from(vec![value_a]);
    
    let value_b = 0x78563412u32;  // Byte-reversed
    let array_b = UInt32Array::from(vec![value_b]);
    
    // These should hash differently (different semantic values)
    assert_ne!(
        ArrowDigester::hash_array(&array_a),
        ArrowDigester::hash_array(&array_b)
    );
}
```

## Current Known Limitations

The current StarFix implementation:

✓ Ensures deterministic hashing on the same platform  
✓ Uses Arrow's native byte order for efficiency  
⚠️ May produce different hashes on different platforms for the same logical data  
⚠️ No explicit API to query or control endianness  

## Future Improvements

Potential enhancements for cross-platform hashing:

1. **Normalize byte order**: Convert all data to a canonical byte order before hashing
2. **Endianness parameter**: Allow users to specify target byte order
3. **Platform-agnostic mode**: Flag for cross-platform hash compatibility
4. **Schema versioning**: Include endianness info in hashed schema

Example future API:

```rust
pub enum HashEndianness {
    Native,      // Use system native (current behavior)
    Little,      // Always little-endian
    Big,         // Always big-endian
}

pub fn hash_array_with_endianness(
    array: &dyn Array,
    endianness: HashEndianness,
) -> Vec<u8> {
    // Implementation
}
```

---

**For more information about Arrow's byte order handling, see the [Apache Arrow documentation](https://arrow.apache.org/docs/format/Columnar.html).**
