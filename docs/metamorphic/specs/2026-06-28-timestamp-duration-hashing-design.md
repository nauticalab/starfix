# Timestamp and Duration Type Support in Leaf Data Hashing

**Date:** 2026-06-28
**Issue:** ITL-438
**Repos:** nauticalab/starfix (Rust), nauticalab/starfix-python (Python)

---

## Overview

`ArrowDigester` panics (Rust `todo!()`) or raises `NotImplementedError` (Python) when
hashing any Arrow table or array containing a `timestamp` or `duration` column. This
blocks any project that uses `datetime.datetime` values as data or tag values — a very
common pattern. The fix is a minimal two-line addition in each repo.

---

## Goals & Success Criteria

- `hash_table`, `hash_record_batch`, `hash_array`, and streaming `update` all work for
  `timestamp` (all 4 units × tz-aware and tz-naive) and `duration` (all 4 units)
- Hashes are byte-for-byte identical between the Rust and Python implementations
- Different timestamp units produce different hashes (unit is captured in schema hash)
- Timezone-aware and timezone-naive timestamps with identical data produce different hashes
  (tz is captured in schema hash)
- `interval` types remain unimplemented (out of scope)

---

## Scope & Boundaries

**In scope:**
- `timestamp[s/ms/us/ns]` with and without timezone
- `duration[s/ms/us/ns]`
- Raw buffer layout tests (nullable bytes) in Rust
- Golden parity tests in Python (`test_golden_parity.py`)
- Schema serialization tests for the new types
- Updating the comprehensive `schema()` golden hash in Rust

**Out of scope:**
- `interval` types (remain `todo!()`)
- Any other currently-unimplemented Arrow types
- Changes to `_data_type_to_value` / schema hashing (already correct)

---

## Root Cause

In `arrow_digester_core.rs`, the `array_digest_update` match has:

```rust
DataType::Timestamp(_, _) => todo!(),
DataType::Duration(_)     => todo!(),
```

In `arrow_digester.py`, `_element_size_for_type` has no handler for
`pa.types.is_timestamp` or `pa.types.is_duration`.

The schema-hashing path (`_data_type_to_value` / `data_type_to_value`) already handles
both types correctly — only the leaf data hashing step is missing.

---

## Arrow Storage Layout

All `timestamp` and `duration` variants are stored as signed 64-bit integers (8 bytes per
element) in Arrow's physical layout. The unit (`s`, `ms`, `us`, `ns`) and timezone are
schema-level metadata, not data-buffer metadata. This is the same physical layout as
`time64`, `int64`, `date64`, etc.

---

## Polars Round-Trip Behaviour

Verified against polars 1.42.0 / pyarrow 24.0.0:

| Type | Round-trip result | Stable? |
|---|---|---|
| `timestamp[ms/us/ns]` (any tz) | Type preserved verbatim | ✅ |
| `timestamp[us/ms/ns]` tz-naive | Type preserved verbatim | ✅ |
| `duration[ms/us/ns]` | Type preserved verbatim | ✅ |
| `timestamp[s, tz=*]` | **Coerced to `timestamp[ms, tz=*]`** | ❌ |
| `duration[s]` | **Coerced to `duration[ms]`** | ❌ |
| Polars native `datetime` | Always produces `timestamp[us, tz=UTC]` | ✅ |
| Timezone strings | Preserved verbatim (`UTC`, `America/New_York`, etc.) | ✅ |

**Design decision:** starfix does **not** normalise `s`→`ms`. The hasher correctly
hashes whatever Arrow type it receives per the Arrow spec. `timestamp[s, value=1]` and
`timestamp[ms, value=1]` represent different points in time and must produce different
hashes. The `s`→`ms` coercion is a Polars limitation.

**Practical implication for users:** Projects that pass Arrow data through Polars before
hashing should avoid `timestamp[s]` and `duration[s]` columns if hash stability across
the Polars boundary is required. Use `ms`, `us`, or `ns` instead — all three survive
Polars round-trips intact.

---

## Design

### Rust fix (`src/arrow_digester_core.rs`)

Replace the two `todo!()` arms with standalone calls, keeping them adjacent to the
existing `Time32`/`Time64` arms for readability:

```rust
DataType::Timestamp(_, _) => Self::hash_fixed_size_array(effective_array, digest, 8),
// ↑ int64 physical storage; unit and tz are schema metadata, not data bytes

DataType::Time32(_) => Self::hash_fixed_size_array(effective_array, digest, 4),
DataType::Time64(_) => Self::hash_fixed_size_array(effective_array, digest, 8),

DataType::Duration(_) => Self::hash_fixed_size_array(effective_array, digest, 8),
// ↑ int64 physical storage; unit is schema metadata, not data bytes
```

### Python fix (`src/starfix/arrow_digester.py`)

In `_element_size_for_type`, add two checks after the existing `time32`/`time64` block:

```python
if pa.types.is_time32(dt):
    return 4
if pa.types.is_time64(dt):
    return 8
if pa.types.is_timestamp(dt):
    return 8  # int64 physical storage; unit/tz are schema metadata
if pa.types.is_duration(dt):
    return 8  # int64 physical storage; unit is schema metadata
```

No other files require changes.

---

## Test Strategy

### TDD sequence (required by starfix CLAUDE.md)

1. Write failing Rust tests → confirm they panic from `todo!()`
2. Implement Rust fix → confirm all Rust tests pass
3. Capture golden hash values from Rust output
4. Write failing Python tests using those golden values → confirm `NotImplementedError`
5. Implement Python fix → confirm all Python tests pass

### Rust tests (`tests/arrow_digester.rs`)

| Test | Description |
|---|---|
| `timestamp_array_hashing` | All 4 units × tz-aware (`UTC`) and tz-naive — 8 `hash_array` assertions with golden hex |
| `duration_array_hashing` | All 4 units — 4 `hash_array` assertions with golden hex |
| `timestamp_units_differ` | Assert `timestamp[s]` ≠ `timestamp[ms]` ≠ `timestamp[us]` ≠ `timestamp[ns]` for same raw data |
| `timestamp_tz_differs` | Assert `timestamp[us, tz=UTC]` ≠ `timestamp[us]` for same raw data |
| `duration_units_differ` | Assert all 4 duration units differ for same raw data |
| `schema()` | Add timestamp and duration columns (all variants); update golden hash |

### Rust raw buffer tests (`src/arrow_digester_core.rs`)

| Test | Description |
|---|---|
| `digest_timestamp_nullable_bytes` | `TimestampMicrosecondArray [0, None, 3_600_000_000]` — verify null bits `[true, false, true]`, data digest = sha256 of two i64 LE values |
| `digest_duration_nullable_bytes` | `DurationMicrosecondArray [0, None, 3_600_000_000]` — same verification |

### Python tests (`tests/test_golden_parity.py`)

| Test | Description |
|---|---|
| `test_timestamp_utc_microsecond_array` | `timestamp[us, tz=UTC]` with 3 values (incl. null); assert hex matches Rust |
| `test_timestamp_naive_microsecond_array` | `timestamp[us]` with same values; assert hex ≠ tz-aware |
| `test_timestamp_all_units` | All 4 units × tz-aware and tz-naive; assert units differ |
| `test_duration_microsecond_array` | `duration[us]` with 3 values (incl. null); assert hex matches Rust |
| `test_duration_all_units` | All 4 duration units; assert units differ |

### Python schema tests (`tests/test_arrow_digester.py`)

| Test | Description |
|---|---|
| `test_timestamp_types_in_schema` | `_serialized_schema` produces `{"Timestamp":["Microsecond","UTC"]}` and `{"Timestamp":["Microsecond",null]}` |
| `test_duration_types_in_schema` | `_serialized_schema` produces `{"Duration":"Microsecond"}` etc. |

---

## Implementation Sequence

1. **Rust:** Write failing tests → implement fix → run `cargo test` → capture golden hashes → update `schema()` golden hash
2. **Python:** Write failing tests (using Rust golden values) → implement fix → run pytest
3. **Formatting:** Run `cargo fmt` before committing (required by CLAUDE.md)
4. **Two PRs:** One per repo, both targeting `dev`; Python PR notes dependency on Rust golden values
