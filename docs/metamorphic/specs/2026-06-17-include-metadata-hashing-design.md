# Design: Include Schema- and Field-Level Metadata in Hash Computation

**Issue:** PLT-1733  
**Date:** 2026-06-17  
**Status:** Approved  

---

## Overview

The starfix hasher computes a logical hash over Arrow data and schema structure (field names,
data types, nullability) but does not factor in Arrow `metadata` attached at the schema level
or per-field. This design adds an opt-in `include_metadata: bool` flag that, when true,
incorporates the full schema-level and per-field metadata into the schema hash. The default
is `false`, preserving existing hash stability.

---

## Scope

**In scope:**
- `HasherConfig` struct with `include_metadata: bool` (default `false`)
- Threading the flag through `ArrowDigester::new`, `hash_schema`, and `hash_record_batch`
- Metadata-aware schema hashing in `ArrowDigesterCore::hash_schema`
- Schema equality enforcement in `update()` when `include_metadata = true`
- Unit tests: regression, inclusion, determinism, empty-metadata invariant, edge cases
- Version bump: `0.0.2` → `0.1.0`

**Out of scope:**
- Python binding surfacing (PLT-1734)
- Cross-language parity golden vectors
- Per-key or per-layer metadata controls (anticipated future work)
- Changes to data hashing (null bits, column values)
- `hash_array` — does not call `hash_schema`; untouched

---

## API Changes

### New: `HasherConfig` (in `lib.rs`, public)

```rust
pub struct HasherConfig {
    pub include_metadata: bool,
}

impl Default for HasherConfig {
    fn default() -> Self {
        Self { include_metadata: false }
    }
}
```

`HasherConfig::default()` reproduces the pre-v0.1.0 behavior exactly. Future options
(per-key filters, separate schema vs. field toggles) will be added as new fields with
defaults, but because `HasherConfig` is a plain public struct, each new field is a
source-breaking change for any caller that constructs it with an exhaustive literal.
Callers should use the struct-update syntax to stay forward-compatible:

```rust
// Forward-compatible construction — new fields automatically receive their Default values
let config = HasherConfig { include_metadata: true, ..HasherConfig::default() };
```

### Changed methods on `ArrowDigester`

| Method | Old signature | New signature |
|---|---|---|
| `new` | `new(schema: &Schema)` | `new(schema: &Schema, config: HasherConfig)` |
| `hash_schema` | `hash_schema(schema: &Schema)` | `hash_schema(schema: &Schema, config: HasherConfig)` |
| `hash_record_batch` | `hash_record_batch(rb: &RecordBatch)` | `hash_record_batch(rb: &RecordBatch, config: HasherConfig)` |

### Unchanged methods

- `update(&mut self, record_batch: &RecordBatch)` — data path; flag baked in at `new()`
- `finalize(self) -> Vec<u8>` — combines pre-computed digests
- `hash_array(array: &dyn Array)` — type-based only; never calls `hash_schema`

---

## Implementation: Two-Phase Schema Hashing

`ArrowDigesterCore::hash_schema(schema: &Schema, include_metadata: bool) -> Vec<u8>` is
updated to hash in two phases. `normalize_schema` preserves both schema-level and per-field
metadata, so the same normalized schema is used for both phases — no separate pre-normalization
copy is needed.

**Phase 1 — Structure (unchanged from v0.0.x):**
```
normalized = normalize_schema(schema)
hasher.update(serialized_schema(normalized))
```
This feeds the canonical JSON of field names, data types, and nullability — identical to
existing behavior.

**Phase 2 — Metadata (only when `include_metadata = true`):**

Fields are traversed recursively (struct children, list/map element fields, etc.) so that
metadata on nested fields is included. Field paths use `/` as a delimiter
(e.g. `"parent/child"`).

For each field path, sorted alphabetically by full path (BTreeMap order):
```
if !field.metadata().is_empty():
    meta_json = serde_json::to_string(BTreeMap::from(field.metadata()))
    hasher.update(field_path.len() as u64, little-endian)
    hasher.update(field_path as bytes)
    hasher.update(meta_json.len() as u64, little-endian)
    hasher.update(meta_json as bytes)
```

Then, if schema-level metadata is non-empty:
```
schema_meta_json = serde_json::to_string(BTreeMap::from(schema.metadata()))
hasher.update(schema_meta_json.len() as u64, little-endian)
hasher.update(schema_meta_json as bytes)
```

Using `BTreeMap` for both field iteration order and key sorting within each metadata map
ensures determinism regardless of the original `HashMap` iteration order.

Every metadata block — both per-field and schema-level — is length-prefixed (`u64` LE).
This prevents concatenation ambiguity: no two distinct (field-metadata, schema-metadata)
pairs can produce the same byte stream fed into the hasher. A path `"a/b"` cannot produce
the same byte stream as two sibling paths `"a"` and `"b"`; a schema-level metadata block
cannot be mistaken for the tail of a per-field entry.

---

## Empty-Metadata Invariant

Phase 2 adds bytes to the hasher **only when metadata is actually present**:
- Per-field: guarded by `!field.metadata().is_empty()`
- Schema-level: guarded by `!schema.metadata().is_empty()`

When a schema has no metadata anywhere, Phase 2 adds nothing, so
`hash_schema(schema, include_metadata: true)` produces the exact same digest as
`hash_schema(schema, include_metadata: false)`. ✓

---

## Schema Equality in `update()`

`ArrowDigesterCore` stores a `schema_equality_key: String` (renamed from `serialized_schema`)
computed at construction time.

- When `include_metadata = false`: `schema_equality_key = serialized_schema(normalized)` —
  same as today; `update()` enforces field-name/type/nullability equality only.
- When `include_metadata = true`: `schema_equality_key = serialized_schema(normalized) +
  canonical_metadata_string(original_schema)`, where `canonical_metadata_string` serializes
  all field metadata (sorted by field name, then key within each field) and schema-level
  metadata (sorted by key) into a single deterministic string appended after the structure.

`update()` asserts `schema_equality_key` matches the recomputed key for the incoming batch's
schema before accepting the batch. When `include_metadata = true`, this rejects batches whose
schema metadata differs from the one used at construction — ensuring the accumulated hash
remains consistent.

---

## Internal Changes to `ArrowDigesterCore`

The existing `serialized_schema` **field** on the struct is renamed to `schema_equality_key`
to reflect that it serves schema identity enforcement in `update()`, not just structure
serialization. The `serialized_schema` **free function** (used for structure serialization
in Phase 1) is unchanged.

`ArrowDigesterCore::new(schema: &Schema, include_metadata: bool)` receives the flag and uses
it to compute both `schema_digest` and `schema_equality_key` at construction time.
`include_metadata` **must be stored** on the struct because `update()` needs it to
recompute the equality key for each incoming record batch's schema and compare it against
the stored `schema_equality_key`.

`ArrowDigesterCore<D>` changes:
- Field renamed: `serialized_schema: String` → `schema_equality_key: String`
- New field: `include_metadata: bool`
- `new` signature: `new(schema: &Schema, include_metadata: bool)`
- `new` passes the **original** schema (before normalization) to `hash_schema` so that
  metadata is available for Phase 2

---

## Crate Version

`Cargo.toml`: `0.0.2` → `0.1.0`

This is a semver minor bump under 0.x versioning. The API changes are intentionally
breaking (new required `config` parameter on three public methods) to make the flag
explicit at every call site.

**`VERSION_BYTES` must not change.** The constant `VERSION_BYTES: [u8; 3] = [0, 0, 1]`
in `lib.rs` is a hash-format version prefix prepended to every output of `ArrowDigester`.
It is independent of the crate's Cargo version. It must remain `[0, 0, 1]` — changing it
would invalidate every hash produced before this release. The `include_metadata` flag
changes what is fed into the hasher, but the output format (version prefix + SHA-256
digest) is unchanged.

---

## Test Plan

All tests live in `tests/arrow_digester.rs` unless otherwise noted.

| Test | Description |
|---|---|
| **Regression** | `include_metadata = false` on existing fixtures produces same hex digests as pre-v0.1.0 golden values |
| **Schema metadata changes hash** | Same schema + data, different schema metadata → different hash when `include_metadata = true` |
| **Field metadata changes hash** | Same schema + data, different field metadata on one field → different hash when `include_metadata = true` |
| **Determinism: key ordering** | Schema with metadata keys in different insertion orders → same hash (BTreeMap sort) |
| **Empty-metadata invariant** | Schema with no metadata: `include_metadata = true` and `false` produce same hash |
| **Metadata excluded by default** | Two schemas differing only in metadata → same hash when `include_metadata = false` |
| **Unicode keys/values** | Metadata with Unicode keys and values hashes without panic |
| **Large metadata values** | Metadata with a kilobyte-scale value hashes correctly |
| **Schema-level vs field-level independence** | Schema metadata change does not affect hash the same way as equivalent field metadata change (hashes differ) |

---

## Worked Examples

These examples trace the exact byte sequence fed into SHA-256 for two schemas with
`include_metadata = true`. Each `hasher.update(...)` call annotates the raw bytes and their
human-readable meaning. Both phases feed into a single SHA-256 instance; the final Starfix
hash prepends the three-byte version prefix `00 00 01`.

### Example 1 — per-field metadata only

**Schema:** `{v: Int32 non-nullable}`, field `v` metadata: `{"unit": "meters"}`

```
hasher ← SHA-256()

── Phase 1: structure (always) ──────────────────────────────────────────────
hasher.update(
  7B 22 76 22 3A 7B 22 64 61 74 61 5F 74 79 70 65
  22 3A 22 49 6E 74 33 32 22 2C 22 6E 75 6C 6C 61
  62 6C 65 22 3A 66 61 6C 73 65 7D 7D
)
── {"v":{"data_type":"Int32","nullable":false}}   (44 bytes)

── Phase 2: per-field metadata (BTreeMap order — one field "v") ─────────────
hasher.update( 01 00 00 00 00 00 00 00 )  ← field path byte-length = 1
hasher.update( 76 )                        ← "v"
hasher.update( 11 00 00 00 00 00 00 00 )  ← meta_json byte-length = 17
hasher.update( 7B 22 75 6E 69 74 22 3A 22 6D 65 74 65 72 73 22 7D )
── {"unit":"meters"}   (17 bytes)

── (schema has no metadata — nothing added) ─────────────────────────────────

schema_digest = hasher.finalize()          ← 32 bytes
output        = 00 00 01 || schema_digest  ← 35 bytes total
```

The structure JSON in Phase 1 is identical to `include_metadata = false` for the same schema.
Phase 2 appends 8 + 1 + 8 + 17 = 34 additional bytes to the hasher state
(path_len u64, path bytes, meta_json_len u64, meta_json bytes).

---

### Example 2 — field metadata and schema-level metadata combined

**Schema:** `{score: Int32 non-nullable}`, field `score` metadata: `{"unit": "points"}`,
schema-level metadata: `{"owner": "team-a"}`

```
hasher ← SHA-256()

── Phase 1: structure (always) ──────────────────────────────────────────────
hasher.update(
  7B 22 73 63 6F 72 65 22 3A 7B 22 64 61 74 61 5F
  74 79 70 65 22 3A 22 49 6E 74 33 32 22 2C 22 6E
  75 6C 6C 61 62 6C 65 22 3A 66 61 6C 73 65 7D 7D
)
── {"score":{"data_type":"Int32","nullable":false}}   (48 bytes)

── Phase 2: per-field metadata (BTreeMap order — one field "score") ─────────
hasher.update( 05 00 00 00 00 00 00 00 )  ← field path byte-length = 5
hasher.update( 73 63 6F 72 65 )           ← "score"
hasher.update( 11 00 00 00 00 00 00 00 )  ← meta_json byte-length = 17
hasher.update( 7B 22 75 6E 69 74 22 3A 22 70 6F 69 6E 74 73 22 7D )
── {"unit":"points"}   (17 bytes)

── Phase 2: schema-level metadata ───────────────────────────────────────────
hasher.update( 12 00 00 00 00 00 00 00 )  ← schema_meta_json byte-length = 18
hasher.update( 7B 22 6F 77 6E 65 72 22 3A 22 74 65 61 6D 2D 61 22 7D )
── {"owner":"team-a"}   (18 bytes)

schema_digest = hasher.finalize()          ← 32 bytes
output        = 00 00 01 || schema_digest  ← 35 bytes total
```

Both the field-level and schema-level blocks are length-prefixed with a `u64` LE byte-count,
so the concatenation is unambiguous regardless of metadata key or value content.

When this schema is used with `hash_record_batch`, the `schema_digest` above is fed as the
first update into the final combiner, followed by the per-field data digests exactly as in
the non-metadata path. The metadata flag only affects what goes into `schema_digest`; data
hashing is unchanged.
