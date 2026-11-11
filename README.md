# StarFix: Arrow Data Deterministic Hashing

## Overview

StarFix computes deterministic, content-based SHA-256 hashes of Apache Arrow schemas, record
batches, streams of record batches, and arrays. Two inputs get the same hash iff they hold the
same logical data under the same schema, regardless of how the data is laid out in memory
(chunking, offsets, small vs large offset types, dictionary encoding).

Every hash is 35 bytes: a 3-byte **hash format version** followed by a SHA-256 digest. This crate
produces **hash format 2** (`00 00 02`). The exact bytes are specified in
[`docs/byte-layout-spec.md`](docs/byte-layout-spec.md), precisely enough for an independent
implementation; [starfix-python](https://github.com/nauticalab/starfix-python) implements the
same format and is checked against the Rust-generated golden vectors in
`tests/golden/format2_v0.5.json`.

## Usage

```rust
use starfix::{ArrowDigester, HasherConfig};

// One batch
let hash = ArrowDigester::hash_record_batch(&batch, HasherConfig::default());

// A stream of batches (the result does not depend on how rows are split into batches)
let mut digester = ArrowDigester::new(&schema, HasherConfig::default());
digester.update(&batch1);
digester.update(&batch2);
let hash = digester.finalize();

// A schema, or a single array
let schema_hash = ArrowDigester::hash_schema(&schema, HasherConfig::default());
let array_hash = ArrowDigester::hash_array(&array);
let array_hash_ordered = ArrowDigester::hash_array_with_config(
    &array,
    HasherConfig { ordered_struct_fields: true, ..HasherConfig::default() },
);
```

## What is hashed

- **The schema**: column names, canonical types (`Utf8`≡`LargeUtf8`, `Binary`≡`LargeBinary`,
  `List`≡`LargeList`, a dictionary is its value type), nullability; with `include_metadata`,
  schema and field metadata (including extension-type keys).
- **The row count.**
- **Every node of the type tree** (each column, struct field and list element): its validity if
  it is nullable, its lengths if it is a list, its values if it is a leaf. Booleans are hashed one
  byte per value.

**Ordering principle.** By default starfix ignores exactly two orders: the order of the
top-level columns and the order of struct fields (at every depth). It never sorts or reorders
anything else: row order, list element order (also for `list<struct<key, value>>`) and the
logical values of dictionary-encoded arrays are hashed as given.

Not hashed (they are not part of the logical data, or pyarrow cannot observe them): the list
element field name, the map entries/key/value field names, the map entries field's metadata, the
dictionary index type and `ordered` flag, and metadata unless `include_metadata` is set.

## Options

All options are per call (or per streaming digester), never global, and default to `false`:

| `HasherConfig` field | Effect |
|---|---|
| `include_metadata` | schema-level and field-level metadata are part of the hash |
| `ordered_columns` | the order of the top-level columns is part of the hash |
| `ordered_struct_fields` | the order of struct fields is part of the hash, at every depth (struct in struct, list, large list, fixed-size list, map, dictionary value type) |

```rust
// Identity of a table value whose column and struct-field order matter:
let config = HasherConfig {
    ordered_columns: true,
    ordered_struct_fields: true,
    ..HasherConfig::default()
};
```

Hashes taken with different order options never coincide: the options are recorded in the
header that starts every digest. Construct `HasherConfig` with `..HasherConfig::default()` to
stay compatible with future options.

## Rejected inputs

In every mode, the following panic (starfix-python raises `ValueError`):

- schemas with two fields at the same path: duplicate column names, duplicate field names in a
  struct, or a name containing `/` that aliases a nested field (a column `"a/b"` next to a struct
  column `a` with child `b`);
- `ListView`, `LargeListView`, `Union` and `RunEndEncoded` types;
- a streamed batch whose schema does not match the digester's (with the order options, this
  includes a different column or struct-field order).

## Supported data types

Data hashing supports Boolean, all integer and float types, Date32/64, Time32/64, Timestamp,
Duration, Decimal32/64/128/256, FixedSizeBinary, Binary/LargeBinary, Utf8/LargeUtf8,
List/LargeList, Struct, and Dictionary of any of these. Schemas of `Null`, `FixedSizeList`, `Map`,
`Interval`, `Utf8View` and `BinaryView` can be hashed, but their data cannot yet.

## Hash format history

- **Format 2** (`00 00 02`, starfix 0.5): the current format.
- **Format 1** (`00 00 01`, starfix 0.4 and earlier). Every hash changed in format 2. Format 1
  lost information for some inputs, which format 2 fixes: Booleans were bit-packed without a
  count (a one-column non-nullable `[true]` equalled `[true, false, false]`, and the hash
  depended on batch boundaries); a null struct equalled a struct with all-null fields and null
  struct rows vanished under non-nullable fields; a list whose element is a nullable list lost
  its own lengths; a list of nullable structs did not distinguish a null element; and the row
  count of zero-column tables and empty structs was not hashed. Format-1 hashes can still be
  computed with starfix 0.4. `tests/golden/format1_to_format2.json` maps every format-1 golden
  value of this repository to its format-2 value.

See `docs/byte-layout-spec.md`, Section 10, and the [CHANGELOG](CHANGELOG.md).

## Testing

- `tests/format2_bytes.rs`, `tests/digest_bytes.rs`: byte-level conformance with the spec.
- `tests/golden_vectors_format2.rs`: the golden vectors and their relations.
- `tests/nested_family.rs`: 420 nested column types, exhaustive small inputs, no collisions.
- `tests/lossless_properties.rs`: every supported leaf type × nullability × position × row
  count, batch split, slicing and empty-vs-absent.
- `tests/ordered_hashing.rs`, `tests/arrow_digester.rs`: options, rejections, invariants.

Regenerate the golden file (append-only) with `cargo run --bin emit_golden_format2`.
