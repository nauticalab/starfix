# StarFix: Arrow Data Deterministic Hashing

## Overview

StarFix computes deterministic, content-based SHA-256 hashes of Apache Arrow schemas, record
batches, streams of record batches, and arrays. Inputs that hold the same logical data under the
same schema get the same hash, regardless of how the data is laid out in memory (chunking,
offsets, small vs large offset types, dictionary encoding); inputs that differ get different
hashes, up to SHA-256 collisions (tested exhaustively on small inputs of every supported type).

Every hash is 35 bytes: a 3-byte **hash format version** followed by a SHA-256 digest. This crate
produces **hash format 2** (`00 00 02`). The exact bytes are specified in
[`docs/byte-layout-spec.md`](docs/byte-layout-spec.md), precisely enough for an independent
implementation. [starfix-python](https://github.com/nauticalab/starfix-python) is being updated
to format 2 in a companion change (its released versions produce format 1); its format-2
implementation must match the Rust-generated golden vectors in `tests/golden/format2_v0.5.json`
exactly.

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
header that starts every digest. The header also names the entry point, so a schema hash, a
record batch or stream hash and an array hash never coincide either. `hash_array` always treats
the array as nullable (its validity is always hashed). Construct `HasherConfig` with `..HasherConfig::default()` to
stay compatible with future options.

## Rejected inputs

In every mode, the following panic (starfix-python raises `ValueError`):

- schemas with two fields at the same path: duplicate column names, duplicate field names in a
  struct, or a name containing `/` that aliases a nested field (a column `"a/b"` next to a struct
  column `a` with child `b`);
- `ListView`, `LargeListView`, `Union` and `RunEndEncoded` types;
- data hashing of a schema that contains a type whose data is not supported (see below), also
  with zero rows;
- a non-nullable field whose array contains nulls not masked by an enclosing null struct (nulls
  in child slots that are not elements, such as under a null list value, are ignored);
- a streamed batch whose schema does not match the digester's (with the order options, this
  includes a different column or struct-field order).

These checks run before any data is hashed, except the null check, which runs while a batch is
traversed; after a panic, discard the digester.

Metadata that repeats a key (possible in stored Arrow metadata) is not portable: arrow-rs keeps
the last value when it reads IPC or imports over the C data interface, before starfix sees it,
while starfix-python rejects such inputs with `include_metadata` (spec Section 7).

## Supported data types

Data hashing supports Boolean, all integer and float types, Date32/64, Time32/64, Timestamp,
Duration, Decimal32/64/128/256, FixedSizeBinary, Binary/LargeBinary, Utf8/LargeUtf8,
List/LargeList, Struct, and Dictionary of any of these (a dictionary of dictionaries resolves
recursively; `FixedSizeBinary(0)` is supported). Schemas of `Null`, `FixedSizeList`, `Map`,
`Interval`, `Utf8View` and `BinaryView` can be hashed, but their data cannot yet. Extension types
hash as their storage type; with `include_metadata`, their `ARROW:extension:*` metadata counts
too (an absent `ARROW:extension:metadata` equals an empty one).

## Legacy 0.4 mode (format 1)

`starfix::legacy::v0_4` reproduces the output of starfix **0.4.0** (Rust) byte for byte, with
the same API as 0.4.0 (`ArrowDigester`, `HasherConfig { include_metadata }`). It exists **only to
read or match data hashed by starfix 0.4.x** — for example to find stored data under its old
hash. **Never hash new data with it.** It is frozen: never fixed, never extended (no order
options), and it keeps every format-1 defect listed below. Migrating code that must keep
producing 0.4 hashes changes one import:

```rust
use starfix::legacy::v0_4::{ArrowDigester, HasherConfig}; // hash format 1, as starfix 0.4.0
```

Exactness is tested against the v0.4.0 test suite and golden files (restored as
`tests/legacy_v0_4_*.rs`), the format-1 side of `tests/golden/format1_to_format2.json`, and a
differential test against the published starfix 0.4.0 crate; its source files are pinned by
`tests/legacy_v0_4_freeze.rs`. It reproduces **Rust** starfix 0.4.0; starfix-python's
`starfix.legacy.v0_4` reproduces **starfix-python** 0.4.0, which differed from Rust 0.4.0 on a
few inputs (see the divergence table in starfix-python's documentation).

## Hash format history

- **Format 2** (`00 00 02`, starfix 0.5): the current format.
- **Format 1** (`00 00 01`, starfix 0.4 and earlier). Every hash changed in format 2. Format 1
  lost information for some inputs, which format 2 fixes: Booleans were bit-packed without a
  count (a one-column non-nullable `[true]` equalled `[true, false, false]`, and the hash
  depended on batch boundaries); a null struct equalled a struct with all-null fields and null
  struct rows vanished under non-nullable fields; a list whose element is a nullable list lost
  its own lengths; a list of nullable structs did not distinguish a null element; and the row
  count of zero-column tables and empty structs was not hashed. Format-1 hashes are computed by
  the legacy 0.4 mode (`starfix::legacy::v0_4`, above). `tests/golden/format1_to_format2.json` maps every format-1 golden
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
`tests/golden_vectors_format2.rs` also requires a minimum vector and relation count, so a
regeneration that drops vectors fails even if the generator was edited too.
