# Optional Order-Sensitive Hashing (and Struct Validity)

**Issue:** ITL-898
**Date:** 2026-11-11
**Status:** D1 and D3–D7 implemented as recommended. **D2 superseded** by the developer's
ruling of 2026-11-11 (below). The authoritative byte layout is `docs/byte-layout-spec.md`
(Section 9: order options; Section 10: the v0.5 fix). Version prefix: open, variant (a) or (b).

> **Developer ruling, 2026-11-11.** "Since this is a bug, the rule of golden value shouldn't
> move gets superseded -- erroneous golden value is extremely wrong and we cannot tolerate it --
> this fix should not be a switch but should be marked as explicit fix!"
>
> Consequences: there is no third switch. The lossless layout (nested validity, Section 3.4 of
> this document; one byte per Boolean, Section 3.5) is the default and only layout, in every mode.
> Existing golden values computed from the erroneous layout move, marked explicitly as a fix
> (CHANGELOG, README, spec Section 10, `tests/golden/fixed_in_v0.5.json`). Flags bit `0x04` is
> reserved and never reused. Inputs of unaffected types keep their v0.4.0 hash (variant (a)) or
> their v0.4.0 digest under a bumped prefix (variant (b)); the developer chooses.
>
> Everything below that describes `nested_validity` (or `struct_validity`) as an option is
> historical.
---

## 1. Motivation

`ArrowDigester` is order independent by design: reordering the columns of a table, or the fields of a
struct at any depth, does not change the hash. That is the right identity for Orcapod *streams*. It
is the wrong identity for values whose own equality depends on order: a `pa.Table`, a Polars or
pandas DataFrame/Series, and `pa.StructArray` values (pyarrow and Polars both treat
`struct<a, b>` and `struct<b, a>` as different types).

This change adds opt-in switches that make column order and struct field order part of the
identity. With every new switch off, every existing hash is byte-for-byte unchanged.

The golden values are produced by Rust (this repo) and consumed unchanged by starfix-python, as
with `include_metadata` (see `2026-06-18-golden-vector-parity-design.md`).

---

## 2. Decisions at a glance

| # | Question | Recommendation | Main alternative |
|---|---|---|---|
| D1 | One switch or two for order? | **Two independent switches**: `ordered_columns`, `ordered_struct_fields` | One switch `ordered` covering both |
| D2 | Null struct vs all-null-fields struct (ITL-848), and the rest of that loss family | **Separate third switch** (placeholder name `nested_validity`) that makes the layout lossless: nested validity (3.4) and Booleans (3.5) | Bundle into the ordered switches; separate switches; or leave unfixed |
| D3 | How is position hashed? | **Through the canonical schema / type JSON only** (ordered JSON arrays). Data entries and their layout are unchanged. | Positional entry paths in the data digests |
| D4 | Domain separation | **An options tag** (`0x00 "starfix-options-v1" flags`) is the first input of the schema digest (and of the `hash_array` digest) whenever any new switch is on. Version prefix stays `00 00 01`. | Different version prefix per mode; or no separation |
| D5 | One-column table, same hash in both modes? | **No.** Any new switch on gives a hash that can never equal a hash taken with a different switch set, even when order is meaningless. | Hash equal when order cannot matter (like the empty-metadata invariant) |
| D6 | Streaming with a batch in a different order | **Rejected** (panic in Rust, `ValueError` in Python), like any schema mismatch | Silently reorder the batch to the digester's order |
| D7 | Duplicate names / path aliasing | **Rejected whenever any new switch is on** (two schema nodes with the same node path); default behaviour unchanged | Support them via positional entry paths (D3 alternative) |
| D8 | `hash_array` API in Rust | **Add `hash_array_with_config(array, config)`**; `hash_array(array)` stays and equals it with the default config | Change `hash_array` to take a config (breaking; 86 call sites) |
| D9 | uniffi bindings in `src/pyarrow.rs` | **Leave default-only**, as was done for `include_metadata` (the Python package ships from starfix-python) | Add config arguments to the uniffi functions |
| D10 | Golden file | **`tests/golden/ordered_v0.5.json`**, generator `cargo run --bin emit_golden_ordered`, append-only | — |
| D11 | Version | Propose crate **0.5.0** (adding config fields breaks exhaustive `HasherConfig { .. }` literals). Not bumped in this PR. | — |

Each is argued below.

---

## 3. API

### 3.1 Rust

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct HasherConfig {
    pub include_metadata: bool,      // existing
    pub ordered_columns: bool,       // new: top-level column order is part of the hash
    pub ordered_struct_fields: bool, // new: struct field order is part of the hash, at every depth
    pub nested_validity: bool,       // new (placeholder name): lossless layout, Sections 3.4-3.5
}
```

All defaults are `false` (order independent, current behaviour).

The options are **per call / per digester**, never global: a process can hash Orcapod streams
unordered and values ordered side by side. Each `ArrowDigester` keeps the config it was created
with; the one-shot functions take it as an argument.

| Entry point | Change |
|---|---|
| `ArrowDigester::new(schema, config)` | honours all four fields |
| `ArrowDigester::update(batch)` | uses the config given to `new` |
| `ArrowDigester::finalize()` | unchanged |
| `ArrowDigester::hash_record_batch(batch, config)` | honours all four fields |
| `ArrowDigester::hash_schema(schema, config)` | honours all four fields (see §5.1 for why `nested_validity` matters here) |
| `ArrowDigester::hash_array(array)` | unchanged; equals `hash_array_with_config(array, HasherConfig::default())` |
| `ArrowDigester::hash_array_with_config(array, config)` | **new**; `include_metadata` is ignored (as today: no schema); the three new switches apply |

Rust has no `hash_table`; a table is a stream of record batches (`new` + `update`* + `finalize`).

Existing code that builds `HasherConfig` with an exhaustive literal (`HasherConfig { include_metadata: true }`)
stops compiling. 22 such literals exist in this repo's tests and the existing generator; they are
changed mechanically to `HasherConfig { include_metadata: true, ..HasherConfig::default() }`.
No assertion or expected value changes.

### 3.2 Python (starfix-python)

Keyword-only arguments, same names, all default `False`, mirroring `include_metadata`:

```python
ArrowDigester(schema, *, include_metadata=False, ordered_columns=False,
              ordered_struct_fields=False, nested_validity=False)
ArrowDigester.hash_schema(schema, *, include_metadata=False, ordered_columns=False, ordered_struct_fields=False, nested_validity=False)
ArrowDigester.hash_record_batch(batch, *, ...same four...)
ArrowDigester.hash_table(table, *, ...same four...)
ArrowDigester.hash_array(array, *, ordered_columns=False, ordered_struct_fields=False, nested_validity=False)
```

`hash_table` = `new(table.schema)` + `update` per batch of `table.to_batches()` + `finalize`.

### 3.3 D1: one switch or two?

Two. Orcapod will turn both on for values and both off for streams, so one switch would cover
today's need, but:

* the developer asked for order independence to be switchable "for columns and for struct
  fields";
* the two are orthogonal in the byte layout (§5), so the second switch costs one flag bit and a
  few golden vectors, not a second design;
* a table whose column order is meaningful but whose struct columns come from sources with
  unstable field order (JSON-derived records) is a plausible need, and a single switch can never
  be split later without defining new hashes.

All four combinations are defined, domain separated from each other (§5.1: each combination has
its own `flags` byte in the options tag) and covered by golden vectors.

Names: `ordered_columns`, `ordered_struct_fields`. Alternatives considered:
`column_order_sensitive` / `struct_field_order_sensitive` (longer), `preserve_column_order`
(reads like a data transformation, not a hash option).

### 3.4 D2: null struct vs all-null-fields struct

Today a struct is transparent: it has no entry of its own and its nulls are only ANDed into its
descendants. Verified on `main` (v0.4.0):

1. **ITL-848.** `s: struct<a: int32 nullable>` with rows `[null, {a: null}]` versus `[{a: null}, {a: null}]`
   (null struct versus all-null fields) give the same hash.
2. **Worse: rows vanish.** `s: struct<a: int32 non-null>` (nullable struct) with rows `[{a:1}, null, {a:3}]`
   (3 rows) gives the same hash as `[{a:1}, {a:3}]` (2 rows): the child has no validity bits, and
   null-struct rows are skipped in its data.

A systematic search (pyarrow-generated tables, 420 column types: list / large list / struct /
struct with a non-null sibling, nullable or not at every level, int and dictionary leaves, up to
three container levels) found that **262 of 420** types lose information on v0.4.0, by exactly
two rules: (1) the type contains a nullable struct anywhere (including the column itself);
(2) a list's element is a nullable list (outer lengths are lost; the total number of inner
lists survives, so 2-level single-row tables are safe but `[[1]],[[2],[3]]` vs
`[[1],[2]],[[3]]` collide, and 3-level single rows collide) or a nullable struct.

Recommendation (accepted provisionally): **one separate switch** with one rule: every nullable
struct, and every nullable list or struct that is a list element, records its own validity
bits (byte-layout spec 9.5). With it the same search finds 0 collisions. Reasons for a separate
switch:

* it is orthogonal to order; Orcapod streams (order independent) have the same bug and should be
  able to opt in without also becoming order sensitive;
* bundling it into an order switch would make "ordered" mean two unrelated things forever.

Alternative: bundle it into `ordered_struct_fields` (fewer switches, but streams cannot get the
fix). Default behaviour does not change in either case.

### 3.5 Booleans (found by the golden generator's relation check)

Default Boolean data is bit-packed per call and a non-nullable Boolean entry has no count:
`[true]` and `[true, false, false]` collide (one-column, non-nullable), `hash_array([true]) ==
hash_array([true, false])`, and Boolean columns are not batch-split independent (a 2-chunk table
differs from its combined form). A sweep of every supported leaf type × nullability × position
(top level, struct, list, nullable struct, nullable list) × {row-count sensitivity, batch split,
slice/offset, empty vs absent} found that in the default mode **only Booleans** break batch-split
independence; all other default-mode hashes are batch-split and slice independent.

Recommendation (accepted provisionally, option A): fold the fix into the third switch: one byte
per valid Boolean value (byte-layout spec 9.5.1). The third switch is thereby the "lossless
layout" switch; with it the sweep (320 cases, `tests/lossless_properties.rs`) is clean.

---

## 4. D3: how position enters the digest

Recommendation: **position enters through the canonical JSON only.** In ordered mode the
canonical schema JSON (and the canonical type JSON of `hash_array`) lists columns and/or struct
fields as JSON arrays in declaration order instead of sorted by name. Data entries, their paths,
their contents and their finalization order (BTreeMap path order) are exactly as today.

Why this is sufficient: the schema digest is the first input of the final digest. Given the
ordered schema, every entry path names exactly one position in it (D7 guarantees paths are
unique), so `(ordered schema, entries keyed by path)` determines the table including its column
and struct-field order. Two inputs that differ only in order differ in the schema digest.

Why it is preferred:

* no change at all to the data path: zero performance cost when off, and almost none when on;
* starfix-python only needs a second serializer for the canonical JSON;
* the same rule covers every depth and container for free: struct in struct, in list,
  in large list, in fixed-size list, in map entries, dictionary value types (dictionaries are
  replaced by their value type before serialization), and extension types (arrow-rs and the IPC
  format carry an extension type as its storage type plus `ARROW:extension:*` field metadata, so
  the storage struct is serialized by the same rule; the metadata only counts with
  `include_metadata`).

Alternative (positional entry paths: entries keyed by column/field index instead of name and
finalized in declaration order) would also support duplicate names, but it is a second, parallel
data layout that starfix-python must reproduce exactly, and it gains nothing else.

### 4.1 D7: duplicate names and path aliasing

Entries are keyed by name paths joined with `/`. Two cases make two schema positions share an
entry, which makes the hash depend on how rows are split into batches and lets different tables
collide:

* duplicate sibling names (`pa.table([[1], [2]], names=["a", "a"])`, or a struct with two
  children `a`);
* a name containing the delimiter (a column `"a/b"` next to a struct column `a` with child `b`;
  a column `"x/"` next to a nullable list column `x`).

These are already ambiguous today; the default behaviour is not changed. **Whenever any new switch
is on, such schemas are rejected** (Rust panics, documented under `# Panics`; Python raises
`ValueError`). Precisely: every schema node (column, struct child, list / fixed-size list / map
element) has a node path (`name`, `parent/child`, `parent/`); the schema is rejected when two
nodes share one (byte-layout spec 9.6). This covers duplicate names at any depth and `/`
aliasing, but accepts a `/` that aliases nothing (`km/h`). It runs in `hash_schema`, on digester
creation, on every batch, and in `hash_array`. (Checking node paths rather than entry paths
avoids flagging the default layout's own entry sharing between a list and its element.) Rejecting now
keeps the door open: a later definition for these inputs would not move any hash.

pandas (`Table.from_pandas`) and Polars already refuse duplicate column names, so Orcapod values
from those sources are unaffected.

---

## 5. Byte layout (to be written into `docs/byte-layout-spec.md` as a new section)

Notation as in the byte-layout spec. `flags` is one byte:

| bit | value | switch |
|---|---|---|
| 0 | `0x01` | `ordered_columns` |
| 1 | `0x02` | `ordered_struct_fields` |
| 2 | `0x04` | `nested_validity` |
| 3–7 | 0 | reserved, must be 0 |

`include_metadata` is **not** a flag bit; it keeps its existing Phase 2 behaviour in every mode.

### 5.1 Options tag (D4)

```
options_tag = 0x00 || "starfix-options-v1" (18 ASCII bytes) || flags     // 20 bytes
```

When `flags == 0x00` nothing is written and every byte is exactly as in v0.4.0.
When `flags != 0x00`:

* **Schema digest** (`hash_schema`, and the schema digest inside `new` / `hash_record_batch` /
  streaming):
  ```
  hasher = SHA-256()
  hasher.update(options_tag)                       // 20 bytes
  hasher.update(phase1_json(schema, flags))        // §5.2
  if include_metadata: Phase 2 exactly as §2.2.1   // unchanged
  schema_digest = hasher.finalize()
  ```
  `hash_schema` returns `00 00 01 || schema_digest`, as today. The record-batch final digest
  starts with this `schema_digest`, as today.
* **`hash_array`**: the final digest starts with the tag, then the type JSON:
  ```
  final = SHA-256()
  final.update(options_tag)
  final.update(type_json(effective_type, flags))   // §5.3
  ... entries exactly as byte-layout §6 (with §5.4 if nested_validity) ...
  output = 00 00 01 || final.finalize()
  ```

All bits of `flags` are written as configured at every entry point, including bits that have no
effect there (`ordered_columns` for `hash_array`, `nested_validity` for `hash_schema`). One rule,
no per-entry-point masking that an independent implementation could get wrong.

**Why the tag separates the modes.** Every untagged input to these SHA-256 instances starts with
canonical JSON, which never starts with `0x00`. Every tagged input starts with `0x00` and carries the
full `flags` byte at a fixed offset. So for any two different `flags` values the SHA-256 inputs
differ, and equal outputs would be a SHA-256 collision. For record batches the separation is
inherited through the schema digest (first 32 bytes of the final input).

**Version prefix.** Stays `00 00 01`: it is the version of the byte layout family, and options do
not change it (precedent: `include_metadata` did not change it). Alternative: a distinct prefix
per mode so the mode is visible in the hash; rejected because the prefix would then mean two
things and three switches would need eight prefixes.

**D5: the one-column table.** With any new switch on, a one-column table (no structs) hashes
differently from the default mode, although order cannot matter. Justification: the alternative
(equal when order is meaningless) means an ordered hash of `struct<a, b>` equals the unordered
hash of `struct<b, a>` (because unordered mode sorts `b, a` into `a, b`). That is a real
cross-mode collision: a value and a stream with different identities would share a hash. A clean
"hash identifies (switches, input)" rule avoids it. The empty-metadata invariant is kept
*within* each mode: for a fixed `flags`, `include_metadata = true` on a schema without metadata
equals `include_metadata = false`.

### 5.2 Phase 1 canonical schema JSON with flags

Same canonicalization as byte-layout §2.1 (type equivalence classes, compact JSON, object keys
sorted recursively), with two changes:

* **`ordered_columns = 1`**: the top level is a **JSON array** with one element per column in
  schema order. Each element has the struct-child form
  `{"data_type": <type>, "name": "<column>", "nullable": <bool>}` (keys sorted).
  With `ordered_columns = 0` the top level is the existing object keyed by column name.
* **`ordered_struct_fields = 1`**: every `{"Struct": [...]}` array lists the children in
  declaration order instead of sorted by name, at every depth: struct children, list / large list
  / fixed-size list elements, map entry structs, dictionary value types.
  With `ordered_struct_fields = 0` children are sorted by name, as today.

Examples. Schema `[b: Int32 non-null, a: LargeUtf8 nullable]`:

```
flags 0x00: {"a":{"data_type":"LargeUtf8","nullable":true},"b":{"data_type":"Int32","nullable":false}}
flags 0x01: [{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}]
```

Schema `[s: Struct<z: Int32 non-null, a: Int32 non-null> non-null]`:

```
flags 0x00, 0x01 sorted:   ...{"Struct":[{"data_type":"Int32","name":"a",...},{"data_type":"Int32","name":"z",...}]}...
flags 0x02, 0x03 declared: ...{"Struct":[{"data_type":"Int32","name":"z",...},{"data_type":"Int32","name":"a",...}]}...
```

Schema digest input for the first example with `flags = 0x01`:

```
00 73 74 61 72 66 69 78 2d 6f 70 74 69 6f 6e 73 2d 76 31 01     // options_tag
[{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}]
```

Phase 2 (`include_metadata`) is unchanged: its `"fields"` map is keyed by name paths, which are
unique (§4.1), and the order is already pinned by Phase 1.

### 5.3 `hash_array` type JSON with flags

`type_json` is `data_type_to_value` of the effective type (byte-layout §6) with the
`ordered_struct_fields` rule of §5.2. `ordered_columns` has no effect on the JSON. The array's
nullability is, as today, "the array has at least one null" (`Array::is_nullable`).

### 5.4 `nested_validity = 1`: struct validity bits (superseded)

Superseded by byte-layout spec Sections 9.5 (which also covers list elements that are nullable
lists) and 9.5.1 (Booleans). The original text follows.


Entry extraction (byte-layout §3), additional rule. For a **Struct node at path P whose field is
nullable**:

* if an entry already exists at P (this happens exactly when the struct is the element type of a
  list: P is the list's element path `parent/`, whose structural entry already has `null_bits`
  because the element is nullable), the struct's validity bits go into **that entry's existing
  `null_bits`**;
* otherwise a **validity-only entry** is created at P (for a top-level struct column `s`, P = `s`;
  for a struct child, P = `parent/child`; for `hash_array` of a struct array, P = `""`).

Traversal: when a nullable struct array of length n is visited at P, push n bits into that
`null_bits`: bit i = struct's own validity AND the ancestor struct validity (the same combined
validity that is propagated to the children today). If the struct array has no null buffer and
there is no ancestor null buffer, push n `1` bits. Children are then traversed exactly as today
(the struct's nulls are still ANDed into its descendants).

Non-nullable structs get no entry (as non-nullable leaves get no validity bits).

Finalization is unchanged: entries in BTreeMap path order (byte-wise UTF-8 order; Python's
`sorted()` on `str` gives the same order), each written per byte-layout §4. A validity-only entry
at P sorts before all entries below P.

Effect on the examples of §3.4, with `nested_validity = 1`: `[null, {a: null}]` gives struct bits
`[0, 1]` versus `[1, 1]` for `[{a: null}, {a: null}]`; the 3-row case records 3 struct bits.

Note: with `nested_validity = 0`, a `List<nullable Struct>` element entry has `null_bits` that are
never filled, so it always contributes `bit_count = 0` (8 zero bytes). That is part of the frozen
default layout and stays as is.

### 5.5 Streaming (D6)

The digester stores the Phase 1 JSON (under its `flags`) plus, with `include_metadata`, the
metadata key, as today. Every batch passed to `update` must produce the same key under the same
`flags`, otherwise it is rejected (Rust panic "Record batch schema does not match ArrowDigester
schema"; Python `ValueError`). Consequences:

* `ordered_columns = 1`: a batch with the same columns in a different order is rejected.
  With `ordered_columns = 0` it is accepted, as today.
* `ordered_struct_fields = 1`: a batch whose struct fields (any depth) are in a different order is
  rejected. With `= 0` it is accepted.

Rejection rather than silent reordering: in ordered mode a different order is a different schema,
and accepting it would make the hash depend only on the first batch's order.

Batch-split independence holds in every mode: all new contributions (tag, JSON, struct validity
bits) are either per-digester constants or appended per row.

### 5.6 Invariants

| # | Invariant |
|---|---|
| I1 | `flags = 0`: byte-identical to v0.4.0 at every entry point (all existing tests and golden vectors pass unchanged). |
| I2 | Hashes taken with different `flags` never coincide (modulo SHA-256 collisions), including one-column tables. |
| I3 | `ordered_columns = 1`: permuting distinct columns changes the hash. `= 0`: it does not, whatever the other switches. |
| I4 | `ordered_struct_fields = 1`: permuting struct fields at any depth changes the hash. `= 0`: it does not. |
| I5 | In every mode: `Utf8`≡`LargeUtf8`, `Binary`≡`LargeBinary`, `List`≡`LargeList`, dictionary≡plain. |
| I6 | In every mode: splitting rows into batches does not change the hash. |
| I7 | Within a fixed `flags`: `include_metadata = true` without any metadata equals `include_metadata = false`. |
| I8 | `nested_validity = 1`: null struct ≠ struct with all-null fields; null struct rows are counted. |
| I9 | Any new switch on: duplicate sibling names and entry-path collisions are rejected. |

---

## 6. Golden vectors (D10)

As built: `tests/golden/ordered_v0.5.json`, 170 vectors and 73 relations (hash_record_batch,
hash_schema, hash_array, hash_stream), generated by `cargo run --bin emit_golden_ordered`, format
in byte-layout spec 9.11. The plan below was the starting point.

### 6.1 File and format

`tests/golden/ordered_v0.5.json` (new; the v0.3 file is never touched). Same shape as
`include_metadata_v0.3.json`, extended:

```json
{
  "version": "0.5",
  "generated_by": "cargo run --bin emit_golden_ordered",
  "rust_commit": "<git rev-parse HEAD>",
  "vectors": [
    {
      "id": "cols_ab_ordered",
      "description": "...",
      "method": "hash_schema | hash_record_batch | hash_stream | hash_array",
      "include_metadata": false,
      "ordered_columns": true,
      "ordered_struct_fields": false,
      "nested_validity": false,
      "ipc_b64": "<Arrow IPC stream>",
      "expected_hash": "000001..."
    }
  ],
  "relations": [
    { "kind": "equal",     "ids": ["cols_ab_unordered", "cols_ba_unordered"], "why": "..." },
    { "kind": "different", "ids": ["cols_ab_ordered", "cols_ba_ordered"],     "why": "..." }
  ]
}
```

Methods (the IPC stream is decoded with `arrow_ipc::reader::StreamReader` /
`pyarrow.ipc.open_stream`):

| method | input |
|---|---|
| `hash_schema` | the stream's schema |
| `hash_record_batch` | exactly one batch |
| `hash_stream` | `new(schema)`, `update` for each batch in stream order (zero or more), `finalize` |
| `hash_array` | exactly one batch with exactly one column; hash that column's array |

`relations` lets both test suites assert the pairwise claims (equal / all different) directly,
on top of each vector's pinned value.

### 6.2 Coverage (planned ids, roughly 50 vectors)

* **Column order**: `[a, b]` vs `[b, a]`, for `hash_schema`, `hash_record_batch`, `hash_stream`:
  different with `ordered_columns`, equal without (and equal to the default-mode hash).
* **Struct order depth 1**: `struct<x, y>` vs `struct<y, x>`, schema and data.
* **Nested struct order**: in struct (`struct<p: struct<b, a>>`, only the inner level permuted),
  in list, in large list (equal to the list variant), list of list of struct.
* **All four order combinations** (`ordered_columns` × `ordered_struct_fields`) on one input that
  has both a column permutation and a struct permutation: each combination equal/different
  exactly as I3/I4 predict, and all combinations pairwise different (I2).
* **Domain separation**: one-column table, no structs, under all flag values: pairwise different.
* **include_metadata**: ordered + field and schema metadata; ordered + no metadata equals the
  same with `include_metadata = false` (I7); extension-type metadata on a struct storage column.
* **Edge cases**: empty table (schema, zero batches), zero-row batch, zero columns, nulls in
  leaves, all-null column.
* **Struct validity**: null struct vs all-null-fields struct (equal off, different on); the
  3-row vs 2-row case; nullable struct as list element; nested nullable structs; `hash_array` of
  a nullable struct array.
* **Physical equivalence in ordered mode**: dictionary vs plain, `Utf8` vs `LargeUtf8`, `List` vs
  `LargeList` (equal).
* **Streaming**: two batches equal one combined batch, in ordered mode and with struct validity.
* **`hash_array`**: struct arrays reordered on/off, list of struct, the default-config vector
  equal to `hash_array` without config.

Rejection cases (D6, D7) are tested in Rust unit tests and specified for Python, not in the
golden file (no hash to pin).

### 6.3 Generator and protection against silent changes

* `src/bin/emit_golden_ordered.rs` builds the vectors with arrow-rs, writes IPC, computes hashes
  with `ArrowDigester`, prints JSON to stdout (as `emit_golden_metadata` does).
* **Append-only guard:** when `tests/golden/ordered_v0.5.json` exists, the generator loads it and
  aborts if any existing `id` would get a different `expected_hash`. New ids may be added; values
  never move.
* `tests/golden_vectors_ordered.rs` re-hashes every vector from its IPC blob and asserts the pinned
  value, then checks every relation. Runs in `cargo test` in CI.
* starfix-python copies the file byte-for-byte and runs the same assertions (and its existing
  golden-sync check can be extended to the new file).

---

## 7. Type support (unchanged by this PR)

Ordered mode changes no type support. **Schema hashing** (`hash_schema`, and the schema part of
every other entry point) accepts every Arrow type that arrow-rs serializes, and the ordered
struct rule applies inside list, large list, fixed-size list, struct, map and dictionary types.

**Data hashing** (`hash_record_batch`, streaming, `hash_array`, Python `hash_table`) supports:

| Supported | Notes |
|---|---|
| `Boolean`, `Int8`–`Int64`, `UInt8`–`UInt64`, `Float16/32/64` | |
| `Date32`, `Date64`, `Time32`, `Time64`, `Timestamp`, `Duration` | |
| `Decimal32/64/128/256`, `FixedSizeBinary` | |
| `Binary`, `LargeBinary`, `Utf8`, `LargeUtf8` | small and large hash the same |
| `List`, `LargeList` | small and large hash the same |
| `Struct` | |
| `Dictionary(_, V)` | for any supported `V`; hashes as plain `V` |

Not supported (Rust panics with `not yet implemented` when data of that type is hashed;
starfix-python should raise an error rather than invent a layout): **`Null`**, **`FixedSizeList`**,
`Map`, `Interval`, `BinaryView`, `Utf8View`, `ListView`, `LargeListView`, `Union`,
`RunEndEncoded`.

Consequences for the Polars consumer:

* **Null-typed (all-null) columns** cannot be data-hashed in any mode, so their position cannot
  enter an ordered data hash. A consumer that drops them hashes a different table: two tables
  that differ only in a null-typed column, or in its position, then hash the same. Casting the
  column to another type instead (e.g. all-null `bool`) keeps the column, its name and its
  position in the hash. Defining a layout for the `Null` type is a new hash definition (it moves
  no existing hash, because these inputs panic today) and is proposed as a follow-up, not part of
  this PR.
* **`FixedSizeList`**: casting to `LargeList` before hashing remains necessary. Defining it
  natively is likewise a follow-up.

---

## 8. What makes this hard, and cost when off

* **Order is erased at serialization time.** `serialized_schema` collects into a `BTreeMap`, and
  `data_type_to_value` sorts struct children. A flags-aware variant is needed; the default code
  path keeps the existing functions.
* **Data entries are keyed by name paths.** Handled by keeping data as is and pinning order in the
  schema digest; that requires unique paths, hence D7.
* **Structs are transparent.** That is the source of the ITL-848 gap and the vanishing-row case;
  `nested_validity` adds entries without disturbing the existing ones.
* **`List<nullable Struct>` quirk** (§5.4 note): reused deliberately by `nested_validity`.
* **`hash_array` has no config.** D8.
* **Exhaustive `HasherConfig` literals** in tests (D11, §3.1).

**Performance when off:** the data path is untouched by the order switches. `nested_validity` adds
one boolean check per struct node per batch (not per row). Schema serialization and the
per-`update` schema check keep their existing fast path when `flags = 0`. No measurable cost is
expected; the existing test suite's timings will be compared before and after.

---

## 9. Open questions for the developer

1. Approve D1 (two order switches) and the names.
2. Approve D2: `nested_validity` as a separate third switch (versus bundled into
   `ordered_struct_fields`).
3. Approve D5 strict separation (one-column table hashes differ between modes).
4. Approve D7 rejection of duplicate names / path aliasing under the new switches.
5. `Null` and `FixedSizeList` data hashing stay unsupported in this PR (§7); follow-up issue?
6. Zero-column tables: today the row count of a table with no columns is not hashed (in every
   mode). Recommendation: leave as is and document; pin it in a golden vector.
