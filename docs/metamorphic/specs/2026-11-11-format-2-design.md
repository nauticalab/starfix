# Starfix Hash Format 2

**Issue:** ITL-898
**Date:** 2026-11-11
> **Update (2026-11-11, developer decision):** S2 is superseded: the 0.4 behaviour is kept as a
> clearly labelled, frozen legacy module `starfix::legacy::v0_4` (v0.4.0 source verbatim), with
> the restored v0.4.0 tests, the format-1 side of `format1_to_format2.json` and a differential
> test against the published 0.4.0 crate. Format 2 itself is unchanged.

**Status:** Proposal for the developer's approval. Nothing below is implemented beyond what
`2026-11-11-ordered-hashing-design.md` already describes; golden files are regenerated only
after approval.

The developer decided (2026-11-11) to bump the hash format version and to build format 2 as the
simplest correct design, with no obligation to stay close to format-1 bytes. starfix-python
matches Rust in every case, so everything here must be observable by a pyarrow-based
implementation.

---

## 1. Summary for approval

| Topic | Format 2 |
|---|---|
| Version prefix | `00 00 02`. The 3 bytes are the format number as a 24-bit big-endian integer (format 1 = `00 00 01`, used by starfix ≤ 0.4). |
| What is hashed | the schema (names, canonical types, nullability; metadata with `include_metadata`), the **row count**, and for every node of the type tree: its validity (if nullable), its list lengths (if a list), its values (if a leaf) |
| What is ignored by default | **only** column order and struct field order. Nothing else is sorted or normalized except the documented physical equivalences (Utf8≡LargeUtf8, Binary≡LargeBinary, List≡LargeList, dictionary≡its values) and the names that pyarrow cannot observe (list element name; map entries/key/value names) |
| Options | `ordered_columns`, `ordered_struct_fields` (unchanged), plus `include_metadata` (unchanged) |
| Rejected (all modes) | two schema nodes with the same node path (duplicate names, `/`-aliasing); `ListView`, `LargeListView`, `Union`, `RunEndEncoded`; a batch whose schema does not match the digester's |
| Unsupported data (schema hashes fine) | `Null`, `FixedSizeList`, `Map`, `Interval`, `Utf8View`, `BinaryView` (data hashing panics / raises) |
| Format-1 hashes | not computable by format-2 code; migration tooling can depend on starfix 0.4 under a renamed dependency (recommended; developer decides) |
| Cost vs v0.4.0 | +4% on a mixed 2M-row batch; Boolean-only worst case 10M values 17.6 → 29.2 ms (one byte instead of one bit per value). Measured on the current corrected layout; the format-2 layout feeds the same bytes plus one 8-byte row count. |

---

## 2. Assessment of the proposed simplifications

| # | Proposal | Verdict |
|---|---|---|
| S1 | Always write the options header, also at flags 0 | **Accept.** One code path for every mode; every format-2 digest is self-describing and can never equal a format-1 digest even with the prefix stripped. The header carries the format number too (Section 3.1). Costs 12 bytes per digest. |
| S2 | No format-1 code paths | **Accept.** Format-1 hashes stay computable with starfix 0.4 (`starfix_v1 = { package = "starfix", version = "0.4" }` in Cargo; a separate environment in Python). No legacy mode in the crate. |
| S3 | One rule: every nullable node has validity bits, every list has lengths; no replace/reuse cases | **Accept, made fully uniform:** one entry per node, holding exactly [validity][lengths][values] as applicable (Section 3.3). This also removes the format-1 "list-leaf" combined entry. |
| S4 | Boolean one byte per valid value | **Accept.** Simplest to specify; counted and batch-split independent. (Alternative with 1/8 of the bytes: pack bits continuously across batches and hash them with a bit count at finalize, like validity bits. Slightly faster, slightly more to specify. Not recommended.) |
| S5 | Reject ambiguous node paths in all modes | **Accept.** In format 1 such schemas hash ill-defined (batch-split dependent, collide). Also: metadata is keyed by node path, so one path notion covers both. |
| S6 | Per-batch schema check in all modes | **Accept** (Rust already does it; Python aligns). |

My additional simplifications:

| # | Proposal | Why |
|---|---|---|
| A1 | Hash the **row count** (u64) right after the schema digest (`hash_array`: the array length after the type JSON) | Closes the last two losses with one rule: zero-column tables and empty structs (`Struct<>`, Copilot finding) no longer lose their row count. Every node's element count is then derivable (struct child = parent count; list element = sum of lengths). |
| A2 | List lengths belong to the **list node's** entry; the element node is an ordinary node at `path/` | Removes the "structural-only" and "list-leaf" entry kinds and the replace/reuse special cases (S3). |
| A3 | Metadata paths = node paths | One path notion (format 1 had two, differing below a column named `""`). |
| A4 | Map canonical JSON without field names: `{"Map":{"key":{...},"keys_sorted":b,"value":{...}}}` | pyarrow renames map entries/key/value fields and drops entries-field metadata (verified, Section 4), so names and entries metadata must not be hashed. |

Not adopted: dropping AND-propagation of struct nulls into children (children of a null struct
row would then need data for undefined values); hashing dictionary order flags or list element
names (not logical).

---

## 3. Format 2 byte layout (to become `docs/byte-layout-spec.md`)

### 3.1 Header and output

```
header = 0x00 || "starfix" || 00 00 02 || flags          // 1 + 7 + 3 + 1 = 12 bytes
flags  = (ordered_columns ? 0x01 : 0) | (ordered_struct_fields ? 0x02 : 0)   // bits 2-7 reserved, 0
output = 00 00 02 || SHA-256(...)                          // 35 bytes
```

### 3.2 Schema digest and record-batch / stream hash

```
schema_digest = SHA-256( header || phase1_json(schema, flags) || [phase2_json if include_metadata and any metadata] )
hash_schema   = 00 00 02 || schema_digest
final = SHA-256( schema_digest || total_row_count as u64 LE || entry_1 || entry_2 || ... )   // entries by node path
```

`phase1_json`: as today (object keyed by column name, or an array with `ordered_columns`; struct
children sorted by name, or in declaration order with `ordered_struct_fields`), with the
format-2 Map form (A4). `phase2_json`: `{"fields":{<node path>:{k:v}},"schema":{k:v}}` as today,
keyed by node path (A3); the map entries field has no metadata in format 2 (A4).

### 3.3 Nodes and entries (the one rule)

Node paths: a column is `name`; a struct child is `join(parent, name)`; a list element is
`parent + "/"`. Every node has one entry; entries are finalized in byte-wise path order. An
entry is the concatenation of, in this order and only where applicable:

1. **validity** — if the node is nullable: `bit_count u64 LE || bit words (u8, Lsb0)`; one bit
   per element of the node, = own validity AND the validity of enclosing structs (within the same
   list level);
2. **lengths** — if the node is a list: `SHA-256(length u64 LE for every valid element)`;
3. **values** — if the node is a leaf: `SHA-256(value bytes of every valid element)`, with the
   format-1 leaf encodings (fixed-size little-endian bytes; u64 length + bytes for
   strings/binary) except **Boolean: one byte `00`/`01` per valid value**.

A non-nullable struct's entry is empty. Items of a null list value are not visited at all. A null
struct value is different: its children still have that element (a nullable child records a `0`
validity bit for it), but no lengths or values are hashed for it below the struct (struct nulls
are ANDed into the children's effective validity). The byte-layout spec, Section 3.2, is
authoritative.

`hash_array` = `00 00 02 || SHA-256( header || type_json || array length u64 LE || entries )`,
with the array as the node at path `""` (nullable iff it has a null).

### 3.4 Golden files

- Keep `tests/golden/include_metadata_v0.3.json` byte-identical as the historical format-1 record
  (no longer asserted; a README line in `tests/golden/` says so).
- Replace the unreleased `ordered_v0.5.json` and `fixed_in_v0.5.json` by
  **`tests/golden/format2_v0.5.json`** (all vectors, including regenerated versions of the 9
  metadata vectors under the same ids, generator `cargo run --bin emit_golden_format2`,
  append-only from then on) and **`tests/golden/format1_to_format2.json`** (every format-1 golden
  value and pinned test value with its format-2 value; all move).

---

## 4. pyarrow observability (verified with arrow-rs 57 writer, pyarrow 26.0.1 reader)

| Construct | pyarrow sees | Format 2 |
|---|---|---|
| Map entries field metadata | dropped | not hashed |
| Map entries / key / value field names (e.g. `kv`, `k`, `v`) | renamed to `entries`/`key`/`value` | not hashed (A4) |
| Map key / value field metadata | kept | hashed at `m//key`, `m//value` |
| Map `keys_sorted` | kept | hashed |
| Registered extension type (e.g. `arrow.uuid`) | `ARROW:extension:*` keys removed from `field.metadata`; type exposes them | hashed as stored in IPC; Python must re-add `ARROW:extension:name` / `:metadata` from the type |
| Unregistered extension (`example.point`) | keys kept in metadata | hashed |
| List / fixed-size-list element name and metadata | kept | name not hashed; metadata hashed at `p/` |
| Dictionary `ordered` flag | kept | not hashed (dictionary = its values) |
| Empty struct, zero-column batch with rows | kept | hashed (row count, A1) |

No other gap found. pyarrow 14 cannot decode Decimal32/64 IPC (Python documents a floor).

---

## 5. Names not hashed, and extension types (added on request)

Not hashed, each with an `equal` golden relation: the list / fixed-size-list element field
name; the map entries, key and value field names (pyarrow renames them); the map entries
field's metadata (pyarrow drops it); the dictionary index type and `ordered` flag.

Extension types: Phase 1 hashes the storage type. With `include_metadata`, the field's metadata
is hashed **including** `ARROW:extension:name` and `ARROW:extension:metadata` exactly as stored in
the Arrow field metadata. pyarrow hides these two keys for registered extension types (e.g.
`arrow.uuid`), so a pyarrow implementation re-adds them from `type.extension_name` and
`type.__arrow_ext_serialize__()`. Golden vectors cover a registered (`arrow.uuid`) and an
unregistered extension type, with and without `include_metadata`.

## 6. Open questions for the developer

1. Approve S1–S6 and A1–A4.
2. Format-1 compatibility: no legacy mode in the crate (recommended), or keep one?
3. Boolean encoding: one byte per value (recommended) or continuous bit packing with a count.

## 7. Amendment after the second independent review (approved)

Two changes to the layout above, both approved before the 0.5.0 release; the golden file was
regenerated in full once (every value changed; ids, inputs and relations kept).
`docs/byte-layout-spec.md` is authoritative.

- **Entry-point kind byte (N1).** The header is 13 bytes:
  `0x00 || "starfix" || 00 00 02 || kind || flags`, with kind `S` (schema digest), `R` (record
  batch / stream final, which now starts with `header('R', flags)` before the schema digest) and
  `A` (`hash_array`). Each public output hashes an input that starts with its own kind, so outputs
  of different entry points never coincide by construction, not merely by input shape.
- **`hash_array` root always nullable (S1).** The root's validity is always written, so an array
  hashes the same with or without an all-valid validity buffer, and the rule "nullable iff it has
  a logical null" (which made the hash depend on a property of the data rather than the type) is
  gone.

Also specified in the same pass: dictionaries resolve recursively; `FixedSizeBinary(0)` hashes
(no value bytes); nulls in list child slots that are not elements are ignored for `List` and
`LargeList` alike, while a null at an element of a non-nullable list element is rejected.
