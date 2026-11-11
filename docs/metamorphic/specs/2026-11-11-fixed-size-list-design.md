# FixedSizeList data hashing in hash format 2 (ITL-927) — design

Status: **proposal, awaiting maintainer approval**. No library code is written before approval.

Base: PR #20 (hash format 2, `e079d56`). References are to `docs/byte-layout-spec.md` (the
"spec"). Maintainer decision (2026-11-11), binding: *"Fixed length list is NOT type identical to
variable length list so should hash differently."* `FixedSizeList<T>[n]` and `List<T>` /
`LargeList<T>` with the same values must hash differently (schema and data), and
`FixedSizeList<T>[n]` / `FixedSizeList<T>[m]`, `n != m`, are different types.

Points marked **[FORMAT]** define hashes and need the maintainer's explicit approval.

---

## (a) Canonical type JSON — no change (already pinned)

Format 2 already serializes and pins `FixedSizeList` schemas; this design keeps that exactly:

```
FixedSizeList(f, n)  →  {"FixedSizeList":[<element>,n]}
<element>            =  {"data_type":<type>,"nullable":<bool>}     // same as a List element
```

* **List size**: `n` is the second array item (decimal integer), so `[2]` and `[3]` differ, and the
  key `"FixedSizeList"` differs from `"LargeList"`: `hash_schema` (and therefore every record
  batch / stream hash, which embeds the schema digest, and every `hash_array`, which embeds the
  type JSON) distinguishes FixedSizeList from List and by size.
* **Element name**: not hashed (`item`, `element`, ...), exactly like the List element (spec 2.4).
* **Element nullability and type**: hashed; the element type is canonicalized recursively
  (`Utf8`→`LargeUtf8`, `List`→`LargeList`, dictionary→values, struct children sorted unless
  `ordered_struct_fields`).
* **Metadata**: the element field's metadata is Phase 2 metadata at node path `p/` (spec 2.3).
* `FixedSizeList` is **not** canonicalized to `LargeList` (that would make them type-identical).

Already pinned by committed vectors: `fsl_int_schema_f0/_f1`, `fsl_struct_ab/ba_schema_f0/_f2`
(+ relations: struct order ignored at this depth by default, respected with
`ordered_struct_fields`), `meta_fsl_element_schema_f0_im1/_f1_im1`,
`ext_tensor_canonical_schema_f0_im1`, and the printed example in spec 2.1
(`hash_schema = 0000022fe3f8…6eae88`). This proposal therefore adds **no schema-side decision**.

## (b) Data **[FORMAT]**

### Nodes (unchanged)

A `FixedSizeList` node `p` has one child node, its element, at `p + "/"` (already claimed by
the path checks of spec 7.1 and used by Phase 2).

### Entry of a FixedSizeList node **[FORMAT]**

Per-node rule `[validity if nullable][lengths digest if list][values digest if leaf]`. A
fixed-size list has **no lengths item** (every valid slot has exactly `n` items and `n` is in the
schema digest) and no values item. Its entry is therefore exactly that of a struct:

```
entry(p) = validity(p)        if p's field is nullable   // bit_count (u64) || validity words
         = (empty)            otherwise
```

where `validity(p)` is the **effective** validity (own validity AND enclosing struct validity at
the same list level), exactly as for every other node (spec 3.2).

### Elements of the element node **[FORMAT]**

Fixed-size lists follow **list semantics** (a list level), not struct semantics:

* The elements of `p/` are the `n` items of every **effectively valid** slot of `p`, in order:
  slot `i` (0-based, logical, after any array offset) covers child positions
  `[i·n, (i+1)·n)` of the logical child array. With `k` effectively valid slots, `p/` has
  `k·n` elements.
* Items of null slots (own null, or effectively null because an enclosing struct is null) are
  **not elements**: they are never visited, whatever they hold (values, nulls, nested
  structure). Arrow requires a null fixed-size-list slot to still occupy `n` child positions;
  those hidden positions do not affect the hash.
* A fixed-size-list boundary resets struct masking (as for List): the element's enclosing-struct
  mask starts all-valid. (Every visited item belongs to an effectively valid slot.)
* Section 7 item 4 applies unchanged: a **visible** null (in an element) of a non-nullable
  element field is rejected; a **hidden** null (under a null slot) is accepted and ignored.

Child entries are then built by the existing rules. Consequently the entries of the subtree
below `p/` are byte-identical to those of `List<T>` with the same values; only the list node's
own entry (no lengths digest) and the schema / type JSON differ.

### Worked example (to be pinned in `tests/format2_bytes.rs`)

`f: FixedSizeList<Int32 nullable, 2>` nullable, rows `[[1, null], null, [3, 4]]`, child array
`[1, null, 7, 8, 3, 4]` (`7, 8` hidden under the null slot):

```
final = SHA-256(
    header('R', 0x00)
    schema_digest   // SHA-256(header('S',0x00) || {"f":{"data_type":{"FixedSizeList":[{"data_type":"Int32","nullable":true},2]},"nullable":true}})
    03 00 00 00 00 00 00 00                      // row count
    03 00 00 00 00 00 00 00  05                  // entry "f":  validity [1,0,1]   (no lengths)
    04 00 00 00 00 00 00 00  0d                  // entry "f/": validity [1,0,1,1]  (items of slots 0 and 2 only)
    SHA-256(01 00 00 00  03 00 00 00  04 00 00 00)   //         values [1, 3, 4]
)
```

`c: List<Int32 nullable>` nullable with the same rows has entry `"c"` =
`03 00.. 05 || SHA-256(02 00.. 02 00..)` (lengths [2, 2]) and a different schema digest.

### Losslessness (the format's own argument, extended)

Fix the schema; its digest is in the final input, so only equal schemas can collide (modulo
SHA-256). Then:

1. **Parsing.** Which entries exist, and which items each has, is a function of the schema
   alone. A validity item is self-delimiting (`bit_count` prefix), digests are 32 bytes, and a
   fixed-size-list entry is either a validity item or empty (schema-determined). So the
   concatenation `row_count || entry(n_1) || entry(n_2) ...` parses uniquely into per-node items.
2. **Reconstruction, top down.** For each node we know its element count `N` and its
   enclosing-struct mask `M` (columns: `N = row_count`, `M` all 1).
   * nullable node: validity gives the effective bits `E` (`bit_count = N`); non-nullable: `E = M`.
   * struct: each child has `N` elements and mask `E`.
   * list: the lengths digest fixes (modulo SHA-256) the lengths of the `|E|` valid elements; the
     child has their sum as element count and mask all 1; the partition is given by the lengths.
   * **fixed-size list**: the child has `n·|E|` elements and mask all 1; the `j`-th valid slot
     owns child elements `[j·n, (j+1)·n)`. No information is needed beyond `E` and `n`.
   * leaf: the values digest fixes (modulo SHA-256) the self-delimiting values of the `|E|`
     valid elements.
   Every logical value (row count, every null at a visible position, every visible value, list
   shapes) is recovered, so two tables that differ in any logical value give different SHA-256
   inputs.
3. **Canonicality.** Conversely every item is a function of the logical value only: positions
   that are not elements (items of null slots) contribute nothing, and a validity bit is the
   *effective* validity, so physical layout (offsets, slices, chunking, hidden child content,
   dictionary encoding) cannot reach the input.

The required cases:

| Case | Behaviour |
|---|---|
| null slots | `p`'s validity bit 0 (if nullable); the slot's `n` hidden child positions are not elements of `p/`: changing them (values, nulls, nested content) never changes the hash; visible ones always do (step 2). |
| `n = 0`, no nulls | entry `p` = validity (all 1) or empty; `p/` (and its whole subtree) has 0 elements, so every entry below is that of an empty node: validity items `00 00 00 00 00 00 00 00` (bit_count 0, no words), lengths / values items `SHA-256("")`. Information = row count + validity, as it must be: `[[], [], []]` vs `[[], []]` differ by row count. |
| `n = 0`, with nulls | as above; the validity of `p` distinguishes `[]` from `null`. A non-nullable `FixedSizeList<_, 0>` column's only information is the row count. |
| FSL of FSL | inner node `p/` is an FSL with `n·|E|` elements; its effective validity decides which of ITS slots reach `p//`. Elements of `p//` = `m` items per valid inner slot of every valid outer slot. |
| FSL of List | `p/` is a list node: [validity] + lengths of its valid elements (only those inside valid outer slots). |
| FSL of Struct | `p/` is a struct node; its children have one element per `p/` element and the struct's mask (ANDed down, reset at `p/`'s own list level). Struct-masked nulls of non-nullable grandchildren are accepted. |
| FSL of Dictionary | `p/` is the decoded value node (spec 3.1); a valid index pointing at a null value is a null (rejected if the element is non-nullable). |
| List / Struct / Dictionary of FSL | the FSL node is an ordinary node in that position: under a list its elements are the list's items; under a struct its effective validity is ANDed with the struct's; a dictionary of FSLs is the decoded FSL array (decoded per row without rebuilding, so hidden child nulls are not validated). |
| non-nullable element, hidden nulls under a null slot | accepted, ignored (step 3). |
| non-nullable element, visible null | rejected (spec 7 item 4), Rust panics "A non-nullable field contains null values". |
| sliced arrays | slot `i` is logical; arrow-rs's `FixedSizeListArray::values()` is already offset-adjusted. (pyarrow: `.values` IGNORES the offset; use `arr.offset`.) |
| multi-batch streams | validity bits form one bit string, items are appended in row order, digests are streamed: identical to one batch. |
| dictionary-encoded FSL | identical to the plain decoded column, for record batches, streams and `hash_array`. |
| `hash_array` root | the root `""` is always nullable: entry `""` = validity of the array; element node `"/"`; `type_json` = `{"FixedSizeList":[<element>,n]}`. |
| extension types over FSL storage (e.g. `arrow.fixed_shape_tensor`) | hash as their storage (spec 2.3); now data-hashable. |

## (c) What else changes

* **Rejections (spec 7 / 8).** `FixedSizeList` is removed from the "data not supported" list
  (Section 8). `Null`, `Map`, `Interval`, `Utf8View`, `BinaryView` stay rejected (a
  `FixedSizeList` containing one of them is still rejected). Item 4's sentence about hidden
  nulls "for `List` and `LargeList` alike" adds `FixedSizeList`.
* **Node paths / ambiguity (spec 3.1, 7.1).** Unchanged: an FSL `f` already claims `f` and `f/`.
  Only the sentence "Fixed-size-list ... nodes exist for the path checks and Phase 2 only" changes.
* **`ordered_columns` / `ordered_struct_fields`.** Unchanged: they affect only headers, Phase 1 and
  the `hash_array` type JSON (already pinned for FSL); entries are identical across flags.
* **`include_metadata`.** Unchanged (element metadata at `p/`, already pinned).
* **Streaming schema check (spec 5).** Unchanged; `FixedSizeList` is not equivalent to any other
  type, so a batch with `List` where the digester has `FixedSizeList` (or another size) is
  rejected.

## (d) No existing hash moves

* Every input whose data hashing reaches the new code contains a `FixedSizeList`, and every such
  input is rejected today **before any data is hashed** (`validate_data_type`, spec 7 item 3,
  also for zero rows / no batches). So no accepted input can change. The schema JSON, path
  checks and metadata code are not touched.
* The only shared code touched is the dictionary dispatch (a dictionary whose values are an FSL
  must be decoded per row like a dictionary of lists); it only changes behaviour for dictionaries
  of FSL, also rejected today.
* **Proof by the guard:** `cargo run --bin emit_golden_format2` aborts if any committed vector
  changes in any field (input, method, options, description, hash) or disappears, or any
  committed relation disappears. All 423 vectors and 114 relations must pass it with the new
  vectors appended; `tests/golden_vectors_format2.rs` re-hashes every committed vector
  independently. Existing vector descriptions that say "(schema only: data unsupported)" stay
  byte-identical (the guard compares descriptions); the spec notes they describe the state when
  they were written.

## (e) `FixedSizeBinary(n)` vs `FixedSizeList<UInt8>[n]`

Distinct, automatically: different Arrow types, different canonical JSON
(`{"FixedSizeBinary":n}` vs `{"FixedSizeList":[{"data_type":"UInt8",...},n]}`), and different
entry structure (one leaf entry `p` vs entries `p` and `p/`). A golden `different` relation pins it.

## (f) Alternatives rejected

1. **Canonicalize FSL to `LargeList`** (cast before hashing, as the old implementation plan
   proposed). Violates the maintainer decision: equal values would hash equal.
2. **List-like entry with a lengths digest** (`SHA-256(n as u64 per valid slot)`). Lossless and
   lets an implementation reuse its List code by casting, but the lengths are redundant (`n` and
   the validity already determine them), it hashes `8·k` extra bytes, and the task fixes "a
   fixed-size list has no per-row lengths".
3. **Struct-like semantics** (the element has `n·N` elements including the hidden positions of
   null slots, with the slot's nullity ANDed downwards). Also lossless, but it writes validity bits
   for positions that are not logical values, makes `p/`'s `bit_count` depend on the number of
   null slots, differs from the List rule ("list boundaries reset masking") and would make hidden
   *nested lists* under null slots contribute lengths unless masked further: more rules, no gain.
4. **Flat leaf encoding** for primitive elements (`n·w` bytes per slot like `FixedSizeBinary`).
   Only works for fixed-width non-nullable elements, needs a second encoding for the rest, and
   breaks the node model (no `p/` entry) that metadata paths and path checks already use.
5. **Writing `n` into the entry.** Redundant with the schema digest / `hash_array` type JSON.

## What the Python implementation must mirror

* No schema change (already mirrored if starfix-python hashes FSL schemas per spec 2.1).
* Accept FSL data (remove it from the unsupported list); keep rejecting FSL containing
  unsupported types.
* Entry of an FSL node: validity only if nullable, nothing otherwise — no lengths.
* Element traversal: for each effectively valid slot `i` (logical index), visit child positions
  `[(arr.offset + i)·n, (arr.offset + i + 1)·n)` of `arr.values` (pyarrow's `.values` ignores the
  array offset); never visit null slots' positions; start the element's struct mask fresh;
  decode dictionaries of FSL first (without validating hidden child nulls).
* The golden file vectors added by this change (ids prefixed `fsl_`), to be copied byte for byte.

## Questions for the maintainer

1. **[FORMAT]** FSL node entry = validity if nullable, else empty (no lengths) — approve?
2. **[FORMAT]** List semantics for the element (only items of effectively valid slots are
   elements; hidden child positions never hashed; struct mask reset at the FSL boundary) — approve?
3. Ship placement (0.5.0 vs 0.5.1): the CHANGELOG entry goes under "Unreleased" either way.
