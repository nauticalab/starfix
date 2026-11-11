# Starfix Byte Layout Specification — Hash Format 2

This document specifies **every byte fed into SHA-256** by starfix to hash Apache Arrow schemas,
record batches, streams and arrays, so that an independent implementation (for example
starfix-python, built on pyarrow) can reproduce every hash exactly. Rust is the reference
implementation; the golden vectors in `tests/golden/format2_v0.5.json` pin the values.

All multi-byte integers are **little-endian** unless stated otherwise. `SHA-256(x)` is the
32-byte digest of the byte string `x`; `||` is concatenation.

**Ordering principle.** By default starfix ignores exactly two orders: the order of the
top-level columns and the order of struct fields (at every depth). It sorts or reorders nothing
else: row order, list element order (also for `list<struct<key, value>>`) and the logical values
of dictionary-encoded arrays are hashed as given. The two ignored orders can be made part of the
hash with the options `ordered_columns` and `ordered_struct_fields` (Section 6). Other
canonicalizations, none of which reorders data: physically different but logically equal types
are unified (Section 2.1); metadata maps are sorted by key (they are unordered);
the canonical JSON's own object keys are sorted; names that pyarrow cannot observe are not
hashed (Section 2.4).

---

## 1. Output format and header

Every hash is **35 bytes**: `version || SHA-256(...)`.

```
version = 00 00 02        // hash format number as a 24-bit big-endian integer
```

Format 1 (`00 00 01`) is the format of starfix 0.4 and earlier (Section 10). The version is
independent of the crate version.

Every digest starts with a 13-byte **header** naming the entry point that produced it:

```
header(kind, flags) = 0x00 || "starfix" || version || kind || flags   // 1 + 7 + 3 + 1 + 1 = 13 bytes
                    = 00 73 74 61 72 66 69 78 00 00 02 <kind> <flags>
kind  = 'S' (0x53)  schema digest (Section 2.5)
      = 'R' (0x52)  record batch / stream final digest (Section 3.3)
      = 'A' (0x41)  hash_array final digest (Section 4)
flags = (ordered_columns ? 0x01 : 0) | (ordered_struct_fields ? 0x02 : 0)
```

Bits 2–7 of `flags` are reserved and always 0. **Bit 2 (`0x04`) must never be used**: it briefly
selected a never-released provisional option during the format-2 work.

**Entry points never collide.** The SHA-256 input of each public output starts with its own
header: `hash_schema` outputs `SHA-256(header('S', flags) || ...)`, record batches and streams
`SHA-256(header('R', flags) || ...)`, and `hash_array` `SHA-256(header('A', flags) || ...)`. Two
outputs of different entry points therefore hash inputs that differ in byte 11, so they never
coincide (modulo SHA-256 collisions), whatever the data; the same holds for different `flags`.

---

## 2. Schema

### 2.1 Canonical type JSON

Each type is serialized to a canonical JSON value.

**Type canonicalization.** Before serialization these logical equivalence classes are collapsed:

| Arrow type(s) | Canonical form |
|---|---|
| `Binary`, `LargeBinary` | `"LargeBinary"` |
| `Utf8`, `LargeUtf8` | `"LargeUtf8"` |
| `List(f)`, `LargeList(f)` | `{"LargeList": <element>}` |
| `Dictionary(k, v)` | canonical form of `v`, recursively (index type and `ordered` flag ignored) |

**Nested types.**

| Type | Canonical JSON |
|---|---|
| Struct | `{"Struct":[<child>, ...]}`, children **sorted by name** (by UTF-8 bytes; with `ordered_struct_fields`, in declaration order); `<child>` = `{"data_type":<type>,"name":"<name>","nullable":<bool>}` |
| List / LargeList | `{"LargeList":<element>}`; `<element>` = `{"data_type":<type>,"nullable":<bool>}` (the element field's name is **not** hashed) |
| FixedSizeList(f, n) | `{"FixedSizeList":[<element>,n]}` |
| Map(entries, keys_sorted) | `{"Map":{"key":<element>,"keys_sorted":<bool>,"value":<element>}}` where key and value are the entries struct's first and second field (their names, the entries field's name and its nullability are **not** hashed) |

Example, `m: Map<Utf8, Struct<z: Int32, a: Int32> nullable>` nullable, `keys_sorted = false`:

```
{"m":{"data_type":{"Map":{"key":{"data_type":"LargeUtf8","nullable":false},"keys_sorted":false,"value":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"Int32","name":"z","nullable":false}]},"nullable":true}}},"nullable":true}}
hash_schema = 000002f772b5bc6bb814d98a3bb68a6148dde15eaa590ad54b289c51f9cc4636a52ba7
```

`f: FixedSizeList<Int32 nullable, 2>` non-nullable:

```
{"f":{"data_type":{"FixedSizeList":[{"data_type":"Int32","nullable":true},2]},"nullable":false}}
hash_schema = 0000022fe3f8d8fcba7e393f3c5808894db2f930a8bd8388e9d04b3763d369cc6eae88
```

**Primitive types** use these exact forms (they are arrow-rs serde's, listed here so that no
implementation depends on arrow-rs). `Utf8` and `Binary` never appear: they are canonicalized
to `LargeUtf8` / `LargeBinary` first.

| Arrow type | Canonical JSON |
|---|---|
| Null, Boolean | `"Null"`, `"Boolean"` |
| Int8, Int16, Int32, Int64 | `"Int8"`, `"Int16"`, `"Int32"`, `"Int64"` |
| UInt8, UInt16, UInt32, UInt64 | `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"` |
| Float16, Float32, Float64 | `"Float16"`, `"Float32"`, `"Float64"` |
| Date32, Date64 | `"Date32"`, `"Date64"` |
| Time32(unit), unit ∈ Second, Millisecond | `{"Time32":"Second"}` |
| Time64(unit), unit ∈ Microsecond, Nanosecond | `{"Time64":"Nanosecond"}` |
| Timestamp(unit, tz) | `{"Timestamp":["Microsecond","UTC"]}`; no time zone: `{"Timestamp":["Second",null]}`; the tz string is copied verbatim (`"+05:30"`, `"America/New_York"`) |
| Duration(unit) | `{"Duration":"Millisecond"}` |
| Interval(unit), unit ∈ YearMonth, DayTime, MonthDayNano | `{"Interval":"DayTime"}` |
| Decimal32/64/128/256(precision, scale) | `{"Decimal128":[38,10]}`; scale may be negative: `{"Decimal64":[18,-3]}` |
| FixedSizeBinary(n) | `{"FixedSizeBinary":16}` |
| Utf8 / LargeUtf8 | `"LargeUtf8"` |
| Binary / LargeBinary | `"LargeBinary"` |
| Utf8View, BinaryView | `"Utf8View"`, `"BinaryView"` (not canonicalized; data unsupported) |

Time units are spelled `Second`, `Millisecond`, `Microsecond`, `Nanosecond`. The golden file has
one `hash_schema` vector per form (`type_*_schema_f0`). `ListView`, `LargeListView`, `Union` and
`RunEndEncoded` are rejected (Section 7).

#### JSON text rules

All JSON fed to a digest (Phase 1 schema JSON, `hash_array` type JSON, Phase 2 metadata JSON) is
produced by `serde_json::to_string`:

* compact: no whitespace; `,` and `:` without spaces;
* object keys sorted by their UTF-8 bytes (equivalently by Unicode code point);
* strings in UTF-8 with **non-ASCII characters written raw** (never `\uXXXX`); escaped are only
  `"` → `\"`, `\` → `\\`, U+0008 → `\b`, U+0009 → `\t`, U+000A → `\n`, U+000C → `\f`, U+000D → `\r`,
  and every other U+0000–U+001F → `\u00xx` with **lowercase** hex; `/` and U+007F are not escaped;
* integers in decimal, booleans `true` / `false`.

For example a column named `é` serializes its key as the two bytes `c3 a9`, and a column named
`a<LF>b<U+001B>` as `"a\nb\u001b"`. The same rules apply to `hash_array`: `hash_array` of
`StructArray [{é: 1}]` (child `é: Int32` non-nullable) feeds `{"Struct":[{"data_type":"Int32","name":"é","nullable":false}]}`
with `é` raw, giving `00000219e467d0280ccc83ad4c67e063997344cedd781f6cb4e449d03d713c989aa12b`.

### 2.2 Phase 1: canonical schema JSON

Without `ordered_columns`: a JSON object keyed by column name (keys sorted), each value
`{"data_type":<type>,"nullable":<bool>}`:

```
{"a":{"data_type":"Int32","nullable":false},"b":{"data_type":"LargeUtf8","nullable":true}}
```

With `ordered_columns`: a JSON array with one `<child>` object per column, in schema order:

```
[{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}]
```

`ordered_struct_fields` keeps struct children in declaration order everywhere a struct occurs
(column, struct child, list / fixed-size list element, map key or value, dictionary value).

### 2.3 Phase 2: metadata (only with `include_metadata`)

```
meta_doc = {}
if any field has metadata: meta_doc["fields"] = { <node path>: {key: value, ...}, ... }
if the schema has metadata: meta_doc["schema"] = {key: value, ...}
// an empty metadata map counts as absent (no entry, no "schema" key)
phase2 = JSON(meta_doc) if meta_doc is non-empty, else nothing
```

All maps are sorted by key (UTF-8 bytes); keys and values are JSON strings. A field's metadata is
recorded under its **node path** (Section 3.1): column `c`, struct child `p + "/" + name`, list
/ fixed-size-list element `p/`, map key `p//key`, map value `p//value` (fixed names). The map's
entries field contributes **nothing** (pyarrow drops its metadata). Dictionary types are replaced
by their value type before the walk.

Example, `m: Map<Utf8, Int32>` with value-field metadata `{"unit":"kg"}`: `{"fields":{"m//value":{"unit":"kg"}}}`.

**Extension types** are Arrow fields whose metadata holds `ARROW:extension:name` (and usually
`ARROW:extension:metadata`). Everywhere — Phase 1, node paths, rejection checks, data, and the
`hash_array` type JSON — an extension type is its **storage type**, and its data is the storage
array. With `include_metadata`, the field's metadata is hashed in Phase 2 like any other field
metadata, **including both `ARROW:extension:*` keys**, with one normalization: if
`ARROW:extension:name` is present and `ARROW:extension:metadata` is absent, it is hashed as
`"ARROW:extension:metadata": ""` (absent and empty are the same). Otherwise the values are
hashed verbatim, as stored in the Arrow field metadata.

For pyarrow-based implementations: pyarrow removes the two keys from `field.metadata` for
extension types it has registered. Add them back: `ARROW:extension:name` =
`type.extension_name`; `ARROW:extension:metadata` = `type.__arrow_ext_serialize__()` decoded as
UTF-8 for Python-defined `pa.ExtensionType` subclasses (these round-trip their stored bytes).
For pyarrow's natively registered canonical extension types, which have no
`__arrow_ext_serialize__` in Python (pyarrow 23: `UuidType`, `FixedShapeTensorType`, ...), read
both keys from the field exported through the **Arrow C data interface**
(`field.__arrow_c_schema__()`): the exported `ArrowSchema` metadata carries them as serialized
by the extension type. These types
(`arrow.uuid`, `arrow.json`, `arrow.bool8`, `arrow.fixed_shape_tensor`,
`arrow.variable_shape_tensor`, `arrow.opaque`, depending on the pyarrow version) expose only the
serialization that the reader reconstructs from its own definition, and rebuild their storage
type the same way. Their hash is therefore **defined only when the stored
`ARROW:extension:metadata` is byte-identical to that reconstructed serialization and the stored
storage type is the one the reader reconstructs**; otherwise a pyarrow implementation and
arrow-rs (which hashes the stored bytes verbatim) disagree. Known stored forms that do not
match: `arrow.json` stored with `{}` (the canonical extension specification allows `""` or
`{}`; Arrow C++/pyarrow reconstructs `""`); `arrow.fixed_shape_tensor`,
`arrow.variable_shape_tensor` and `arrow.opaque` metadata written with another JSON key order or
with whitespace; tensor storage whose element is non-nullable (`arrow.fixed_shape_tensor`
storage is reconstructed as `FixedSizeList<item: T nullable, n>`). Unregistered extension types keep both keys in `field.metadata`.
Without `include_metadata`, an extension type hashes exactly like its storage type.

### 2.4 Names that are not hashed

| Not hashed | Why |
|---|---|
| list and fixed-size-list element field name (`item`, `element`, ...) | Arrow-internal; not part of the logical type |
| map entries, key and value field names | pyarrow renames them to `entries` / `key` / `value` on read |
| map entries field metadata | pyarrow drops it on read |
| dictionary index type and `ordered` flag | a dictionary is hashed as its values |

Each has an `equal` relation in the golden file.

### 2.5 Schema digest

```
schema_digest = SHA-256( header('S', flags) || phase1 || phase2 )
hash_schema   = version || schema_digest
```

Within one `flags` value, `include_metadata = true` on a schema without metadata gives the same
hash as `false`.

---

## 3. Nodes and entries

### 3.1 Nodes and node paths

The schema is a tree of **nodes**: each column, each struct child and each list element is a
node. Node paths: a column is its `name`; a struct child is `p + "/" + name` (always, also when
`p` or `name` is empty); a list element is `p + "/"`. For `hash_array` the array is the node `""`,
so a child of a root struct is `"/" + name`. Example: column `items: List<Struct<id, label>>` has
nodes `items`, `items/`, `items//id`, `items//label`; a struct column `""` with child `x` has
nodes `""` and `/x` (distinct from a column `x`). A dictionary node is a node of its value type
(children of a dictionary of structs are below it as for a struct). Dictionaries resolve
**recursively**: a dictionary whose values are themselves dictionary-encoded is its innermost
value type, and its data is the fully decoded array (a dictionary of lists is the decoded list).

### 3.2 The one rule

Every node has one **entry**. An entry is the concatenation, in this order and only where
applicable, of:

1. **validity** — if the node's field is nullable:
   `bit_count (u64) || validity words (u8 each, bits Lsb0, last word zero-padded)`;
2. **lengths** — if the node is a list: `SHA-256( length (u64) of every valid element, in order )`;
3. **values** — if the node is a leaf (neither struct nor list): `SHA-256( value bytes of every effectively valid element, in order )`.

A struct has no lengths or values: a non-nullable struct's entry is empty; a nullable struct's
entry is its validity.

**Elements of a node.** The elements of a column are the rows. A struct child has one element per
element of its struct (also where the struct value is null; the child's validity bit is then 0).
A list element node's elements are the items of all **effectively valid** values of its list
(below), in order; items of null list values (and child slots outside every list value's offset
range) are not elements and are never visited, whatever they hold. The `hash_array` root's
elements are the array's elements.

**Validity bits.** Every node has an *effective validity* per element: `1` iff the element is
valid **and** every enclosing struct value at the same list level is valid (struct nulls are
ANDed downwards; list boundaries reset this). It is computed for **every** node and **written**
(item 1) only for nullable nodes. Elements whose effective validity is `0` contribute no lengths
and no values — also for non-nullable nodes, whose nulls can only come from an enclosing struct.
"Valid" in items 2 and 3 always means effectively valid.

Example: `s: Struct<a: Int32 non-null>` nullable, rows `[{a:1}, null, {a:3}]` (child array
`[1, 0, 3]`): entry `s` = validity `03 00 00 00 00 00 00 00  05`; entry `s/a` = `SHA-256(01 00 00 00
03 00 00 00)` (no validity, values 1 and 3 only).

**Leaf value bytes** (valid elements only):

| Type | Bytes per valid value |
|---|---|
| Int8/UInt8, Int16/UInt16/Float16, Int32/UInt32/Float32/Date32/Time32/Decimal32, Int64/UInt64/Float64/Date64/Time64/Timestamp/Duration/Decimal64, Decimal128, Decimal256 | the value's 1, 2, 4, 8, 16, 32 little-endian bytes (IEEE 754 for floats; NaN payloads and -0.0 are hashed as is) |
| FixedSizeBinary(n) | the n bytes (none for `n = 0`; validity and the row count still count) |
| Boolean | one byte: `01` for true, `00` for false |
| Utf8/LargeUtf8, Binary/LargeBinary | `length (u64) || bytes` (UTF-8 for strings) |
| Dictionary | the bytes of the resolved value (resolved recursively) |

### 3.3 Final digest of a record batch or stream

```
final = SHA-256( header('R', flags) || schema_digest || row_count (u64) || entry(n_1) || entry(n_2) || ... )
hash  = version || final
```

`row_count` is the total number of rows of all batches. Entries are in **byte-wise order of their
node paths** (Python: `sorted()` of the path strings). A stream is `new(schema)`, `update(batch)`
for each batch, `finalize()`; the result does not depend on how rows are split into batches. A
Python `hash_table(table)` is a stream over `table.to_batches()`.

---

## 4. `hash_array`

```
final = SHA-256( header('A', flags) || type_json || length (u64) || entries )
hash  = version || final
```

`type_json` is the canonical type JSON of the array's type (dictionary arrays: the value type,
recursively) under the `ordered_struct_fields` rule; `include_metadata` has no effect. The array
itself is the node at path `""` and is **always nullable**: its entry always starts with its
validity (`length` bits; all 1 when the array has no nulls or no validity buffer). An array
therefore hashes the same with or without an all-valid validity buffer. For a dictionary array
the validity is that of the decoded array (a valid index pointing at a null dictionary value is
a null). Its children follow Section 3 (a struct child of the root is at `/name`, the element of
a root list at `/`).

---

## 5. Streaming: per-batch schema check

Every batch passed to `update` must have the digester's schema: its Phase 1 JSON under the
digester's flags must be equal, and with `include_metadata` its Phase 2 document must be equal
(or both absent); it must also pass Section 7. Otherwise it is rejected (Rust panics with
"Record batch schema does not match ArrowDigester schema"; Python raises `ValueError`). Hence a
batch with reordered columns is accepted without `ordered_columns` and rejected with it, and
the same for struct fields and `ordered_struct_fields`. Physically different but canonically
equal types (`Utf8` vs `LargeUtf8`, dictionary vs plain, `List` vs `LargeList`) are accepted.

---

## 6. Options

| Option (Rust `HasherConfig` field = Python keyword) | Effect |
|---|---|
| `include_metadata` | Phase 2 (Section 2.3) |
| `ordered_columns` | flags bit `0x01`; Phase 1 is the ordered array form (Section 2.2) |
| `ordered_struct_fields` | flags bit `0x02`; struct children in declaration order (Section 2.1) |

Options are per call / per digester, never global; all default to false. Every entry point takes
all of them (`hash_array` ignores `include_metadata`; `ordered_columns` changes only its header).
The order options change only the headers and Phase 1; entries are identical. Because the flags
byte is part of the header, hashes taken with different order options never coincide (modulo
SHA-256 collisions), even when order cannot matter.

---

## 7. Rejected inputs (every mode)

Rust panics (documented under `# Panics`); Python raises `ValueError`:

1. two nodes with the same node path (Section 3.1) — duplicate column names, duplicate field
   names within a struct, or a name containing `/` that aliases another node (a column `"a/b"`
   next to a struct `a` with child `b`; a struct column `""` with child `a` next to a column
   `"/a"`); a `/` that aliases nothing (`"km/h"`) and empty names that alias nothing are fine.
   Map nodes count as `m`, `m/`, `m//key`, `m//value`; the `hash_array` root is the node `""`;
2. a `ListView`, `LargeListView`, `Union` or `RunEndEncoded` type anywhere in the schema;
3. data hashing (record batch, stream, `hash_array`) of a schema containing a type whose data is
   not supported (Section 8) — **whatever the number of rows, also zero rows or no batches**;
4. a non-nullable field whose array has a null at an element (Section 3.2) that no enclosing
   struct null masks (arrow-rs already refuses to build such arrays; pyarrow can build them).
   A null in a child slot that is not an element — under a null list value, or outside every
   list value's offset range — is ignored, for `List` and `LargeList` alike;
5. a streamed batch whose schema does not match (Section 5).

**Duplicate metadata keys are not portable.** Metadata is a map (Section 2.3); stored Arrow
metadata (IPC, C data interface) can nevertheless repeat a key, and readers disagree on which
value survives (arrow-rs keeps the last, pyarrow's `to_dict()` the first). arrow-rs collapses
such metadata into a map when it reads IPC or imports over the C data interface, before
starfix sees the schema, so the Rust crate cannot observe the duplicate and hashes the surviving
value. Implementations that can observe duplicates reject them with `include_metadata`:
starfix-python raises `ValueError` for any repeated key (even with equal values) in the metadata
of a field at any depth or of the schema, and for a field of a registered extension type whose
stored `ARROW:extension:*` keys conflict with the type's own (equal ones are accepted). Hashes of
inputs with duplicate metadata keys are therefore defined only where they are rejected.

**When the checks run.** 1–3 run before any data is hashed: in `hash_schema` (1, 2), when a
digester is created (1–3), in `hash_array` (1–3), and on each streamed batch before it changes
the digester (1, 2, 5). Check 4 runs while a batch is traversed; after any panic a digester must
be discarded (a `Result`-returning API is a proposed follow-up).

## 8. Type support

Data hashing supports Boolean, all integer and float types, Date32/64, Time32/64, Timestamp,
Duration, Decimal32/64/128/256, FixedSizeBinary, Binary/LargeBinary, Utf8/LargeUtf8,
List/LargeList, Struct, Dictionary of any of these (also of structs and lists; a dictionary of
dictionaries resolves recursively), and extension types over any of these. Schemas of `Null`, `FixedSizeList`, `Map`, `Interval`, `Utf8View` and
`BinaryView` hash, but data hashing of a schema that contains them is rejected (Section 7, item 3;
Rust panics with `not yet implemented`; Python raises).

---

## 9. Worked examples

Every hash printed in this document is pinned by `tests/format2_bytes.rs`
(`printed_spec_hashes` and the byte-level tests).

**Example 1 — record batch.** `a: Int32 non-null [1, 2]`, `b: LargeUtf8 nullable ["x", null]`.

```
schema_digest = SHA-256( header('S', 0x00) || {"a":{"data_type":"Int32","nullable":false},"b":{"data_type":"LargeUtf8","nullable":true}} )
final = SHA-256(
    header('R', 0x00)
    schema_digest
    02 00 00 00 00 00 00 00                       // row count
    SHA-256(01 00 00 00  02 00 00 00)             // entry "a": values
    02 00 00 00 00 00 00 00  01                   // entry "b": validity [1,0]
    SHA-256(01 00 00 00 00 00 00 00  78)          //           values: len 1, "x"
)
hash = 00 00 02 || final = 0000028934d60de8eb8c21ca04b859c02d21015032cb53ee029a4702281f2b0d5822bd
```

**Example 2 — list.** `c: List<Int32 nullable>` nullable, rows `[[1, null], null, []]`.

```
final = SHA-256(
    header('R', 0x00)
    schema_digest(c)
    03 00 00 00 00 00 00 00                       // row count
    03 00 00 00 00 00 00 00  05                   // entry "c": validity [1,0,1]
    SHA-256(02 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00)   //  lengths [2, 0]
    02 00 00 00 00 00 00 00  01                   // entry "c/": validity [1,0]
    SHA-256(01 00 00 00)                          //   values [1]
)
```

**Example 3 — nullable struct.** `s: Struct<a: Int32 nullable>` nullable, rows
`[null, {a: null}, {a: 1}]` (child array `[5, null, 1]`).

```
final = SHA-256(
    header('R', 0x00)
    schema_digest(s)
    03 00 00 00 00 00 00 00                       // row count
    03 00 00 00 00 00 00 00  06                   // entry "s": validity [0,1,1]
    03 00 00 00 00 00 00 00  04                   // entry "s/a": [1,0,1] AND [0,1,1] = [0,0,1]
    SHA-256(01 00 00 00)                          //   values [1]
)
```

**Example 4 — list of nullable struct.** `l: List<Struct<a: Int32 non-null> nullable>` non-null, one row
`[null, {a: 4}]`: entries `"l"` = `SHA-256(02 00 00 00 00 00 00 00)` (lengths), `"l/"` =
`02 00 00 00 00 00 00 00  02` (validity [0,1]), `"l//a"` = `SHA-256(04 00 00 00)` (no validity
because `a` is non-nullable; the value under the null element is skipped).

**Example 5 — `hash_array`.** `Int32Array [1, 2, 3]`:

```
SHA-256( header('A', 0x00) || "Int32" (with quotes) || 03 00 00 00 00 00 00 00
         || 03 00 00 00 00 00 00 00 07            // root validity [1,1,1]: the root is always nullable
         || SHA-256(01 00 00 00 02 00 00 00 03 00 00 00) )
hash = 0000028e4c60ec932ba53b9f514d4f6953091bed36d7295ad1d4cdde10907b83bf550f
```

`BooleanArray [true, null, false]`: `SHA-256( header('A', 0x00) || "Boolean" || 03 00.. (length) || 03 00.. 05 (validity) || SHA-256(01 00) )`.

**Example 6 — options.** `hash_schema` of `[b: Int32, a: Utf8 nullable]` with `ordered_columns`:
`SHA-256( header('S', 0x01) || [{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}] )`.

**Example 7 — zero columns.** A batch with no columns and 3 rows: `SHA-256( header('R', 0x00) || SHA-256(header('S', 0x00) || {}) || 03 00 00 00 00 00 00 00 )`.

---

## 10. Format history

**Format 1** (`00 00 01`, starfix ≤ 0.4). No header; the row count was not hashed; struct
validity was not recorded; list lengths were stored in the element's entry, which a nullable
nested list *replaced*; Booleans were bit-packed per call; map JSON included the
entries/key/value names; metadata paths differed from node paths below a column named `""`;
ambiguous paths were accepted (they hashed ill-defined). Known information losses (types
affected):

| Defect | Affected types | Example |
|---|---|---|
| Booleans packed without count, per batch | any type containing a Boolean | one non-nullable column `[true]` == `[true, false, false]`; batch splits changed the hash |
| no struct validity | any type containing a nullable struct | `[null]` == `[{a: null}]`; null rows vanished under non-nullable fields |
| outer lengths replaced | a list whose element is a nullable list | rows `[[1]],[[2],[3]]` == `[[1],[2]],[[3]]` |
| element validity not recorded | a list whose element is a nullable struct | `[null]` == `[{a: null}]` as elements |
| row count not hashed | zero-column tables; non-nullable empty structs | 0 rows == 3 rows |

A search over 420 nested column types found 268 lossy in format 1; a 320-case leaf-type sweep
found 68 failures. Both find none in format 2 (`tests/nested_family.rs`,
`tests/lossless_properties.rs`). Format-1 hashes are computed by the frozen legacy module `starfix::legacy::v0_4` (Rust 0.4.0
byte for byte) or by starfix 0.4.

**Format 2** (`00 00 02`): this document.

`tests/golden/include_metadata_v0.3.json` is kept unchanged as the historical format-1 record;
`tests/golden/format1_to_format2.json` maps every format-1 golden and pinned test value to its
format-2 value.

---

## 11. Invariants

| # | Invariant |
|---|---|
| I1 | Order: without `ordered_columns`, permuting columns does not change the hash; with it, it does. The same for struct fields at every depth and `ordered_struct_fields`. |
| I2 | Hashes taken with different order options never coincide. |
| I2a | Outputs of different entry points (`hash_schema`, record batch / stream, `hash_array`) never coincide (each starts its SHA-256 input with its own kind byte, Section 1). |
| I3 | `Utf8`≡`LargeUtf8`, `Binary`≡`LargeBinary`, `List`≡`LargeList`, dictionary≡plain, in every mode. |
| I4 | For every supported type: splitting rows into batches, slicing (non-zero offsets) and empty batches do not change the hash; adding or removing rows does; logically different data never collides (tested exhaustively on small inputs). |
| I5 | Within one `flags` value, `include_metadata = true` without metadata equals `false`. |

---

## 12. Golden vectors

`tests/golden/format2_v0.5.json` is generated by Rust (`cargo run --bin emit_golden_format2`;
append-only: the generator aborts if any committed vector changes in any field, or any committed
relation disappears; every vector must hash the same after its IPC round trip) and must be
matched exactly by every implementation. Before the 0.5.0 release it was regenerated in full
once, when the entry-point kind byte was added (every value changed; ids, inputs and relations
were kept). From the release on it is append-only:

```json
{
  "version": "0.5",
  "format": 2,
  "generated_by": "cargo run --bin emit_golden_format2",
  "vectors": [
    { "id": "...", "description": "...",
      "method": "hash_schema | hash_record_batch | hash_stream | hash_array",
      "include_metadata": false, "ordered_columns": true, "ordered_struct_fields": false,
      "ipc_b64": "<base64 Arrow IPC stream>", "expected_hash": "000002..." }
  ],
  "relations": [ { "kind": "equal | different", "ids": ["...", "..."], "why": "..." } ]
}
```

| method | what to hash (IPC decoded with an Arrow stream reader) |
|---|---|
| `hash_schema` | the stream's schema |
| `hash_record_batch` | the single batch |
| `hash_stream` | `new(schema)`, `update` with each batch in order (possibly none), `finalize` |
| `hash_array` | column 0 of the single batch, as an array |

The vectors `hidden_null_list_rb_f0` and `hidden_null_large_list_rb_f0` hold nulls under null list
values of a non-nullable element (Section 7, item 4): decode them without full validation
(arrow-rs's validating IPC reader rejects them; pyarrow reads them).

A relation `equal` requires all listed vectors to have the same `expected_hash`; `different`
requires them to be pairwise different.

## 13. Platform considerations

All lengths and counts are `u64` little-endian; validity words are `u8` (Lsb0); values are
little-endian. Hashes are platform-independent.
