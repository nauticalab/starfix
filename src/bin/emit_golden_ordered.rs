//! Golden fixture generator for the ordered / nested-validity hashing options (ITL-898).
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin emit_golden_ordered > tests/golden/ordered_v0.5.json.new \
//!   && mv tests/golden/ordered_v0.5.json.new tests/golden/ordered_v0.5.json
//! ```
//!
//! Then copy the file byte-for-byte to `starfix-python/tests/golden/ordered_v0.5.json`.
//!
//! # Append-only guard
//!
//! Hash definitions are permanent. When `tests/golden/ordered_v0.5.json` exists, this generator
//! loads it and **aborts** if any vector id already present would get a different
//! `expected_hash`. New vectors may be added; existing values never move. `cargo test`
//! (`tests/golden_vectors_ordered.rs`) separately re-hashes every committed vector.
//!
//! Byte layout: `docs/byte-layout-spec.md`, Section 9.

#![expect(clippy::unwrap_used, reason = "CLI tool — panics are acceptable")]
#![expect(
    clippy::too_many_lines,
    reason = "each golden vector group is a self-contained fixture block"
)]
#![expect(
    clippy::semicolon_outside_block,
    clippy::literal_string_with_formatting_args,
    clippy::shadow_unrelated,
    reason = "fixture blocks and descriptions with metadata/struct literal syntax"
)]
#![expect(
    clippy::struct_excessive_bools,
    reason = "fixture rows mirror the HasherConfig options one-to-one"
)]

use std::collections::{HashMap, HashSet};
use std::process::Command;
use std::sync::Arc;
use std::{fs, slice};

use arrow::array::{
    Array as _, ArrayRef, BooleanArray, DictionaryArray, Int32Array, Int32Builder, Int8Array,
    LargeListArray, LargeStringArray, ListArray, ListBuilder, RecordBatch, RecordBatchOptions,
    StringArray, StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::Int32Type;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use starfix::{ArrowDigester, HasherConfig};

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/golden/ordered_v0.5.json"
);

#[derive(Serialize)]
struct GoldenFixture {
    version: String,
    generated_by: String,
    rust_commit: String,
    vectors: Vec<GoldenVector>,
    relations: Vec<Relation>,
}

#[derive(Serialize)]
struct GoldenVector {
    id: String,
    description: String,
    method: String,
    include_metadata: bool,
    ordered_columns: bool,
    ordered_struct_fields: bool,
    nested_validity: bool,
    ipc_b64: String,
    expected_hash: String,
}

#[derive(Serialize)]
struct Relation {
    kind: String,
    ids: Vec<String>,
    why: String,
}

#[derive(Deserialize)]
struct ExistingFixture {
    vectors: Vec<ExistingVector>,
}

#[derive(Deserialize)]
struct ExistingVector {
    id: String,
    expected_hash: String,
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_or_else(
            |_| "unknown".to_owned(),
            |o| {
                if o.status.success() {
                    String::from_utf8_lossy(&o.stdout).trim().to_owned()
                } else {
                    "unknown".to_owned()
                }
            },
        )
}

/// Config from the flags byte of the spec (bit0 columns, bit1 struct fields, bit2 nested
/// validity).
const fn cfg(flags: u8, include_metadata: bool) -> HasherConfig {
    HasherConfig {
        include_metadata,
        ordered_columns: flags & 0x01 != 0,
        ordered_struct_fields: flags & 0x02 != 0,
        nested_validity: flags & 0x04 != 0,
    }
}

fn ipc_b64(schema: &SchemaRef, batches: &[RecordBatch]) -> String {
    let mut writer = StreamWriter::try_new(Vec::new(), schema).unwrap();
    for b in batches {
        writer.write(b).unwrap();
    }
    writer.finish().unwrap();
    BASE64_STANDARD.encode(writer.into_inner().unwrap())
}

fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// What to hash for one vector.
enum Input {
    Schema(SchemaRef),
    Batch(RecordBatch),
    Stream(SchemaRef, Vec<RecordBatch>),
    Array(ArrayRef),
}

#[derive(Default)]
struct Builder {
    vectors: Vec<GoldenVector>,
    relations: Vec<Relation>,
}

impl Builder {
    fn add(
        &mut self,
        id: &str,
        description: &str,
        flags: u8,
        include_metadata: bool,
        input: &Input,
    ) {
        let config = cfg(flags, include_metadata);
        let (method, ipc, hash) = match input {
            Input::Schema(schema) => (
                "hash_schema",
                ipc_b64(schema, &[]),
                ArrowDigester::hash_schema(schema, config),
            ),
            Input::Batch(b) => (
                "hash_record_batch",
                ipc_b64(&b.schema(), slice::from_ref(b)),
                ArrowDigester::hash_record_batch(b, config),
            ),
            Input::Stream(schema, batches) => {
                let mut digester = ArrowDigester::new(schema, config);
                for b in batches {
                    digester.update(b);
                }
                ("hash_stream", ipc_b64(schema, batches), digester.finalize())
            }
            Input::Array(array) => {
                let b = batch(
                    vec![Field::new(
                        "array",
                        array.data_type().clone(),
                        array.null_count() > 0,
                    )],
                    vec![Arc::clone(array)],
                );
                (
                    "hash_array",
                    ipc_b64(&b.schema(), &[b]),
                    ArrowDigester::hash_array_with_config(array.as_ref(), config),
                )
            }
        };
        assert!(
            self.vectors.iter().all(|v| v.id != id),
            "duplicate vector id {id}"
        );
        self.vectors.push(GoldenVector {
            id: id.to_owned(),
            description: description.to_owned(),
            method: method.to_owned(),
            include_metadata,
            ordered_columns: config.ordered_columns,
            ordered_struct_fields: config.ordered_struct_fields,
            nested_validity: config.nested_validity,
            ipc_b64: ipc,
            expected_hash: hex::encode(hash),
        });
    }

    fn relate(&mut self, kind: &str, ids: &[&str], why: &str) {
        let hashes: Vec<&str> = ids
            .iter()
            .map(|id| {
                self.vectors
                    .iter()
                    .find(|v| v.id == *id)
                    .unwrap_or_else(|| unreachable_id(id))
                    .expected_hash
                    .as_str()
            })
            .collect();
        let distinct: HashSet<&str> = hashes.iter().copied().collect();
        match kind {
            "equal" => assert!(distinct.len() == 1, "relation equal violated: {ids:?}"),
            "different" => assert!(
                distinct.len() == hashes.len(),
                "relation different violated: {ids:?}"
            ),
            other => unreachable_kind(other),
        }
        self.relations.push(Relation {
            kind: kind.to_owned(),
            ids: ids.iter().map(|&s| s.to_owned()).collect(),
            why: why.to_owned(),
        });
    }
}

#[expect(clippy::panic, reason = "CLI tool — invalid fixture definitions abort")]
fn unreachable_id(id: &str) -> ! {
    panic!("relation refers to unknown vector id {id}")
}

#[expect(clippy::panic, reason = "CLI tool — invalid fixture definitions abort")]
fn unreachable_kind(kind: &str) -> ! {
    panic!("unknown relation kind {kind}")
}

// ── data builders ───────────────────────────────────────────────────────────

/// `[a: Int32 non-null, b: LargeUtf8 nullable]` or reversed.
fn two_cols(a_first: bool, a: Vec<i32>, b: Vec<Option<&str>>) -> RecordBatch {
    let a_field = Field::new("a", DataType::Int32, false);
    let b_field = Field::new("b", DataType::LargeUtf8, true);
    let a_col: ArrayRef = Arc::new(Int32Array::from(a));
    let b_col: ArrayRef = Arc::new(LargeStringArray::from(b));
    if a_first {
        batch(vec![a_field, b_field], vec![a_col, b_col])
    } else {
        batch(vec![b_field, a_field], vec![b_col, a_col])
    }
}

fn xy_struct(x_first: bool, xs: Vec<i32>, ys: Vec<bool>) -> StructArray {
    let x = (
        Arc::new(Field::new("x", DataType::Int32, false)),
        Arc::new(Int32Array::from(xs)) as ArrayRef,
    );
    let y = (
        Arc::new(Field::new("y", DataType::Boolean, false)),
        Arc::new(BooleanArray::from(ys)) as ArrayRef,
    );
    if x_first {
        StructArray::from(vec![x, y])
    } else {
        StructArray::from(vec![y, x])
    }
}

/// Columns `n: Int32` and `s: Struct<x, y>`, with column order and struct order chosen.
fn cols_and_struct(
    cols_ns: bool,
    x_first: bool,
    ns: Vec<i32>,
    xs: Vec<i32>,
    ys: Vec<bool>,
) -> RecordBatch {
    let s = xy_struct(x_first, xs, ys);
    let s_field = Field::new("s", s.data_type().clone(), false);
    let n_field = Field::new("n", DataType::Int32, false);
    let s_col: ArrayRef = Arc::new(s);
    let n_col: ArrayRef = Arc::new(Int32Array::from(ns));
    if cols_ns {
        batch(vec![n_field, s_field], vec![n_col, s_col])
    } else {
        batch(vec![s_field, n_field], vec![s_col, n_col])
    }
}

fn ab_struct_type(ab: bool) -> DataType {
    let a = Field::new("a", DataType::Int32, false);
    let b = Field::new("b", DataType::Utf8, true);
    if ab {
        DataType::Struct(Fields::from(vec![a, b]))
    } else {
        DataType::Struct(Fields::from(vec![b, a]))
    }
}

/// `l: List<Struct<a, b>?>?` rows `[[{a:1,b:"x"},{a:2,b:null}], null, []]`.
fn list_of_struct(ab: bool, large: bool) -> RecordBatch {
    let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
    let b: ArrayRef = Arc::new(StringArray::from(vec![Some("x"), None]));
    let af = Arc::new(Field::new("a", DataType::Int32, false));
    let bf = Arc::new(Field::new("b", DataType::Utf8, true));
    let st = if ab {
        StructArray::from(vec![(af, a), (bf, b)])
    } else {
        StructArray::from(vec![(bf, b), (af, a)])
    };
    let item = Arc::new(Field::new("item", st.data_type().clone(), true));
    let lengths = [2_usize, 0, 0];
    let nulls = Some(NullBuffer::from(vec![true, false, true]));
    let col: ArrayRef = if large {
        Arc::new(LargeListArray::new(
            item,
            OffsetBuffer::<i64>::from_lengths(lengths),
            Arc::new(st),
            nulls,
        ))
    } else {
        Arc::new(ListArray::new(
            item,
            OffsetBuffer::<i32>::from_lengths(lengths),
            Arc::new(st),
            nulls,
        ))
    };
    batch(
        vec![Field::new("l", col.data_type().clone(), true)],
        vec![col],
    )
}

fn nested_struct_schema(inner_ab: bool) -> SchemaRef {
    let a = Field::new("a", DataType::Int32, false);
    let b = Field::new("b", DataType::Int32, false);
    let inner = if inner_ab {
        Fields::from(vec![a, b])
    } else {
        Fields::from(vec![b, a])
    };
    let outer = DataType::Struct(Fields::from(vec![
        Field::new("q", DataType::Int32, false),
        Field::new("p", DataType::Struct(inner), false),
    ]));
    Arc::new(Schema::new(vec![Field::new("s", outer, false)]))
}

fn single_struct_col(fields: Fields, child: ArrayRef, nulls: Option<NullBuffer>) -> RecordBatch {
    let s = StructArray::new(fields, vec![child], nulls);
    batch(
        vec![Field::new("s", s.data_type().clone(), true)],
        vec![Arc::new(s)],
    )
}

fn list_of_list(rows: &[Vec<Vec<i32>>]) -> RecordBatch {
    let mut b = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
    for row in rows {
        for inner in row {
            for v in inner {
                b.values().values().append_value(*v);
            }
            b.values().append(true);
        }
        b.append(true);
    }
    let arr = b.finish();
    batch(
        vec![Field::new("c", arr.data_type().clone(), false)],
        vec![Arc::new(arr)],
    )
}

fn list_of_nullable_struct(element_null: bool) -> RecordBatch {
    let f = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
    let st = StructArray::new(
        f,
        vec![Arc::new(Int32Array::from(vec![None::<i32>, Some(4)]))],
        Some(NullBuffer::from(vec![!element_null, true])),
    );
    let item = Arc::new(Field::new("item", st.data_type().clone(), true));
    let col: ArrayRef = Arc::new(ListArray::new(
        item,
        OffsetBuffer::<i32>::from_lengths([2]),
        Arc::new(st),
        None,
    ));
    batch(
        vec![Field::new("l", col.data_type().clone(), false)],
        vec![col],
    )
}

fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|&(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

// ── vectors ─────────────────────────────────────────────────────────────────

fn build() -> Builder {
    let mut gen = Builder::default();

    // 1. Column order: record batch, schema, stream.
    let ab = two_cols(true, vec![1, 2], vec![Some("x"), None]);
    let ba = two_cols(false, vec![1, 2], vec![Some("x"), None]);
    for flags in [0x00_u8, 0x01] {
        gen.add(
            &format!("cols_ab_rb_f{flags}"),
            "Record batch [a: Int32, b: LargeUtf8?], columns a,b",
            flags,
            false,
            &Input::Batch(ab.clone()),
        );
        gen.add(
            &format!("cols_ba_rb_f{flags}"),
            "Same data, columns b,a",
            flags,
            false,
            &Input::Batch(ba.clone()),
        );
        gen.add(
            &format!("cols_ab_schema_f{flags}"),
            "Schema [a, b]",
            flags,
            false,
            &Input::Schema(ab.schema()),
        );
        gen.add(
            &format!("cols_ba_schema_f{flags}"),
            "Schema [b, a]",
            flags,
            false,
            &Input::Schema(ba.schema()),
        );
    }
    gen.relate(
        "equal",
        &["cols_ab_rb_f0", "cols_ba_rb_f0"],
        "column order ignored by default",
    );
    gen.relate(
        "different",
        &["cols_ab_rb_f1", "cols_ba_rb_f1"],
        "ordered_columns: column order is part of the hash",
    );
    gen.relate(
        "equal",
        &["cols_ab_schema_f0", "cols_ba_schema_f0"],
        "schema: column order ignored by default",
    );
    gen.relate(
        "different",
        &["cols_ab_schema_f1", "cols_ba_schema_f1"],
        "schema: ordered_columns",
    );
    {
        let p1 = two_cols(false, vec![1], vec![Some("x")]);
        let p2 = two_cols(false, vec![2], vec![None]);
        gen.add(
            "cols_ba_stream_f1",
            "Stream of the cols_ba data split into two batches, ordered_columns",
            0x01,
            false,
            &Input::Stream(ba.schema(), vec![p1, p2]),
        );
        gen.relate(
            "equal",
            &["cols_ba_stream_f1", "cols_ba_rb_f1"],
            "batch-split independence in ordered mode",
        );
    }

    // 2. All four order combinations, on an input with a column AND a struct permutation.
    for flags in [0x00_u8, 0x01, 0x02, 0x03] {
        for (cols_ns, x_first, tag) in [
            (true, true, "ns_xy"),
            (false, true, "sn_xy"),
            (true, false, "ns_yx"),
            (false, false, "sn_yx"),
        ] {
            gen.add(
                &format!("combo_{tag}_f{flags}"),
                "Columns n: Int32 and s: Struct<x: Int32, y: Boolean>; ns/sn = column order, xy/yx = struct field order",
                flags,
                false,
                &Input::Batch(cols_and_struct(cols_ns, x_first, vec![7, 8], vec![1, 2], vec![true, false])),
            );
        }
    }
    gen.relate(
        "equal",
        &[
            "combo_ns_xy_f0",
            "combo_sn_xy_f0",
            "combo_ns_yx_f0",
            "combo_sn_yx_f0",
        ],
        "flags 0: no order matters",
    );
    gen.relate(
        "equal",
        &["combo_ns_xy_f1", "combo_ns_yx_f1"],
        "flags 1: struct order ignored",
    );
    gen.relate(
        "equal",
        &["combo_sn_xy_f1", "combo_sn_yx_f1"],
        "flags 1: struct order ignored",
    );
    gen.relate(
        "different",
        &["combo_ns_xy_f1", "combo_sn_xy_f1"],
        "flags 1: column order matters",
    );
    gen.relate(
        "equal",
        &["combo_ns_xy_f2", "combo_sn_xy_f2"],
        "flags 2: column order ignored",
    );
    gen.relate(
        "equal",
        &["combo_ns_yx_f2", "combo_sn_yx_f2"],
        "flags 2: column order ignored",
    );
    gen.relate(
        "different",
        &["combo_ns_xy_f2", "combo_ns_yx_f2"],
        "flags 2: struct order matters",
    );
    gen.relate(
        "different",
        &[
            "combo_ns_xy_f3",
            "combo_sn_xy_f3",
            "combo_ns_yx_f3",
            "combo_sn_yx_f3",
        ],
        "flags 3: both orders matter",
    );
    gen.relate(
        "different",
        &[
            "combo_ns_xy_f0",
            "combo_ns_xy_f1",
            "combo_ns_xy_f2",
            "combo_ns_xy_f3",
        ],
        "the four combinations are domain separated",
    );

    // 3. Struct order at depth: struct in struct, in list / large list, fixed-size list, map.
    for flags in [0x00_u8, 0x02] {
        gen.add(
            &format!("nested_struct_ab_schema_f{flags}"),
            "s: Struct<q, p: Struct<a, b>>",
            flags,
            false,
            &Input::Schema(nested_struct_schema(true)),
        );
        gen.add(
            &format!("nested_struct_ba_schema_f{flags}"),
            "s: Struct<q, p: Struct<b, a>> (only the inner level permuted)",
            flags,
            false,
            &Input::Schema(nested_struct_schema(false)),
        );
        gen.add(
            &format!("list_struct_ab_rb_f{flags}"),
            "l: List<Struct<a: Int32, b: Utf8?>?>? rows [[{1,'x'},{2,null}], null, []]",
            flags,
            false,
            &Input::Batch(list_of_struct(true, false)),
        );
        gen.add(
            &format!("list_struct_ba_rb_f{flags}"),
            "Same, struct fields declared b, a",
            flags,
            false,
            &Input::Batch(list_of_struct(false, false)),
        );
        gen.add(
            &format!("large_list_struct_ab_rb_f{flags}"),
            "Same as list_struct_ab with LargeList",
            flags,
            false,
            &Input::Batch(list_of_struct(true, true)),
        );
        let fsl = |ab: bool| {
            Arc::new(Schema::new(vec![Field::new(
                "f",
                DataType::FixedSizeList(Arc::new(Field::new("item", ab_struct_type(ab), true)), 2),
                false,
            )]))
        };
        gen.add(
            &format!("fsl_struct_ab_schema_f{flags}"),
            "f: FixedSizeList<Struct<a, b>, 2> (schema only: data unsupported)",
            flags,
            false,
            &Input::Schema(fsl(true)),
        );
        gen.add(
            &format!("fsl_struct_ba_schema_f{flags}"),
            "f: FixedSizeList<Struct<b, a>, 2>",
            flags,
            false,
            &Input::Schema(fsl(false)),
        );
        let map = |ab: bool| {
            let entries = Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", ab_struct_type(ab), true),
                ])),
                false,
            );
            Arc::new(Schema::new(vec![Field::new(
                "m",
                DataType::Map(Arc::new(entries), false),
                true,
            )]))
        };
        gen.add(
            &format!("map_struct_ab_schema_f{flags}"),
            "m: Map<Utf8, Struct<a, b>> (schema only)",
            flags,
            false,
            &Input::Schema(map(true)),
        );
        gen.add(
            &format!("map_struct_ba_schema_f{flags}"),
            "m: Map<Utf8, Struct<b, a>>",
            flags,
            false,
            &Input::Schema(map(false)),
        );
    }
    for (a, b) in [
        ("nested_struct_ab_schema", "nested_struct_ba_schema"),
        ("list_struct_ab_rb", "list_struct_ba_rb"),
        ("fsl_struct_ab_schema", "fsl_struct_ba_schema"),
        ("map_struct_ab_schema", "map_struct_ba_schema"),
    ] {
        gen.relate(
            "equal",
            &[&format!("{a}_f0"), &format!("{b}_f0")],
            "struct order ignored by default",
        );
        gen.relate(
            "different",
            &[&format!("{a}_f2"), &format!("{b}_f2")],
            "ordered_struct_fields applies at this depth",
        );
    }
    gen.relate(
        "equal",
        &["list_struct_ab_rb_f2", "large_list_struct_ab_rb_f2"],
        "List and LargeList agree in ordered mode",
    );
    gen.relate(
        "equal",
        &["list_struct_ab_rb_f0", "large_list_struct_ab_rb_f0"],
        "List and LargeList agree by default",
    );

    // 4. Domain separation: one column, no struct, all eight flag values, three entry points.
    {
        let b = batch(
            vec![Field::new("v", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        );
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let mut rb_ids = Vec::new();
        let mut schema_ids = Vec::new();
        let mut array_ids = Vec::new();
        for flags in 0x00_u8..=0x07 {
            let (r, s, a) = (
                format!("one_col_rb_f{flags}"),
                format!("one_col_schema_f{flags}"),
                format!("one_col_array_f{flags}"),
            );
            gen.add(
                &r,
                "One column v: Int32 [1, 2, 3]",
                flags,
                false,
                &Input::Batch(b.clone()),
            );
            gen.add(
                &s,
                "Schema [v: Int32]",
                flags,
                false,
                &Input::Schema(b.schema()),
            );
            gen.add(
                &a,
                "hash_array of Int32 [1, 2, 3]",
                flags,
                false,
                &Input::Array(Arc::clone(&arr)),
            );
            rb_ids.push(r);
            schema_ids.push(s);
            array_ids.push(a);
        }
        for ids in [rb_ids, schema_ids, array_ids] {
            let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            gen.relate(
                "different",
                &refs,
                "every flag value is domain separated, even when order cannot matter",
            );
        }
    }

    // 5. include_metadata combined with the options.
    {
        let plain = Arc::new(Schema::new(vec![
            Field::new("b", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]));
        let with_meta = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("b", DataType::Int32, false).with_metadata(meta(&[("unit", "kg")])),
                Field::new("a", DataType::Int32, false),
            ],
            meta(&[("source", "golden")]),
        ));
        for flags in [0x01_u8, 0x07] {
            gen.add(
                &format!("plain_schema_f{flags}_im0"),
                "Schema [b, a] without metadata, include_metadata=false",
                flags,
                false,
                &Input::Schema(Arc::clone(&plain)),
            );
            gen.add(
                &format!("plain_schema_f{flags}_im1"),
                "Same, include_metadata=true",
                flags,
                true,
                &Input::Schema(Arc::clone(&plain)),
            );
            gen.add(
                &format!("meta_schema_f{flags}_im1"),
                "Schema [b{unit:kg}, a] + schema metadata {source:golden}, include_metadata=true",
                flags,
                true,
                &Input::Schema(Arc::clone(&with_meta)),
            );
            gen.relate(
                "equal",
                &[
                    &format!("plain_schema_f{flags}_im0"),
                    &format!("plain_schema_f{flags}_im1"),
                ],
                "empty-metadata invariant holds within each mode",
            );
            gen.relate(
                "different",
                &[
                    &format!("plain_schema_f{flags}_im1"),
                    &format!("meta_schema_f{flags}_im1"),
                ],
                "metadata counts within each mode",
            );
        }
        // extension type: ARROW:extension:* metadata on a struct storage column
        let ext = |x_first: bool| {
            let s = xy_struct(x_first, vec![1], vec![true]);
            let field = Field::new("e", s.data_type().clone(), false).with_metadata(meta(&[
                ("ARROW:extension:name", "example.point"),
                ("ARROW:extension:metadata", ""),
            ]));
            batch(vec![field], vec![Arc::new(s)])
        };
        for flags in [0x00_u8, 0x02] {
            gen.add(
                &format!("ext_struct_xy_rb_f{flags}_im1"),
                "Extension type (metadata) over Struct<x, y> storage, include_metadata=true",
                flags,
                true,
                &Input::Batch(ext(true)),
            );
            gen.add(
                &format!("ext_struct_yx_rb_f{flags}_im1"),
                "Extension type over Struct<y, x> storage, include_metadata=true",
                flags,
                true,
                &Input::Batch(ext(false)),
            );
        }
        gen.relate(
            "equal",
            &["ext_struct_xy_rb_f0_im1", "ext_struct_yx_rb_f0_im1"],
            "extension storage struct order ignored by default",
        );
        gen.relate(
            "different",
            &["ext_struct_xy_rb_f2_im1", "ext_struct_yx_rb_f2_im1"],
            "extension storage struct order counts with ordered_struct_fields",
        );
    }

    // 6. Edge cases.
    {
        let schema = ba.schema();
        gen.add(
            "empty_stream_f1",
            "Schema [b, a], zero batches, ordered_columns",
            0x01,
            false,
            &Input::Stream(Arc::clone(&schema), vec![]),
        );
        let zero_rows = two_cols(false, vec![], vec![]);
        gen.add(
            "zero_row_rb_f1",
            "Schema [b, a], one batch with zero rows, ordered_columns",
            0x01,
            false,
            &Input::Batch(zero_rows),
        );
        gen.relate(
            "equal",
            &["empty_stream_f1", "zero_row_rb_f1"],
            "a zero-row batch contributes nothing",
        );
        let empty_schema = Arc::new(Schema::empty());
        let zero_cols = |rows: usize| {
            RecordBatch::try_new_with_options(
                Arc::clone(&empty_schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .unwrap()
        };
        for flags in [0x00_u8, 0x01] {
            gen.add(
                &format!("zero_cols_0rows_rb_f{flags}"),
                "Zero columns, 0 rows",
                flags,
                false,
                &Input::Batch(zero_cols(0)),
            );
            gen.add(
                &format!("zero_cols_3rows_rb_f{flags}"),
                "Zero columns, 3 rows (row count of a zero-column table is not hashed)",
                flags,
                false,
                &Input::Batch(zero_cols(3)),
            );
            gen.relate(
                "equal",
                &[
                    &format!("zero_cols_0rows_rb_f{flags}"),
                    &format!("zero_cols_3rows_rb_f{flags}"),
                ],
                "documented limitation: zero-column tables do not hash their row count",
            );
        }
        gen.relate(
            "different",
            &["zero_cols_0rows_rb_f0", "zero_cols_0rows_rb_f1"],
            "zero-column table is still domain separated",
        );
        let nulls = batch(
            vec![
                Field::new("b", DataType::Int32, true),
                Field::new("a", DataType::Utf8, true),
                Field::new("z", DataType::Int32, true),
            ],
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None])),
                Arc::new(StringArray::from(vec![None, Some("x")])),
                Arc::new(Int32Array::from(vec![None::<i32>, None])),
            ],
        );
        gen.add(
            "leaf_nulls_rb_f3",
            "Nullable leaves with nulls and an all-null Int32 column, flags 3",
            0x03,
            false,
            &Input::Batch(nulls),
        );
    }

    // 7. nested_validity.
    {
        let f_nullable = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        let child: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let null_struct = single_struct_col(
            f_nullable.clone(),
            Arc::clone(&child),
            Some(NullBuffer::from(vec![true, false])),
        );
        let null_fields = single_struct_col(f_nullable, child, None);
        let f_non_null = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let rows3 = single_struct_col(
            f_non_null.clone(),
            Arc::new(Int32Array::from(vec![1, 0, 3])),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let rows2 = single_struct_col(
            f_non_null.clone(),
            Arc::new(Int32Array::from(vec![1, 3])),
            None,
        );
        let ll1 = list_of_list(&[vec![vec![1], vec![2]]]);
        let ll2 = list_of_list(&[vec![vec![1]], vec![vec![2]]]);
        let ls_null = list_of_nullable_struct(true);
        let ls_fields = list_of_nullable_struct(false);
        for flags in [0x00_u8, 0x04, 0x07] {
            gen.add(
                &format!("nv_null_struct_rb_f{flags}"),
                "s: Struct<a: Int32?>? rows [{a:1}, null]",
                flags,
                false,
                &Input::Batch(null_struct.clone()),
            );
            gen.add(
                &format!("nv_null_fields_rb_f{flags}"),
                "s: Struct<a: Int32?>? rows [{a:1}, {a:null}]",
                flags,
                false,
                &Input::Batch(null_fields.clone()),
            );
            gen.add(
                &format!("nv_rows3_rb_f{flags}"),
                "s: Struct<a: Int32>? rows [{a:1}, null, {a:3}]",
                flags,
                false,
                &Input::Batch(rows3.clone()),
            );
            gen.add(
                &format!("nv_rows2_rb_f{flags}"),
                "s: Struct<a: Int32>? rows [{a:1}, {a:3}]",
                flags,
                false,
                &Input::Batch(rows2.clone()),
            );
            gen.add(
                &format!("nv_list_list_1row_rb_f{flags}"),
                "c: List<List<Int32?>?> one row [[1], [2]]",
                flags,
                false,
                &Input::Batch(ll1.clone()),
            );
            gen.add(
                &format!("nv_list_list_2rows_rb_f{flags}"),
                "c: List<List<Int32?>?> two rows [[1]], [[2]]",
                flags,
                false,
                &Input::Batch(ll2.clone()),
            );
            gen.add(
                &format!("nv_list_null_elem_rb_f{flags}"),
                "l: List<Struct<a: Int32?>?> one row [null, {a:4}]",
                flags,
                false,
                &Input::Batch(ls_null.clone()),
            );
            gen.add(
                &format!("nv_list_null_fields_elem_rb_f{flags}"),
                "l: List<Struct<a: Int32?>?> one row [{a:null}, {a:4}]",
                flags,
                false,
                &Input::Batch(ls_fields.clone()),
            );
        }
        for (a, b) in [
            ("nv_null_struct_rb", "nv_null_fields_rb"),
            ("nv_rows3_rb", "nv_rows2_rb"),
            ("nv_list_list_1row_rb", "nv_list_list_2rows_rb"),
            ("nv_list_null_elem_rb", "nv_list_null_fields_elem_rb"),
        ] {
            gen.relate(
                "equal",
                &[&format!("{a}_f0"), &format!("{b}_f0")],
                "default layout loses this distinction (frozen)",
            );
            gen.relate(
                "different",
                &[&format!("{a}_f4"), &format!("{b}_f4")],
                "nested_validity records it",
            );
            gen.relate(
                "different",
                &[&format!("{a}_f7"), &format!("{b}_f7")],
                "nested_validity records it (all options on)",
            );
        }
        // nested nullable structs, two levels, plus a list of nullable lists of nullable structs
        let inner_t = DataType::Struct(Fields::from(vec![Field::new("v", DataType::Int32, true)]));
        let inner = StructArray::new(
            Fields::from(vec![Field::new("v", DataType::Int32, true)]),
            vec![Arc::new(Int32Array::from(vec![
                Some(1),
                None,
                Some(3),
                None,
            ]))],
            Some(NullBuffer::from(vec![true, true, false, false])),
        );
        let outer = StructArray::new(
            Fields::from(vec![
                Field::new("w", DataType::Int32, false),
                Field::new("i", inner_t, true),
            ]),
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
                Arc::new(inner),
            ],
            Some(NullBuffer::from(vec![true, false, true, true])),
        );
        let nested = batch(
            vec![Field::new("o", outer.data_type().clone(), true)],
            vec![Arc::new(outer)],
        );
        for flags in [0x04_u8, 0x07] {
            gen.add(
                &format!("nv_nested_structs_rb_f{flags}"),
                "o: Struct<w: Int32, i: Struct<v: Int32?>?>? with nulls at both levels",
                flags,
                false,
                &Input::Batch(nested.clone()),
            );
        }
        // stream split with nested_validity
        let p1 = single_struct_col(
            f_non_null.clone(),
            Arc::new(Int32Array::from(vec![1])),
            None,
        );
        let p2 = single_struct_col(
            f_non_null.clone(),
            Arc::new(Int32Array::from(vec![0, 3])),
            Some(NullBuffer::from(vec![false, true])),
        );
        gen.add(
            "nv_rows3_stream_f4",
            "nv_rows3 data split into two batches, nested_validity",
            0x04,
            false,
            &Input::Stream(rows3.schema(), vec![p1, p2]),
        );
        gen.relate(
            "equal",
            &["nv_rows3_stream_f4", "nv_rows3_rb_f4"],
            "batch-split independence with nested_validity",
        );
        // hash_array of a nullable struct
        let arr3: ArrayRef = Arc::new(StructArray::new(
            f_non_null.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 0, 3]))],
            Some(NullBuffer::from(vec![true, false, true])),
        ));
        let arr2: ArrayRef = Arc::new(StructArray::new(
            f_non_null,
            vec![Arc::new(Int32Array::from(vec![1, 3]))],
            None,
        ));
        for flags in [0x00_u8, 0x04] {
            gen.add(
                &format!("nv_array_rows3_f{flags}"),
                "hash_array of Struct<a: Int32> [{a:1}, null, {a:3}]",
                flags,
                false,
                &Input::Array(Arc::clone(&arr3)),
            );
            gen.add(
                &format!("nv_array_rows2_f{flags}"),
                "hash_array of Struct<a: Int32> [{a:1}, {a:3}]",
                flags,
                false,
                &Input::Array(Arc::clone(&arr2)),
            );
        }
        gen.relate(
            "equal",
            &["nv_array_rows3_f0", "nv_array_rows2_f0"],
            "default hash_array drops null struct rows (frozen)",
        );
        gen.relate(
            "different",
            &["nv_array_rows3_f4", "nv_array_rows2_f4"],
            "nested_validity counts them",
        );
    }

    // 8. Physical equivalences in every mode.
    {
        let dict: DictionaryArray<Int32Type> = vec!["p", "q", "p"].into_iter().collect();
        let mk = |d: ArrayRef| {
            batch(
                vec![
                    Field::new("z", DataType::Int32, false),
                    Field::new("d", d.data_type().clone(), false),
                ],
                vec![Arc::new(Int32Array::from(vec![1, 2, 3])), d],
            )
        };
        for flags in [0x00_u8, 0x03, 0x07] {
            gen.add(
                &format!("phys_dict_rb_f{flags}"),
                "[z: Int32, d: Dictionary<Int32, Utf8>]",
                flags,
                false,
                &Input::Batch(mk(Arc::new(dict.clone()))),
            );
            gen.add(
                &format!("phys_utf8_rb_f{flags}"),
                "[z: Int32, d: Utf8] same values",
                flags,
                false,
                &Input::Batch(mk(Arc::new(StringArray::from(vec!["p", "q", "p"])))),
            );
            gen.add(
                &format!("phys_large_utf8_rb_f{flags}"),
                "[z: Int32, d: LargeUtf8] same values",
                flags,
                false,
                &Input::Batch(mk(Arc::new(LargeStringArray::from(vec!["p", "q", "p"])))),
            );
            gen.relate(
                "equal",
                &[
                    &format!("phys_dict_rb_f{flags}"),
                    &format!("phys_utf8_rb_f{flags}"),
                    &format!("phys_large_utf8_rb_f{flags}"),
                ],
                "dictionary, Utf8 and LargeUtf8 agree in every mode",
            );
        }
    }

    // 9. hash_array with struct order.
    {
        for flags in [0x00_u8, 0x02, 0x06] {
            gen.add(
                &format!("array_struct_xy_f{flags}"),
                "hash_array of Struct<x: Int32, y: Boolean> [{1,true},{2,false}]",
                flags,
                false,
                &Input::Array(Arc::new(xy_struct(true, vec![1, 2], vec![true, false]))),
            );
            gen.add(
                &format!("array_struct_yx_f{flags}"),
                "Same, fields declared y, x",
                flags,
                false,
                &Input::Array(Arc::new(xy_struct(false, vec![1, 2], vec![true, false]))),
            );
        }
        gen.relate(
            "equal",
            &["array_struct_xy_f0", "array_struct_yx_f0"],
            "hash_array: struct order ignored by default",
        );
        gen.relate(
            "different",
            &["array_struct_xy_f2", "array_struct_yx_f2"],
            "hash_array: ordered_struct_fields",
        );
        gen.relate(
            "different",
            &["array_struct_xy_f6", "array_struct_yx_f6"],
            "hash_array: ordered_struct_fields with nested_validity",
        );
        let l = list_of_struct(false, false);
        for flags in [0x02_u8, 0x07] {
            gen.add(
                &format!("array_list_struct_f{flags}"),
                "hash_array of List<Struct<b, a>?> with a null row",
                flags,
                false,
                &Input::Array(Arc::clone(l.column(0))),
            );
        }
    }

    // 10. Stream of the combo data, all options on.
    {
        let whole = cols_and_struct(
            false,
            false,
            vec![7, 8, 9],
            vec![1, 2, 3],
            vec![true, false, true],
        );
        let p1 = cols_and_struct(false, false, vec![7], vec![1], vec![true]);
        let p2 = cols_and_struct(false, false, vec![8, 9], vec![2, 3], vec![false, true]);
        gen.add(
            "combo_sn_yx_3rows_rb_f7",
            "Columns s, n; struct y, x; 3 rows; all options on",
            0x07,
            false,
            &Input::Batch(whole.clone()),
        );
        gen.add(
            "combo_sn_yx_3rows_stream_f7",
            "Same data as two batches",
            0x07,
            false,
            &Input::Stream(whole.schema(), vec![p1, p2]),
        );
        gen.relate(
            "equal",
            &["combo_sn_yx_3rows_rb_f7", "combo_sn_yx_3rows_stream_f7"],
            "batch-split independence with all options on",
        );
    }

    // 11. Boolean data under the third switch (one byte per valid value).
    {
        let bools = |nullable: bool, values: Vec<Option<bool>>| {
            batch(
                vec![Field::new("b", DataType::Boolean, nullable)],
                vec![Arc::new(BooleanArray::from(values))],
            )
        };
        let one = bools(false, vec![Some(true)]);
        let three = bools(false, vec![Some(true), Some(false), Some(false)]);
        for flags in [0x00_u8, 0x04] {
            gen.add(
                &format!("bool_1row_rb_f{flags}"),
                "b: Boolean non-null [true]",
                flags,
                false,
                &Input::Batch(one.clone()),
            );
            gen.add(
                &format!("bool_3rows_rb_f{flags}"),
                "b: Boolean non-null [true, false, false]",
                flags,
                false,
                &Input::Batch(three.clone()),
            );
            gen.add(
                &format!("bool_array_1_f{flags}"),
                "hash_array of Boolean [true]",
                flags,
                false,
                &Input::Array(Arc::new(BooleanArray::from(vec![true]))),
            );
            gen.add(
                &format!("bool_array_2_f{flags}"),
                "hash_array of Boolean [true, false]",
                flags,
                false,
                &Input::Array(Arc::new(BooleanArray::from(vec![true, false]))),
            );
        }
        gen.relate(
            "equal",
            &["bool_1row_rb_f0", "bool_3rows_rb_f0"],
            "default layout: trailing false Boolean rows are lost (frozen)",
        );
        gen.relate(
            "different",
            &["bool_1row_rb_f4", "bool_3rows_rb_f4"],
            "third switch: one byte per Boolean value counts them",
        );
        gen.relate(
            "equal",
            &["bool_array_1_f0", "bool_array_2_f0"],
            "default hash_array: trailing false lost (frozen)",
        );
        gen.relate(
            "different",
            &["bool_array_1_f4", "bool_array_2_f4"],
            "third switch: counted",
        );
        let whole = bools(true, vec![Some(true), None, Some(false), Some(true)]);
        let p1 = bools(true, vec![Some(true)]);
        let p2 = bools(true, vec![None, Some(false), Some(true)]);
        for flags in [0x00_u8, 0x04] {
            gen.add(
                &format!("bool_nullable_rb_f{flags}"),
                "b: Boolean nullable [true, null, false, true]",
                flags,
                false,
                &Input::Batch(whole.clone()),
            );
            gen.add(
                &format!("bool_nullable_stream_f{flags}"),
                "Same as two batches [true] + [null, false, true]",
                flags,
                false,
                &Input::Stream(whole.schema(), vec![p1.clone(), p2.clone()]),
            );
        }
        gen.relate(
            "different",
            &["bool_nullable_rb_f0", "bool_nullable_stream_f0"],
            "default layout: Boolean data is packed per batch, so batch splits change the hash (frozen)",
        );
        gen.relate(
            "equal",
            &["bool_nullable_rb_f4", "bool_nullable_stream_f4"],
            "third switch: Boolean is batch-split independent",
        );
    }

    // 12. Names: non-ASCII, control characters, quotes, a non-colliding '/'; container schemas.
    {
        let names = Arc::new(Schema::new(vec![
            Field::new("é", DataType::Int32, false),
            Field::new("名前", DataType::Utf8, true),
            Field::new("a\nb\u{1b}\t\"q\"\\", DataType::Int32, false),
            Field::new("km/h", DataType::Float64, false),
        ]));
        for flags in [0x00_u8, 0x01, 0x07] {
            gen.add(
                &format!("names_schema_f{flags}"),
                "Column names: non-ASCII, CJK, control characters and quotes, and 'km/h' (a '/' that aliases nothing)",
                flags,
                false,
                &Input::Schema(Arc::clone(&names)),
            );
            let s = StructArray::from(vec![(
                Arc::new(Field::new("é", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            )]);
            gen.add(
                &format!("names_array_struct_f{flags}"),
                "hash_array of Struct<é: Int32> [{é: 1}] (type JSON keeps non-ASCII raw)",
                flags,
                false,
                &Input::Array(Arc::new(s)),
            );
        }
        let s8 = StructArray::from(vec![(
            Arc::new(Field::new("é", DataType::Int8, false)),
            Arc::new(Int8Array::from(vec![1_i8])) as ArrayRef,
        )]);
        gen.add(
            "names_array_struct_int8_f0",
            "hash_array of Struct<é: Int8 non-null> [{é: 1}], default options",
            0x00,
            false,
            &Input::Array(Arc::new(s8)),
        );
        let fsl_int = Arc::new(Schema::new(vec![Field::new(
            "f",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2),
            false,
        )]));
        for flags in [0x00_u8, 0x01] {
            gen.add(
                &format!("fsl_int_schema_f{flags}"),
                "f: FixedSizeList<Int32?, 2> (schema only)",
                flags,
                false,
                &Input::Schema(Arc::clone(&fsl_int)),
            );
        }
    }

    // 13. List element order is never ignored, even for list<struct<key, value>>.
    {
        let kv = |reverse: bool| {
            let mut keys = vec!["a", "b"];
            let mut values = vec![1, 2];
            if reverse {
                keys.reverse();
                values.reverse();
            }
            let st = StructArray::from(vec![
                (
                    Arc::new(Field::new("key", DataType::Utf8, false)),
                    Arc::new(StringArray::from(keys)) as ArrayRef,
                ),
                (
                    Arc::new(Field::new("value", DataType::Int32, false)),
                    Arc::new(Int32Array::from(values)) as ArrayRef,
                ),
            ]);
            let item = Arc::new(Field::new("item", st.data_type().clone(), true));
            let col: ArrayRef = Arc::new(ListArray::new(
                item,
                OffsetBuffer::<i32>::from_lengths([2]),
                Arc::new(st),
                None,
            ));
            batch(
                vec![Field::new("kv", col.data_type().clone(), false)],
                vec![col],
            )
        };
        for flags in 0x00_u8..=0x07 {
            let a = format!("list_kv_ab_rb_f{flags}");
            let b = format!("list_kv_ba_rb_f{flags}");
            gen.add(
                &a,
                "kv: List<Struct<key: Utf8, value: Int32>> one row [{a,1},{b,2}]",
                flags,
                false,
                &Input::Batch(kv(false)),
            );
            gen.add(
                &b,
                "Same elements in reverse order [{b,2},{a,1}]",
                flags,
                false,
                &Input::Batch(kv(true)),
            );
            gen.relate(
                "different",
                &[&a, &b],
                "list element order is part of the hash in every mode (starfix never sorts lists)",
            );
        }
    }

    gen
}

#[expect(clippy::panic, reason = "CLI tool — the append-only guard aborts")]
fn check_append_only(vectors: &[GoldenVector]) {
    let Ok(raw) = fs::read_to_string(FIXTURE_PATH) else {
        return; // no committed fixture yet
    };
    let existing: ExistingFixture = serde_json::from_str(&raw).unwrap();
    for old in &existing.vectors {
        match vectors.iter().find(|v| v.id == old.id) {
            None => panic!(
                "append-only guard: committed vector '{}' is no longer generated",
                old.id
            ),
            Some(new) if new.expected_hash != old.expected_hash => panic!(
                "append-only guard: vector '{}' would change from {} to {}; hash definitions are permanent",
                old.id, old.expected_hash, new.expected_hash
            ),
            Some(_) => {}
        }
    }
}

fn main() {
    let g = build();
    check_append_only(&g.vectors);
    let fixture = GoldenFixture {
        version: "0.5".to_owned(),
        generated_by: "cargo run --bin emit_golden_ordered".to_owned(),
        rust_commit: git_sha(),
        vectors: g.vectors,
        relations: g.relations,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
