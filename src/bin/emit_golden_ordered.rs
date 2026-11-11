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
use arrow_schema::{DataType, Field, Fields, IntervalUnit, Schema, SchemaRef, TimeUnit};
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

/// Config from the flags byte of the spec (bit0 columns, bit1 struct fields).
const fn cfg(flags: u8, include_metadata: bool) -> HasherConfig {
    HasherConfig {
        include_metadata,
        ordered_columns: flags & 0x01 != 0,
        ordered_struct_fields: flags & 0x02 != 0,
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
            ipc_b64: ipc,
            expected_hash: hex::encode(hash),
        });
    }

    #[expect(clippy::panic, reason = "CLI tool — invalid fixture definitions abort")]
    fn relate(&mut self, kind: &str, ids: &[&str], why: &str) {
        let hashes: Vec<&str> = ids
            .iter()
            .map(|id| {
                self.vectors
                    .iter()
                    .find(|v| v.id == *id)
                    .unwrap_or_else(|| panic!("relation refers to unknown vector id {id}"))
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
            other => panic!("unknown relation kind {other}"),
        }
        self.relations.push(Relation {
            kind: kind.to_owned(),
            ids: ids.iter().map(|&s| s.to_owned()).collect(),
            why: why.to_owned(),
        });
    }
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
        for flags in 0x00_u8..=0x03 {
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
        for flags in [0x01_u8, 0x03] {
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

    // 7. v0.5 fixes: nested validity (byte-layout spec Section 10).
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
        for flags in [0x00_u8, 0x03] {
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
                "different",
                &[&format!("{a}_f0"), &format!("{b}_f0")],
                "v0.5 fix: these collided in v0.4.0",
            );
            gen.relate(
                "different",
                &[&format!("{a}_f3"), &format!("{b}_f3")],
                "v0.5 fix, also with both order options on",
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
        for flags in [0x00_u8, 0x03] {
            gen.add(
                &format!("nv_nested_structs_rb_f{flags}"),
                "o: Struct<w: Int32, i: Struct<v: Int32?>?>? with nulls at both levels",
                flags,
                false,
                &Input::Batch(nested.clone()),
            );
        }
        // stream split of the nullable-struct rows
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
            "nv_rows3_stream_f0",
            "nv_rows3 data split into two batches",
            0x00,
            false,
            &Input::Stream(rows3.schema(), vec![p1, p2]),
        );
        gen.relate(
            "equal",
            &["nv_rows3_stream_f0", "nv_rows3_rb_f0"],
            "batch-split independence with nullable struct rows",
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
        for flags in [0x00_u8, 0x03] {
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
            "different",
            &["nv_array_rows3_f0", "nv_array_rows2_f0"],
            "v0.5 fix: hash_array counts null struct rows (v0.4.0 dropped them)",
        );
        gen.relate(
            "different",
            &["nv_array_rows3_f3", "nv_array_rows2_f3"],
            "v0.5 fix, also with both order options on",
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
        for flags in [0x00_u8, 0x03] {
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
        for flags in [0x00_u8, 0x02, 0x03] {
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
            &["array_struct_xy_f3", "array_struct_yx_f3"],
            "hash_array: ordered_struct_fields with ordered_columns",
        );
        let l = list_of_struct(false, false);
        for flags in [0x02_u8, 0x03] {
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
            "combo_sn_yx_3rows_rb_f3",
            "Columns s, n; struct y, x; 3 rows (Boolean field y); both order options on",
            0x03,
            false,
            &Input::Batch(whole.clone()),
        );
        gen.add(
            "combo_sn_yx_3rows_stream_f3",
            "Same data as two batches",
            0x03,
            false,
            &Input::Stream(whole.schema(), vec![p1, p2]),
        );
        gen.relate(
            "equal",
            &["combo_sn_yx_3rows_rb_f3", "combo_sn_yx_3rows_stream_f3"],
            "batch-split independence with both order options on (v0.5 Boolean fix)",
        );
    }

    add_boolean_names_types_and_metadata(&mut gen);

    gen
}

/// Vector groups 11–15 (split out of `build` to keep each function readable).
fn add_boolean_names_types_and_metadata(gen: &mut Builder) {
    // 11. v0.5 fix: Boolean data, one byte per valid value.
    {
        let bools = |nullable: bool, values: Vec<Option<bool>>| {
            batch(
                vec![Field::new("b", DataType::Boolean, nullable)],
                vec![Arc::new(BooleanArray::from(values))],
            )
        };
        let one = bools(false, vec![Some(true)]);
        let three = bools(false, vec![Some(true), Some(false), Some(false)]);
        for flags in [0x00_u8, 0x03] {
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
        for flags in [0x00_u8, 0x03] {
            gen.relate(
                "different",
                &[
                    &format!("bool_1row_rb_f{flags}"),
                    &format!("bool_3rows_rb_f{flags}"),
                ],
                "v0.5 fix: trailing false Boolean rows are counted (v0.4.0 lost them)",
            );
            gen.relate(
                "different",
                &[
                    &format!("bool_array_1_f{flags}"),
                    &format!("bool_array_2_f{flags}"),
                ],
                "v0.5 fix: hash_array counts trailing false values",
            );
        }
        let whole = bools(true, vec![Some(true), None, Some(false), Some(true)]);
        let p1 = bools(true, vec![Some(true)]);
        let p2 = bools(true, vec![None, Some(false), Some(true)]);
        for flags in [0x00_u8, 0x03] {
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
        for flags in [0x00_u8, 0x03] {
            gen.relate(
                "equal",
                &[
                    &format!("bool_nullable_rb_f{flags}"),
                    &format!("bool_nullable_stream_f{flags}"),
                ],
                "v0.5 fix: Boolean columns are batch-split independent (v0.4.0 was not)",
            );
        }
    }

    // 12. Names: non-ASCII, control characters, quotes, a non-colliding '/'; container schemas.
    {
        let names = Arc::new(Schema::new(vec![
            Field::new("é", DataType::Int32, false),
            Field::new("名前", DataType::Utf8, true),
            Field::new("a\nb\u{1b}\t\"q\"\\", DataType::Int32, false),
            Field::new("km/h", DataType::Float64, false),
        ]));
        for flags in [0x00_u8, 0x01, 0x03] {
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
        for flags in 0x00_u8..=0x03 {
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

    // 14. One hash_schema vector per primitive type JSON form (byte-layout spec Section 2.1).
    {
        let types: Vec<(&str, DataType)> = vec![
            ("null", DataType::Null),
            ("boolean", DataType::Boolean),
            ("int8", DataType::Int8),
            ("int16", DataType::Int16),
            ("int32", DataType::Int32),
            ("int64", DataType::Int64),
            ("uint8", DataType::UInt8),
            ("uint16", DataType::UInt16),
            ("uint32", DataType::UInt32),
            ("uint64", DataType::UInt64),
            ("float16", DataType::Float16),
            ("float32", DataType::Float32),
            ("float64", DataType::Float64),
            ("date32", DataType::Date32),
            ("date64", DataType::Date64),
            ("time32_s", DataType::Time32(TimeUnit::Second)),
            ("time32_ms", DataType::Time32(TimeUnit::Millisecond)),
            ("time64_us", DataType::Time64(TimeUnit::Microsecond)),
            ("time64_ns", DataType::Time64(TimeUnit::Nanosecond)),
            (
                "timestamp_s_naive",
                DataType::Timestamp(TimeUnit::Second, None),
            ),
            (
                "timestamp_ms_utc",
                DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            ),
            (
                "timestamp_us_offset",
                DataType::Timestamp(TimeUnit::Microsecond, Some("+05:30".into())),
            ),
            (
                "timestamp_ns_named_tz",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("America/New_York".into())),
            ),
            ("duration_s", DataType::Duration(TimeUnit::Second)),
            ("duration_ms", DataType::Duration(TimeUnit::Millisecond)),
            ("duration_us", DataType::Duration(TimeUnit::Microsecond)),
            ("duration_ns", DataType::Duration(TimeUnit::Nanosecond)),
            (
                "interval_year_month",
                DataType::Interval(IntervalUnit::YearMonth),
            ),
            (
                "interval_day_time",
                DataType::Interval(IntervalUnit::DayTime),
            ),
            (
                "interval_month_day_nano",
                DataType::Interval(IntervalUnit::MonthDayNano),
            ),
            ("decimal32", DataType::Decimal32(9, 2)),
            ("decimal64_negative_scale", DataType::Decimal64(18, -3)),
            ("decimal128", DataType::Decimal128(38, 10)),
            ("decimal256", DataType::Decimal256(76, 0)),
            ("fixed_size_binary", DataType::FixedSizeBinary(16)),
            ("utf8", DataType::Utf8),
            ("large_utf8", DataType::LargeUtf8),
            ("binary", DataType::Binary),
            ("large_binary", DataType::LargeBinary),
            ("utf8_view", DataType::Utf8View),
            ("binary_view", DataType::BinaryView),
        ];
        for (name, dt) in types {
            let schema = Arc::new(Schema::new(vec![Field::new("c", dt.clone(), true)]));
            gen.add(
                &format!("type_{name}_schema_f0"),
                &format!("Schema [c: {dt:?} nullable], default options"),
                0x00,
                false,
                &Input::Schema(schema),
            );
        }
        gen.relate(
            "equal",
            &["type_utf8_schema_f0", "type_large_utf8_schema_f0"],
            "Utf8 and LargeUtf8 share the canonical form",
        );
        gen.relate(
            "equal",
            &["type_binary_schema_f0", "type_large_binary_schema_f0"],
            "Binary and LargeBinary share the canonical form",
        );
        gen.relate(
            "different",
            &["type_utf8_schema_f0", "type_utf8_view_schema_f0"],
            "Utf8View is not canonicalized",
        );
    }

    // 15. Phase 2 metadata field paths (byte-layout spec Section 2.2.1).
    {
        let kv = |key_meta: bool, value_meta: bool, entries_meta: bool| {
            let mut key = Field::new("key", DataType::Utf8, false);
            if key_meta {
                key = key.with_metadata(meta(&[("role", "id")]));
            }
            let mut value = Field::new("value", DataType::Int32, true);
            if value_meta {
                value = value.with_metadata(meta(&[("unit", "kg")]));
            }
            let mut entries = Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![key, value])),
                false,
            );
            if entries_meta {
                entries = entries.with_metadata(meta(&[("role", "pairs")]));
            }
            Arc::new(Schema::new(vec![Field::new(
                "m",
                DataType::Map(Arc::new(entries), false),
                true,
            )]))
        };
        let struct_child = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, false).with_metadata(meta(&[("unit", "m")])),
                Field::new("b", DataType::Int32, false),
            ])),
            false,
        )]));
        let list_element = Arc::new(Schema::new(vec![Field::new(
            "l",
            DataType::List(Arc::new(
                Field::new("item", DataType::Int32, true).with_metadata(meta(&[("unit", "s")])),
            )),
            true,
        )]));
        let fsl_element = Arc::new(Schema::new(vec![Field::new(
            "f",
            DataType::FixedSizeList(
                Arc::new(
                    Field::new("item", DataType::Int32, true).with_metadata(meta(&[("unit", "s")])),
                ),
                2,
            ),
            true,
        )]));
        for flags in [0x00_u8, 0x01] {
            for (id, desc, schema) in [
                (
                    "meta_map_value",
                    "m: Map<Utf8, Int32>, metadata on the value field (path m//value)",
                    kv(false, true, false),
                ),
                (
                    "meta_map_key",
                    "m: Map<Utf8, Int32>, metadata on the key field (path m//key)",
                    kv(true, false, false),
                ),
                (
                    "meta_map_entries",
                    "m: Map<Utf8, Int32>, metadata on the entries field (path m/)",
                    kv(false, false, true),
                ),
                (
                    "meta_map_all",
                    "m: Map<Utf8, Int32>, metadata on entries, key and value",
                    kv(true, true, true),
                ),
                (
                    "meta_struct_child",
                    "s: Struct<a{unit:m}, b>, metadata on a struct child (path s/a)",
                    Arc::clone(&struct_child),
                ),
                (
                    "meta_list_element",
                    "l: List<Int32{unit:s}>, metadata on the element field (path l/)",
                    Arc::clone(&list_element),
                ),
                (
                    "meta_fsl_element",
                    "f: FixedSizeList<Int32{unit:s}, 2>, metadata on the element field (path f/)",
                    Arc::clone(&fsl_element),
                ),
            ] {
                gen.add(
                    &format!("{id}_schema_f{flags}_im1"),
                    desc,
                    flags,
                    true,
                    &Input::Schema(schema),
                );
            }
        }
        gen.relate(
            "different",
            &[
                "meta_map_value_schema_f0_im1",
                "meta_map_key_schema_f0_im1",
                "meta_map_entries_schema_f0_im1",
                "meta_map_all_schema_f0_im1",
            ],
            "metadata on the map's entries, key and value fields is recorded at distinct paths",
        );
    }
}

/// Reviewed exception list for the v0.5 fix (ITL-898, developer ruling 2026-11-11): the
/// committed fixture at commit 666821f predates the fix. Each entry is (id, the pre-fix hash);
/// the guard lets exactly that committed value change (or disappear, for ids of the removed
/// provisional third switch, flags 4–7). Once the fixed fixture is committed, no entry
/// matches any committed value, so every later change aborts again.
const V0_5_FIX_EXCEPTIONS: &[(&str, &str)] = &[
    (
        "combo_ns_xy_f0",
        "00000104a7fc5a264db0cda632bf4ac856e8c2f1285af97c65430cfb361446ec0216cd",
    ),
    (
        "combo_sn_xy_f0",
        "00000104a7fc5a264db0cda632bf4ac856e8c2f1285af97c65430cfb361446ec0216cd",
    ),
    (
        "combo_ns_yx_f0",
        "00000104a7fc5a264db0cda632bf4ac856e8c2f1285af97c65430cfb361446ec0216cd",
    ),
    (
        "combo_sn_yx_f0",
        "00000104a7fc5a264db0cda632bf4ac856e8c2f1285af97c65430cfb361446ec0216cd",
    ),
    (
        "combo_ns_xy_f1",
        "000001a12d9d439905c16338fec9ab33c757d3c9c11ed4ed14e8d618a699fbf170e40d",
    ),
    (
        "combo_sn_xy_f1",
        "0000016d9c52cf9a055df5764d2b52e57929f650d6ac8b92206eb7f67adb9b679cfd41",
    ),
    (
        "combo_ns_yx_f1",
        "000001a12d9d439905c16338fec9ab33c757d3c9c11ed4ed14e8d618a699fbf170e40d",
    ),
    (
        "combo_sn_yx_f1",
        "0000016d9c52cf9a055df5764d2b52e57929f650d6ac8b92206eb7f67adb9b679cfd41",
    ),
    (
        "combo_ns_xy_f2",
        "000001f3338aeec6740e360c7919ed1527d3a180d721611dbe2573fdfdafb01c250b93",
    ),
    (
        "combo_sn_xy_f2",
        "000001f3338aeec6740e360c7919ed1527d3a180d721611dbe2573fdfdafb01c250b93",
    ),
    (
        "combo_ns_yx_f2",
        "000001ce00ff6234af4635713dbc8ef6ed4acc6f751e2153fca3fd292e9d7bf20a6a28",
    ),
    (
        "combo_sn_yx_f2",
        "000001ce00ff6234af4635713dbc8ef6ed4acc6f751e2153fca3fd292e9d7bf20a6a28",
    ),
    (
        "combo_ns_xy_f3",
        "00000157528aa46dc70002df3ac22328918d1128fdad6b3fb0f428b944214ff283ec1f",
    ),
    (
        "combo_sn_xy_f3",
        "000001bbbb6e0038be6938efa52f5b1cd8107f54d7db033790b27d5f9068b852efec9f",
    ),
    (
        "combo_ns_yx_f3",
        "0000018ad68d9e117c5551e8bddf182148c7f4873d14ea322df73dfe52ea6ef8be26da",
    ),
    (
        "combo_sn_yx_f3",
        "0000018484372d6f9b79b7e2dd09a4574784f5d16d8cd32c8e9f5d806d7961ba15e0f6",
    ),
    (
        "list_struct_ab_rb_f0",
        "000001ef7a82cdbf533f8f10c07b716c56be4aa9959fa0a3542ccd15baf974a04feafb",
    ),
    (
        "list_struct_ba_rb_f0",
        "000001ef7a82cdbf533f8f10c07b716c56be4aa9959fa0a3542ccd15baf974a04feafb",
    ),
    (
        "large_list_struct_ab_rb_f0",
        "000001ef7a82cdbf533f8f10c07b716c56be4aa9959fa0a3542ccd15baf974a04feafb",
    ),
    (
        "list_struct_ab_rb_f2",
        "00000174b0ee3721df257940875351a9d9a639b5422637c6e83725d0f346ae5cc38a37",
    ),
    (
        "list_struct_ba_rb_f2",
        "000001098bdb016a9ef1086c264d7d6dc16e42d30b82cb6b44a194545f2262dee91101",
    ),
    (
        "large_list_struct_ab_rb_f2",
        "00000174b0ee3721df257940875351a9d9a639b5422637c6e83725d0f346ae5cc38a37",
    ),
    (
        "nv_null_struct_rb_f0",
        "00000103d480f6459069c661bb0e93923fe8cfe505cf8d7daf363bef0d9cd7552295c9",
    ),
    (
        "nv_null_fields_rb_f0",
        "00000103d480f6459069c661bb0e93923fe8cfe505cf8d7daf363bef0d9cd7552295c9",
    ),
    (
        "nv_rows3_rb_f0",
        "000001480a4649266dce9eb73e48072e37598caafe7be4cd92549920cfbbe2a7f9dd0a",
    ),
    (
        "nv_rows2_rb_f0",
        "000001480a4649266dce9eb73e48072e37598caafe7be4cd92549920cfbbe2a7f9dd0a",
    ),
    (
        "nv_list_list_1row_rb_f0",
        "000001cdc413731107ace52443c76e11051f1ef8b1bd98b81bfafa1e4800917d813006",
    ),
    (
        "nv_list_list_2rows_rb_f0",
        "000001cdc413731107ace52443c76e11051f1ef8b1bd98b81bfafa1e4800917d813006",
    ),
    (
        "nv_list_null_elem_rb_f0",
        "0000017d4c97792f67c22b08676bd80a9ba94f5630905facc4455b7e90de966b5286d6",
    ),
    (
        "nv_list_null_fields_elem_rb_f0",
        "0000017d4c97792f67c22b08676bd80a9ba94f5630905facc4455b7e90de966b5286d6",
    ),
    (
        "nv_array_rows3_f0",
        "0000015797829374325d96463f52c24c6d2749c861585b7c77914b431971dead0d861a",
    ),
    (
        "array_struct_xy_f0",
        "0000014e329829b09545ca5b33545e37bb856b9e939568ed967b6dc7309674595ba0dd",
    ),
    (
        "array_struct_yx_f0",
        "0000014e329829b09545ca5b33545e37bb856b9e939568ed967b6dc7309674595ba0dd",
    ),
    (
        "array_struct_xy_f2",
        "000001e79c0ad382c937100b1dba1ae06a2c0b27be7f78cc72a4b37b9c3d0d87348098",
    ),
    (
        "array_struct_yx_f2",
        "0000015fe62c542a14a2b619eca6792dba8d93fa89a87176995a8b5ae6462a8da51788",
    ),
    (
        "array_list_struct_f2",
        "00000183624352b7ebacda7b8c3a3e8a2e51561b694c3835b8468e5a9fcfac3d488c5b",
    ),
    (
        "bool_3rows_rb_f0",
        "000001fdfad32e7d7dbc6dc064b22c955e4606083098ab660970559b57a58bab619a31",
    ),
    (
        "bool_array_2_f0",
        "000001bfbf7b19b3c1de85d67ca1a1544c15adaf58ab9fe554ca2ad83967be3a24e005",
    ),
    (
        "bool_nullable_rb_f0",
        "0000012374ef8a63a8cfe593283ae0336f8d9ea5a84c534655acd6eff15636812d388b",
    ),
    (
        "bool_nullable_stream_f0",
        "000001185dc35a80c8f2e12d34ceaa7062f09b6edbba7131bd7298752debea7e9c9d04",
    ),
    (
        "list_kv_ab_rb_f0",
        "000001622d311ff5b38c734946fffc9088f1dc99bfe8e0cb04ccfda0e420aa8914f4dd",
    ),
    (
        "list_kv_ba_rb_f0",
        "0000015dd41853de5c821a2f13436e7d1886bb606be477024b795d0993cf00f55df2b5",
    ),
    (
        "list_kv_ab_rb_f1",
        "0000013262951dd2bd0d5816a27db1b0dbe6baf6fd321960bdbca93eb77424120754d4",
    ),
    (
        "list_kv_ba_rb_f1",
        "000001b732bdeb1adbe10736db0d382ae346bff3579bcf0c0f726c36097c05621a21ef",
    ),
    (
        "list_kv_ab_rb_f2",
        "000001e3e2ab9ce5627462a6478b88358f11d9e830f9fe0a6e93caea1d99c074bd4ae1",
    ),
    (
        "list_kv_ba_rb_f2",
        "000001934aae7c67907dfb8d81e9065d5882577878dbd6fb9e45a16b70db7a572fe963",
    ),
    (
        "list_kv_ab_rb_f3",
        "000001cc01e2e14642c75fa87a20386fffbc6cbf768573f83746c77236dcfc59ecf72f",
    ),
    (
        "list_kv_ba_rb_f3",
        "000001888c32dcb51501e91305f2b10e384fb4b64f38051fde6ae198a2e354a075e649",
    ),
    (
        "one_col_rb_f4",
        "0000017b8433e7bd2c431a74138d9935ee6d619c001af4f2e4e128c35c60dbdb53c70b",
    ),
    (
        "one_col_schema_f4",
        "0000017022b81e60e4a1dba9ec4602a4976a0dea50f490c916d0ce01e697ebcf9069ee",
    ),
    (
        "one_col_array_f4",
        "0000019e3adff54fa94dec7e5d72a37b3d7b1f8153afedd28032b59997d2ba47c51db0",
    ),
    (
        "one_col_rb_f5",
        "000001c38897ca9f720d18fb37eba87e4b28765a1e664b92764a914590130131816bd2",
    ),
    (
        "one_col_schema_f5",
        "000001d90f32c33eaaa8b3c3c51f2cf33ff920ca09015eeb50d98727a556765a83bd3e",
    ),
    (
        "one_col_array_f5",
        "0000016add16fe46328698a1f252768018a01aa8eab9eacef5adaf602c67fa4196f869",
    ),
    (
        "one_col_rb_f6",
        "000001dbbf891eed1bf9ef6a93202984a39deb5c993b7481cd734eba54878bce23e6f1",
    ),
    (
        "one_col_schema_f6",
        "0000018996061e12067c82279f8c5f08d3defe8ad2f9509c6701416a7b75055dcf0d95",
    ),
    (
        "one_col_array_f6",
        "000001973166b6e70cba833de35adaed48af4416d986c5b411196e4b5a1951259119c7",
    ),
    (
        "one_col_rb_f7",
        "00000165a886e87aaa8efd06668bc6bb3f154d0aad098cc2f301eaf45150d1bacb15ca",
    ),
    (
        "one_col_schema_f7",
        "0000010da7daf6e5c269dcc172b9756942366d3ed59d809ed79f0a8c961c21a64ba581",
    ),
    (
        "one_col_array_f7",
        "0000016135b88789127c04f395b05ffd47ceac8ed34829f9bb952ad5e5f4e17c881394",
    ),
    (
        "plain_schema_f7_im0",
        "0000012e5ca80a107600a263f4ae5362e7be8f5531dc190fb745134eeb99b5dd29cf42",
    ),
    (
        "plain_schema_f7_im1",
        "0000012e5ca80a107600a263f4ae5362e7be8f5531dc190fb745134eeb99b5dd29cf42",
    ),
    (
        "meta_schema_f7_im1",
        "0000015a690f2487025a0de8162d8c5eaa4a7684991819b210b7726f83f3249a34b22f",
    ),
    (
        "nv_null_struct_rb_f4",
        "000001bac1fd3395c1558237b83d1058ddfe0ed147777235b268f915d0db961d5d5a92",
    ),
    (
        "nv_null_fields_rb_f4",
        "0000013c71bf0f16c580e51fecfe6a319bcabf8726c971c7dae68aadabfae3a2d463ba",
    ),
    (
        "nv_rows3_rb_f4",
        "000001ca0b11291a17b3a74df4d3e07e598df6254f18bea48d6de658bad6ce0764ed2c",
    ),
    (
        "nv_rows2_rb_f4",
        "0000014938f09116f0bd0ba424d14c2147fba37f50156ea6a8a16ad6491e7a19fd7b4e",
    ),
    (
        "nv_list_list_1row_rb_f4",
        "000001a12b9454ccc5aba3531a3402e2b245afa5c13aeb7e34760893614025460c3510",
    ),
    (
        "nv_list_list_2rows_rb_f4",
        "000001024d511e3ecf756438dae91a23fc5129a7f075aa30be86f6605370b076567c95",
    ),
    (
        "nv_list_null_elem_rb_f4",
        "0000014be0fee254d85051c22a6be84af446096947a5a04b993571fdbb6ff1967f3113",
    ),
    (
        "nv_list_null_fields_elem_rb_f4",
        "00000110b24fe51f19bcd11887cc0b00514da0939a1f4e6465852f5b67c2ba5ee8d4fe",
    ),
    (
        "nv_null_struct_rb_f7",
        "00000178a32a80015616d24eec5c25eca12bbea6685c04bde437c1f1575dfa1afafda3",
    ),
    (
        "nv_null_fields_rb_f7",
        "000001196f0d60192e3fd1b9cab06112c76238fdb91f17e489099e71551c29faeef270",
    ),
    (
        "nv_rows3_rb_f7",
        "00000198afb787b75ff51607154462604f6c5a87c59c8b3d559d40ea3633f750cb2794",
    ),
    (
        "nv_rows2_rb_f7",
        "000001323fc9c3c1bcd52c819613c152c5f57af4249574a11a1cec936827576f3e507a",
    ),
    (
        "nv_list_list_1row_rb_f7",
        "0000018c0df99d4e6f5a1e71c6dfcbe32a9812cd8c2027ec6d83d699e52aeff7c6b0bb",
    ),
    (
        "nv_list_list_2rows_rb_f7",
        "000001fbafc458f570526d42a9a81f3ecd8e5e15ca485efab0e1a96a4d84c3c2e2d1c7",
    ),
    (
        "nv_list_null_elem_rb_f7",
        "00000183c17f4fc47ee0bae5018422ed0d792ceeb5d1d8bedec85cc476996b96056a00",
    ),
    (
        "nv_list_null_fields_elem_rb_f7",
        "000001002591d3d2c26cc8b54288f76f17955ee1563be257ed6ff1671e6511c0481641",
    ),
    (
        "nv_nested_structs_rb_f4",
        "00000142749678d54b57f5edea35b4cdf05ee82198181e33812ce1b9dea23f3b0b5c71",
    ),
    (
        "nv_nested_structs_rb_f7",
        "00000137e9864f3f6ecc066f6e970390de95773bf728c664df3ce23b9f5093e038b2a8",
    ),
    (
        "nv_rows3_stream_f4",
        "000001ca0b11291a17b3a74df4d3e07e598df6254f18bea48d6de658bad6ce0764ed2c",
    ),
    (
        "nv_array_rows3_f4",
        "0000012fb1af7ca813fefcbd19e1f6d01edc842e1729b0531d2e1bee183692647d4173",
    ),
    (
        "nv_array_rows2_f4",
        "000001db8ad691e48339ba83ece8c4d4aa7a720788c0e415edbfba4d8ad51d2fc0a692",
    ),
    (
        "phys_dict_rb_f7",
        "000001abd90a756d2b9b699859c501edf37b6dcfc31e009c4e4191d4b0e087a65f7575",
    ),
    (
        "phys_utf8_rb_f7",
        "000001abd90a756d2b9b699859c501edf37b6dcfc31e009c4e4191d4b0e087a65f7575",
    ),
    (
        "phys_large_utf8_rb_f7",
        "000001abd90a756d2b9b699859c501edf37b6dcfc31e009c4e4191d4b0e087a65f7575",
    ),
    (
        "array_struct_xy_f6",
        "000001ec888add39caf7a9dedd6ec56bfa446143f3f2abf03848fbf82f61af9cda7215",
    ),
    (
        "array_struct_yx_f6",
        "0000015735e773b173cc5fe0e1e882674f130575b03e4133667062a777592ac31c5faf",
    ),
    (
        "array_list_struct_f7",
        "00000106b91fc017c699ca97737d4bc78d5f52525b6406dd3efecb46b902e928731e4e",
    ),
    (
        "combo_sn_yx_3rows_rb_f7",
        "000001d6b4594d17ae3482c202877531923bb9212bcf8389fb7597b173dc5796079929",
    ),
    (
        "combo_sn_yx_3rows_stream_f7",
        "000001d6b4594d17ae3482c202877531923bb9212bcf8389fb7597b173dc5796079929",
    ),
    (
        "bool_1row_rb_f4",
        "000001e392174345825efe12d76a8cf613d226902dcc594342c537f9d81c528b137531",
    ),
    (
        "bool_3rows_rb_f4",
        "000001b5597cc0ee4c9dc458c9e06baa947609cd923db09dacbe2378f33aaf6c690f14",
    ),
    (
        "bool_array_1_f4",
        "0000014e102943402882cc417bc1d710b20cae5f60c411ccea8c77bdc020a884ca1236",
    ),
    (
        "bool_array_2_f4",
        "0000019f574770ec94ceca20efe47c91263b67b642a106bc40d27345e22d1e72dd2423",
    ),
    (
        "bool_nullable_rb_f4",
        "000001f38e97f8e2fe332a95bf129fbc7bd02bf809c7ff2ad345cbed6045ec3085ddf4",
    ),
    (
        "bool_nullable_stream_f4",
        "000001f38e97f8e2fe332a95bf129fbc7bd02bf809c7ff2ad345cbed6045ec3085ddf4",
    ),
    (
        "names_schema_f7",
        "00000158f3d79b469fa3908f93bf0521948e82d2c742a631debb55abfe87d98aa75ae5",
    ),
    (
        "names_array_struct_f7",
        "0000011791c000afa0ce2c897e5e6635537d691efe7e7bc7ae6be6589005f8aa805034",
    ),
    (
        "list_kv_ab_rb_f4",
        "000001e7d3f89675621f5a4156f6f078abae23727db8fc317f1cac71e1984e21992ce6",
    ),
    (
        "list_kv_ba_rb_f4",
        "000001ed916278281456016a7a4464aa609718917bb3241fc3dfdab5aa9b195b2b59b6",
    ),
    (
        "list_kv_ab_rb_f5",
        "0000010aebc959d788c3dc406624ce5bc8e8fcd8075b1548cffc3591e1458ff810a49a",
    ),
    (
        "list_kv_ba_rb_f5",
        "00000149081383d6607d80608e51b8b98ef17851e7dd4812a7eef4338767a7f1dfc13a",
    ),
    (
        "list_kv_ab_rb_f6",
        "000001748fdb72d4caf3a1c6801f9e100aa1eec834ece90a80bf9c9ba22dce25c9d006",
    ),
    (
        "list_kv_ba_rb_f6",
        "000001756f035b80282125cf229c8aa85c8d3a57f86d03bde234f60d97c58fce63cab0",
    ),
    (
        "list_kv_ab_rb_f7",
        "00000168c7045a867d7ad44e355b2dd396eeb6a5eaa0755ab51b0e06be7a3fb9ba498f",
    ),
    (
        "list_kv_ba_rb_f7",
        "00000165094d74353dc990af3bf495d67dbef64b278e757fed267583a7ce44f7d6088d",
    ),
];

#[expect(clippy::panic, reason = "CLI tool — the append-only guard aborts")]
fn check_append_only(vectors: &[GoldenVector]) {
    let Ok(raw) = fs::read_to_string(FIXTURE_PATH) else {
        return; // no committed fixture yet
    };
    let existing: ExistingFixture = serde_json::from_str(&raw).unwrap();
    for old in &existing.vectors {
        if V0_5_FIX_EXCEPTIONS.contains(&(old.id.as_str(), old.expected_hash.as_str())) {
            continue;
        }
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
