//! Golden fixture generator for hash format 2 (ITL-898).
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin emit_golden_format2 > tests/golden/format2_v0.5.json.new \
//!   && mv tests/golden/format2_v0.5.json.new tests/golden/format2_v0.5.json
//! ```
//!
//! Then copy the file byte-for-byte to `starfix-python/tests/golden/format2_v0.5.json`.
//!
//! # Append-only guard
//!
//! Hash definitions are permanent. When `tests/golden/format2_v0.5.json` exists, this generator
//! loads it and **aborts** if any vector id already present would get a different
//! `expected_hash` or would disappear. New vectors may be added; existing values never move.
//! `cargo test` (`tests/golden_vectors_format2.rs`) separately re-hashes every committed vector.
//!
//! Byte layout: `docs/byte-layout-spec.md`.

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
use std::io::Cursor;
use std::sync::Arc;
use std::{fs, io, slice};

use arrow::array::{
    make_array, Array as _, ArrayData, ArrayRef, BooleanArray, DictionaryArray,
    FixedSizeBinaryArray, Int32Array, Int32Builder, Int8Array, LargeListArray, LargeStringArray,
    ListArray, ListBuilder, RecordBatch, RecordBatchOptions, StringArray, StructArray,
};
use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer};
use arrow::compute::cast;
use arrow::datatypes::{Int32Type, Int8Type};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Fields, IntervalUnit, Schema, SchemaRef, TimeUnit};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use starfix::{ArrowDigester, HasherConfig};

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/golden/format2_v0.5.json"
);

#[derive(Serialize)]
struct GoldenFixture {
    version: String,
    format: u32,
    generated_by: String,
    vectors: Vec<GoldenVector>,
    relations: Vec<Relation>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
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

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
struct Relation {
    kind: String,
    ids: Vec<String>,
    why: String,
}

#[derive(Deserialize)]
struct ExistingFixture {
    vectors: Vec<GoldenVector>,
    relations: Vec<Relation>,
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
        assert_eq!(
            round_trip_hash(method, &ipc, config),
            hash,
            "vector {id} does not survive its IPC encoding"
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
                "Zero columns, 3 rows",
                flags,
                false,
                &Input::Batch(zero_cols(3)),
            );
            gen.relate(
                "different",
                &[
                    &format!("zero_cols_0rows_rb_f{flags}"),
                    &format!("zero_cols_3rows_rb_f{flags}"),
                ],
                "format 2 hashes the row count, also of a zero-column table",
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
    add_format1_metadata_names_and_extensions(&mut gen);
    add_review_coverage(&mut gen);
    add_python_literal_inputs(&mut gen);
    add_entry_points_and_late_review_coverage(&mut gen);

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

    // 15. Phase 2 metadata field paths (byte-layout spec Section 2.3).
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

/// The nine schemas of the format-1 fixture `include_metadata_v0.3.json` (same ids, re-hashed),
/// names that are not hashed, extension types, and empty structs.
fn add_format1_metadata_names_and_extensions(gen: &mut Builder) {
    // 16. The format-1 metadata vectors, same inputs (decoded from the historical file).
    let v03_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/include_metadata_v0.3.json"
    );
    let v03: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(v03_path).unwrap()).unwrap();
    for v in v03
        .get("vectors")
        .and_then(serde_json::Value::as_array)
        .unwrap()
    {
        let field = |name: &str| v.get(name).unwrap();
        let ipc = BASE64_STANDARD
            .decode(field("ipc_b64").as_str().unwrap())
            .unwrap();
        let schema = StreamReader::try_new(Cursor::new(ipc), None)
            .unwrap()
            .schema();
        gen.add(
            field("id").as_str().unwrap(),
            field("description").as_str().unwrap(),
            0x00,
            field("include_metadata").as_bool().unwrap(),
            &Input::Schema(schema),
        );
    }

    // 17. Names that are not hashed (byte-layout spec Section 2.4).
    let list_named = |name: &str| {
        let item =
            Arc::new(Field::new(name, DataType::Int32, true).with_metadata(meta(&[("u", "s")])));
        let col: ArrayRef = Arc::new(ListArray::new(
            Arc::clone(&item),
            OffsetBuffer::<i32>::from_lengths([2]),
            Arc::new(Int32Array::from(vec![1, 2])),
            None,
        ));
        batch(
            vec![Field::new("l", col.data_type().clone(), false)],
            vec![col],
        )
    };
    gen.add(
        "name_list_item_rb_f0_im1",
        "l: List<item: Int32{u:s}> [[1, 2]], include_metadata",
        0x00,
        true,
        &Input::Batch(list_named("item")),
    );
    gen.add(
        "name_list_element_rb_f0_im1",
        "Same with the element field named 'element'",
        0x00,
        true,
        &Input::Batch(list_named("element")),
    );
    gen.relate(
        "equal",
        &["name_list_item_rb_f0_im1", "name_list_element_rb_f0_im1"],
        "the list element field name is not hashed",
    );
    let map_named = |entries: &str, key: &str, value: &str, entries_meta: bool| {
        let mut entries_field = Field::new(
            entries,
            DataType::Struct(Fields::from(vec![
                Field::new(key, DataType::Utf8, false).with_metadata(meta(&[("role", "id")])),
                Field::new(value, DataType::Int32, true).with_metadata(meta(&[("unit", "kg")])),
            ])),
            false,
        );
        if entries_meta {
            entries_field = entries_field.with_metadata(meta(&[("role", "pairs")]));
        }
        Arc::new(Schema::new(vec![Field::new(
            "m",
            DataType::Map(Arc::new(entries_field), false),
            true,
        )]))
    };
    gen.add(
        "name_map_standard_schema_f0_im1",
        "m: Map<Utf8, Int32> with entries/key/value names and key/value metadata",
        0x00,
        true,
        &Input::Schema(map_named("entries", "key", "value", false)),
    );
    gen.add(
        "name_map_custom_schema_f0_im1",
        "Same with names kv/k/v",
        0x00,
        true,
        &Input::Schema(map_named("kv", "k", "v", false)),
    );
    gen.add(
        "name_map_entries_meta_schema_f0_im1",
        "Same with metadata on the entries field",
        0x00,
        true,
        &Input::Schema(map_named("entries", "key", "value", true)),
    );
    gen.relate(
        "equal",
        &["name_map_standard_schema_f0_im1", "name_map_custom_schema_f0_im1", "name_map_entries_meta_schema_f0_im1"],
        "map entries/key/value names and the entries field's metadata are not hashed (pyarrow cannot observe them)",
    );
    let dict = |ordered: bool, int8: bool| {
        let values = StringArray::from(vec!["p", "q", "p"]);
        let key_type = if int8 {
            DataType::Int8
        } else {
            DataType::Int32
        };
        let array = cast(
            &values,
            &DataType::Dictionary(Box::new(key_type), Box::new(DataType::Utf8)),
        )
        .unwrap();
        let field = Field::new("d", array.data_type().clone(), false).with_dict_is_ordered(ordered);
        batch(vec![field], vec![array])
    };
    gen.add(
        "name_dict_int8_rb_f0",
        "d: Dictionary<Int8, Utf8> [p, q, p]",
        0x00,
        false,
        &Input::Batch(dict(false, true)),
    );
    gen.add(
        "name_dict_int32_ordered_rb_f0",
        "d: Dictionary<Int32, Utf8> [p, q, p], field marked dict ordered",
        0x00,
        false,
        &Input::Batch(dict(true, false)),
    );
    gen.relate(
        "equal",
        &["name_dict_int8_rb_f0", "name_dict_int32_ordered_rb_f0"],
        "dictionary index type and ordered flag are not hashed",
    );

    // 18. Extension types: registered (arrow.uuid) and unregistered (example.point).
    let uuid = |with_ext: bool| {
        let values: Vec<Option<[u8; 16]>> = vec![Some([7_u8; 16]), None];
        let arr =
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(values.into_iter(), 16).unwrap();
        let mut field = Field::new("u", DataType::FixedSizeBinary(16), true);
        if with_ext {
            field = field.with_metadata(meta(&[
                ("ARROW:extension:name", "arrow.uuid"),
                ("ARROW:extension:metadata", ""),
            ]));
        }
        batch(vec![field], vec![Arc::new(arr)])
    };
    let point = |with_ext: bool| {
        let mut field = Field::new("p", DataType::Int32, true);
        if with_ext {
            field = field.with_metadata(meta(&[
                ("ARROW:extension:name", "example.point"),
                ("ARROW:extension:metadata", "{\"dims\":1}"),
            ]));
        }
        batch(
            vec![field],
            vec![Arc::new(Int32Array::from(vec![Some(3), None]))],
        )
    };
    for im in [false, true] {
        let s = if im { "im1" } else { "im0" };
        gen.add(
            &format!("ext_uuid_rb_f0_{s}"),
            "u: arrow.uuid extension (registered in pyarrow) over FixedSizeBinary(16)",
            0x00,
            im,
            &Input::Batch(uuid(true)),
        );
        gen.add(
            &format!("ext_uuid_storage_rb_f0_{s}"),
            "Same data as plain FixedSizeBinary(16)",
            0x00,
            im,
            &Input::Batch(uuid(false)),
        );
        gen.add(
            &format!("ext_point_rb_f0_{s}"),
            "p: example.point extension (unregistered) over Int32",
            0x00,
            im,
            &Input::Batch(point(true)),
        );
        gen.add(
            &format!("ext_point_storage_rb_f0_{s}"),
            "Same data as plain Int32",
            0x00,
            im,
            &Input::Batch(point(false)),
        );
    }
    gen.relate(
        "equal",
        &["ext_uuid_rb_f0_im0", "ext_uuid_storage_rb_f0_im0"],
        "without include_metadata an extension type hashes as its storage type",
    );
    gen.relate(
        "equal",
        &["ext_point_rb_f0_im0", "ext_point_storage_rb_f0_im0"],
        "without include_metadata an extension type hashes as its storage type",
    );
    gen.relate(
        "different",
        &["ext_uuid_rb_f0_im1", "ext_uuid_storage_rb_f0_im1"],
        "with include_metadata both ARROW:extension:* keys are hashed",
    );
    gen.relate(
        "different",
        &["ext_point_rb_f0_im1", "ext_point_storage_rb_f0_im1"],
        "with include_metadata both ARROW:extension:* keys are hashed",
    );

    // 19. Empty structs keep their row count.
    let empty_struct = |rows: usize| {
        let s = StructArray::new_empty_fields(rows, None);
        batch(
            vec![Field::new("e", s.data_type().clone(), false)],
            vec![Arc::new(s)],
        )
    };
    gen.add(
        "empty_struct_1row_rb_f0",
        "e: Struct<> non-null, 1 row",
        0x00,
        false,
        &Input::Batch(empty_struct(1)),
    );
    gen.add(
        "empty_struct_2rows_rb_f0",
        "e: Struct<> non-null, 2 rows",
        0x00,
        false,
        &Input::Batch(empty_struct(2)),
    );
    gen.relate(
        "different",
        &["empty_struct_1row_rb_f0", "empty_struct_2rows_rb_f0"],
        "format 2 hashes the row count",
    );
}

/// Coverage requested in review: nested nulls, data under null parents, dictionaries at depth,
/// extension metadata forms, empty names (groups 20–25).
fn add_review_coverage(gen: &mut Builder) {
    let col = |name: &str, array: ArrayRef, nullable: bool| {
        batch(
            vec![Field::new(name, array.data_type().clone(), nullable)],
            vec![array],
        )
    };
    let list = |item_nullable: bool,
                lengths: Vec<usize>,
                items: ArrayRef,
                nulls: Option<NullBuffer>|
     -> ArrayRef {
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", items.data_type().clone(), item_nullable)),
            OffsetBuffer::<i32>::from_lengths(lengths),
            items,
            nulls,
        ))
    };

    // 20. A null inner list in a list of nullable lists: [[[1], null, [2]]]
    {
        let inner = list(
            true,
            vec![1, 0, 1],
            Arc::new(Int32Array::from(vec![1, 2])),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let outer = list(true, vec![3], inner, None);
        gen.add(
            "nested_null_inner_list_rb_f0",
            "c: List<List<Int32?>?> one row [[1], null, [2]]",
            0x00,
            false,
            &Input::Batch(col("c", outer, false)),
        );
        let inner_empty = list(
            true,
            vec![1, 0, 1],
            Arc::new(Int32Array::from(vec![1, 2])),
            None,
        );
        let outer_empty = list(true, vec![3], inner_empty, None);
        gen.add(
            "nested_empty_inner_list_rb_f0",
            "Same with an empty inner list instead of null: [[1], [], [2]]",
            0x00,
            false,
            &Input::Batch(col("c", outer_empty, false)),
        );
        gen.relate(
            "different",
            &[
                "nested_null_inner_list_rb_f0",
                "nested_empty_inner_list_rb_f0",
            ],
            "a null list differs from an empty list, also when nested",
        );
    }

    // 21. Depth 4: List<Struct<v: List<Struct<x: Int32?>?>?>?>, nulls at every level
    {
        let x = StructArray::new(
            Fields::from(vec![Field::new("x", DataType::Int32, true)]),
            vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let v = list(
            true,
            vec![2, 1, 0],
            Arc::new(x),
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let s = StructArray::new(
            Fields::from(vec![Field::new("v", v.data_type().clone(), true)]),
            vec![v],
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let outer = list(true, vec![2, 1], Arc::new(s), None);
        gen.add(
            "depth4_rb_f0",
            "d: List<Struct<v: List<Struct<x: Int32?>?>?>?> with nulls at every level",
            0x00,
            false,
            &Input::Batch(col("d", Arc::clone(&outer), false)),
        );
        gen.add(
            "depth4_rb_f3",
            "Same, both order options",
            0x03,
            false,
            &Input::Batch(col("d", outer, false)),
        );
    }

    // 22. A nullable list inside a nullable struct, and data under null parents
    {
        let l = list(
            true,
            vec![1, 0, 2],
            Arc::new(Int32Array::from(vec![1, 7, 8])),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let s = StructArray::new(
            Fields::from(vec![Field::new("l", l.data_type().clone(), true)]),
            vec![l],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        gen.add(
            "list_in_struct_rb_f0",
            "s: Struct<l: List<Int32>?>? rows [{l:[1]}, {l:null}, null]",
            0x00,
            false,
            &Input::Batch(col("s", Arc::new(s), true)),
        );
        // a null list slot that still covers values hashes like an empty one
        let covered = list(
            true,
            vec![1, 2, 1],
            Arc::new(Int32Array::from(vec![1, 98, 99, 4])),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let empty_slot = list(
            true,
            vec![1, 0, 1],
            Arc::new(Int32Array::from(vec![1, 4])),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        gen.add(
            "null_list_slot_with_values_rb_f0",
            "l: List<Int32>? rows [[1], null (slot covers 98, 99), [4]]",
            0x00,
            false,
            &Input::Batch(col("l", covered, true)),
        );
        gen.add(
            "null_list_slot_empty_rb_f0",
            "l: List<Int32>? rows [[1], null (empty slot), [4]]",
            0x00,
            false,
            &Input::Batch(col("l", empty_slot, true)),
        );
        gen.relate(
            "equal",
            &[
                "null_list_slot_with_values_rb_f0",
                "null_list_slot_empty_rb_f0",
            ],
            "values under a null list are not hashed",
        );
        let under_null = |hidden: i32| {
            let s = StructArray::new(
                Fields::from(vec![Field::new("a", DataType::Int32, false)]),
                vec![Arc::new(Int32Array::from(vec![1, hidden, 3]))],
                Some(NullBuffer::from(vec![true, false, true])),
            );
            col("s", Arc::new(s), true)
        };
        gen.add(
            "struct_null_hides_5_rb_f0",
            "s: Struct<a: Int32>? rows [{a:1}, null (a=5 underneath), {a:3}]",
            0x00,
            false,
            &Input::Batch(under_null(5)),
        );
        gen.add(
            "struct_null_hides_6_rb_f0",
            "Same with a=6 underneath the null struct",
            0x00,
            false,
            &Input::Batch(under_null(6)),
        );
        gen.relate("equal", &["struct_null_hides_5_rb_f0", "struct_null_hides_6_rb_f0"], "values under a null struct are not hashed (non-nullable child: no validity bits, no values)");
        let empty_vs_null = |null: bool| {
            let l = list(
                true,
                vec![0],
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Some(NullBuffer::from(vec![!null])),
            );
            col("l", l, true)
        };
        gen.add(
            "empty_list_rb_f0",
            "l: List<Int32>? one row []",
            0x00,
            false,
            &Input::Batch(empty_vs_null(false)),
        );
        gen.add(
            "null_list_rb_f0",
            "l: List<Int32>? one row null",
            0x00,
            false,
            &Input::Batch(empty_vs_null(true)),
        );
        gen.relate(
            "different",
            &["empty_list_rb_f0", "null_list_rb_f0"],
            "an empty list differs from a null list",
        );
    }

    // 23. Dictionaries: null, unused and duplicate dictionary values; at depth; equal to plain
    {
        use arrow::array::{DictionaryArray, Float64Array, Int64Array};
        use arrow::datatypes::Int32Type;
        let dict64 = |keys: Vec<Option<i32>>, values: Vec<Option<i64>>| -> ArrayRef {
            Arc::new(
                DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(keys),
                    Arc::new(Int64Array::from(values)),
                )
                .unwrap(),
            )
        };
        // keys -> [30, 10, null (null key), null (key -> null value), 30]
        let d = dict64(
            vec![Some(2), Some(0), None, Some(4), Some(3)],
            vec![Some(10), Some(99), Some(30), Some(30), None],
        );
        let plain: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(30),
            Some(10),
            None,
            None,
            Some(30),
        ]));
        gen.add("dict_null_unused_dup_rb_f0", "v: Dictionary<Int32, Int64> with a null key, a key to a null value, an unused and a duplicated value", 0x00, false, &Input::Batch(col("v", Arc::clone(&d), true)));
        gen.add(
            "dict_null_unused_dup_plain_rb_f0",
            "v: Int64 with the same logical values",
            0x00,
            false,
            &Input::Batch(col("v", Arc::clone(&plain), true)),
        );
        gen.relate(
            "equal",
            &[
                "dict_null_unused_dup_rb_f0",
                "dict_null_unused_dup_plain_rb_f0",
            ],
            "a dictionary hashes as its logical values",
        );
        let same_keys_other_values = dict64(
            vec![Some(2), Some(0), None, Some(4), Some(3)],
            vec![Some(11), Some(99), Some(30), Some(30), None],
        );
        gen.add(
            "dict_same_keys_other_values_rb_f0",
            "Same keys, dictionary value 10 replaced by 11",
            0x00,
            false,
            &Input::Batch(col("v", same_keys_other_values, true)),
        );
        gen.relate(
            "different",
            &[
                "dict_null_unused_dup_rb_f0",
                "dict_same_keys_other_values_rb_f0",
            ],
            "same index pattern, different values",
        );
        // list<dictionary<int64>> (nullable items) and sliced
        gen.add(
            "list_dict_rb_f0",
            "l: List<Dictionary<Int32, Int64>?> rows of 2 and 3 items",
            0x00,
            false,
            &Input::Batch(col(
                "l",
                list(true, vec![2, 3], Arc::clone(&d), None),
                false,
            )),
        );
        gen.add(
            "list_plain_rb_f0",
            "l: List<Int64?> with the same values",
            0x00,
            false,
            &Input::Batch(col(
                "l",
                list(true, vec![2, 3], Arc::clone(&plain), None),
                false,
            )),
        );
        gen.relate(
            "equal",
            &["list_dict_rb_f0", "list_plain_rb_f0"],
            "dictionary equals plain inside a list",
        );
        let sliced_dict = col("l", list(true, vec![2, 3], Arc::clone(&d), None), false).slice(1, 1);
        let sliced_plain =
            col("l", list(true, vec![2, 3], Arc::clone(&plain), None), false).slice(1, 1);
        gen.add(
            "list_dict_sliced_rb_f0",
            "Row 1 of list_dict, as a slice (non-zero offsets)",
            0x00,
            false,
            &Input::Batch(sliced_dict),
        );
        gen.add(
            "list_plain_sliced_rb_f0",
            "Row 1 of list_plain, as a slice",
            0x00,
            false,
            &Input::Batch(sliced_plain),
        );
        gen.relate(
            "equal",
            &["list_dict_sliced_rb_f0", "list_plain_sliced_rb_f0"],
            "dictionary equals plain inside a sliced list",
        );
        // list<struct<dictionary<float64>>>
        let fdict: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![1, 0, 1]),
                Arc::new(Float64Array::from(vec![0.5, -2.25])),
            )
            .unwrap(),
        );
        let fplain: ArrayRef = Arc::new(Float64Array::from(vec![-2.25, 0.5, -2.25]));
        let lsd = |values: ArrayRef| {
            let st = StructArray::new(
                Fields::from(vec![Field::new("f", values.data_type().clone(), false)]),
                vec![values],
                None,
            );
            col("l", list(false, vec![1, 2], Arc::new(st), None), false)
        };
        gen.add(
            "list_struct_dict_rb_f0",
            "l: List<Struct<f: Dictionary<Int32, Float64>>>",
            0x00,
            false,
            &Input::Batch(lsd(fdict)),
        );
        gen.add(
            "list_struct_plain_rb_f0",
            "l: List<Struct<f: Float64>> with the same values",
            0x00,
            false,
            &Input::Batch(lsd(fplain)),
        );
        gen.relate(
            "equal",
            &["list_struct_dict_rb_f0", "list_struct_plain_rb_f0"],
            "dictionary equals plain inside a struct inside a list",
        );
        // list<list<dictionary<utf8>>>
        let sdict: ArrayRef = cast(
            &StringArray::from(vec![Some("x"), None, Some("y"), Some("x")]),
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        )
        .unwrap();
        let splain: ArrayRef = Arc::new(StringArray::from(vec![
            Some("x"),
            None,
            Some("y"),
            Some("x"),
        ]));
        let lls = |values: ArrayRef| {
            col(
                "l",
                list(true, vec![2], list(true, vec![1, 3], values, None), None),
                false,
            )
        };
        gen.add(
            "list_list_dict_rb_f0",
            "l: List<List<Dictionary<Int8, Utf8>?>?>",
            0x00,
            false,
            &Input::Batch(lls(sdict)),
        );
        gen.add(
            "list_list_plain_rb_f0",
            "l: List<List<Utf8?>?> with the same values",
            0x00,
            false,
            &Input::Batch(lls(splain)),
        );
        gen.relate(
            "equal",
            &["list_list_dict_rb_f0", "list_list_plain_rb_f0"],
            "dictionary equals plain two lists deep",
        );
        // dictionary of structs and of lists (E5): hashed as the value type
        let st = StructArray::from(vec![(
            Arc::new(Field::new("a", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )]);
        let dst: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(Int32Array::from(vec![1, 0, 1]), Arc::new(st))
                .unwrap(),
        );
        let pst: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("a", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![2, 1, 2])) as ArrayRef,
        )]));
        gen.add(
            "dict_of_struct_array_f0",
            "hash_array of Dictionary<Int32, Struct<a: Int32>>",
            0x00,
            false,
            &Input::Array(dst),
        );
        gen.add(
            "dict_of_struct_plain_array_f0",
            "hash_array of the equivalent Struct<a: Int32>",
            0x00,
            false,
            &Input::Array(pst),
        );
        gen.relate(
            "equal",
            &["dict_of_struct_array_f0", "dict_of_struct_plain_array_f0"],
            "a dictionary of structs hashes as its decoded structs",
        );
        // hash_array nullability counts logical nulls (a valid key pointing at a null value) (R4)
        let dn: ArrayRef = dict64(vec![Some(0), Some(1)], vec![Some(5), None]);
        let pn: ArrayRef = Arc::new(Int64Array::from(vec![Some(5), None]));
        gen.add(
            "dict_key_to_null_array_f0",
            "hash_array of Dictionary [5, null] where the null comes from the dictionary values",
            0x00,
            false,
            &Input::Array(dn),
        );
        gen.add(
            "dict_key_to_null_plain_array_f0",
            "hash_array of Int64 [5, null]",
            0x00,
            false,
            &Input::Array(pn),
        );
        gen.relate(
            "equal",
            &[
                "dict_key_to_null_array_f0",
                "dict_key_to_null_plain_array_f0",
            ],
            "hash_array nullability counts logical nulls after decoding the dictionary",
        );
    }

    // 24. Extension metadata forms (E1); with include_metadata
    {
        let ext = |name: &str, pairs: &[(&str, &str)], storage: DataType| {
            let mut m = meta(&[("ARROW:extension:name", name)]);
            m.extend(meta(pairs));
            Arc::new(Schema::new(vec![
                Field::new("e", storage, true).with_metadata(m)
            ]))
        };
        gen.add(
            "ext_custom_python_schema_f0_im1",
            "orcapod.ndarray-style extension with metadata {\"version\":1} over LargeBinary",
            0x00,
            true,
            &Input::Schema(ext(
                "orcapod.ndarray",
                &[("ARROW:extension:metadata", "{\"version\":1}")],
                DataType::LargeBinary,
            )),
        );
        gen.add(
            "ext_uuid_name_only_schema_f0_im1",
            "arrow.uuid with ARROW:extension:name only",
            0x00,
            true,
            &Input::Schema(ext("arrow.uuid", &[], DataType::FixedSizeBinary(16))),
        );
        gen.add(
            "ext_uuid_empty_metadata_schema_f0_im1",
            "arrow.uuid with ARROW:extension:metadata = \"\"",
            0x00,
            true,
            &Input::Schema(ext(
                "arrow.uuid",
                &[("ARROW:extension:metadata", "")],
                DataType::FixedSizeBinary(16),
            )),
        );
        gen.relate(
            "equal",
            &[
                "ext_uuid_name_only_schema_f0_im1",
                "ext_uuid_empty_metadata_schema_f0_im1",
            ],
            "an absent ARROW:extension:metadata equals an empty one",
        );
        // Canonical storage as Arrow C++ / pyarrow define it: the item field is NULLABLE.
        let tensor =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4);
        gen.add("ext_tensor_canonical_schema_f0_im1", "arrow.fixed_shape_tensor with canonical metadata {\"shape\":[2,2]} over its canonical storage FixedSizeList<item: Float32 nullable, 4> (schema only)", 0x00, true, &Input::Schema(ext("arrow.fixed_shape_tensor", &[("ARROW:extension:metadata", "{\"shape\":[2,2]}")], tensor)));
    }

    // 25. Empty names (R7) and the format-2 nullable-struct example with a non-nullable child (E2)
    {
        let empty_child = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Struct(Fields::from(vec![Field::new("", DataType::Int32, false)])),
            false,
        )]));
        gen.add(
            "empty_name_child_schema_f0",
            "s: Struct<\"\": Int32>",
            0x00,
            false,
            &Input::Schema(empty_child),
        );
        let empty_col = Arc::new(Schema::new(vec![
            Field::new(
                "",
                DataType::Struct(Fields::from(vec![Field::new("x", DataType::Int32, false)])),
                false,
            ),
            Field::new("x", DataType::Int32, false),
        ]));
        gen.add(
            "empty_name_column_schema_f0",
            "[\"\": Struct<x: Int32>, x: Int32] (child path \"/x\", column \"x\")",
            0x00,
            false,
            &Input::Schema(empty_col),
        );
        let s = StructArray::new(
            Fields::from(vec![Field::new("a", DataType::Int32, false)]),
            vec![Arc::new(Int32Array::from(vec![1, 0, 3]))],
            Some(NullBuffer::from(vec![true, false, true])),
        );
        gen.add(
            "struct_nonnull_child_rb_f0",
            "s: Struct<a: Int32 non-null>? rows [{a:1}, null, {a:3}] (spec worked example)",
            0x00,
            false,
            &Input::Batch(col("s", Arc::new(s), true)),
        );
    }
}

/// Group 26: the inputs of starfix-python's hand-coded format-1 test literals
/// (`tests/golden/python_literals_format1_inputs.json`, recovered on the Python side and checked
/// against starfix 0.4), re-hashed in format 2 so that every such literal has a Rust value.
/// A test with several literals gets ids `pylit_<test>`, `pylit_<test>_2`, ...
#[expect(
    clippy::panic,
    reason = "CLI tool — an unknown method in the inputs file aborts"
)]
fn add_python_literal_inputs(gen: &mut Builder) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/python_literals_format1_inputs.json"
    );
    let inputs: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for e in inputs
        .get("entries")
        .and_then(serde_json::Value::as_array)
        .unwrap()
    {
        let field = |name: &str| e.get(name).unwrap();
        let test = field("test").as_str().unwrap();
        let name = test
            .split_once("::")
            .map_or(test, |(_, rest)| rest)
            .to_lowercase();
        let raw_slug: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let slug = raw_slug
            .split('_')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        let count = seen.entry(slug.clone()).or_insert(0);
        *count = count.checked_add(1).unwrap();
        let id = if *count == 1 {
            format!("pylit_{slug}")
        } else {
            format!("pylit_{slug}_{count}")
        };
        let ipc = BASE64_STANDARD
            .decode(field("ipc_b64").as_str().unwrap())
            .unwrap();
        let reader = StreamReader::try_new(Cursor::new(ipc), None).unwrap();
        let schema = reader.schema();
        let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
        let input = match field("method").as_str().unwrap() {
            "hash_schema" => Input::Schema(schema),
            "hash_array" => Input::Array(Arc::clone(batches.first().unwrap().column(0))),
            "hash_stream" => Input::Stream(schema, batches),
            other => panic!("unknown method {other}"),
        };
        let flags = u8::from(field("ordered_columns").as_bool().unwrap())
            | (u8::from(field("ordered_struct_fields").as_bool().unwrap()) << 1_u8);
        gen.add(
            &id,
            &format!("starfix-python test literal input: {test}"),
            flags,
            field("include_metadata").as_bool().unwrap(),
            &input,
        );
    }
}

/// Hash of a vector as a consumer sees it: decoded from its IPC stream (without validation, as
/// some vectors are deliberately outside what arrow-rs validates).
fn round_trip_hash(method: &str, ipc: &str, config: HasherConfig) -> Vec<u8> {
    let bytes = BASE64_STANDARD.decode(ipc).unwrap();
    #[expect(unsafe_code, reason = "Decoding vectors arrow-rs would not validate")]
    // SAFETY: the stream was just written by arrow-rs from arrays with consistent buffers; only
    // the nullability rule for hidden list items may be relaxed.
    let reader = unsafe {
        StreamReader::try_new(Cursor::new(bytes), None)
            .unwrap()
            .with_skip_validation(true)
    };
    let schema = reader.schema();
    let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
    match method {
        "hash_schema" => ArrowDigester::hash_schema(&schema, config),
        "hash_record_batch" => ArrowDigester::hash_record_batch(batches.first().unwrap(), config),
        "hash_array" => ArrowDigester::hash_array_with_config(
            batches.first().unwrap().column(0).as_ref(),
            config,
        ),
        _ => {
            let mut digester = ArrowDigester::new(&schema, config);
            for b in &batches {
                digester.update(b);
            }
            digester.finalize()
        }
    }
}

/// `FixedSizeBinary(0)` with `n` rows and optional validity.
fn zero_width(n: usize, valid: Option<Vec<bool>>) -> ArrayRef {
    let data = ArrayData::builder(DataType::FixedSizeBinary(0))
        .len(n)
        .add_buffer(Buffer::from(Vec::<u8>::new()))
        .nulls(valid.map(NullBuffer::from))
        .build()
        .unwrap();
    make_array(data)
}

/// Group 27: the `hash_array` root (always nullable), `FixedSizeBinary(0)`, dictionaries of lists,
/// and nulls hidden under null list values of a non-nullable element. (Nested dictionaries are
/// tested in Rust only: arrow-rs cannot round-trip them through IPC.)
#[expect(
    clippy::too_many_lines,
    reason = "fixture definitions are long but linear"
)]
fn add_entry_points_and_late_review_coverage(gen: &mut Builder) {
    let col = |name: &str, array: ArrayRef, nullable: bool| {
        batch(
            vec![Field::new(name, array.data_type().clone(), nullable)],
            vec![array],
        )
    };

    // 27a. hash_array: the root is always nullable, so a validity buffer without nulls is
    // irrelevant and a null still counts.
    {
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let all_valid: ArrayRef = Arc::new(Int32Array::new(
            vec![1, 2, 3].into(),
            Some(NullBuffer::from(vec![true, true, true])),
        ));
        let with_null: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
        gen.add(
            "root_int32_no_validity_array_f0",
            "hash_array Int32 [1, 2, 3] without a validity buffer (the root is always nullable)",
            0x00,
            false,
            &Input::Array(values),
        );
        gen.add(
            "root_int32_all_valid_array_f0",
            "hash_array Int32 [1, 2, 3] with an all-valid validity buffer",
            0x00,
            false,
            &Input::Array(all_valid),
        );
        gen.add(
            "root_int32_with_null_array_f0",
            "hash_array Int32 [1, null, 3]",
            0x00,
            false,
            &Input::Array(with_null),
        );
        gen.relate(
            "equal",
            &[
                "root_int32_no_validity_array_f0",
                "root_int32_all_valid_array_f0",
            ],
            "the hash_array root is always nullable: a validity buffer without nulls changes nothing",
        );
        gen.relate(
            "different",
            &[
                "root_int32_no_validity_array_f0",
                "root_int32_with_null_array_f0",
            ],
            "a null in the root array counts",
        );
    }

    // 27b. FixedSizeBinary(0): no value bytes, but validity and the row count count.
    {
        gen.add(
            "fsb0_nullable_rb_f0",
            "z: FixedSizeBinary(0)? rows [b\"\", null, b\"\"]",
            0x00,
            false,
            &Input::Batch(col("z", zero_width(3, Some(vec![true, false, true])), true)),
        );
        gen.add(
            "fsb0_1row_rb_f0",
            "z: FixedSizeBinary(0) one row [b\"\"]",
            0x00,
            false,
            &Input::Batch(col("z", zero_width(1, None), false)),
        );
        gen.add(
            "fsb0_5rows_rb_f0",
            "z: FixedSizeBinary(0) five rows of b\"\"",
            0x00,
            false,
            &Input::Batch(col("z", zero_width(5, None), false)),
        );
        gen.add(
            "fsb0_nullable_array_f0",
            "hash_array FixedSizeBinary(0) [b\"\", null, b\"\"]",
            0x00,
            false,
            &Input::Array(zero_width(3, Some(vec![true, false, true]))),
        );
        gen.add(
            "fsb0_array_f0",
            "hash_array FixedSizeBinary(0) [b\"\", b\"\", b\"\"]",
            0x00,
            false,
            &Input::Array(zero_width(3, None)),
        );
        gen.relate(
            "different",
            &["fsb0_1row_rb_f0", "fsb0_5rows_rb_f0"],
            "zero-width values still count rows",
        );
        gen.relate(
            "different",
            &["fsb0_nullable_array_f0", "fsb0_array_f0"],
            "zero-width values still count nulls",
        );
    }

    // 27c. A dictionary of lists hashes as the decoded list.
    {
        let plain: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths(vec![2, 1, 2]),
            Arc::new(Int32Array::from(vec![1, 2, 3, 1, 2])),
            None,
        ));
        let dict_values: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths(vec![2, 1]),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            None,
        ));
        let dict: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0, 1, 0]), dict_values)
                .unwrap(),
        );
        gen.add(
            "dict_of_list_rb_f0",
            "c: Dictionary<Int8, List<Int32?>> keys [0, 1, 0] over [[1, 2], [3]]",
            0x00,
            false,
            &Input::Batch(col("c", Arc::clone(&dict), false)),
        );
        gen.add(
            "plain_list_rb_f0",
            "c: List<Int32?> [[1, 2], [3], [1, 2]]",
            0x00,
            false,
            &Input::Batch(col("c", Arc::clone(&plain), false)),
        );
        gen.add(
            "dict_of_list_array_f0",
            "hash_array of the dictionary of lists",
            0x00,
            false,
            &Input::Array(dict),
        );
        gen.add(
            "plain_list_array_f0",
            "hash_array of the plain list",
            0x00,
            false,
            &Input::Array(plain),
        );
        gen.relate(
            "equal",
            &["dict_of_list_rb_f0", "plain_list_rb_f0"],
            "a dictionary of lists is its decoded list",
        );
        gen.relate(
            "equal",
            &["dict_of_list_array_f0", "plain_list_array_f0"],
            "a dictionary of lists is its decoded list (hash_array)",
        );
    }

    // 27d. Nulls hidden under a null list value of a non-nullable element are not elements.
    for large in [false, true] {
        let kind = if large { "large_list" } else { "list" };
        let hidden = hidden_null_list(large, true);
        let clean = hidden_null_list(large, false);
        let ty = if large { "LargeList" } else { "List" };
        gen.add(
            &format!("hidden_null_{kind}_rb_f0"),
            &format!(
                "l: {ty}<Int32 non-null>? rows [[1], null, [4]]; the null row's slot covers a null item"
            ),
            0x00,
            false,
            &Input::Batch(col("l", hidden, true)),
        );
        gen.add(
            &format!("clean_{kind}_rb_f0"),
            &format!(
                "l: {ty}<Int32 non-null>? rows [[1], null, [4]]; the null row's slot is empty"
            ),
            0x00,
            false,
            &Input::Batch(col("l", clean, true)),
        );
    }
    gen.relate(
        "equal",
        &[
            "hidden_null_list_rb_f0",
            "clean_list_rb_f0",
            "hidden_null_large_list_rb_f0",
            "clean_large_list_rb_f0",
        ],
        "items under a null list value are not visited, for List and LargeList alike",
    );
}

/// `List` or `LargeList` with a non-nullable `Int32` item and rows `[[1], null, [4]]`; with
/// `hidden`, the null row's slot covers a null item (pyarrow can produce such arrays; arrow-rs
/// does not validate them, so they are built unchecked).
fn hidden_null_list(large: bool, hidden: bool) -> ArrayRef {
    let item = Arc::new(Field::new("item", DataType::Int32, false));
    let values: ArrayRef = if hidden {
        Arc::new(Int32Array::from(vec![Some(1), None, Some(4)]))
    } else {
        Arc::new(Int32Array::from(vec![1, 4]))
    };
    let ends: Vec<i64> = if hidden {
        vec![0, 1, 2, 3]
    } else {
        vec![0, 1, 1, 2]
    };
    let (dt, offsets) = if large {
        (DataType::LargeList(item), Buffer::from_vec(ends))
    } else {
        let small: Vec<i32> = ends.iter().map(|&e| i32::try_from(e).unwrap()).collect();
        (DataType::List(item), Buffer::from_vec(small))
    };
    let builder = ArrayData::builder(dt)
        .len(3)
        .add_buffer(offsets)
        .add_child_data(values.to_data())
        .nulls(Some(NullBuffer::from(vec![true, false, true])));
    #[expect(unsafe_code, reason = "Building an array arrow-rs would not validate")]
    // SAFETY: offsets, child length and validity are consistent; only the nullability rule for
    // hidden items is relaxed, which the hasher does not rely on.
    let data = unsafe { builder.build_unchecked() };
    make_array(data)
}

#[expect(clippy::panic, reason = "CLI tool — the append-only guard aborts")]
fn check_append_only(vectors: &[GoldenVector], relations: &[Relation]) {
    let raw = match fs::read_to_string(FIXTURE_PATH) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return, // no committed fixture yet
        Err(e) => panic!("append-only guard: cannot read {FIXTURE_PATH}: {e}"),
    };
    let existing: ExistingFixture = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("append-only guard: cannot parse {FIXTURE_PATH}: {e}"));
    for old in &existing.vectors {
        match vectors.iter().find(|v| v.id == old.id) {
            None => panic!(
                "append-only guard: committed vector '{}' is no longer generated",
                old.id
            ),
            // every field must be identical: input, method, options, description, hash
            Some(new) if new != old => panic!(
                "append-only guard: committed vector '{}' would change; vectors are permanent\n  committed: {old:?}\n  generated: {new:?}",
                old.id
            ),
            Some(_) => {}
        }
    }
    for old in &existing.relations {
        assert!(
            relations.contains(old),
            "append-only guard: committed relation {old:?} is no longer generated"
        );
    }
}

fn main() {
    let g = build();
    check_append_only(&g.vectors, &g.relations);
    let fixture = GoldenFixture {
        version: "0.5".to_owned(),
        format: 2,
        generated_by: "cargo run --bin emit_golden_format2".to_owned(),
        vectors: g.vectors,
        relations: g.relations,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
