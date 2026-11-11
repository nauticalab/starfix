//! Differential corpus for the legacy 0.4 mode (shared by `tests/legacy_v0_4_differential.rs`,
//! which runs it with `starfix::legacy::v0_4`, and `tools/legacy_v0_4_reference`, which runs it
//! with the published starfix 0.4.0 crate).
//!
//! Each group hashes many inputs and folds every outcome (the hash, or "panicked") into one
//! SHA-256 digest, so the two implementations can be compared without linking them into one
//! binary (both crates export the same uniffi symbols).

#![expect(clippy::unwrap_used, reason = "Okay in test")]

use std::fs;
use std::io::Cursor;
use std::panic::{self, catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow_ipc::reader::StreamReader;
use arrow_schema::{Field, Schema, SchemaRef};
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use sha2::{Digest as _, Sha256};

use crate::leaf_types::{leaf_types, make, wrap, Position};
use crate::nested_types::{all_types, arrow_type, build, row_sets, vals};

/// The four entry points of the 0.4.0 API (the `bool` is `include_metadata`).
pub struct Api {
    pub schema: fn(&Schema, bool) -> Vec<u8>,
    pub batch: fn(&RecordBatch, bool) -> Vec<u8>,
    pub stream: fn(&SchemaRef, &[RecordBatch], bool) -> Vec<u8>,
    pub array: fn(&dyn Array) -> Vec<u8>,
}

struct Group {
    cases: usize,
    hasher: Sha256,
}

impl Group {
    fn new() -> Self {
        Self {
            cases: 0,
            hasher: Sha256::new(),
        }
    }

    fn record<F>(&mut self, label: &str, call: F)
    where
        F: FnOnce() -> Vec<u8>,
    {
        self.hasher.update((label.len() as u64).to_le_bytes());
        self.hasher.update(label.as_bytes());
        match catch_unwind(AssertUnwindSafe(call)) {
            Ok(hash) => {
                self.hasher.update([1_u8]);
                self.hasher.update(&hash);
            }
            Err(_) => self.hasher.update([0_u8]),
        }
        self.cases = self.cases.checked_add(1).unwrap();
    }

    fn finish(self) -> (usize, String) {
        (self.cases, hex::encode(self.hasher.finalize()))
    }
}

fn decode(ipc_b64: &str) -> (SchemaRef, Vec<RecordBatch>) {
    let reader =
        StreamReader::try_new(Cursor::new(STANDARD.decode(ipc_b64).unwrap()), None).unwrap();
    let schema = reader.schema();
    (schema, reader.map(Result::unwrap).collect())
}

/// Inputs of the format-1 golden data: `include_metadata_v0.3.json` and
/// `python_literals_format1_inputs.json`, each hashed both with and without metadata.
fn format1_inputs(api: &Api, root: &Path, group: &mut Group) {
    for file in [
        "tests/golden/include_metadata_v0.3.json",
        "tests/golden/python_literals_format1_inputs.json",
    ] {
        let doc: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join(file)).unwrap()).unwrap();
        let list = doc
            .get("vectors")
            .or_else(|| doc.get("entries"))
            .and_then(serde_json::Value::as_array)
            .unwrap();
        for (i, v) in list.iter().enumerate() {
            let (schema, batches) = decode(
                v.get("ipc_b64")
                    .and_then(serde_json::Value::as_str)
                    .unwrap(),
            );
            for im in [false, true] {
                let label = format!("{file}#{i} im={im}");
                group.record(&format!("{label} schema"), || (api.schema)(&schema, im));
                group.record(&format!("{label} stream"), || {
                    (api.stream)(&schema, &batches, im)
                });
                if let Some(first) = batches.first() {
                    group.record(&format!("{label} batch"), || (api.batch)(first, im));
                    for (c, column) in first.columns().iter().enumerate() {
                        group.record(&format!("{label} array {c}"), || {
                            (api.array)(column.as_ref())
                        });
                    }
                }
            }
        }
    }
}

/// The 420 nested column types of the collision search, ~60 small tables each.
fn nested(api: &Api, group: &mut Group) {
    for node in &all_types() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c",
            arrow_type(node),
            node.nullable,
        )]));
        group.record(&format!("{node:?} schema"), || (api.schema)(&schema, false));
        let mut batches = Vec::new();
        for rows in row_sets(&vals(node)) {
            let column = build(node, &rows);
            group.record(&format!("{node:?} {rows:?} array"), || {
                (api.array)(column.as_ref())
            });
            let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
            group.record(&format!("{node:?} {rows:?} batch"), || {
                (api.batch)(&batch, false)
            });
            batches.push(batch);
        }
        group.record(&format!("{node:?} stream"), || {
            (api.stream)(&schema, &batches, false)
        });
    }
}

/// Every leaf type x nullability x five positions x three seed sets, whole, split into batches
/// of two rows, and sliced at offsets around byte and word boundaries.
fn leaves(api: &Api, group: &mut Group) {
    let seed_sets: [Vec<Option<i64>>; 3] = [
        vec![
            Some(1),
            Some(0),
            Some(5),
            Some(2),
            Some(3),
            Some(0),
            Some(1),
        ],
        vec![Some(1), Some(0), None, Some(2), Some(3), Some(0), None],
        vec![None, None, Some(0)],
    ];
    for dt in leaf_types() {
        for nullable in [false, true] {
            for position in [
                Position::Top,
                Position::InStruct,
                Position::InList,
                Position::InNullableStruct,
                Position::InNullableList,
            ] {
                for seeds in &seed_sets {
                    if !nullable && seeds.contains(&None) {
                        continue;
                    }
                    let label = format!("{dt:?} {nullable} {position:?} {seeds:?}");
                    let whole = wrap(position, &make(&dt, seeds), nullable, seeds);
                    group.record(&format!("{label} batch"), || (api.batch)(&whole, false));
                    let column: ArrayRef = Arc::clone(whole.column(0));
                    group.record(&format!("{label} array"), || (api.array)(column.as_ref()));
                    let parts: Vec<RecordBatch> = seeds
                        .chunks(2)
                        .map(|part| wrap(position, &make(&dt, part), nullable, part))
                        .collect();
                    let schema = whole.schema();
                    group.record(&format!("{label} split"), || {
                        (api.stream)(&schema, &parts, false)
                    });
                    for offset in [1_usize, 7, 8, 9, 16, 63, 64, 65] {
                        let mut padded = vec![Some(9); offset];
                        padded.extend(seeds.iter().copied());
                        let sliced = wrap(position, &make(&dt, &padded), nullable, &padded)
                            .slice(offset, seeds.len());
                        group.record(&format!("{label} slice {offset}"), || {
                            (api.batch)(&sliced, false)
                        });
                        let sliced_column: ArrayRef = Arc::clone(sliced.column(0));
                        group.record(&format!("{label} slice {offset} array"), || {
                            (api.array)(sliced_column.as_ref())
                        });
                    }
                }
            }
        }
    }
}

/// `(group name, number of cases, digest)` for every group. `root` is the starfix repository.
pub fn group_digests(api: &Api, root: &Path) -> Vec<(&'static str, usize, String)> {
    // expected panics (e.g. the 0.4.0 Boolean offset bug) would flood the output
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let run = |fill: &dyn Fn(&mut Group)| {
        let mut group = Group::new();
        fill(&mut group);
        group.finish()
    };
    let named = [
        ("format1_inputs", run(&|g| format1_inputs(api, root, g))),
        ("nested_types", run(&|g| nested(api, g))),
        ("leaf_sweep", run(&|g| leaves(api, g))),
    ];
    let out = named
        .into_iter()
        .map(|(name, (cases, digest))| (name, cases, digest))
        .collect();
    panic::set_hook(previous_hook);
    out
}
