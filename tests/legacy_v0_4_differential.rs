//! Differential test of the legacy 0.4 mode against the real starfix 0.4.0 release (crates.io).
//!
//! `starfix::legacy::v0_4` must reproduce the released crate byte for byte, including its
//! defects and its panics, on: every option-free golden input, every nested column type of the
//! collision search with ~60 small tables each, and every supported leaf type in five positions
//! with nulls, slices at many offsets and batch splits.

#[path = "common/leaf_types.rs"]
mod leaf_types;
#[path = "common/nested_types.rs"]
mod nested_types;

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::panic, reason = "Okay in test")]

    use std::fs;
    use std::io::Cursor;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{ArrayRef, RecordBatch};
    use arrow_ipc::reader::StreamReader;
    use arrow_schema::{Field, Schema, SchemaRef};
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;
    use starfix::legacy::v0_4 as legacy;
    use starfix_v0_4_release as release;

    use crate::leaf_types::{leaf_types, make, wrap, Position};
    use crate::nested_types::{all_types, arrow_type, build, row_sets, vals};

    /// The hash, or `None` if the call panicked.
    fn outcome<F>(f: F) -> Option<Vec<u8>>
    where
        F: FnOnce() -> Vec<u8>,
    {
        catch_unwind(AssertUnwindSafe(f)).ok()
    }

    fn same_batch(batch: &RecordBatch, include_metadata: bool) -> bool {
        outcome(|| {
            legacy::ArrowDigester::hash_record_batch(
                batch,
                legacy::HasherConfig { include_metadata },
            )
        }) == outcome(|| {
            release::ArrowDigester::hash_record_batch(
                batch,
                release::HasherConfig { include_metadata },
            )
        })
    }

    fn same_stream(schema: &SchemaRef, batches: &[RecordBatch], include_metadata: bool) -> bool {
        let ours = outcome(|| {
            let mut d =
                legacy::ArrowDigester::new(schema, legacy::HasherConfig { include_metadata });
            for b in batches {
                d.update(b);
            }
            d.finalize()
        });
        let theirs = outcome(|| {
            let mut d =
                release::ArrowDigester::new(schema, release::HasherConfig { include_metadata });
            for b in batches {
                d.update(b);
            }
            d.finalize()
        });
        ours == theirs
    }

    fn same_schema(schema: &Schema, include_metadata: bool) -> bool {
        outcome(|| {
            legacy::ArrowDigester::hash_schema(schema, legacy::HasherConfig { include_metadata })
        }) == outcome(|| {
            release::ArrowDigester::hash_schema(schema, release::HasherConfig { include_metadata })
        })
    }

    fn same_array(array: &ArrayRef) -> bool {
        outcome(|| legacy::ArrowDigester::hash_array(array.as_ref()))
            == outcome(|| release::ArrowDigester::hash_array(array.as_ref()))
    }

    #[test]
    fn golden_inputs_match_the_release() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/format2_v0.5.json"
        );
        let fixture: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let mut checked = 0_usize;
        for v in fixture
            .get("vectors")
            .and_then(serde_json::Value::as_array)
            .unwrap()
        {
            let flag = |name: &str| v.get(name).and_then(serde_json::Value::as_bool).unwrap();
            if flag("ordered_columns") || flag("ordered_struct_fields") {
                continue; // order options do not exist in 0.4
            }
            let ipc = STANDARD
                .decode(
                    v.get("ipc_b64")
                        .and_then(serde_json::Value::as_str)
                        .unwrap(),
                )
                .unwrap();
            let reader = StreamReader::try_new(Cursor::new(ipc), None).unwrap();
            let schema = reader.schema();
            let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
            let id = v.get("id").and_then(serde_json::Value::as_str).unwrap();
            for im in [false, true] {
                let same = match v.get("method").and_then(serde_json::Value::as_str).unwrap() {
                    "hash_schema" => same_schema(&schema, im),
                    "hash_record_batch" => same_batch(batches.first().unwrap(), im),
                    "hash_stream" => same_stream(&schema, &batches, im),
                    "hash_array" => same_array(batches.first().unwrap().column(0)),
                    other => panic!("{id}: unknown method {other}"),
                };
                assert!(
                    same,
                    "{id} (include_metadata {im}) differs from starfix 0.4.0"
                );
            }
            checked += 1;
        }
        assert!(checked > 200, "only {checked} golden inputs checked");
    }

    #[test]
    fn nested_types_match_the_release() {
        for node in &all_types() {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "c",
                arrow_type(node),
                node.nullable,
            )]));
            assert!(same_schema(&schema, false), "{node:?}: schema");
            let mut batches = Vec::new();
            for rows in row_sets(&vals(node)) {
                let column = build(node, &rows);
                assert!(same_array(&column), "{node:?} {rows:?}: hash_array");
                let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
                assert!(same_batch(&batch, false), "{node:?} {rows:?}: record batch");
                batches.push(batch);
            }
            assert!(same_stream(&schema, &batches, false), "{node:?}: stream");
        }
    }

    #[test]
    fn leaf_types_match_the_release() {
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
                        let what = format!("{dt:?} nullable {nullable} {position:?} {seeds:?}");
                        let whole = wrap(position, &make(&dt, seeds), nullable, seeds);
                        assert!(same_batch(&whole, false), "{what}");
                        assert!(same_array(whole.column(0)), "{what}: hash_array");
                        let parts: Vec<RecordBatch> = seeds
                            .chunks(2)
                            .map(|part| wrap(position, &make(&dt, part), nullable, part))
                            .collect();
                        assert!(same_stream(&whole.schema(), &parts, false), "{what}: split");
                        for offset in [1_usize, 7, 8, 9, 16, 63, 64, 65] {
                            let mut padded = vec![Some(9); offset];
                            padded.extend(seeds.iter().copied());
                            let sliced = wrap(position, &make(&dt, &padded), nullable, &padded)
                                .slice(offset, seeds.len());
                            assert!(same_batch(&sliced, false), "{what}: slice {offset}");
                            assert!(same_array(sliced.column(0)), "{what}: slice {offset} array");
                        }
                    }
                }
            }
        }
    }
}
