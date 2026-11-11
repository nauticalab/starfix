//! Byte-level tests of hash format 2 (ITL-898; `docs/byte-layout-spec.md`).
//!
//! Every expected value is assembled here from the spec's rules, byte by byte.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        Array as _, ArrayRef, BooleanArray, Int32Array, Int32Builder, LargeStringArray, ListArray,
        ListBuilder, RecordBatch, RecordBatchOptions, StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use pretty_assertions::assert_eq;
    use sha2::{Digest as _, Sha256};
    use starfix::{ArrowDigester, HasherConfig};

    const PREFIX: [u8; 3] = [0, 0, 2];

    /// Entry-point kinds: schema digest, record-batch / stream final digest, `hash_array`.
    const SCHEMA: u8 = b'S';
    const BATCH: u8 = b'R';
    const ARRAY: u8 = b'A';

    fn header(kind: u8, flags: u8) -> Vec<u8> {
        let mut h = vec![0x00_u8];
        h.extend_from_slice(b"starfix");
        h.extend_from_slice(&PREFIX);
        h.push(kind);
        h.push(flags);
        h
    }

    fn sha(parts: &[&[u8]]) -> Vec<u8> {
        let mut h = Sha256::new();
        for p in parts {
            h.update(p);
        }
        h.finalize().to_vec()
    }

    fn out(digest: &[u8]) -> Vec<u8> {
        let mut v = PREFIX.to_vec();
        v.extend_from_slice(digest);
        v
    }

    fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|&(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    fn with_metadata() -> HasherConfig {
        HasherConfig {
            include_metadata: true,
            ..HasherConfig::default()
        }
    }

    #[test]
    fn header_is_written_at_flags_zero() {
        let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);
        let json = r#"{"v":{"data_type":"Int32","nullable":false}}"#;
        assert_eq!(
            ArrowDigester::hash_schema(&schema, HasherConfig::default()),
            out(&sha(&[&header(SCHEMA, 0), json.as_bytes()]))
        );
        let ordered = HasherConfig {
            ordered_columns: true,
            ..HasherConfig::default()
        };
        let ordered_json = r#"[{"data_type":"Int32","name":"v","nullable":false}]"#;
        assert_eq!(
            ArrowDigester::hash_schema(&schema, ordered),
            out(&sha(&[&header(SCHEMA, 1), ordered_json.as_bytes()]))
        );
    }

    #[test]
    fn record_batch_with_row_count_and_leaf_entries() {
        // a: Int32 non-null [1, 2]; b: LargeUtf8 nullable ["x", null]
        let b = batch(
            vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::LargeUtf8, true),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(LargeStringArray::from(vec![Some("x"), None])),
            ],
        );
        let json = r#"{"a":{"data_type":"Int32","nullable":false},"b":{"data_type":"LargeUtf8","nullable":true}}"#;
        let schema_digest = sha(&[&header(SCHEMA, 0), json.as_bytes()]);
        let expected = out(&sha(&[
            &header(BATCH, 0),
            &schema_digest,
            &2_u64.to_le_bytes(), // row count
            // entry "a": values
            &sha(&[&1_i32.to_le_bytes(), &2_i32.to_le_bytes()]),
            // entry "b": validity [1, 0], values
            &2_u64.to_le_bytes(),
            &[0b01],
            &sha(&[&1_u64.to_le_bytes(), b"x"]),
        ]));
        assert_eq!(
            ArrowDigester::hash_record_batch(&b, HasherConfig::default()),
            expected
        );
    }

    #[test]
    fn list_node_holds_lengths_element_node_holds_values() {
        // c: List<Int32 nullable> nullable, rows [[1, null], null, []]
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.values().append_value(1);
        builder.values().append_null();
        builder.append(true);
        builder.append(false);
        builder.append(true);
        let list = builder.finish();
        let b = batch(
            vec![Field::new("c", list.data_type().clone(), true)],
            vec![Arc::new(list)],
        );
        let json = r#"{"c":{"data_type":{"LargeList":{"data_type":"Int32","nullable":true}},"nullable":true}}"#;
        let expected = out(&sha(&[
            &header(BATCH, 0),
            &sha(&[&header(SCHEMA, 0), json.as_bytes()]),
            &3_u64.to_le_bytes(),
            // entry "c": validity [1, 0, 1], lengths of valid rows [2, 0]
            &3_u64.to_le_bytes(),
            &[0b101],
            &sha(&[&2_u64.to_le_bytes(), &0_u64.to_le_bytes()]),
            // entry "c/": element validity [1, 0], values [1]
            &2_u64.to_le_bytes(),
            &[0b01],
            &sha(&[&1_i32.to_le_bytes()]),
        ]));
        assert_eq!(
            ArrowDigester::hash_record_batch(&b, HasherConfig::default()),
            expected
        );
    }

    #[test]
    fn nullable_struct_node_has_validity() {
        // s: Struct<a: Int32 nullable> nullable, rows [null, {a: null}, {a: 1}]
        let s = StructArray::new(
            Fields::from(vec![Field::new("a", DataType::Int32, true)]),
            vec![Arc::new(Int32Array::from(vec![Some(5), None, Some(1)]))],
            Some(NullBuffer::from(vec![false, true, true])),
        );
        let b = batch(
            vec![Field::new("s", s.data_type().clone(), true)],
            vec![Arc::new(s)],
        );
        let json = r#"{"s":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":true}]},"nullable":true}}"#;
        let expected = out(&sha(&[
            &header(BATCH, 0),
            &sha(&[&header(SCHEMA, 0), json.as_bytes()]),
            &3_u64.to_le_bytes(),
            &3_u64.to_le_bytes(), // "s": [0, 1, 1]
            &[0b110],
            &3_u64.to_le_bytes(), // "s/a": [1, 0, 1] AND [0, 1, 1]
            &[0b100],
            &sha(&[&1_i32.to_le_bytes()]),
        ]));
        assert_eq!(
            ArrowDigester::hash_record_batch(&b, HasherConfig::default()),
            expected
        );
    }

    #[test]
    fn boolean_one_byte_per_valid_value() {
        let a = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let expected = out(&sha(&[
            &header(ARRAY, 0),
            br#""Boolean""#,
            &3_u64.to_le_bytes(), // array length
            &3_u64.to_le_bytes(), // entry "": validity [1, 0, 1]
            &[0b101],
            &sha(&[&[0x01, 0x00]]),
        ]));
        assert_eq!(ArrowDigester::hash_array(&a), expected);
    }

    #[test]
    fn hash_array_non_nullable() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let expected = out(&sha(&[
            &header(ARRAY, 0),
            br#""Int32""#,
            &3_u64.to_le_bytes(),
            // the root is always nullable: validity [1, 1, 1]
            &3_u64.to_le_bytes(),
            &[0b111],
            &sha(&[
                &1_i32.to_le_bytes(),
                &2_i32.to_le_bytes(),
                &3_i32.to_le_bytes(),
            ]),
        ]));
        assert_eq!(ArrowDigester::hash_array(&a), expected);
    }

    #[test]
    fn row_count_of_zero_column_tables_and_empty_structs() {
        let empty = Arc::new(Schema::empty());
        let zero_cols = |rows: usize| {
            RecordBatch::try_new_with_options(
                Arc::clone(&empty),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .unwrap()
        };
        assert_eq!(
            ArrowDigester::hash_record_batch(&zero_cols(3), HasherConfig::default()),
            out(&sha(&[
                &header(BATCH, 0),
                &sha(&[&header(SCHEMA, 0), b"{}"]),
                &3_u64.to_le_bytes()
            ]))
        );
        assert_ne!(
            ArrowDigester::hash_record_batch(&zero_cols(0), HasherConfig::default()),
            ArrowDigester::hash_record_batch(&zero_cols(3), HasherConfig::default())
        );
        let empty_struct = |rows: usize| {
            let s = StructArray::new_empty_fields(rows, None);
            batch(
                vec![Field::new("e", s.data_type().clone(), false)],
                vec![Arc::new(s)],
            )
        };
        assert_ne!(
            ArrowDigester::hash_record_batch(&empty_struct(1), HasherConfig::default()),
            ArrowDigester::hash_record_batch(&empty_struct(2), HasherConfig::default())
        );
    }

    fn map_schema(entries: &str, key: &str, value: &str, entries_meta: bool) -> Schema {
        let value_type = DataType::Struct(Fields::from(vec![
            Field::new("z", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]));
        let mut entries_field = Field::new(
            entries,
            DataType::Struct(Fields::from(vec![
                Field::new(key, DataType::Utf8, false),
                Field::new(value, value_type, true).with_metadata(meta(&[("unit", "kg")])),
            ])),
            false,
        );
        if entries_meta {
            entries_field = entries_field.with_metadata(meta(&[("role", "pairs")]));
        }
        Schema::new(vec![Field::new(
            "m",
            DataType::Map(Arc::new(entries_field), false),
            true,
        )])
    }

    #[test]
    fn map_json_omits_field_names_and_entries_metadata() {
        let json = r#"{"m":{"data_type":{"Map":{"key":{"data_type":"LargeUtf8","nullable":false},"keys_sorted":false,"value":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false},{"data_type":"Int32","name":"z","nullable":false}]},"nullable":true}}},"nullable":true}}"#;
        let standard = map_schema("entries", "key", "value", false);
        assert_eq!(
            ArrowDigester::hash_schema(&standard, HasherConfig::default()),
            out(&sha(&[&header(SCHEMA, 0), json.as_bytes()]))
        );
        // names of the entries/key/value fields and the entries metadata are not hashed
        let custom = map_schema("kv", "k", "v", true);
        for config in [HasherConfig::default(), with_metadata()] {
            assert_eq!(
                ArrowDigester::hash_schema(&standard, config),
                ArrowDigester::hash_schema(&custom, config)
            );
        }
        // the value field's metadata is hashed at m//value
        let phase2 = r#"{"fields":{"m//value":{"unit":"kg"}}}"#;
        assert_eq!(
            ArrowDigester::hash_schema(&standard, with_metadata()),
            out(&sha(&[
                &header(SCHEMA, 0),
                json.as_bytes(),
                phase2.as_bytes()
            ]))
        );
    }

    #[test]
    fn list_element_name_is_not_hashed() {
        let list = |name: &str| {
            Schema::new(vec![Field::new(
                "l",
                DataType::List(Arc::new(
                    Field::new(name, DataType::Int32, true).with_metadata(meta(&[("u", "s")])),
                )),
                true,
            )])
        };
        for config in [HasherConfig::default(), with_metadata()] {
            assert_eq!(
                ArrowDigester::hash_schema(&list("item"), config),
                ArrowDigester::hash_schema(&list("element"), config)
            );
        }
    }

    #[test]
    fn metadata_paths_are_node_paths() {
        // struct column "" with child "a": node path of the child is "" + "/" + "a" = "/a"
        let schema = Schema::new(vec![Field::new(
            "",
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, false).with_metadata(meta(&[("u", "m")]))
            ])),
            false,
        )]);
        let json = r#"{"":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false}]},"nullable":false}}"#;
        let phase2 = r#"{"fields":{"/a":{"u":"m"}}}"#;
        assert_eq!(
            ArrowDigester::hash_schema(&schema, with_metadata()),
            out(&sha(&[
                &header(SCHEMA, 0),
                json.as_bytes(),
                phase2.as_bytes()
            ]))
        );
    }

    #[test]
    fn extension_metadata_is_hashed_as_field_metadata() {
        let schema = Schema::new(vec![Field::new("u", DataType::FixedSizeBinary(16), true)
            .with_metadata(meta(&[
                ("ARROW:extension:name", "arrow.uuid"),
                ("ARROW:extension:metadata", ""),
            ]))]);
        let json = r#"{"u":{"data_type":{"FixedSizeBinary":16},"nullable":true}}"#;
        let phase2 = r#"{"fields":{"u":{"ARROW:extension:metadata":"","ARROW:extension:name":"arrow.uuid"}}}"#;
        assert_eq!(
            ArrowDigester::hash_schema(&schema, with_metadata()),
            out(&sha(&[
                &header(SCHEMA, 0),
                json.as_bytes(),
                phase2.as_bytes()
            ]))
        );
    }

    fn panics<F>(f: F) -> bool
    where
        F: FnOnce(),
    {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }

    #[test]
    fn ambiguous_paths_rejected_in_every_mode() {
        let dup = batch(
            vec![
                Field::new("a", DataType::Int32, false),
                Field::new("a", DataType::Int32, false),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![2])),
            ],
        );
        assert!(panics(|| {
            ArrowDigester::hash_record_batch(&dup, HasherConfig::default());
        }));
        assert!(panics(|| {
            ArrowDigester::hash_schema(&dup.schema(), HasherConfig::default());
        }));
        let aliased = Schema::new(vec![
            Field::new("a/b", DataType::Int32, false),
            Field::new(
                "a",
                DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int32, false)])),
                false,
            ),
        ]);
        assert!(panics(|| {
            ArrowDigester::hash_schema(&aliased, HasherConfig::default());
        }));
        let fine = Schema::new(vec![Field::new("km/h", DataType::Int32, false)]);
        assert!(!panics(|| {
            ArrowDigester::hash_schema(&fine, HasherConfig::default());
        }));
    }

    #[test]
    fn non_portable_types_rejected_in_every_mode() {
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]);
        assert!(panics(|| {
            ArrowDigester::hash_schema(&schema, HasherConfig::default());
        }));
    }

    #[test]
    fn nested_list_lengths_stay_with_their_list() {
        // c: List<List<Int32>> (default nullable items); rows [[1]],[[2],[3]] vs [[1],[2]],[[3]]
        let mk = |rows: &[Vec<Vec<i32>>]| {
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
        };
        assert_ne!(
            ArrowDigester::hash_record_batch(
                &mk(&[vec![vec![1]], vec![vec![2], vec![3]]]),
                HasherConfig::default()
            ),
            ArrowDigester::hash_record_batch(
                &mk(&[vec![vec![1], vec![2]], vec![vec![3]]]),
                HasherConfig::default()
            )
        );
    }

    #[test]
    fn list_of_struct_entries() {
        // l: List<Struct<a: Int32>> non-null, element nullable, one row [null, {a: 4}]
        let st = StructArray::new(
            Fields::from(vec![Field::new("a", DataType::Int32, false)]),
            vec![Arc::new(Int32Array::from(vec![0, 4]))],
            Some(NullBuffer::from(vec![false, true])),
        );
        let item = Arc::new(Field::new("item", st.data_type().clone(), true));
        let col: ArrayRef = Arc::new(ListArray::new(
            item,
            OffsetBuffer::<i32>::from_lengths([2]),
            Arc::new(st),
            None,
        ));
        let b = batch(
            vec![Field::new("l", col.data_type().clone(), false)],
            vec![col],
        );
        let json = r#"{"l":{"data_type":{"LargeList":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":false}]},"nullable":true}},"nullable":false}}"#;
        let expected = out(&sha(&[
            &header(BATCH, 0),
            &sha(&[&header(SCHEMA, 0), json.as_bytes()]),
            &1_u64.to_le_bytes(),
            // entry "l": lengths [2]
            &sha(&[&2_u64.to_le_bytes()]),
            // entry "l/": element (struct) validity [0, 1]
            &2_u64.to_le_bytes(),
            &[0b10],
            // entry "l//a": values of valid elements [4]
            &sha(&[&4_i32.to_le_bytes()]),
        ]));
        assert_eq!(
            ArrowDigester::hash_record_batch(&b, HasherConfig::default()),
            expected
        );
    }
}
