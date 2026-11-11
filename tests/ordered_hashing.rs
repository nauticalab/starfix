//! Tests for the order-sensitive and nested-validity options of `HasherConfig` (ITL-898).
//!
//! The byte layout is specified in `docs/byte-layout-spec.md`, Section 9.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::indexing_slicing, reason = "Okay in test")]

    use std::collections::{HashMap, HashSet};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, BooleanArray, DictionaryArray, Int32Array, Int32Builder, LargeListArray,
        LargeStringArray, ListArray, ListBuilder, MapBuilder, RecordBatch, StringArray,
        StringBuilder, StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::Int32Type;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use pretty_assertions::assert_eq;
    use sha2::{Digest as _, Sha256};
    use starfix::{ArrowDigester, HasherConfig};

    const VERSION: [u8; 3] = [0, 0, 1];

    // ── helpers ────────────────────────────────────────────────────────────

    /// Build a config from the option flags byte of the spec (bit0 columns, bit1 struct
    /// fields, bit2 nested validity).
    fn cfg(flags: u8) -> HasherConfig {
        HasherConfig {
            include_metadata: false,
            ordered_columns: flags & 0x01 != 0,
            ordered_struct_fields: flags & 0x02 != 0,
            nested_validity: flags & 0x04 != 0,
        }
    }

    fn tag(flags: u8) -> Vec<u8> {
        let mut t = vec![0x00_u8];
        t.extend_from_slice(b"starfix-options-v1");
        t.push(flags);
        t
    }

    fn sha(parts: &[&[u8]]) -> Vec<u8> {
        let mut h = Sha256::new();
        for p in parts {
            h.update(p);
        }
        h.finalize().to_vec()
    }

    fn versioned(digest: &[u8]) -> Vec<u8> {
        let mut v = VERSION.to_vec();
        v.extend_from_slice(digest);
        v
    }

    fn batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    fn rb_hash(b: &RecordBatch, c: HasherConfig) -> String {
        hex::encode(ArrowDigester::hash_record_batch(b, c))
    }

    fn schema_hash(s: &Schema, c: HasherConfig) -> String {
        hex::encode(ArrowDigester::hash_schema(s, c))
    }

    fn array_hash(a: &dyn Array, c: HasherConfig) -> String {
        hex::encode(ArrowDigester::hash_array_with_config(a, c))
    }

    /// Two columns: `a: Int32 [1, 2]`, `b: LargeUtf8 ["x", null]`, in the given order.
    fn two_col_batch(a_first: bool) -> RecordBatch {
        let a_field = Field::new("a", DataType::Int32, false);
        let b_field = Field::new("b", DataType::LargeUtf8, true);
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let b: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("x"), None]));
        if a_first {
            batch(vec![a_field, b_field], vec![a, b])
        } else {
            batch(vec![b_field, a_field], vec![b, a])
        }
    }

    /// Struct with children `x: Int32`, `y: Boolean` (non-nullable), in the given order.
    fn xy_struct(x_first: bool) -> StructArray {
        let x = (
            Arc::new(Field::new("x", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        );
        let y = (
            Arc::new(Field::new("y", DataType::Boolean, false)),
            Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef,
        );
        if x_first {
            StructArray::from(vec![x, y])
        } else {
            StructArray::from(vec![y, x])
        }
    }

    fn struct_col_batch(x_first: bool) -> RecordBatch {
        let s = xy_struct(x_first);
        batch(
            vec![Field::new("s", s.data_type().clone(), false)],
            vec![Arc::new(s)],
        )
    }

    /// A batch with a column permutation AND a struct field permutation.
    fn cols_and_struct_batch(cols_ab: bool, x_first: bool) -> RecordBatch {
        let s = xy_struct(x_first);
        let s_field = Field::new("s", s.data_type().clone(), false);
        let n_field = Field::new("n", DataType::Int32, false);
        let s_col: ArrayRef = Arc::new(s);
        let n_col: ArrayRef = Arc::new(Int32Array::from(vec![7, 8]));
        if cols_ab {
            batch(vec![n_field, s_field], vec![n_col, s_col])
        } else {
            batch(vec![s_field, n_field], vec![s_col, n_col])
        }
    }

    fn assert_all_distinct(hashes: &[String], what: &str) {
        let set: HashSet<&String> = hashes.iter().collect();
        assert_eq!(
            set.len(),
            hashes.len(),
            "{what}: hashes not pairwise distinct"
        );
    }

    // ── defaults are unchanged ──────────────────────────────────────────────

    #[test]
    fn default_config_has_all_options_off() {
        let c = HasherConfig::default();
        assert!(!c.include_metadata && !c.ordered_columns);
        assert!(!c.ordered_struct_fields && !c.nested_validity);
    }

    #[test]
    fn hash_array_equals_hash_array_with_default_config() {
        let a = Int32Array::from(vec![Some(1), None]);
        assert_eq!(
            ArrowDigester::hash_array(&a),
            ArrowDigester::hash_array_with_config(&a, HasherConfig::default())
        );
    }

    // ── ordered_columns ─────────────────────────────────────────────────────

    #[test]
    fn column_order_ignored_when_off() {
        for flags in [0x00, 0x02, 0x04, 0x06] {
            assert_eq!(
                rb_hash(&two_col_batch(true), cfg(flags)),
                rb_hash(&two_col_batch(false), cfg(flags)),
                "flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn column_order_matters_when_on() {
        for flags in [0x01, 0x03, 0x05, 0x07] {
            assert_ne!(
                rb_hash(&two_col_batch(true), cfg(flags)),
                rb_hash(&two_col_batch(false), cfg(flags)),
                "record batch, flags {flags:#04x}"
            );
            assert_ne!(
                schema_hash(&two_col_batch(true).schema(), cfg(flags)),
                schema_hash(&two_col_batch(false).schema(), cfg(flags)),
                "schema, flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn ordered_columns_schema_digest_bytes() {
        // [b: Int32 non-null, a: LargeUtf8 nullable]
        let schema = Schema::new(vec![
            Field::new("b", DataType::Int32, false),
            Field::new("a", DataType::Utf8, true),
        ]);
        let json = r#"[{"data_type":"Int32","name":"b","nullable":false},{"data_type":"LargeUtf8","name":"a","nullable":true}]"#;
        let expected = versioned(&sha(&[&tag(0x01), json.as_bytes()]));
        assert_eq!(ArrowDigester::hash_schema(&schema, cfg(0x01)), expected);
    }

    #[test]
    fn ordered_columns_record_batch_bytes() {
        // [b: Int32 non-null [5], a: Int32 non-null [6]]
        let b = batch(
            vec![
                Field::new("b", DataType::Int32, false),
                Field::new("a", DataType::Int32, false),
            ],
            vec![
                Arc::new(Int32Array::from(vec![5])),
                Arc::new(Int32Array::from(vec![6])),
            ],
        );
        let json = r#"[{"data_type":"Int32","name":"b","nullable":false},{"data_type":"Int32","name":"a","nullable":false}]"#;
        let schema_digest = sha(&[&tag(0x01), json.as_bytes()]);
        // entries in path order: "a", "b"
        let expected = versioned(&sha(&[
            &schema_digest,
            &sha(&[&6_i32.to_le_bytes()]),
            &sha(&[&5_i32.to_le_bytes()]),
        ]));
        assert_eq!(ArrowDigester::hash_record_batch(&b, cfg(0x01)), expected);
    }

    #[test]
    fn streaming_rejects_reordered_batch_when_columns_ordered() {
        let mut d = ArrowDigester::new(&two_col_batch(true).schema(), cfg(0x01));
        d.update(&two_col_batch(true));
        let res = catch_unwind(AssertUnwindSafe(|| {
            d.update(&two_col_batch(false));
        }));
        assert!(res.is_err(), "reordered batch must be rejected");
    }

    #[test]
    fn streaming_accepts_reordered_batch_when_columns_unordered() {
        let mut d1 = ArrowDigester::new(&two_col_batch(true).schema(), cfg(0x06));
        d1.update(&two_col_batch(true));
        d1.update(&two_col_batch(false));
        let mut d2 = ArrowDigester::new(&two_col_batch(true).schema(), cfg(0x06));
        d2.update(&two_col_batch(true));
        d2.update(&two_col_batch(true));
        assert_eq!(d1.finalize(), d2.finalize());
    }

    // ── ordered_struct_fields ───────────────────────────────────────────────

    #[test]
    fn struct_order_ignored_when_off() {
        for flags in [0x00, 0x01, 0x04, 0x05] {
            assert_eq!(
                rb_hash(&struct_col_batch(true), cfg(flags)),
                rb_hash(&struct_col_batch(false), cfg(flags)),
                "flags {flags:#04x}"
            );
            assert_eq!(
                array_hash(&xy_struct(true), cfg(flags)),
                array_hash(&xy_struct(false), cfg(flags)),
                "hash_array, flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn struct_order_matters_when_on() {
        for flags in [0x02, 0x03, 0x06, 0x07] {
            assert_ne!(
                rb_hash(&struct_col_batch(true), cfg(flags)),
                rb_hash(&struct_col_batch(false), cfg(flags)),
                "flags {flags:#04x}"
            );
            assert_ne!(
                schema_hash(&struct_col_batch(true).schema(), cfg(flags)),
                schema_hash(&struct_col_batch(false).schema(), cfg(flags)),
                "schema, flags {flags:#04x}"
            );
            assert_ne!(
                array_hash(&xy_struct(true), cfg(flags)),
                array_hash(&xy_struct(false), cfg(flags)),
                "hash_array, flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn all_four_order_combinations() {
        // Each combination: equal/different exactly as the two switches predict.
        for flags in [0x00_u8, 0x01, 0x02, 0x03] {
            let base = rb_hash(&cols_and_struct_batch(true, true), cfg(flags));
            let cols_perm = rb_hash(&cols_and_struct_batch(false, true), cfg(flags));
            let struct_perm = rb_hash(&cols_and_struct_batch(true, false), cfg(flags));
            assert_eq!(base == cols_perm, flags & 0x01 == 0, "cols, flags {flags}");
            assert_eq!(
                base == struct_perm,
                flags & 0x02 == 0,
                "struct, flags {flags}"
            );
        }
        // And the four combinations are pairwise distinct for the same input.
        let all: Vec<String> = [0x00_u8, 0x01, 0x02, 0x03]
            .iter()
            .map(|&f| rb_hash(&cols_and_struct_batch(true, true), cfg(f)))
            .collect();
        assert_all_distinct(&all, "four order combinations");
    }

    #[test]
    fn ordered_struct_schema_json_keeps_declaration_order() {
        let schema = struct_col_batch(false).schema(); // s: Struct<y, x>
        let json = r#"{"s":{"data_type":{"Struct":[{"data_type":"Boolean","name":"y","nullable":false},{"data_type":"Int32","name":"x","nullable":false}]},"nullable":false}}"#;
        let expected = versioned(&sha(&[&tag(0x02), json.as_bytes()]));
        assert_eq!(ArrowDigester::hash_schema(&schema, cfg(0x02)), expected);
    }

    #[test]
    fn ordered_struct_hash_array_bytes() {
        // StructArray [{z: 1, a: 2}] non-nullable children, declared order z, a
        let s = StructArray::from(vec![
            (
                Arc::new(Field::new("z", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![2])) as ArrayRef,
            ),
        ]);
        let json = r#"{"Struct":[{"data_type":"Int32","name":"z","nullable":false},{"data_type":"Int32","name":"a","nullable":false}]}"#;
        let expected = versioned(&sha(&[
            &tag(0x02),
            json.as_bytes(),
            &sha(&[&2_i32.to_le_bytes()]), // entry "a"
            &sha(&[&1_i32.to_le_bytes()]), // entry "z"
        ]));
        assert_eq!(
            ArrowDigester::hash_array_with_config(&s, cfg(0x02)),
            expected
        );
    }

    fn nested_struct_schema(inner_ab: bool) -> Schema {
        let a = Field::new("a", DataType::Int32, false);
        let b = Field::new("b", DataType::Int32, false);
        let inner = if inner_ab {
            Fields::from(vec![a, b])
        } else {
            Fields::from(vec![b, a])
        };
        let p = Field::new("p", DataType::Struct(inner), false);
        let outer = DataType::Struct(Fields::from(vec![
            Field::new("q", DataType::Int32, false),
            p,
        ]));
        Schema::new(vec![Field::new("s", outer, false)])
    }

    fn ab_struct_type(inner_ab: bool) -> DataType {
        let a = Field::new("a", DataType::Int32, false);
        let b = Field::new("b", DataType::Utf8, true);
        if inner_ab {
            DataType::Struct(Fields::from(vec![a, b]))
        } else {
            DataType::Struct(Fields::from(vec![b, a]))
        }
    }

    fn list_of_struct_type(inner_ab: bool, large: bool) -> DataType {
        let item = Arc::new(Field::new("item", ab_struct_type(inner_ab), true));
        if large {
            DataType::LargeList(item)
        } else {
            DataType::List(item)
        }
    }

    #[test]
    fn nested_struct_order_at_every_depth() {
        let on = cfg(0x02);
        let off = cfg(0x00);
        // struct in struct, only the inner level permuted
        let s1 = nested_struct_schema(true);
        let s2 = nested_struct_schema(false);
        assert_ne!(schema_hash(&s1, on), schema_hash(&s2, on));
        assert_eq!(schema_hash(&s1, off), schema_hash(&s2, off));

        // struct in list / large list; large and small agree
        let mk = |ab: bool, large: bool| {
            Schema::new(vec![Field::new("l", list_of_struct_type(ab, large), true)])
        };
        assert_ne!(
            schema_hash(&mk(true, false), on),
            schema_hash(&mk(false, false), on)
        );
        assert_eq!(
            schema_hash(&mk(true, false), off),
            schema_hash(&mk(false, false), off)
        );
        assert_eq!(
            schema_hash(&mk(true, false), on),
            schema_hash(&mk(true, true), on)
        );

        // struct in fixed-size list (schema only)
        let fsl = |ab: bool| {
            let item = Arc::new(Field::new("item", ab_struct_type(ab), true));
            Schema::new(vec![Field::new(
                "f",
                DataType::FixedSizeList(item, 2),
                false,
            )])
        };
        assert_ne!(schema_hash(&fsl(true), on), schema_hash(&fsl(false), on));
        assert_eq!(schema_hash(&fsl(true), off), schema_hash(&fsl(false), off));

        // map value struct (schema only)
        let map = |ab: bool| {
            let item = Field::new("value", ab_struct_type(ab), true);
            let entries = Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", item.data_type().clone(), true),
                ])),
                false,
            );
            Schema::new(vec![Field::new(
                "m",
                DataType::Map(Arc::new(entries), false),
                true,
            )])
        };
        assert_ne!(schema_hash(&map(true), on), schema_hash(&map(false), on));
        assert_eq!(schema_hash(&map(true), off), schema_hash(&map(false), off));

        // dictionary value type is a struct-free leaf here; dictionary of list of struct
        // is not constructible as a column type, so the list case above covers containers.
    }

    #[test]
    fn list_of_struct_data_reorder() {
        // l: List<Struct<a: Int32, b: Utf8?>?>? rows [[{a:1,b:"x"}], null, []]
        let mk = |ab: bool, large: bool| {
            let a: ArrayRef = Arc::new(Int32Array::from(vec![1]));
            let b: ArrayRef = Arc::new(StringArray::from(vec![Some("x")]));
            let af = Arc::new(Field::new("a", DataType::Int32, false));
            let bf = Arc::new(Field::new("b", DataType::Utf8, true));
            let st = if ab {
                StructArray::from(vec![(af, a), (bf, b)])
            } else {
                StructArray::from(vec![(bf, b), (af, a)])
            };
            let item = Arc::new(Field::new("item", st.data_type().clone(), true));
            let offsets = [0_usize, 1, 1, 1];
            let nulls = Some(NullBuffer::from(vec![true, false, true]));
            let col: ArrayRef = if large {
                Arc::new(LargeListArray::new(
                    item,
                    OffsetBuffer::<i64>::from_lengths(offsets.windows(2).map(|w| w[1] - w[0])),
                    Arc::new(st),
                    nulls,
                ))
            } else {
                Arc::new(ListArray::new(
                    item,
                    OffsetBuffer::<i32>::from_lengths(offsets.windows(2).map(|w| w[1] - w[0])),
                    Arc::new(st),
                    nulls,
                ))
            };
            batch(
                vec![Field::new("l", col.data_type().clone(), true)],
                vec![col],
            )
        };
        for flags in 0x00_u8..=0x07 {
            let same =
                rb_hash(&mk(true, false), cfg(flags)) == rb_hash(&mk(false, false), cfg(flags));
            assert_eq!(same, flags & 0x02 == 0, "flags {flags:#04x}");
            assert_eq!(
                rb_hash(&mk(true, false), cfg(flags)),
                rb_hash(&mk(true, true), cfg(flags)),
                "list vs large list, flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn streaming_rejects_reordered_struct_batch_when_struct_fields_ordered() {
        let mut d = ArrowDigester::new(&struct_col_batch(true).schema(), cfg(0x02));
        d.update(&struct_col_batch(true));
        let res = catch_unwind(AssertUnwindSafe(|| {
            d.update(&struct_col_batch(false));
        }));
        assert!(
            res.is_err(),
            "batch with reordered struct fields must be rejected"
        );

        let mut d_cols = ArrowDigester::new(&struct_col_batch(true).schema(), cfg(0x01));
        d_cols.update(&struct_col_batch(true));
        d_cols.update(&struct_col_batch(false)); // accepted: struct order not part of the hash
    }

    // ── domain separation ───────────────────────────────────────────────────

    #[test]
    fn every_flag_combination_is_domain_separated() {
        let b = batch(
            vec![Field::new("v", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        );
        let rbs: Vec<String> = (0x00_u8..=0x07).map(|f| rb_hash(&b, cfg(f))).collect();
        assert_all_distinct(&rbs, "one-column record batch");
        let schemas: Vec<String> = (0x00_u8..=0x07)
            .map(|f| schema_hash(&b.schema(), cfg(f)))
            .collect();
        assert_all_distinct(&schemas, "one-column schema");
        let arrays: Vec<String> = (0x00_u8..=0x07)
            .map(|f| array_hash(b.column(0).as_ref(), cfg(f)))
            .collect();
        assert_all_distinct(&arrays, "hash_array");
    }

    #[test]
    fn options_tag_bytes_for_one_column_schema() {
        let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);
        let object_json = r#"{"v":{"data_type":"Int32","nullable":false}}"#;
        let array_json = r#"[{"data_type":"Int32","name":"v","nullable":false}]"#;
        for flags in 0x01_u8..=0x07 {
            let json = if flags & 0x01 != 0 {
                array_json
            } else {
                object_json
            };
            assert_eq!(
                ArrowDigester::hash_schema(&schema, cfg(flags)),
                versioned(&sha(&[&tag(flags), json.as_bytes()])),
                "flags {flags:#04x}"
            );
        }
        // flags 0: no tag, v0.4.0 bytes
        assert_eq!(
            ArrowDigester::hash_schema(&schema, cfg(0)),
            versioned(&sha(&[object_json.as_bytes()]))
        );
    }

    // ── include_metadata combination ────────────────────────────────────────

    #[test]
    fn include_metadata_combines_with_order_options() {
        let plain = Schema::new(vec![
            Field::new("b", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]);
        let meta: HashMap<String, String> = [("unit".to_owned(), "kg".to_owned())].into();
        let with_meta = Schema::new(vec![
            Field::new("b", DataType::Int32, false).with_metadata(meta),
            Field::new("a", DataType::Int32, false),
        ]);
        for flags in 0x00_u8..=0x07 {
            let off = cfg(flags);
            let on = HasherConfig {
                include_metadata: true,
                ..off
            };
            // empty-metadata invariant holds within each mode
            assert_eq!(schema_hash(&plain, off), schema_hash(&plain, on), "{flags}");
            // metadata counts within each mode
            assert_ne!(
                schema_hash(&with_meta, on),
                schema_hash(&plain, on),
                "{flags}"
            );
        }
        // byte level: tag, ordered JSON, then the unchanged Phase 2 document
        let on = HasherConfig {
            include_metadata: true,
            ..cfg(0x01)
        };
        let json = r#"[{"data_type":"Int32","name":"b","nullable":false},{"data_type":"Int32","name":"a","nullable":false}]"#;
        let phase2 = r#"{"fields":{"b":{"unit":"kg"}}}"#;
        assert_eq!(
            ArrowDigester::hash_schema(&with_meta, on),
            versioned(&sha(&[&tag(0x01), json.as_bytes(), phase2.as_bytes()]))
        );
    }

    // ── rejection of ambiguous names ────────────────────────────────────────

    fn panics<F>(f: F) -> bool
    where
        F: FnOnce(),
    {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }

    #[test]
    fn duplicate_column_names_rejected_when_any_option_on() {
        let b = batch(
            vec![
                Field::new("a", DataType::Int32, false),
                Field::new("a", DataType::Int32, false),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![2])),
            ],
        );
        assert!(!panics(|| {
            ArrowDigester::hash_record_batch(&b, cfg(0));
        }));
        for flags in 0x01_u8..=0x07 {
            assert!(
                panics(|| {
                    ArrowDigester::hash_record_batch(&b, cfg(flags));
                }),
                "record batch, flags {flags:#04x}"
            );
            assert!(
                panics(|| {
                    ArrowDigester::hash_schema(&b.schema(), cfg(flags));
                }),
                "schema, flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn duplicate_struct_field_names_rejected_when_any_option_on() {
        let st = DataType::Struct(Fields::from(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("x", DataType::Int32, false),
        ]));
        let schema = Schema::new(vec![Field::new(
            "l",
            DataType::List(Arc::new(Field::new("item", st, true))),
            true,
        )]);
        assert!(!panics(|| {
            ArrowDigester::hash_schema(&schema, cfg(0));
        }));
        for flags in 0x01_u8..=0x07 {
            assert!(
                panics(|| {
                    ArrowDigester::hash_schema(&schema, cfg(flags));
                }),
                "flags {flags:#04x}"
            );
        }
    }

    #[test]
    fn delimiter_path_aliasing_rejected_when_any_option_on() {
        // column "a/b" next to struct column a { b }
        let st = DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int32, false)]));
        let s1 = Schema::new(vec![
            Field::new("a/b", DataType::Int32, false),
            Field::new("a", st, false),
        ]);
        // column "x/" next to nullable list column x (element path "x/")
        let s2 = Schema::new(vec![
            Field::new("x/", DataType::Int32, false),
            Field::new(
                "x",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            ),
        ]);
        // a "/" in a name that collides with nothing is fine
        let s3 = Schema::new(vec![Field::new("km/h", DataType::Int32, false)]);
        for flags in 0x01_u8..=0x07 {
            assert!(panics(|| {
                ArrowDigester::hash_schema(&s1, cfg(flags));
            }));
            assert!(panics(|| {
                ArrowDigester::hash_schema(&s2, cfg(flags));
            }));
            assert!(!panics(|| {
                ArrowDigester::hash_schema(&s3, cfg(flags));
            }));
        }
    }

    #[test]
    fn streaming_rejects_batch_with_duplicate_columns() {
        // Digester schema [a]; batch [a, a] has the same unordered schema JSON (object keyed
        // by name) but two columns feeding one entry.
        let one = batch(
            vec![Field::new("a", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1]))],
        );
        let two = batch(
            vec![
                Field::new("a", DataType::Int32, false),
                Field::new("a", DataType::Int32, false),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![2])),
            ],
        );
        for flags in 0x01_u8..=0x07 {
            let mut d = ArrowDigester::new(&one.schema(), cfg(flags));
            assert!(
                panics(|| d.update(&two)),
                "flags {flags:#04x}: duplicate-column batch must be rejected"
            );
        }
    }

    // ── nested_validity ─────────────────────────────────────────────────────

    fn struct_batch(fields: Fields, child: ArrayRef, nulls: Option<NullBuffer>) -> RecordBatch {
        let s = StructArray::new(fields, vec![child], nulls);
        batch(
            vec![Field::new("s", s.data_type().clone(), true)],
            vec![Arc::new(s)],
        )
    }

    #[test]
    fn null_struct_vs_struct_with_null_fields() {
        let f = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        let child: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let null_struct = struct_batch(
            f.clone(),
            Arc::clone(&child),
            Some(NullBuffer::from(vec![true, false])),
        );
        let null_fields = struct_batch(f, child, None);
        for flags in 0x00_u8..=0x07 {
            let same = rb_hash(&null_struct, cfg(flags)) == rb_hash(&null_fields, cfg(flags));
            assert_eq!(same, flags & 0x04 == 0, "flags {flags:#04x}");
        }
    }

    #[test]
    fn null_struct_rows_with_non_nullable_fields_are_counted() {
        let f = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let three = struct_batch(
            f.clone(),
            Arc::new(Int32Array::from(vec![1, 0, 3])),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let two = struct_batch(f, Arc::new(Int32Array::from(vec![1, 3])), None);
        for flags in 0x00_u8..=0x07 {
            let same = rb_hash(&three, cfg(flags)) == rb_hash(&two, cfg(flags));
            assert_eq!(same, flags & 0x04 == 0, "flags {flags:#04x}");
        }
    }

    fn list_of_list(rows: &[Vec<Vec<i32>>]) -> RecordBatch {
        // inner list items nullable (ListBuilder default), inner list field nullable
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

    #[test]
    fn list_of_nullable_list_keeps_outer_lengths() {
        let one_row = list_of_list(&[vec![vec![1], vec![2]]]);
        let two_rows = list_of_list(&[vec![vec![1]], vec![vec![2]]]);
        for flags in 0x00_u8..=0x07 {
            let same = rb_hash(&one_row, cfg(flags)) == rb_hash(&two_rows, cfg(flags));
            assert_eq!(same, flags & 0x04 == 0, "flags {flags:#04x}");
        }
    }

    #[test]
    fn list_of_nullable_list_bytes() {
        // c: List<List<Int32?>?> non-null, one row [[1], [2]]
        let b = list_of_list(&[vec![vec![1], vec![2]]]);
        let json = r#"{"c":{"data_type":{"LargeList":{"data_type":{"LargeList":{"data_type":"Int32","nullable":true}},"nullable":true}},"nullable":false}}"#;
        let schema_digest = sha(&[&tag(0x04), json.as_bytes()]);
        let expected = versioned(&sha(&[
            &schema_digest,
            // entry "c/": null_bits = inner-list validity [1, 1], structural = outer lengths [2]
            &2_u64.to_le_bytes(),
            &[0x03],
            &sha(&[&2_u64.to_le_bytes()]),
            // entry "c//": list-leaf, null_bits = item validity [1, 1], lengths [1, 1], data [1, 2]
            &2_u64.to_le_bytes(),
            &[0x03],
            &sha(&[&1_u64.to_le_bytes(), &1_u64.to_le_bytes()]),
            &sha(&[&1_i32.to_le_bytes(), &2_i32.to_le_bytes()]),
        ]));
        assert_eq!(ArrowDigester::hash_record_batch(&b, cfg(0x04)), expected);
    }

    #[test]
    fn nested_validity_struct_bytes() {
        // s: Struct<a: Int32?>? rows [null, {a: null}, {a: 1}]
        let f = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        let b = struct_batch(
            f,
            Arc::new(Int32Array::from(vec![Some(5), None, Some(1)])),
            Some(NullBuffer::from(vec![false, true, true])),
        );
        let json = r#"{"s":{"data_type":{"Struct":[{"data_type":"Int32","name":"a","nullable":true}]},"nullable":true}}"#;
        let schema_digest = sha(&[&tag(0x04), json.as_bytes()]);
        let expected = versioned(&sha(&[
            &schema_digest,
            // entry "s": validity-only, struct bits [0, 1, 1]
            &3_u64.to_le_bytes(),
            &[0b110],
            // entry "s/a": own [1, 0, 1] AND struct [0, 1, 1] = [0, 0, 1], data [1]
            &3_u64.to_le_bytes(),
            &[0b100],
            &sha(&[&1_i32.to_le_bytes()]),
        ]));
        assert_eq!(ArrowDigester::hash_record_batch(&b, cfg(0x04)), expected);
    }

    #[test]
    fn list_of_nullable_struct_element_validity() {
        // l: List<Struct<a: Int32?>?> non-null; row [null] vs row [{a: null}]
        let mk = |struct_null: bool| {
            let f = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
            let st = StructArray::new(
                f,
                vec![Arc::new(Int32Array::from(vec![None::<i32>]))],
                Some(NullBuffer::from(vec![!struct_null])),
            );
            let item = Arc::new(Field::new("item", st.data_type().clone(), true));
            let col: ArrayRef = Arc::new(ListArray::new(
                item,
                OffsetBuffer::<i32>::from_lengths([1]),
                Arc::new(st),
                None,
            ));
            batch(
                vec![Field::new("l", col.data_type().clone(), false)],
                vec![col],
            )
        };
        for flags in 0x00_u8..=0x07 {
            let same = rb_hash(&mk(true), cfg(flags)) == rb_hash(&mk(false), cfg(flags));
            assert_eq!(same, flags & 0x04 == 0, "flags {flags:#04x}");
        }
    }

    #[test]
    fn nested_validity_hash_array_of_struct() {
        let f = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let three = StructArray::new(
            f.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 0, 3]))],
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let two = StructArray::new(f, vec![Arc::new(Int32Array::from(vec![1, 3]))], None);
        assert_eq!(array_hash(&three, cfg(0)), array_hash(&two, cfg(0)));
        assert_ne!(array_hash(&three, cfg(0x04)), array_hash(&two, cfg(0x04)));
    }

    // ── Boolean data under the third switch ────────────────────────────────

    fn bool_batch(nullable: bool, values: Vec<Option<bool>>) -> RecordBatch {
        batch(
            vec![Field::new("b", DataType::Boolean, nullable)],
            vec![Arc::new(BooleanArray::from(values))],
        )
    }

    #[test]
    fn boolean_trailing_false_rows_are_counted() {
        let one = bool_batch(false, vec![Some(true)]);
        let three = bool_batch(false, vec![Some(true), Some(false), Some(false)]);
        for flags in 0x00_u8..=0x07 {
            let same = rb_hash(&one, cfg(flags)) == rb_hash(&three, cfg(flags));
            assert_eq!(same, flags & 0x04 == 0, "flags {flags:#04x}");
        }
        let a1 = BooleanArray::from(vec![true]);
        let a2 = BooleanArray::from(vec![true, false]);
        assert_eq!(array_hash(&a1, cfg(0)), array_hash(&a2, cfg(0)));
        assert_ne!(array_hash(&a1, cfg(0x04)), array_hash(&a2, cfg(0x04)));
    }

    #[test]
    fn boolean_batch_split_independence_with_third_switch() {
        for nullable in [false, true] {
            let whole = bool_batch(nullable, vec![Some(true), Some(false), Some(true)]);
            for flags in [0x04_u8, 0x07] {
                let mut d = ArrowDigester::new(&whole.schema(), cfg(flags));
                d.update(&bool_batch(nullable, vec![Some(true)]));
                d.update(&bool_batch(nullable, vec![Some(false), Some(true)]));
                assert_eq!(
                    hex::encode(d.finalize()),
                    rb_hash(&whole, cfg(flags)),
                    "nullable {nullable}, flags {flags:#04x}"
                );
            }
        }
    }

    #[test]
    fn boolean_one_byte_per_value_bytes() {
        // b: Boolean nullable [true, null, false]; flags 0x04
        let b = bool_batch(true, vec![Some(true), None, Some(false)]);
        let json = r#"{"b":{"data_type":"Boolean","nullable":true}}"#;
        let expected = versioned(&sha(&[
            &sha(&[&tag(0x04), json.as_bytes()]),
            &3_u64.to_le_bytes(),
            &[0b101],
            &sha(&[&[0x01, 0x00]]), // one byte per VALID value
        ]));
        assert_eq!(ArrowDigester::hash_record_batch(&b, cfg(0x04)), expected);

        // hash_array, non-nullable [true, false, true]
        let a = BooleanArray::from(vec![true, false, true]);
        let expected_array = versioned(&sha(&[
            &tag(0x04),
            br#""Boolean""#,
            &sha(&[&[0x01, 0x00, 0x01]]),
        ]));
        assert_eq!(
            ArrowDigester::hash_array_with_config(&a, cfg(0x04)),
            expected_array
        );
    }

    // ── JSON text rules (byte-layout spec Section 2.1) ──────────────────────

    #[test]
    fn json_escaping_is_serde_json() {
        // default mode = v0.4.0 bytes; non-ASCII raw, short escapes, lowercase \u00xx
        let schema = Schema::new(vec![
            Field::new("é", DataType::Int32, false),
            Field::new(
                "a\nb\u{1b}\u{8}\u{c}\r\t\u{7f}\"\\/",
                DataType::Int32,
                false,
            ),
        ]);
        let json = "{\"a\\nb\\u001b\\b\\f\\r\\t\u{7f}\\\"\\\\/\":{\"data_type\":\"Int32\",\"nullable\":false},\"é\":{\"data_type\":\"Int32\",\"nullable\":false}}";
        assert_eq!(
            ArrowDigester::hash_schema(&schema, cfg(0)),
            versioned(&sha(&[json.as_bytes()]))
        );
    }

    #[test]
    fn map_metadata_paths_bytes() {
        // default options; Phase 2 paths: entries field at "m/", value field at "m//value"
        let entries = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int32, true)
                    .with_metadata([("unit".to_owned(), "kg".to_owned())].into()),
            ])),
            false,
        )
        .with_metadata([("role".to_owned(), "pairs".to_owned())].into());
        let schema = Schema::new(vec![Field::new(
            "m",
            DataType::Map(Arc::new(entries), false),
            true,
        )]);
        let phase1 = r#"{"m":{"data_type":{"Map":[{"data_type":{"Struct":[{"data_type":"LargeUtf8","name":"key","nullable":false},{"data_type":"Int32","name":"value","nullable":true}]},"name":"entries","nullable":false},false]},"nullable":true}}"#;
        let phase2 = r#"{"fields":{"m/":{"role":"pairs"},"m//value":{"unit":"kg"}}}"#;
        let config = HasherConfig {
            include_metadata: true,
            ..HasherConfig::default()
        };
        assert_eq!(
            ArrowDigester::hash_schema(&schema, config),
            versioned(&sha(&[phase1.as_bytes(), phase2.as_bytes()]))
        );
    }

    // ── invariants that hold in every mode ─────────────────────────────────

    #[test]
    fn batch_split_independence_in_every_mode() {
        let f = Fields::from(vec![
            Field::new("y", DataType::Int32, true),
            Field::new("x", DataType::Int32, false),
        ]);
        let mk = |ys: Vec<Option<i32>>, xs: Vec<i32>, nulls: Vec<bool>| {
            struct_batch_two(f.clone(), ys, xs, nulls)
        };
        let whole = mk(
            vec![Some(1), None, Some(3)],
            vec![1, 2, 3],
            vec![true, false, true],
        );
        let p1 = mk(vec![Some(1)], vec![1], vec![true]);
        let p2 = mk(vec![None, Some(3)], vec![2, 3], vec![false, true]);
        for flags in 0x00_u8..=0x07 {
            let mut d = ArrowDigester::new(&whole.schema(), cfg(flags));
            d.update(&p1);
            d.update(&p2);
            assert_eq!(
                hex::encode(d.finalize()),
                rb_hash(&whole, cfg(flags)),
                "flags {flags:#04x}"
            );
        }
    }

    fn struct_batch_two(
        f: Fields,
        ys: Vec<Option<i32>>,
        xs: Vec<i32>,
        nulls: Vec<bool>,
    ) -> RecordBatch {
        let s = StructArray::new(
            f,
            vec![
                Arc::new(Int32Array::from(ys)),
                Arc::new(Int32Array::from(xs)),
            ],
            Some(NullBuffer::from(nulls)),
        );
        batch(
            vec![
                Field::new("s", s.data_type().clone(), true),
                Field::new("k", DataType::Int32, false),
            ],
            vec![
                Arc::new(s.clone()),
                Arc::new(Int32Array::from(vec![0; s.len()])),
            ],
        )
    }

    #[test]
    fn physical_equivalences_in_every_mode() {
        let dict: DictionaryArray<Int32Type> = vec!["p", "q", "p"].into_iter().collect();
        let plain = StringArray::from(vec!["p", "q", "p"]);
        let large = LargeStringArray::from(vec!["p", "q", "p"]);
        let mk = |a: ArrayRef| {
            batch(
                vec![
                    Field::new("z", DataType::Int32, false),
                    Field::new("d", a.data_type().clone(), false),
                ],
                vec![Arc::new(Int32Array::from(vec![1, 2, 3])), a],
            )
        };
        for flags in 0x00_u8..=0x07 {
            let h_dict = rb_hash(&mk(Arc::new(dict.clone())), cfg(flags));
            let h_plain = rb_hash(&mk(Arc::new(plain.clone())), cfg(flags));
            let h_large = rb_hash(&mk(Arc::new(large.clone())), cfg(flags));
            assert_eq!(h_dict, h_plain, "dictionary, flags {flags:#04x}");
            assert_eq!(h_plain, h_large, "large, flags {flags:#04x}");
        }
    }

    #[test]
    fn map_schema_hashes_in_every_mode() {
        let mut mb = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        mb.keys().append_value("k");
        mb.values().append_value(1);
        mb.append(true).unwrap();
        let m = mb.finish();
        let schema = Schema::new(vec![Field::new("m", m.data_type().clone(), true)]);
        let all: Vec<String> = (0x00_u8..=0x07)
            .map(|f| schema_hash(&schema, cfg(f)))
            .collect();
        assert_all_distinct(&all, "map schema");
    }
}
