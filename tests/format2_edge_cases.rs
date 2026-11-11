//! Format-2 edge cases from review (ITL-898): offsets, empty names, unsupported data types,
//! extension metadata.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        Array as _, ArrayRef, BooleanArray, Int32Array, ListArray, NullArray, RecordBatch,
        StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    fn panics<F>(f: F) -> bool
    where
        F: FnOnce(),
    {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }

    fn rb(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|&(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    /// `s: Struct<f: Boolean?>?`, 12 rows, `f[9]` null, struct row 11 null.
    fn struct_of_bool() -> RecordBatch {
        let bools: Vec<Option<bool>> = (0..12)
            .map(|i| if i == 9 { None } else { Some(i % 2 == 0) })
            .collect();
        let mut valid = vec![true; 12];
        valid[11] = false;
        let s = StructArray::new(
            Fields::from(vec![Field::new("f", DataType::Boolean, true)]),
            vec![Arc::new(BooleanArray::from(bools))],
            Some(NullBuffer::from(valid)),
        );
        rb(
            vec![Field::new("s", s.data_type().clone(), true)],
            vec![Arc::new(s)],
        )
    }

    #[test]
    fn sliced_struct_of_boolean_at_byte_aligned_offset() {
        let whole = struct_of_bool();
        for (offset, len) in [(8, 4), (1, 4), (7, 5), (9, 3)] {
            let sliced = whole.slice(offset, len);
            let fresh = rb(
                vec![sliced.schema().field(0).clone()],
                vec![arrow::compute::concat(&[sliced.column(0).as_ref()]).unwrap()],
            );
            assert_eq!(
                ArrowDigester::hash_record_batch(&sliced, HasherConfig::default()),
                ArrowDigester::hash_record_batch(&fresh, HasherConfig::default()),
                "slice({offset}, {len})"
            );
        }
    }

    #[test]
    fn list_of_struct_of_boolean_crossing_a_byte() {
        let flags = BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            None,
            Some(true),
        ]);
        let mut valid = vec![true; 10];
        valid[9] = false;
        let s = StructArray::new(
            Fields::from(vec![Field::new("flag", DataType::Boolean, true)]),
            vec![Arc::new(flags)],
            Some(NullBuffer::from(valid)),
        );
        let l = ListArray::new(
            Arc::new(Field::new("item", s.data_type().clone(), true)),
            OffsetBuffer::from_lengths(vec![8, 2]),
            Arc::new(s),
            None,
        );
        let b = rb(
            vec![Field::new("l", l.data_type().clone(), false)],
            vec![Arc::new(l)],
        );
        assert!(!panics(|| {
            ArrowDigester::hash_record_batch(&b, HasherConfig::default());
        }));
    }

    #[test]
    fn empty_names_are_unambiguous() {
        let empty_child =
            DataType::Struct(Fields::from(vec![Field::new("", DataType::Int32, false)]));
        let schemas = [
            Schema::new(vec![Field::new("s", empty_child.clone(), false)]),
            Schema::new(vec![Field::new("", empty_child, false)]),
            Schema::new(vec![
                Field::new(
                    "",
                    DataType::Struct(Fields::from(vec![Field::new("x", DataType::Int32, false)])),
                    false,
                ),
                Field::new("x", DataType::Int32, false),
            ]),
        ];
        for schema in &schemas {
            for flags in 0_u8..=3 {
                let config = HasherConfig {
                    include_metadata: true,
                    ordered_columns: flags & 1 != 0,
                    ordered_struct_fields: flags & 2 != 0,
                };
                assert!(
                    !panics(|| {
                        ArrowDigester::hash_schema(schema, config);
                    }),
                    "{schema:?}"
                );
            }
        }
        // metadata on the child "x" of column "" differs from metadata on column "x"
        let mk = |on_child: bool| {
            let m = meta(&[("u", "kg")]);
            let mut child = Field::new("x", DataType::Int32, false);
            let mut column = Field::new("x", DataType::Int32, false);
            if on_child {
                child = child.with_metadata(m);
            } else {
                column = column.with_metadata(m);
            }
            Schema::new(vec![
                Field::new("", DataType::Struct(Fields::from(vec![child])), false),
                column,
            ])
        };
        let im = HasherConfig {
            include_metadata: true,
            ..HasherConfig::default()
        };
        assert_ne!(
            ArrowDigester::hash_schema(&mk(true), im),
            ArrowDigester::hash_schema(&mk(false), im)
        );
    }

    #[test]
    fn unsupported_data_types_rejected_before_hashing_any_data() {
        let null_col = rb(
            vec![Field::new("n", DataType::Null, true)],
            vec![Arc::new(NullArray::new(0))],
        );
        // schema hashing works
        assert!(!panics(|| {
            ArrowDigester::hash_schema(&null_col.schema(), HasherConfig::default());
        }));
        // data hashing is rejected even with zero rows, and when creating a digester
        assert!(panics(|| {
            ArrowDigester::hash_record_batch(&null_col, HasherConfig::default());
        }));
        assert!(panics(|| {
            ArrowDigester::new(&null_col.schema(), HasherConfig::default());
        }));
        assert!(panics(|| {
            ArrowDigester::hash_array(&NullArray::new(0));
        }));
    }

    #[test]
    fn absent_extension_metadata_equals_empty() {
        let mk = |pairs: &[(&str, &str)]| {
            Schema::new(vec![
                Field::new("u", DataType::FixedSizeBinary(16), true).with_metadata(meta(pairs))
            ])
        };
        let im = HasherConfig {
            include_metadata: true,
            ..HasherConfig::default()
        };
        assert_eq!(
            ArrowDigester::hash_schema(&mk(&[("ARROW:extension:name", "arrow.uuid")]), im),
            ArrowDigester::hash_schema(
                &mk(&[
                    ("ARROW:extension:name", "arrow.uuid"),
                    ("ARROW:extension:metadata", "")
                ]),
                im
            )
        );
        assert_ne!(
            ArrowDigester::hash_schema(&mk(&[("ARROW:extension:name", "arrow.uuid")]), im),
            ArrowDigester::hash_schema(
                &mk(&[
                    ("ARROW:extension:name", "arrow.uuid"),
                    ("ARROW:extension:metadata", "{}")
                ]),
                im
            )
        );
    }

    #[test]
    fn dictionary_at_depth_equals_plain() {
        use arrow::array::{DictionaryArray, Int64Array};
        use arrow::datatypes::Int32Type;
        // list<dictionary<int32, int64>> with unused and duplicate dictionary values vs plain
        let keys = Int32Array::from(vec![Some(2), Some(0), None, Some(2)]);
        let values = Int64Array::from(vec![10, 99, 30, 30]); // 99 unused; 30 duplicated
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
        let plain = Int64Array::from(vec![Some(30), Some(10), None, Some(30)]);
        let wrap = |items: ArrayRef| {
            let l = ListArray::new(
                Arc::new(Field::new("item", items.data_type().clone(), true)),
                OffsetBuffer::from_lengths(vec![1, 3]),
                items,
                None,
            );
            rb(
                vec![Field::new("l", l.data_type().clone(), false)],
                vec![Arc::new(l)],
            )
        };
        assert_eq!(
            ArrowDigester::hash_record_batch(&wrap(Arc::new(dict)), HasherConfig::default()),
            ArrowDigester::hash_record_batch(&wrap(Arc::new(plain)), HasherConfig::default())
        );
    }

    #[test]
    fn nullable_child_masked_by_struct_is_fine_but_hash_array_counts_logical_nulls() {
        // hash_array of a dictionary whose valid key points at a null value: the array is nullable
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int32Type;
        let keys = Int32Array::from(vec![0, 1]);
        let values = Int32Array::from(vec![Some(5), None]);
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
        let plain = Int32Array::from(vec![Some(5), None]);
        assert_eq!(
            ArrowDigester::hash_array(&dict),
            ArrowDigester::hash_array(&plain)
        );
    }
}
