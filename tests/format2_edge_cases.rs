//! Format-2 edge cases from review (ITL-898): offsets, empty names, unsupported data types,
//! extension metadata.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::indexing_slicing, reason = "Okay in test")]
    #![expect(
        clippy::integer_division_remainder_used,
        reason = "Okay in test: alternating values"
    )]

    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        Array as _, ArrayRef, BooleanArray, Int32Array, ListArray, NullArray, RecordBatch,
        StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::compute::concat;
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
                vec![concat(&[sliced.column(0).as_ref()]).unwrap()],
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

    #[test]
    fn dictionary_validity_is_that_of_the_decoded_array() {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int32Type;
        let dict = || -> ArrayRef {
            Arc::new(
                DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(vec![0, 1]),
                    Arc::new(Int32Array::from(vec![Some(5), None])),
                )
                .unwrap(),
            )
        };
        // a valid index pointing at a null value is a null
        let nullable = rb(
            vec![Field::new("c", dict().data_type().clone(), true)],
            vec![dict()],
        );
        let plain = rb(
            vec![Field::new("c", DataType::Int32, true)],
            vec![Arc::new(Int32Array::from(vec![Some(5), None]))],
        );
        assert_eq!(
            ArrowDigester::hash_record_batch(&nullable, HasherConfig::default()),
            ArrowDigester::hash_record_batch(&plain, HasherConfig::default())
        );
        // arrow-rs builds it as a non-nullable column (it counts only null indices); rejected
        let non_nullable = rb(
            vec![Field::new("c", dict().data_type().clone(), false)],
            vec![dict()],
        );
        assert!(panics(|| {
            ArrowDigester::hash_record_batch(&non_nullable, HasherConfig::default());
        }));
    }

    #[test]
    fn fixed_size_list_element_path_collides() {
        let schema = Schema::new(vec![
            Field::new(
                "f",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2),
                false,
            ),
            Field::new("f/", DataType::Int32, false),
        ]);
        assert!(panics(|| {
            ArrowDigester::hash_schema(&schema, HasherConfig::default());
        }));
    }

    fn plain_list() -> ArrayRef {
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths(vec![2, 1, 2]),
            Arc::new(Int32Array::from(vec![1, 2, 3, 1, 2])),
            None,
        ))
    }

    /// Dictionary<Int8, List<Int32>> with keys [0, 1, 0] over [[1, 2], [3]].
    fn dict_of_list() -> ArrayRef {
        use arrow::array::{DictionaryArray, Int8Array};
        use arrow::datatypes::Int8Type;
        let values: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths(vec![2, 1]),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            None,
        ));
        Arc::new(
            DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0, 1, 0]), values).unwrap(),
        )
    }

    #[test]
    fn dictionary_of_list_equals_plain_list() {
        let col = |a: ArrayRef| rb(vec![Field::new("c", a.data_type().clone(), false)], vec![a]);
        assert_eq!(
            ArrowDigester::hash_record_batch(&col(dict_of_list()), HasherConfig::default()),
            ArrowDigester::hash_record_batch(&col(plain_list()), HasherConfig::default())
        );
        assert_eq!(
            ArrowDigester::hash_array(dict_of_list().as_ref()),
            ArrowDigester::hash_array(plain_list().as_ref())
        );
        let in_struct = |a: ArrayRef| {
            let s = StructArray::new(
                Fields::from(vec![Field::new("d", a.data_type().clone(), false)]),
                vec![a],
                None,
            );
            rb(
                vec![Field::new("s", s.data_type().clone(), false)],
                vec![Arc::new(s)],
            )
        };
        assert_eq!(
            ArrowDigester::hash_record_batch(&in_struct(dict_of_list()), HasherConfig::default()),
            ArrowDigester::hash_record_batch(&in_struct(plain_list()), HasherConfig::default())
        );
    }

    #[test]
    fn nested_dictionary_resolves_recursively() {
        use arrow::array::{DictionaryArray, Int8Array, StringArray};
        use arrow::datatypes::Int8Type;
        let inner = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![1, 0]),
            Arc::new(StringArray::from(vec!["x", "y"])),
        )
        .unwrap();
        let outer = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![Some(0), Some(1), None, Some(0)]),
            Arc::new(inner),
        )
        .unwrap();
        let plain = StringArray::from(vec![Some("y"), Some("x"), None, Some("y")]);
        let col = |a: ArrayRef| rb(vec![Field::new("c", a.data_type().clone(), true)], vec![a]);
        assert_eq!(
            ArrowDigester::hash_record_batch(
                &col(Arc::new(outer.clone())),
                HasherConfig::default()
            ),
            ArrowDigester::hash_record_batch(
                &col(Arc::new(plain.clone())),
                HasherConfig::default()
            )
        );
        assert_eq!(
            ArrowDigester::hash_array(&outer),
            ArrowDigester::hash_array(&plain)
        );
    }

    /// `FixedSizeBinary(0)` with `n` rows and optional validity.
    fn zero_width(n: usize, valid: Option<Vec<bool>>) -> ArrayRef {
        use arrow::array::{make_array, ArrayData};
        use arrow::buffer::Buffer;
        let data = ArrayData::builder(DataType::FixedSizeBinary(0))
            .len(n)
            .add_buffer(Buffer::from(Vec::<u8>::new()))
            .nulls(valid.map(NullBuffer::from))
            .build()
            .unwrap();
        make_array(data)
    }

    #[test]
    fn zero_width_fixed_size_binary() {
        let with_nulls = zero_width(3, Some(vec![true, false, true]));
        let b = rb(
            vec![Field::new("z", DataType::FixedSizeBinary(0), true)],
            vec![Arc::clone(&with_nulls)],
        );
        assert!(!panics(|| {
            ArrowDigester::hash_record_batch(&b, HasherConfig::default());
        }));
        assert!(!panics(|| {
            ArrowDigester::hash_array(with_nulls.as_ref());
        }));
        // the row count still counts
        let col = |n: usize| {
            rb(
                vec![Field::new("z", DataType::FixedSizeBinary(0), false)],
                vec![zero_width(n, None)],
            )
        };
        assert_ne!(
            ArrowDigester::hash_record_batch(&col(1), HasherConfig::default()),
            ArrowDigester::hash_record_batch(&col(5), HasherConfig::default())
        );
    }

    /// `List` or `LargeList` with a non-nullable `Int32` item and rows `[[1], null, [4]]`; with
    /// `hidden`, the null row's slot covers a null item.
    fn list_with_hidden_null(large: bool, hidden: bool) -> RecordBatch {
        use arrow::array::{make_array, ArrayData};
        use arrow::buffer::Buffer;
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
        // arrow-rs refuses a null under a null list value of a non-nullable item, but pyarrow
        // can produce such arrays (they reach starfix over the C data interface).
        #[expect(unsafe_code, reason = "Building an array arrow-rs would not validate")]
        // SAFETY: offsets, child length and validity are consistent; only the nullability rule
        // for hidden items is relaxed, which the hasher does not rely on.
        let data = unsafe { builder.build_unchecked() };
        let col = make_array(data);
        rb(
            vec![Field::new("l", col.data_type().clone(), true)],
            vec![col],
        )
    }

    /// `List` or `LargeList` with a non-nullable `Int32` item and rows `[[1, null], [4]]`: the
    /// null item lies under a valid list value, so it is a real element.
    fn list_with_visible_null(large: bool) -> RecordBatch {
        use arrow::array::{make_array, ArrayData};
        use arrow::buffer::Buffer;
        let item = Arc::new(Field::new("item", DataType::Int32, false));
        let values = Int32Array::from(vec![Some(1), None, Some(4)]);
        let ends: Vec<i64> = vec![0, 2, 3];
        let (dt, offsets) = if large {
            (DataType::LargeList(item), Buffer::from_vec(ends))
        } else {
            let small: Vec<i32> = ends.iter().map(|&e| i32::try_from(e).unwrap()).collect();
            (DataType::List(item), Buffer::from_vec(small))
        };
        let builder = ArrayData::builder(dt)
            .len(2)
            .add_buffer(offsets)
            .add_child_data(values.to_data());
        #[expect(unsafe_code, reason = "Building an array arrow-rs would not validate")]
        // SAFETY: offsets, child length and validity are consistent; only the nullability rule
        // is broken, which is what the hasher must reject.
        let data = unsafe { builder.build_unchecked() };
        let col = make_array(data);
        rb(
            vec![Field::new("l", col.data_type().clone(), true)],
            vec![col],
        )
    }

    #[test]
    fn visible_null_under_valid_list_value_is_rejected() {
        for large in [false, true] {
            let b = list_with_visible_null(large);
            assert!(
                panics(|| {
                    ArrowDigester::hash_record_batch(&b, HasherConfig::default());
                }),
                "large {large}: record batch"
            );
            let col = Arc::clone(b.column(0));
            assert!(
                panics(|| {
                    ArrowDigester::hash_array(col.as_ref());
                }),
                "large {large}: hash_array"
            );
        }
    }

    #[test]
    fn hidden_nulls_under_null_list_values_are_not_elements() {
        for large in [false, true] {
            let hidden = list_with_hidden_null(large, true);
            let clean = list_with_hidden_null(large, false);
            assert_eq!(
                ArrowDigester::hash_record_batch(&hidden, HasherConfig::default()),
                ArrowDigester::hash_record_batch(&clean, HasherConfig::default()),
                "large {large}"
            );
        }
        assert_eq!(
            ArrowDigester::hash_record_batch(
                &list_with_hidden_null(false, true),
                HasherConfig::default()
            ),
            ArrowDigester::hash_record_batch(
                &list_with_hidden_null(true, true),
                HasherConfig::default()
            )
        );
    }
}
