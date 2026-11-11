//! Batch-independent traversal of hidden nulls at arbitrary nesting depth.
#![expect(clippy::unwrap_used, reason = "Tests")]
#[path = "common/hidden_nulls.rs"]
mod hidden_nulls;

#[cfg(test)]
mod tests {
    #![expect(
        clippy::shadow_reuse,
        clippy::shadow_unrelated,
        reason = "Transform fixtures within each test"
    )]
    use super::hidden_nulls;
    use arrow::array::{Array as _, ArrayRef, DictionaryArray, Int32Array, RecordBatch};
    use arrow::datatypes::{Field, Int32Type, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    use std::sync::Arc;

    fn batch(array: ArrayRef) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "x",
                array.data_type().clone(),
                true,
            )])),
            vec![array],
        )
        .unwrap()
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "Test helper accepts temporary arrays"
    )]
    fn check(hidden: ArrayRef, clean: ArrayRef) {
        for flags in 0..4 {
            for include_metadata in [false, true] {
                let config = HasherConfig {
                    ordered_columns: flags & 1 != 0,
                    ordered_struct_fields: flags & 2 != 0,
                    include_metadata,
                };
                assert_eq!(
                    ArrowDigester::hash_array_with_config(hidden.as_ref(), config),
                    ArrowDigester::hash_array_with_config(clean.as_ref(), config),
                    "hidden and clean arrays: flags={flags}, metadata={include_metadata}"
                );
                let hidden = batch(Arc::clone(&hidden));
                let clean = batch(Arc::clone(&clean));
                let expected = ArrowDigester::hash_record_batch(&clean, config);
                assert_eq!(
                    ArrowDigester::hash_record_batch(&hidden, config),
                    expected,
                    "hidden and clean batches: flags={flags}, metadata={include_metadata}"
                );
                for input in [&hidden, &clean] {
                    let mut digester = ArrowDigester::new(&input.schema(), config);
                    digester.update(&input.slice(0, 0));
                    for row in 0..input.num_rows() {
                        digester.update(&input.slice(row, 1));
                        digester.update(&input.slice(row, 0));
                    }
                    assert_eq!(
                        digester.finalize(),
                        expected,
                        "split stream: flags={flags}, metadata={include_metadata}"
                    );
                }
            }
        }
    }

    #[test]
    fn nested_hidden_null_matrix() {
        for shape in [
            "list_list",
            "list_struct_list",
            "struct_list_list",
            "list_struct",
            "depth3",
            "depth4",
        ] {
            for large in [false, true] {
                let hidden = hidden_nulls::nested(shape, large, true);
                let clean = hidden_nulls::nested(shape, large, false);
                check(Arc::clone(&hidden), Arc::clone(&clean));
                check(hidden.slice(1, 2), clean.slice(1, 2));
                let padded_hidden = hidden_nulls::list(
                    Arc::clone(&hidden),
                    true,
                    large,
                    vec![0, 0, 1, 2, 3],
                    vec![true; 4],
                );
                let padded_clean = hidden_nulls::list(
                    Arc::clone(&clean),
                    true,
                    large,
                    vec![0, 0, 1, 2, 3],
                    vec![true; 4],
                );
                check(padded_hidden.slice(1, 3), padded_clean.slice(1, 3));
                let identity = Int32Array::from(vec![0, 1, 2]);
                check(
                    Arc::new(
                        DictionaryArray::<Int32Type>::try_new(identity, Arc::clone(&hidden))
                            .unwrap(),
                    ),
                    Arc::clone(&clean),
                );
                // A nonzero offset retaining the null inner value and the separated outer ranges.
                let keys = Int32Array::from(vec![2, 0, 1, 2]);
                let hidden: ArrayRef =
                    Arc::new(DictionaryArray::<Int32Type>::try_new(keys.clone(), hidden).unwrap());
                let clean: ArrayRef =
                    Arc::new(DictionaryArray::<Int32Type>::try_new(keys, clean).unwrap());
                check(hidden.slice(1, 3), clean.slice(1, 3));
            }
        }
    }

    #[test]
    fn dictionary_hidden_null_parents() {
        for large in [false, true] {
            let hidden = hidden_nulls::nested("depth4", large, true);
            let clean = hidden_nulls::nested("depth4", large, false);
            let keys = Int32Array::from(vec![Some(0), None, Some(2), Some(0)]);
            check(
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys.clone(), hidden).unwrap()),
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys, clean).unwrap()),
            );
        }
    }

    #[test]
    fn dictionary_skips_structs_under_null_lists_before_validation() {
        use arrow::array::{make_array, ArrayData};
        use arrow::datatypes::DataType;
        for large in [false, true] {
            let fields = vec![Arc::new(Field::new("value", DataType::Int32, false))];
            let builder = ArrayData::builder(DataType::Struct(fields.into()))
                .len(3)
                .add_child_data(Int32Array::from(vec![Some(1), None, Some(8)]).to_data());
            #[expect(unsafe_code, reason = "Null under a null outer list is not an element")]
            // SAFETY: all buffers have matching lengths; the non-nullable null is hidden by
            // the null list row constructed below and must never be visited.
            let structure = make_array(unsafe { builder.build_unchecked() });
            let hidden = hidden_nulls::list(
                structure,
                false,
                large,
                vec![0, 1, 2, 3],
                vec![true, false, true],
            );
            let clean = hidden_nulls::list(
                hidden_nulls::structure(
                    Arc::new(Int32Array::from(vec![1, 8])),
                    false,
                    vec![true; 2],
                ),
                false,
                large,
                vec![0, 1, 1, 2],
                vec![true, false, true],
            );
            let keys = Int32Array::from(vec![0, 1, 2]);
            check(
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys.clone(), hidden).unwrap()),
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys, clean).unwrap()),
            );
        }
    }

    #[test]
    fn nested_dictionary_null_keys_and_struct_masks() {
        use arrow::array::new_null_array;
        use arrow::compute::cast;
        for large in [false, true] {
            let hidden = hidden_nulls::nested("list_list", large, true);
            let clean = hidden_nulls::nested("list_list", large, false);
            let value_type = clean.data_type().clone();
            for keys in [
                Int32Array::from(vec![Some(0), None, Some(2)]),
                Int32Array::from(vec![None, None]),
                Int32Array::from(Vec::<i32>::new()),
            ] {
                let hidden: ArrayRef = Arc::new(
                    DictionaryArray::<Int32Type>::try_new(keys.clone(), Arc::clone(&hidden))
                        .unwrap(),
                );
                let clean: ArrayRef = Arc::new(
                    DictionaryArray::<Int32Type>::try_new(keys, Arc::clone(&clean)).unwrap(),
                );
                check(hidden, cast(clean.as_ref(), &value_type).unwrap());
            }
            let empty_values = clean.slice(0, 0);
            let empty: ArrayRef = Arc::new(
                DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(vec![None, None]),
                    empty_values,
                )
                .unwrap(),
            );
            check(empty, new_null_array(&value_type, 2));
            let keys = Int32Array::from(vec![None, Some(2)]);
            let hidden: ArrayRef =
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys.clone(), hidden).unwrap());
            let clean: ArrayRef =
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys, clean).unwrap());
            let plain = cast(clean.as_ref(), &value_type).unwrap();
            check(
                hidden_nulls::structure(Arc::clone(&hidden), false, vec![false, true]),
                hidden_nulls::structure(Arc::clone(&plain), false, vec![false, true]),
            );
            let keys = Int32Array::from(vec![Some(1), Some(0), None, Some(1)]);
            let nested: ArrayRef =
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys.clone(), hidden).unwrap());
            let expected: ArrayRef =
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys, plain).unwrap());
            check(nested, cast(expected.as_ref(), &value_type).unwrap());
        }
    }
    #[test]
    fn dictionary_grandparent_struct_masks_hidden_nulls() {
        use arrow::array::Int8Array;
        use arrow::datatypes::Int8Type;
        for depth in 2..=4 {
            let hidden = hidden_nulls::masked_struct(depth, true);
            let clean = hidden_nulls::masked_struct(depth, false);
            let keys = Int8Array::from(vec![0, 1, 2]);
            let dictionary: ArrayRef =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::clone(&hidden)).unwrap());
            check(Arc::clone(&dictionary), Arc::clone(&hidden));
            check(Arc::clone(&dictionary), Arc::clone(&clean));
            // The exact one-row repro, and a nonzero-offset slice of the same shape.
            check(dictionary.slice(0, 1), hidden.slice(0, 1));
            check(dictionary.slice(1, 2), clean.slice(1, 2));
        }
    }
}
