//! `FixedSizeList` data hashing in format 2 (ITL-927; byte-layout spec Sections 3.2 and 8).
//!
//! A fixed-size list node's entry is its validity (if nullable) and nothing else; its element
//! node's elements are the `n` items of every effectively valid slot. Items under null slots are
//! never visited. `FixedSizeList` is a different type from `List` / `LargeList` and from
//! `FixedSizeList` of another size, and from `FixedSizeBinary`.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        make_array, Array as _, ArrayData, ArrayRef, DictionaryArray, FixedSizeBinaryArray,
        Int32Array, Int8Array, LargeListArray, ListArray, NullArray, RecordBatch, StringArray,
        StructArray, UInt8Array,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::Int8Type;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use pretty_assertions::assert_eq;
    use sha2::{Digest as _, Sha256};
    use starfix::{ArrowDigester, HasherConfig};

    const PREFIX: [u8; 3] = [0, 0, 2];

    fn header(kind: u8) -> Vec<u8> {
        let mut h = vec![0x00_u8];
        h.extend_from_slice(b"starfix");
        h.extend_from_slice(&PREFIX);
        h.push(kind);
        h.push(0x00);
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

    fn panics<F: FnOnce()>(f: F) -> bool {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }

    fn cfg() -> HasherConfig {
        HasherConfig::default()
    }

    fn col(name: &str, array: &ArrayRef, nullable: bool) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                name,
                array.data_type().clone(),
                nullable,
            )])),
            vec![Arc::clone(array)],
        )
        .unwrap()
    }

    fn rb_hash(array: &ArrayRef, nullable: bool) -> Vec<u8> {
        ArrowDigester::hash_record_batch(&col("f", array, nullable), cfg())
    }

    fn arr_hash(array: &ArrayRef) -> Vec<u8> {
        ArrowDigester::hash_array_with_config(array.as_ref(), cfg())
    }

    /// `FixedSizeList<item, size>` of `len` slots over `values`, built without validation so that
    /// nulls hidden under null slots of a non-nullable element (which pyarrow can produce) and
    /// child arrays longer than `len * size` can be expressed.
    fn fsl(
        values: &ArrayRef,
        size: i32,
        item_nullable: bool,
        valid: Option<Vec<bool>>,
        len: usize,
    ) -> ArrayRef {
        let field = Arc::new(Field::new(
            "item",
            values.data_type().clone(),
            item_nullable,
        ));
        let builder = ArrayData::builder(DataType::FixedSizeList(field, size))
            .len(len)
            .add_child_data(values.to_data())
            .nulls(valid.map(NullBuffer::from));
        #[expect(unsafe_code, reason = "Building arrays arrow-rs would not validate")]
        // SAFETY: the child holds at least `len * size` positions; only the nullability rule for
        // hidden items (and the exact child length) is relaxed, which the hasher does not rely on.
        let data = unsafe { builder.build_unchecked() };
        make_array(data)
    }

    fn ints(values: &[Option<i32>]) -> ArrayRef {
        Arc::new(Int32Array::from(values.to_vec()))
    }

    // ── byte layout ──────────────────────────────────────────────────────────

    /// The worked example of the spec: `f: FixedSizeList<Int32 nullable, 2>` nullable, rows
    /// `[[1, null], null, [3, 4]]` over child `[1, null, 7, 8, 3, 4]`.
    #[test]
    fn record_batch_bytes() {
        let array = fsl(
            &ints(&[Some(1), None, Some(7), Some(8), Some(3), Some(4)]),
            2,
            true,
            Some(vec![true, false, true]),
            3,
        );
        let schema_json = br#"{"f":{"data_type":{"FixedSizeList":[{"data_type":"Int32","nullable":true},2]},"nullable":true}}"#;
        let schema_digest = sha(&[&header(b'S'), schema_json]);
        let expected = out(&sha(&[
            &header(b'R'),
            &schema_digest,
            &3_u64.to_le_bytes(),
            // entry "f": validity [1, 0, 1] — no lengths
            &3_u64.to_le_bytes(),
            &[0x05],
            // entry "f/": validity [1, 0, 1, 1] of the items of slots 0 and 2, values [1, 3, 4]
            &4_u64.to_le_bytes(),
            &[0x0d],
            &sha(&[&[1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]]),
        ]));
        assert_eq!(rb_hash(&array, true), expected);
    }

    /// `hash_array` of `FixedSizeList<Int32 non-null, 2>` `[[1, 2], [3, 4]]`: the root `""` is
    /// always nullable (validity only), the element `"/"` has values only.
    #[test]
    fn hash_array_bytes() {
        let array = fsl(
            &ints(&[Some(1), Some(2), Some(3), Some(4)]),
            2,
            false,
            None,
            2,
        );
        let type_json = br#"{"FixedSizeList":[{"data_type":"Int32","nullable":false},2]}"#;
        let expected = out(&sha(&[
            &header(b'A'),
            type_json,
            &2_u64.to_le_bytes(),
            &2_u64.to_le_bytes(),
            &[0x03],
            &sha(&[&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]]),
        ]));
        assert_eq!(arr_hash(&array), expected);
    }

    /// `n = 0`: `f: FixedSizeList<Int32 nullable, 0>` nullable, rows `[[], null, []]`. The element
    /// has no elements: validity `bit_count = 0` and the values of nothing.
    #[test]
    fn zero_size_bytes() {
        let array = fsl(&ints(&[]), 0, true, Some(vec![true, false, true]), 3);
        let schema_json = br#"{"f":{"data_type":{"FixedSizeList":[{"data_type":"Int32","nullable":true},0]},"nullable":true}}"#;
        let expected = out(&sha(&[
            &header(b'R'),
            &sha(&[&header(b'S'), schema_json]),
            &3_u64.to_le_bytes(),
            &3_u64.to_le_bytes(),
            &[0x05],
            &0_u64.to_le_bytes(),
            &sha(&[]),
        ]));
        assert_eq!(rb_hash(&array, true), expected);
    }

    /// Non-nullable `FixedSizeList<Int32 non-null, 2>`: the list node has an empty entry.
    #[test]
    fn non_nullable_bytes() {
        let array = fsl(&ints(&[Some(5), Some(6)]), 2, false, None, 1);
        let schema_json = br#"{"f":{"data_type":{"FixedSizeList":[{"data_type":"Int32","nullable":false},2]},"nullable":false}}"#;
        let expected = out(&sha(&[
            &header(b'R'),
            &sha(&[&header(b'S'), schema_json]),
            &1_u64.to_le_bytes(),
            &sha(&[&[5, 0, 0, 0, 6, 0, 0, 0]]),
        ]));
        assert_eq!(rb_hash(&array, false), expected);
    }

    // ── type identity ────────────────────────────────────────────────────────

    fn list_of(values: &[Option<i32>], lengths: Vec<usize>, valid: Option<Vec<bool>>) -> ArrayRef {
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths(lengths),
            ints(values),
            valid.map(NullBuffer::from),
        ))
    }

    fn large_list_of(
        values: &[Option<i32>],
        lengths: Vec<usize>,
        valid: Option<Vec<bool>>,
    ) -> ArrayRef {
        Arc::new(LargeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths(lengths),
            ints(values),
            valid.map(NullBuffer::from),
        ))
    }

    #[test]
    fn differs_from_list_and_large_list_with_equal_values() {
        let values = [Some(1), Some(2), Some(3), Some(4)];
        let fixed = fsl(&ints(&values), 2, true, Some(vec![true, true]), 2);
        let list = list_of(&values, vec![2, 2], Some(vec![true, true]));
        let large = large_list_of(&values, vec![2, 2], Some(vec![true, true]));
        assert_eq!(rb_hash(&list, true), rb_hash(&large, true));
        assert_ne!(rb_hash(&fixed, true), rb_hash(&list, true));
        assert_eq!(arr_hash(&list), arr_hash(&large));
        assert_ne!(arr_hash(&fixed), arr_hash(&list));
        let schema = |a: &ArrayRef| ArrowDigester::hash_schema(&col("f", a, true).schema(), cfg());
        assert_ne!(schema(&fixed), schema(&list));
    }

    #[test]
    fn size_is_part_of_the_type() {
        let values = ints(&[Some(1), Some(2), Some(3), Some(4)]);
        let two = fsl(&values, 2, true, None, 2);
        let four = fsl(&values, 4, true, None, 1);
        let two_empty = fsl(&ints(&[]), 2, true, None, 0);
        let three_empty = fsl(&ints(&[]), 3, true, None, 0);
        assert_ne!(rb_hash(&two, false), rb_hash(&four, false));
        assert_ne!(rb_hash(&two_empty, false), rb_hash(&three_empty, false));
        assert_ne!(arr_hash(&two_empty), arr_hash(&three_empty));
    }

    #[test]
    fn differs_from_fixed_size_binary() {
        let binary: ArrayRef = Arc::new(
            FixedSizeBinaryArray::try_from_iter(vec![vec![1_u8, 2], vec![3, 4]].into_iter())
                .unwrap(),
        );
        let bytes: ArrayRef = Arc::new(UInt8Array::from(vec![1_u8, 2, 3, 4]));
        let fixed = fsl(&bytes, 2, false, None, 2);
        assert_ne!(rb_hash(&binary, false), rb_hash(&fixed, false));
        assert_ne!(arr_hash(&binary), arr_hash(&fixed));
    }

    // ── hidden and visible child positions ───────────────────────────────────

    #[test]
    fn items_under_null_slots_are_not_elements() {
        let base = |hidden: [Option<i32>; 2]| {
            let [h0, h1] = hidden;
            fsl(
                &ints(&[Some(1), Some(2), h0, h1, Some(3), Some(4)]),
                2,
                true,
                Some(vec![true, false, true]),
                3,
            )
        };
        let a = base([Some(7), Some(8)]);
        let b = base([None, None]);
        let c = base([Some(0), Some(0)]);
        assert_eq!(rb_hash(&a, true), rb_hash(&b, true));
        assert_eq!(rb_hash(&a, true), rb_hash(&c, true));
        assert_eq!(arr_hash(&a), arr_hash(&b));
        // visible items always count
        let d = fsl(
            &ints(&[Some(1), Some(9), Some(7), Some(8), Some(3), Some(4)]),
            2,
            true,
            Some(vec![true, false, true]),
            3,
        );
        assert_ne!(rb_hash(&a, true), rb_hash(&d, true));
    }

    /// Depth two: an outer null slot hides inner slots of any content (values, inner nulls).
    #[test]
    fn nested_items_under_null_outer_slots_are_not_elements() {
        let mk = |hidden: &[Option<i32>], hidden_inner_valid: [bool; 2]| {
            let [v0, v1] = hidden_inner_valid;
            let mut leaf = vec![Some(1), Some(2), Some(3), Some(4)];
            leaf.extend_from_slice(hidden);
            leaf.extend_from_slice(&[Some(5), Some(6), Some(7), Some(8)]);
            let inner = fsl(
                &ints(&leaf),
                2,
                false,
                Some(vec![true, true, v0, v1, true, true]),
                6,
            );
            fsl(&inner, 2, true, Some(vec![true, false, true]), 3)
        };
        let a = mk(&[Some(9), Some(9), Some(9), Some(9)], [true, true]);
        let b = mk(&[None, None, None, None], [false, true]);
        assert_eq!(rb_hash(&a, true), rb_hash(&b, true));
        assert_eq!(arr_hash(&a), arr_hash(&b));
    }

    fn struct_of(child: ArrayRef, child_nullable: bool, valid: Vec<bool>) -> ArrayRef {
        Arc::new(StructArray::new(
            Fields::from(vec![Field::new(
                "x",
                child.data_type().clone(),
                child_nullable,
            )]),
            vec![child],
            Some(NullBuffer::from(valid)),
        ))
    }

    /// A slot masked by a null enclosing struct is effectively null: its items are hidden.
    #[test]
    fn items_under_struct_masked_slots_are_not_elements() {
        let mk = |hidden: [Option<i32>; 2]| {
            let [h0, h1] = hidden;
            let inner = fsl(&ints(&[Some(1), Some(2), h0, h1]), 2, false, None, 2);
            struct_of(inner, false, vec![true, false])
        };
        assert_eq!(
            rb_hash(&mk([Some(7), Some(8)]), true),
            rb_hash(&mk([None, None]), true)
        );
        // two levels of masking: a struct of a struct of a fixed-size list
        let deep = |hidden: [Option<i32>; 2]| {
            let [h0, h1] = hidden;
            let inner = fsl(&ints(&[h0, h1, Some(3), Some(4)]), 2, false, None, 2);
            struct_of(
                struct_of(inner, false, vec![true, true]),
                true,
                vec![false, true],
            )
        };
        assert_eq!(
            rb_hash(&deep([Some(7), Some(8)]), true),
            rb_hash(&deep([None, None]), true)
        );
    }

    #[test]
    fn non_nullable_element_hidden_nulls_accepted_visible_rejected() {
        let hidden = fsl(
            &ints(&[Some(1), Some(2), None, None]),
            2,
            false,
            Some(vec![true, false]),
            2,
        );
        assert!(!panics(|| {
            rb_hash(&hidden, true);
        }));
        let visible = fsl(&ints(&[Some(1), None]), 2, false, Some(vec![true]), 1);
        assert!(panics(|| {
            rb_hash(&visible, true);
        }));
        assert!(panics(|| {
            arr_hash(&visible);
        }));
    }

    /// A non-nullable fixed-size-list field (here the element of an outer fixed-size list) with
    /// a visible null slot is rejected, as for any other type.
    #[test]
    fn non_nullable_fixed_size_list_with_visible_null_slot_is_rejected() {
        let inner = fsl(
            &ints(&[Some(1), Some(2)]),
            1,
            true,
            Some(vec![true, false]),
            2,
        );
        let outer = fsl(&inner, 2, false, None, 1);
        assert!(panics(|| {
            rb_hash(&outer, false);
        }));
    }

    /// Child positions beyond `len * n` are not part of the array.
    #[test]
    fn trailing_child_positions_are_ignored() {
        let exact = fsl(&ints(&[Some(1), Some(2)]), 2, true, Some(vec![true]), 1);
        let longer = fsl(
            &ints(&[Some(1), Some(2), Some(9), None, Some(9)]),
            2,
            true,
            Some(vec![true]),
            1,
        );
        assert_eq!(rb_hash(&exact, true), rb_hash(&longer, true));
        assert_eq!(arr_hash(&exact), arr_hash(&longer));
    }

    // ── n = 0 ────────────────────────────────────────────────────────────────

    #[test]
    fn zero_size_carries_row_count_and_validity() {
        let empty = ints(&[]);
        let three = fsl(&empty, 0, true, None, 3);
        let two = fsl(&empty, 0, true, None, 2);
        let with_null = fsl(&empty, 0, true, Some(vec![true, false, true]), 3);
        let null_first = fsl(&empty, 0, true, Some(vec![false, true, true]), 3);
        assert_ne!(rb_hash(&three, true), rb_hash(&two, true));
        assert_ne!(rb_hash(&three, true), rb_hash(&with_null, true));
        assert_ne!(rb_hash(&with_null, true), rb_hash(&null_first, true));
        assert_ne!(arr_hash(&with_null), arr_hash(&null_first));
        assert_ne!(arr_hash(&three), arr_hash(&two));
        // non-nullable: only the row count
        assert_ne!(rb_hash(&three, false), rb_hash(&two, false));
    }

    // ── physical equivalences ────────────────────────────────────────────────

    #[test]
    fn slices_and_batch_splits_hash_like_the_logical_value() {
        // rows: [0,0] [1,2] null [3,null] [5,6] [9,9]
        let padded = fsl(
            &ints(&[
                Some(0),
                Some(0),
                Some(1),
                Some(2),
                Some(7),
                None,
                Some(3),
                None,
                Some(5),
                Some(6),
                Some(9),
                Some(9),
            ]),
            2,
            true,
            Some(vec![true, true, false, true, true, true]),
            6,
        );
        let fresh = fsl(
            &ints(&[
                Some(1),
                Some(2),
                None,
                None,
                Some(3),
                None,
                Some(5),
                Some(6),
            ]),
            2,
            true,
            Some(vec![true, false, true, true]),
            4,
        );
        let sliced = padded.slice(1, 4);
        assert_eq!(rb_hash(&sliced, true), rb_hash(&fresh, true));
        assert_eq!(arr_hash(&sliced), arr_hash(&fresh));

        let whole = col("f", &fresh, true);
        let mut digester = ArrowDigester::new(&whole.schema(), cfg());
        for (start, len) in [(0, 1), (1, 0), (1, 2), (3, 1)] {
            digester.update(&whole.slice(start, len));
        }
        assert_eq!(
            digester.finalize(),
            ArrowDigester::hash_record_batch(&whole, cfg())
        );

        // a sliced struct of fixed-size lists
        let s_padded = struct_of(padded, true, vec![true, true, true, true, false, true]);
        let s_fresh = struct_of(
            fsl(
                &ints(&[
                    Some(1),
                    Some(2),
                    None,
                    None,
                    Some(3),
                    None,
                    Some(0),
                    Some(0),
                ]),
                2,
                true,
                Some(vec![true, false, true, true]),
                4,
            ),
            true,
            vec![true, true, true, false],
        );
        assert_eq!(
            rb_hash(&s_padded.slice(1, 4), true),
            rb_hash(&s_fresh, true)
        );
    }

    #[test]
    fn dictionary_of_fixed_size_list_equals_plain() {
        // values: [[1, 2], null, [3, 4]] with a hidden null under the null slot
        let values = fsl(
            &ints(&[Some(1), Some(2), None, Some(7), Some(3), Some(4)]),
            2,
            false,
            Some(vec![true, false, true]),
            3,
        );
        let dict: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(2), None, Some(1), Some(0), Some(2)]),
                Arc::clone(&values),
            )
            .unwrap(),
        );
        let plain = fsl(
            &ints(&[
                Some(3),
                Some(4),
                None,
                None,
                None,
                None,
                Some(1),
                Some(2),
                Some(3),
                Some(4),
            ]),
            2,
            false,
            Some(vec![true, false, false, true, true]),
            5,
        );
        assert_eq!(rb_hash(&dict, true), rb_hash(&plain, true));
        assert_eq!(arr_hash(&dict), arr_hash(&plain));
        // under a struct
        let valid = vec![true, true, false, true, true];
        assert_eq!(
            rb_hash(&struct_of(dict, true, valid.clone()), true),
            rb_hash(&struct_of(plain, true, valid), true)
        );
    }

    #[test]
    fn fixed_size_list_of_dictionary_equals_plain() {
        let strings = StringArray::from(vec![Some("a"), None, Some("b"), Some("a")]);
        let dict: ArrayRef = Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(0), None, Some(1), Some(0)]),
                Arc::new(StringArray::from(vec!["a", "b"])),
            )
            .unwrap(),
        );
        let plain: ArrayRef = Arc::new(strings);
        assert_eq!(
            rb_hash(&fsl(&dict, 2, true, None, 2), false),
            rb_hash(&fsl(&plain, 2, true, None, 2), false)
        );
    }

    /// `List<FixedSizeList>`: a null list value hides its fixed-size-list items entirely.
    #[test]
    fn fixed_size_list_under_null_list_value() {
        let mk = |hidden: [Option<i32>; 2], hidden_slot_valid: bool| {
            let inner = fsl(
                &ints(&[Some(1), Some(2), hidden[0], hidden[1], Some(3), Some(4)]),
                2,
                false,
                Some(vec![true, hidden_slot_valid, true]),
                3,
            );
            Arc::new(ListArray::new(
                Arc::new(Field::new("item", inner.data_type().clone(), true)),
                OffsetBuffer::from_lengths(vec![1, 1, 1]),
                inner,
                Some(NullBuffer::from(vec![true, false, true])),
            )) as ArrayRef
        };
        assert_eq!(
            rb_hash(&mk([Some(9), Some(9)], true), true),
            rb_hash(&mk([None, None], false), true)
        );
        // the same logical rows without the hidden slot
        let inner = fsl(
            &ints(&[Some(1), Some(2), Some(3), Some(4)]),
            2,
            false,
            None,
            2,
        );
        let clean: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", inner.data_type().clone(), true)),
            OffsetBuffer::from_lengths(vec![1, 0, 1]),
            inner,
            Some(NullBuffer::from(vec![true, false, true])),
        ));
        assert_eq!(
            rb_hash(&mk([None, None], false), true),
            rb_hash(&clean, true)
        );
    }

    #[test]
    fn fixed_size_list_of_list_and_struct_hash() {
        let list = list_of(
            &[Some(1), Some(2), Some(3)],
            vec![2, 0, 1, 0],
            Some(vec![true, true, true, false]),
        );
        let of_list = fsl(&list, 2, true, None, 2);
        let s = struct_of(
            ints(&[Some(1), None, Some(3), Some(4)]),
            true,
            vec![true, true, false, true],
        );
        let of_struct = fsl(&s, 2, true, Some(vec![true, true]), 2);
        assert!(!panics(|| {
            rb_hash(&of_list, false);
            rb_hash(&of_struct, true);
            arr_hash(&of_list);
            arr_hash(&of_struct);
        }));
        // the struct null at item 2 is a value: replacing it changes the hash
        let s2 = struct_of(
            ints(&[Some(1), None, Some(3), Some(4)]),
            true,
            vec![true, true, true, true],
        );
        assert_ne!(
            rb_hash(&of_struct, true),
            rb_hash(&fsl(&s2, 2, true, Some(vec![true, true]), 2), true)
        );
    }

    // ── rejections ───────────────────────────────────────────────────────────

    #[test]
    fn fixed_size_list_of_unsupported_type_is_still_rejected() {
        let nulls: ArrayRef = Arc::new(NullArray::new(2));
        let array = fsl(&nulls, 2, true, None, 1);
        assert!(panics(|| {
            rb_hash(&array, true);
        }));
        assert!(panics(|| {
            ArrowDigester::new(&col("f", &array, true).schema(), cfg());
        }));
    }

    #[test]
    fn stream_rejects_a_list_batch_for_a_fixed_size_list_schema() {
        let values = [Some(1), Some(2)];
        let fixed = col("f", &fsl(&ints(&values), 2, true, None, 1), true);
        let list = col("f", &list_of(&values, vec![2], None), true);
        let other_size = col("f", &fsl(&ints(&values), 1, true, None, 2), true);
        assert!(panics(|| {
            let mut d = ArrowDigester::new(&fixed.schema(), cfg());
            d.update(&list);
        }));
        assert!(panics(|| {
            let mut d = ArrowDigester::new(&fixed.schema(), cfg());
            d.update(&other_size);
        }));
    }
}
