//! Property sweep over every supported leaf type (ITL-898).
//!
//! For each leaf type x nullable/non-nullable x position (top-level column, inside a struct,
//! inside a list) it checks:
//!
//! 1. row-count sensitivity: `n` rows vs `n + k` rows of zero / false / empty values (and, when
//!    nullable, of nulls) all hash differently;
//! 2. batch-split independence: the same rows in 1, 2, 3 or 4 batches (including an empty
//!    batch) hash the same;
//! 3. slice / offset independence: a sliced column (non-zero offset) hashes like a fresh copy;
//! 4. empty vs absent: a zero-row batch equals no batch, and (nullable) an empty / zero value
//!    differs from a null.
//!
//! `third_switch_is_lossless` asserts all four for the third switch (flag `0x04`, and with all
//! options on). `report_all_modes` (ignored; run with `--ignored --nocapture`) prints the table
//! for the default mode too.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::indexing_slicing, reason = "Okay in test")]
    #![expect(clippy::print_stdout, reason = "Okay in test: report table")]
    #![expect(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "Okay in test: small non-negative seeds"
    )]
    #![expect(
        clippy::struct_excessive_bools,
        clippy::too_many_lines,
        clippy::shadow_unrelated,
        reason = "Okay in test: one flat property check per case"
    )]

    use std::collections::HashSet;
    use std::iter::{once, repeat_n};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{
        Array as _, ArrayRef, BooleanArray, FixedSizeBinaryArray, Float32Array, Int64Array,
        ListArray, RecordBatch, StringArray, StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::compute::{cast, concat};
    use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};
    use starfix::{ArrowDigester, HasherConfig};

    fn cfg(flags: u8) -> HasherConfig {
        HasherConfig {
            include_metadata: false,
            ordered_columns: flags & 0x01 != 0,
            ordered_struct_fields: flags & 0x02 != 0,
            nested_validity: flags & 0x04 != 0,
        }
    }

    fn leaf_types() -> Vec<DataType> {
        vec![
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Boolean,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::FixedSizeBinary(2),
            DataType::Date32,
            DataType::Date64,
            DataType::Time32(TimeUnit::Second),
            DataType::Time32(TimeUnit::Millisecond),
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
            DataType::Duration(TimeUnit::Millisecond),
            DataType::Decimal32(9, 2),
            DataType::Decimal64(18, 2),
            DataType::Decimal128(38, 5),
            DataType::Decimal256(76, 10),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int32)),
        ]
    }

    /// Build an array of `dt` from integer seeds; `Some(0)` is the zero / false / empty value.
    fn make(dt: &DataType, seeds: &[Option<i64>]) -> ArrayRef {
        let strings = || {
            StringArray::from(
                seeds
                    .iter()
                    .map(|s| {
                        s.map(|v| {
                            if v == 0 {
                                String::new()
                            } else {
                                format!("v{v}")
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            )
        };
        match dt {
            DataType::Boolean => Arc::new(BooleanArray::from(
                seeds.iter().map(|s| s.map(|v| v != 0)).collect::<Vec<_>>(),
            )),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                cast(&strings(), dt).unwrap()
            }
            DataType::Dictionary(_, value) if value.as_ref() == &DataType::Utf8 => {
                cast(&strings(), dt).unwrap()
            }
            DataType::FixedSizeBinary(2) => Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    seeds
                        .iter()
                        .map(|s| s.map(|v| [(v & 0xff) as u8, ((v >> 8) & 0xff) as u8])),
                    2,
                )
                .unwrap(),
            ),
            DataType::Float16 => cast(
                &Float32Array::from(
                    seeds
                        .iter()
                        .map(|s| s.map(|v| v as f32))
                        .collect::<Vec<_>>(),
                ),
                dt,
            )
            .unwrap(),
            DataType::Date32 | DataType::Time32(_) => {
                let ints = cast(&Int64Array::from(seeds.to_vec()), &DataType::Int32).unwrap();
                cast(&ints, dt).unwrap()
            }
            _ => cast(&Int64Array::from(seeds.to_vec()), dt).unwrap(),
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Position {
        Top,
        InStruct,
        InList,
        /// Nullable struct; the struct row is null where the seed is 3.
        InNullableStruct,
        /// Nullable list; the list row is null where the seed is 3.
        InNullableList,
    }

    /// Wrap a leaf array as a one-column batch at `position`. `seeds` decide which container
    /// rows are null (seed 3) for the nullable-container positions.
    fn wrap(
        position: Position,
        leaf: &ArrayRef,
        nullable: bool,
        seeds: &[Option<i64>],
    ) -> RecordBatch {
        let container_nulls = || {
            Some(NullBuffer::from(
                seeds.iter().map(|s| *s != Some(3)).collect::<Vec<_>>(),
            ))
        };
        let leaf_field = Field::new("c", leaf.data_type().clone(), nullable);
        let column: ArrayRef = match position {
            Position::Top => Arc::clone(leaf),
            Position::InStruct => Arc::new(StructArray::new(
                Fields::from(vec![leaf_field.clone()]),
                vec![Arc::clone(leaf)],
                None,
            )),
            Position::InList => {
                // one element per row
                Arc::new(ListArray::new(
                    Arc::new(Field::new("item", leaf.data_type().clone(), nullable)),
                    OffsetBuffer::<i32>::from_lengths(vec![1; leaf.len()]),
                    Arc::clone(leaf),
                    None,
                ))
            }
            Position::InNullableStruct => Arc::new(StructArray::new(
                Fields::from(vec![leaf_field.clone()]),
                vec![Arc::clone(leaf)],
                container_nulls(),
            )),
            Position::InNullableList => Arc::new(ListArray::new(
                Arc::new(Field::new("item", leaf.data_type().clone(), nullable)),
                OffsetBuffer::<i32>::from_lengths(vec![1; leaf.len()]),
                Arc::clone(leaf),
                container_nulls(),
            )),
        };
        let field = match position {
            Position::Top => leaf_field,
            Position::InStruct | Position::InList => {
                Field::new("c", column.data_type().clone(), false)
            }
            Position::InNullableStruct | Position::InNullableList => {
                Field::new("c", column.data_type().clone(), true)
            }
        };
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column]).unwrap()
    }

    fn stream_hash(schema: &Arc<Schema>, batches: &[RecordBatch], flags: u8) -> Vec<u8> {
        let mut d = ArrowDigester::new(schema, cfg(flags));
        for b in batches {
            d.update(b);
        }
        d.finalize()
    }

    #[derive(Debug, Default)]
    struct Outcome {
        row_count: bool,
        batch_split: bool,
        slice: bool,
        empty_vs_absent: bool,
    }

    fn check(dt: &DataType, nullable: bool, position: Position, flags: u8) -> Option<Outcome> {
        let run = || {
            let mut out = Outcome::default();
            let base: Vec<Option<i64>> = if nullable {
                vec![Some(1), None, Some(3)]
            } else {
                vec![Some(1), Some(2), Some(3)]
            };

            // 1. row-count sensitivity
            let mut hashes = Vec::new();
            for k in 0..10 {
                let mut seeds = base.clone();
                seeds.extend(repeat_n(Some(0), k));
                let b = wrap(position, &make(dt, &seeds), nullable, &seeds);
                hashes.push(ArrowDigester::hash_record_batch(&b, cfg(flags)));
                if position == Position::Top {
                    hashes.push(ArrowDigester::hash_array_with_config(
                        b.column(0).as_ref(),
                        cfg(flags),
                    ));
                }
                if matches!(
                    position,
                    Position::InNullableStruct | Position::InNullableList
                ) && k > 0
                {
                    let mut seeds = base.clone();
                    seeds.extend(repeat_n(Some(3), k));
                    let b = wrap(position, &make(dt, &seeds), nullable, &seeds);
                    hashes.push(ArrowDigester::hash_record_batch(&b, cfg(flags)));
                }
                if nullable && k > 0 {
                    let mut seeds = base.clone();
                    seeds.extend(repeat_n(None, k));
                    let b = wrap(position, &make(dt, &seeds), nullable, &seeds);
                    hashes.push(ArrowDigester::hash_record_batch(&b, cfg(flags)));
                }
            }
            let distinct: HashSet<&Vec<u8>> = hashes.iter().collect();
            out.row_count = distinct.len() == hashes.len();

            // 2. batch-split independence (incl. an empty batch)
            let seeds: Vec<Option<i64>> = if nullable {
                vec![Some(1), Some(0), None, Some(2), Some(3), Some(0), None]
            } else {
                vec![
                    Some(1),
                    Some(0),
                    Some(5),
                    Some(2),
                    Some(3),
                    Some(0),
                    Some(1),
                ]
            };
            let whole_leaf = make(dt, &seeds);
            let whole = wrap(position, &whole_leaf, nullable, &seeds);
            let schema = whole.schema();
            let expected = ArrowDigester::hash_record_batch(&whole, cfg(flags));
            let mut split_ok = true;
            for cuts in [
                vec![1_usize],
                vec![3],
                vec![2, 5],
                vec![1, 1, 4],
                vec![0, 7],
            ] {
                let mut batches = Vec::new();
                let mut start = 0;
                for &end in cuts.iter().chain(once(&seeds.len())) {
                    let part = &seeds[start..end];
                    batches.push(wrap(position, &make(dt, part), nullable, part));
                    start = end;
                }
                split_ok &= stream_hash(&schema, &batches, flags) == expected;
            }
            out.batch_split = split_ok;

            // 3. slice / offset independence
            let mut padded = vec![Some(9)];
            padded.extend(seeds.iter().copied());
            padded.push(Some(9));
            let sliced_leaf = make(dt, &padded).slice(1, seeds.len());
            let fresh_leaf = concat(&[whole_leaf.as_ref()]).unwrap();
            let mut slice_ok = ArrowDigester::hash_record_batch(
                &wrap(position, &sliced_leaf, nullable, &seeds),
                cfg(flags),
            ) == ArrowDigester::hash_record_batch(
                &wrap(position, &fresh_leaf, nullable, &seeds),
                cfg(flags),
            );
            // also slice the wrapped column itself (struct / list offsets)
            let padded_batch = wrap(position, &make(dt, &padded), nullable, &padded);
            let sliced_batch = padded_batch.slice(1, seeds.len());
            slice_ok &= ArrowDigester::hash_record_batch(&sliced_batch, cfg(flags)) == expected;
            if position == Position::Top {
                slice_ok &= ArrowDigester::hash_array_with_config(sliced_leaf.as_ref(), cfg(flags))
                    == ArrowDigester::hash_array_with_config(fresh_leaf.as_ref(), cfg(flags));
            }
            out.slice = slice_ok;

            // 4. empty vs absent
            let empty = wrap(position, &make(dt, &[]), nullable, &[]);
            let mut empty_ok = ArrowDigester::hash_record_batch(&empty, cfg(flags))
                == stream_hash(&empty.schema(), &[], flags);
            if nullable {
                let zero = wrap(position, &make(dt, &[Some(0)]), nullable, &[Some(0)]);
                let null = wrap(position, &make(dt, &[None]), nullable, &[None]);
                empty_ok &= ArrowDigester::hash_record_batch(&zero, cfg(flags))
                    != ArrowDigester::hash_record_batch(&null, cfg(flags));
            }
            out.empty_vs_absent = empty_ok;
            out
        };
        catch_unwind(AssertUnwindSafe(run)).ok()
    }

    fn sweep(flags: u8) -> Vec<(String, Option<Outcome>)> {
        let mut rows = Vec::new();
        for dt in leaf_types() {
            for nullable in [false, true] {
                for position in [
                    Position::Top,
                    Position::InStruct,
                    Position::InList,
                    Position::InNullableStruct,
                    Position::InNullableList,
                ] {
                    let label = format!("{dt:?}{} {position:?}", if nullable { "?" } else { "" });
                    rows.push((label, check(&dt, nullable, position, flags)));
                }
            }
        }
        rows
    }

    fn failures(rows: &[(String, Option<Outcome>)]) -> Vec<String> {
        rows.iter()
            .filter_map(|(label, o)| match o {
                None => Some(format!("{label}: panic")),
                Some(out)
                    if !(out.row_count && out.batch_split && out.slice && out.empty_vs_absent) =>
                {
                    Some(format!("{label}: {out:?}"))
                }
                Some(_) => None,
            })
            .collect()
    }

    #[test]
    fn third_switch_is_lossless() {
        for flags in [0x04_u8, 0x07] {
            let rows = sweep(flags);
            let failed = failures(&rows);
            assert!(
                failed.is_empty(),
                "flags {flags:#04x}: {} of {} cases fail:\n{}",
                failed.len(),
                rows.len(),
                failed.join("\n")
            );
        }
    }

    #[test]
    #[ignore = "report only: cargo test --test lossless_properties -- --ignored --nocapture"]
    fn report_all_modes() {
        for flags in [0x00_u8, 0x04, 0x07] {
            let rows = sweep(flags);
            let failed = failures(&rows);
            println!(
                "== flags {flags:#04x}: {} of {} cases fail",
                failed.len(),
                rows.len()
            );
            for f in failed {
                println!("  {f}");
            }
        }
    }
}
