//! Property sweep over every supported leaf type (ITL-898).
//!
//! For each leaf type x nullable/non-nullable x position (top-level column, inside a struct,
//! inside a list, inside a fixed-size list; each container also nullable) it checks:
//!
//! 1. row-count sensitivity: `n` rows vs `n + k` rows of zero / false / empty values (and, when
//!    nullable, of nulls) all hash differently;
//! 2. batch-split independence: the same rows in 1, 2, 3 or 4 batches (including an empty
//!    batch) hash the same;
//! 3. slice / offset independence: a sliced column (non-zero offset) hashes like a fresh copy;
//! 4. empty vs absent: a zero-row batch equals no batch, and (nullable) an empty / zero value
//!    differs from a null.
//!
//! `every_mode_is_lossless` asserts all four in the default mode and with every combination of
//! the order options (v0.5 fixes, byte-layout spec Section 10; v0.4.0 failed 68 of these cases).

#[path = "common/leaf_types.rs"]
mod leaf_types;

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::indexing_slicing, reason = "Okay in test")]
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

    use arrow::array::{Array as _, ArrayRef, FixedSizeListArray, RecordBatch};
    use arrow::buffer::NullBuffer;
    use arrow::compute::concat;
    use arrow_schema::{DataType, Field, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    use crate::leaf_types::{self, leaf_types, make, Position};

    fn cfg(flags: u8) -> HasherConfig {
        HasherConfig {
            include_metadata: false,
            ordered_columns: flags & 0x01 != 0,
            ordered_struct_fields: flags & 0x02 != 0,
        }
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

    /// A position of the sweep: the shared leaf positions, or inside a `FixedSizeList<_, 1>`
    /// (nullable: the slot is null where the seed is 3; ITL-927).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Pos {
        Leaf(Position),
        InFixedSizeList,
        InNullableFixedSizeList,
    }

    const TOP: Pos = Pos::Leaf(Position::Top);

    fn wrap(position: Pos, leaf: &ArrayRef, nullable: bool, seeds: &[Option<i64>]) -> RecordBatch {
        let container_nullable = match position {
            Pos::Leaf(p) => return leaf_types::wrap(p, leaf, nullable, seeds),
            Pos::InFixedSizeList => false,
            Pos::InNullableFixedSizeList => true,
        };
        let nulls = container_nullable
            .then(|| NullBuffer::from(seeds.iter().map(|s| *s != Some(3)).collect::<Vec<_>>()));
        let column: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", leaf.data_type().clone(), nullable)),
                1,
                Arc::clone(leaf),
                nulls,
            )
            .unwrap(),
        );
        let field = Field::new("c", column.data_type().clone(), container_nullable);
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column]).unwrap()
    }

    fn check(dt: &DataType, nullable: bool, position: Pos, flags: u8) -> Option<Outcome> {
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
                if position == TOP {
                    hashes.push(ArrowDigester::hash_array_with_config(
                        b.column(0).as_ref(),
                        cfg(flags),
                    ));
                }
                if matches!(
                    position,
                    Pos::Leaf(Position::InNullableStruct | Position::InNullableList)
                        | Pos::InNullableFixedSizeList
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

            // 3. slice / offset independence, at bit offsets around byte and word boundaries
            let fresh_leaf = concat(&[whole_leaf.as_ref()]).unwrap();
            let mut slice_ok = true;
            for offset in [1_usize, 7, 8, 9, 16, 63, 64, 65] {
                let mut padded = vec![Some(9); offset];
                padded.extend(seeds.iter().copied());
                padded.push(Some(9));
                let sliced_leaf = make(dt, &padded).slice(offset, seeds.len());
                slice_ok &= ArrowDigester::hash_record_batch(
                    &wrap(position, &sliced_leaf, nullable, &seeds),
                    cfg(flags),
                ) == ArrowDigester::hash_record_batch(
                    &wrap(position, &fresh_leaf, nullable, &seeds),
                    cfg(flags),
                );
                // also slice the wrapped column itself (struct / list offsets)
                let padded_batch = wrap(position, &make(dt, &padded), nullable, &padded);
                let sliced_batch = padded_batch.slice(offset, seeds.len());
                slice_ok &= ArrowDigester::hash_record_batch(&sliced_batch, cfg(flags)) == expected;
                if position == TOP {
                    slice_ok &=
                        ArrowDigester::hash_array_with_config(sliced_leaf.as_ref(), cfg(flags))
                            == ArrowDigester::hash_array_with_config(
                                fresh_leaf.as_ref(),
                                cfg(flags),
                            );
                }
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
                    TOP,
                    Pos::Leaf(Position::InStruct),
                    Pos::Leaf(Position::InList),
                    Pos::Leaf(Position::InNullableStruct),
                    Pos::Leaf(Position::InNullableList),
                    Pos::InFixedSizeList,
                    Pos::InNullableFixedSizeList,
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
    fn every_mode_is_lossless() {
        for flags in 0x00_u8..=0x03 {
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
}
