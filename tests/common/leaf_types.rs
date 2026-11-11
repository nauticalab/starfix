//! Leaf Arrow types, values from integer seeds, and the five positions of the leaf sweep (shared
//! by `tests/lossless_properties.rs` and `tests/legacy_v0_4_differential.rs`).

#![expect(clippy::unwrap_used, reason = "Okay in test")]
#![expect(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "Okay in test: small non-negative seeds"
)]

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, FixedSizeBinaryArray, Float32Array, Int64Array, ListArray, RecordBatch,
    StringArray, StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::cast;
use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};

pub fn leaf_types() -> Vec<DataType> {
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
pub fn make(dt: &DataType, seeds: &[Option<i64>]) -> ArrayRef {
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
pub enum Position {
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
pub fn wrap(
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
        Position::InStruct | Position::InList => Field::new("c", column.data_type().clone(), false),
        Position::InNullableStruct | Position::InNullableList => {
            Field::new("c", column.data_type().clone(), true)
        }
    };
    RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column]).unwrap()
}
