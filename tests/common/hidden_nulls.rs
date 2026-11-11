//! Fixtures whose physically present nulls are hidden by nullable containers.
#![expect(clippy::unwrap_used, reason = "Test fixtures")]
#![expect(
    clippy::shadow_reuse,
    reason = "Wrap the same logical fixture at successive depths"
)]

use std::sync::Arc;

use arrow::array::{make_array, Array as _, ArrayData, ArrayRef, Int32Array, StructArray};
use arrow::buffer::{Buffer, NullBuffer};
use arrow::datatypes::{DataType, Field};

#[expect(
    clippy::needless_pass_by_value,
    reason = "Fixture builder accepts temporary arrays"
)]
pub fn list(
    values: ArrayRef,
    nullable: bool,
    large: bool,
    offsets: Vec<i32>,
    valid: Vec<bool>,
) -> ArrayRef {
    let field = Arc::new(Field::new("item", values.data_type().clone(), nullable));
    let (data_type, buffer) = if large {
        (
            DataType::LargeList(field),
            Buffer::from_vec(offsets.into_iter().map(i64::from).collect::<Vec<_>>()),
        )
    } else {
        (DataType::List(field), Buffer::from_vec(offsets))
    };
    let builder = ArrayData::builder(data_type)
        .len(valid.len())
        .add_buffer(buffer)
        .add_child_data(values.to_data())
        .nulls(Some(NullBuffer::from(valid)));
    #[expect(
        unsafe_code,
        reason = "Hidden nulls are accepted by the byte-layout specification"
    )]
    // SAFETY: buffers, offsets and lengths agree. Only Arrow's stricter child nullability
    // validation is bypassed; each non-nullable null lies under a null container.
    let data = unsafe { builder.build_unchecked() };
    make_array(data)
}

pub fn structure(values: ArrayRef, nullable: bool, valid: Vec<bool>) -> ArrayRef {
    Arc::new(StructArray::new(
        vec![Arc::new(Field::new(
            "value",
            values.data_type().clone(),
            nullable,
        ))]
        .into(),
        vec![values],
        Some(NullBuffer::from(valid)),
    ))
}

/// Shapes are described from outermost to innermost; every shape has three logical rows.
pub fn nested(shape: &str, large: bool, hidden: bool) -> ArrayRef {
    let valid = if hidden {
        vec![false, true, true]
    } else {
        vec![false, true]
    };
    let leaf: ArrayRef = if hidden {
        Arc::new(Int32Array::from(vec![None, Some(7), Some(8)]))
    } else {
        Arc::new(Int32Array::from(vec![8]))
    };
    let inner = if shape == "list_struct" {
        let leaf: ArrayRef = if hidden {
            leaf
        } else {
            Arc::new(Int32Array::from(vec![None, Some(8)]))
        };
        structure(leaf, false, valid)
    } else {
        list(
            leaf,
            false,
            large,
            if hidden {
                vec![0, 1, 2, 3]
            } else {
                vec![0, 0, 1]
            },
            valid,
        )
    };
    let count = inner.len();
    let identity: Vec<i32> = (0..=count).map(|i| i32::try_from(i).unwrap()).collect();
    let middle = match shape {
        "list_struct_list" => structure(inner, true, vec![true; count]),
        "depth3" => list(inner, true, large, identity, vec![true; count]),
        "depth4" => {
            let inner = list(inner, true, large, identity.clone(), vec![true; count]);
            list(inner, true, large, identity, vec![true; count])
        }
        _ => inner,
    };
    let outer = list(
        middle,
        true,
        large,
        if hidden {
            vec![0, 1, 2, 3]
        } else {
            vec![0, 1, 1, 2]
        },
        vec![true, false, true],
    );
    if shape == "struct_list_list" {
        structure(outer, true, vec![true; 3])
    } else {
        outer
    }
}

/// A null root struct masks a non-nullable leaf through valid, non-nullable structs.
pub fn masked_struct(depth: usize, hidden: bool) -> ArrayRef {
    let mut values: ArrayRef = Arc::new(Int32Array::from(if hidden {
        vec![None, Some(7), None]
    } else {
        vec![Some(0), Some(7), Some(0)]
    }));
    for _ in 1..depth {
        let field = Arc::new(Field::new("i", values.data_type().clone(), false));
        let builder = ArrayData::builder(DataType::Struct(vec![field].into()))
            .len(3)
            .add_child_data(values.to_data());
        #[expect(unsafe_code, reason = "Nulls are masked by the root struct below")]
        // SAFETY: child and parent lengths agree; only the nullability rule is relaxed.
        // The invalid leaf slots are masked by the outer struct before they are visited.
        let data = unsafe { builder.build_unchecked() };
        values = make_array(data);
    }
    Arc::new(StructArray::new(
        vec![Arc::new(Field::new(
            "child",
            values.data_type().clone(),
            false,
        ))]
        .into(),
        vec![values],
        Some(NullBuffer::from(vec![false, true, false])),
    ))
}
