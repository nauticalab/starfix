//! Nested column types containing `FixedSizeList` at every nesting position, small logical values,
//! and builders with two physical layouts per logical value (ITL-927; used by
//! `tests/fixed_size_list_properties.rs`).
//!
//! `build(node, rows, false)` fills positions that are not elements (items of null fixed-size-list
//! slots, children of null struct rows, physical values of null leaves) with one filler,
//! `build(node, rows, true)` with another, and also gives null list values hidden items and
//! dictionaries a different key mapping with an unused entry. The two must always hash alike.

#![expect(clippy::unwrap_used, reason = "Okay in test")]
#![expect(clippy::indexing_slicing, reason = "Okay in test")]
#![expect(
    clippy::arithmetic_side_effects,
    clippy::integer_division_remainder_used,
    reason = "Okay in test: tiny deterministic generator"
)]

use std::collections::BTreeSet;
use std::iter::{repeat_n, repeat_with};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, DictionaryArray, FixedSizeListArray, Int32Array, Int8Array, ListArray, StringArray,
    StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::cast;
use arrow::datatypes::Int8Type;
use arrow_schema::{DataType, Field, Fields};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Int,
    Dict,
    List,
    Struct,
    /// `FixedSizeList<_, 2>`.
    Fsl2,
    /// `FixedSizeList<_, 0>`.
    Fsl0,
    /// Dictionary-encoded values of the (non-nullable) child; nulls come from the keys.
    DictOf,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node {
    pub kind: Kind,
    pub nullable: bool,
    pub child: Option<Box<Self>>,
}

/// Logical value of one row.
#[derive(Clone, Debug)]
pub enum Val {
    Null,
    I(i32),
    S(&'static str),
    L(Vec<Self>),
    St(Box<Self>),
}

const fn leaf(kind: Kind, nullable: bool) -> Node {
    Node {
        kind,
        nullable,
        child: None,
    }
}

fn wrap(kind: Kind, nullable: bool, child: &Node) -> Node {
    Node {
        kind,
        nullable,
        child: Some(Box::new(child.clone())),
    }
}

fn child(node: &Node) -> &Node {
    node.child.as_deref().unwrap()
}

fn contains_fsl(node: &Node) -> bool {
    matches!(node.kind, Kind::Fsl2 | Kind::Fsl0) || node.child.as_deref().is_some_and(contains_fsl)
}

const WRAPPERS: [Kind; 3] = [Kind::List, Kind::Struct, Kind::Fsl2];

fn leaves() -> Vec<Node> {
    [Kind::Int, Kind::Dict]
        .iter()
        .flat_map(|&k| [leaf(k, true), leaf(k, false)])
        .collect()
}

fn trees(depth: usize) -> Vec<Node> {
    let mut out = leaves();
    if depth > 0 {
        for c in trees(depth - 1) {
            for kind in WRAPPERS {
                for nullable in [true, false] {
                    out.push(wrap(kind, nullable, &c));
                }
            }
        }
    }
    out
}

/// Every type of up to three container levels (list, struct, fixed-size list of size 2) that
/// contains a fixed-size list; size-0 fixed-size lists at the top, inside every container and
/// containing every one-level type; dictionary-encoded fixed-size lists at the top and inside
/// every container.
pub fn types() -> Vec<Node> {
    let mut set: BTreeSet<Node> = trees(3).into_iter().filter(contains_fsl).collect();
    for nullable in [true, false] {
        for c in trees(1) {
            set.insert(wrap(Kind::Fsl0, nullable, &c));
        }
        for l in leaves() {
            let zero = wrap(Kind::Fsl0, nullable, &l);
            for kind in WRAPPERS {
                for outer in [true, false] {
                    set.insert(wrap(kind, outer, &zero));
                }
            }
        }
        for c in trees(1) {
            let dict = wrap(Kind::DictOf, nullable, &wrap(Kind::Fsl2, false, &c));
            set.insert(dict.clone());
            for kind in WRAPPERS {
                for outer in [true, false] {
                    set.insert(wrap(kind, outer, &dict));
                }
            }
        }
    }
    set.into_iter().collect()
}

/// The same type with every dictionary of fixed-size lists replaced by the plain values type
/// (with the dictionary's nullability): it must hash like the dictionary-encoded one.
pub fn plain(node: &Node) -> Node {
    match node.kind {
        Kind::DictOf => {
            let c = child(node);
            Node {
                kind: c.kind,
                nullable: node.nullable,
                child: c.child.as_deref().map(|cc| Box::new(plain(cc))),
            }
        }
        _ => Node {
            kind: node.kind,
            nullable: node.nullable,
            child: node.child.as_deref().map(|c| Box::new(plain(c))),
        },
    }
}

pub fn arrow_type(node: &Node) -> DataType {
    match node.kind {
        Kind::Int => DataType::Int32,
        Kind::Dict => DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        Kind::List => DataType::List(Arc::new(item_field(child(node)))),
        Kind::Struct => DataType::Struct(struct_fields(node)),
        Kind::Fsl2 => DataType::FixedSizeList(Arc::new(item_field(child(node))), 2),
        Kind::Fsl0 => DataType::FixedSizeList(Arc::new(item_field(child(node))), 0),
        Kind::DictOf => {
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(arrow_type(child(node))))
        }
    }
}

fn item_field(c: &Node) -> Field {
    Field::new("item", arrow_type(c), c.nullable)
}

fn struct_fields(node: &Node) -> Fields {
    let c = child(node);
    Fields::from(vec![Field::new("a", arrow_type(c), c.nullable)])
}

/// Up to six distinct values of `node` (including null when nullable).
pub fn vals(node: &Node) -> Vec<Val> {
    let mut v: Vec<Val> = match node.kind {
        Kind::Int => vec![Val::I(1), Val::I(2)],
        Kind::Dict => vec![Val::S("x"), Val::S("y")],
        Kind::List => {
            let cv = vals(child(node));
            let mut v = vec![Val::L(vec![])];
            v.extend(cv.iter().take(3).map(|x| Val::L(vec![x.clone()])));
            for x in cv.iter().take(3) {
                for y in cv.iter().take(3) {
                    v.push(Val::L(vec![x.clone(), y.clone()]));
                }
            }
            v
        }
        Kind::Fsl2 => {
            let cv = vals(child(node));
            let mut v = Vec::new();
            for x in cv.iter().take(3) {
                for y in cv.iter().take(3) {
                    v.push(Val::L(vec![x.clone(), y.clone()]));
                }
            }
            v
        }
        Kind::Fsl0 => vec![Val::L(vec![])],
        Kind::Struct => vals(child(node))
            .into_iter()
            .map(|x| Val::St(Box::new(x)))
            .collect(),
        Kind::DictOf => vals(child(node)),
    };
    v.truncate(5);
    if node.nullable {
        v.insert(0, Val::Null);
    }
    v.truncate(6);
    v
}

/// A filler for a position that is not an element: null when allowed (`alt = false`), else the
/// first (`alt = false`) or last (`alt = true`) non-null value.
fn filler(node: &Node, alt: bool) -> Val {
    if node.nullable && !alt {
        return Val::Null;
    }
    let mut non_null = vals(node).into_iter().filter(|v| !matches!(v, Val::Null));
    if alt {
        non_null.next_back().unwrap()
    } else {
        non_null.next().unwrap()
    }
}

fn validity(rows: &[Val]) -> NullBuffer {
    NullBuffer::from(
        rows.iter()
            .map(|v| !matches!(v, Val::Null))
            .collect::<Vec<_>>(),
    )
}

/// Build `rows` as an array of `node`; `alt` selects the second physical layout.
#[expect(clippy::too_many_lines, reason = "one flat match arm per kind")]
pub fn build(node: &Node, rows: &[Val], alt: bool) -> ArrayRef {
    let nulls = node.nullable.then(|| validity(rows));
    match node.kind {
        Kind::Int => Arc::new(Int32Array::new(
            rows.iter()
                .map(|v| match v {
                    Val::I(i) => *i,
                    _ if alt => 99,
                    _ => 0,
                })
                .collect(),
            nulls,
        )),
        Kind::Dict => {
            let strings = StringArray::from(
                rows.iter()
                    .map(|v| match v {
                        Val::S(s) => Some(*s),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            );
            cast(&strings, &arrow_type(node)).unwrap()
        }
        Kind::List => {
            let c = child(node);
            let mut lengths = Vec::new();
            let mut flat = Vec::new();
            for v in rows {
                match v {
                    Val::L(items) => {
                        lengths.push(items.len());
                        flat.extend(items.iter().cloned());
                    }
                    // a null list value covering a hidden item
                    _ if alt => {
                        lengths.push(1);
                        flat.push(filler(c, true));
                    }
                    _ => lengths.push(0),
                }
            }
            Arc::new(ListArray::new(
                Arc::new(item_field(c)),
                OffsetBuffer::<i32>::from_lengths(lengths),
                build(c, &flat, alt),
                nulls,
            ))
        }
        Kind::Fsl2 | Kind::Fsl0 => {
            let c = child(node);
            let size: i32 = if node.kind == Kind::Fsl2 { 2 } else { 0 };
            let mut flat = Vec::new();
            for v in rows {
                match v {
                    Val::L(items) => flat.extend(items.iter().cloned()),
                    _ => flat.extend(repeat_n(filler(c, alt), usize::try_from(size).unwrap())),
                }
            }
            Arc::new(
                FixedSizeListArray::try_new_with_length(
                    Arc::new(item_field(c)),
                    size,
                    build(c, &flat, alt),
                    nulls,
                    rows.len(),
                )
                .unwrap(),
            )
        }
        Kind::Struct => {
            let c = child(node);
            let inner: Vec<Val> = rows
                .iter()
                .map(|v| match v {
                    Val::St(x) => (**x).clone(),
                    _ => filler(c, alt),
                })
                .collect();
            Arc::new(StructArray::new(
                struct_fields(node),
                vec![build(c, &inner, alt)],
                nulls,
            ))
        }
        Kind::DictOf => {
            let c = child(node);
            let mut values: Vec<Val> = rows
                .iter()
                .map(|v| match v {
                    Val::Null => filler(c, alt),
                    other => other.clone(),
                })
                .collect();
            let n = values.len();
            let keys: Vec<Option<i8>> = rows
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let key = if alt { n - 1 - i } else { i };
                    (!matches!(v, Val::Null)).then(|| i8::try_from(key).unwrap())
                })
                .collect();
            if alt {
                values.reverse();
                values.push(filler(c, false)); // an unused dictionary entry
            }
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(Int8Array::from(keys), build(c, &values, alt))
                    .unwrap(),
            )
        }
    }
}

/// Deterministic row sets: empty, all singles, all pairs, and 25 pseudo-random triples.
pub fn row_sets(vs: &[Val]) -> Vec<Vec<Val>> {
    let mut sets: Vec<Vec<Val>> = vec![vec![]];
    sets.extend(vs.iter().map(|v| vec![v.clone()]));
    for a in vs {
        for b in vs {
            sets.push(vec![a.clone(), b.clone()]);
        }
    }
    let mut state: u64 = 7;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(state >> 33).unwrap() % vs.len()
    };
    for _ in 0..25 {
        let row: Vec<Val> = repeat_with(|| vs[next()].clone()).take(3).collect();
        sets.push(row);
    }
    sets
}
