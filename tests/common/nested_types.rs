//! Nested Arrow column types and small logical values for the collision tests (shared by
//! `tests/nested_family.rs` and `tests/legacy_v0_4_differential.rs`).

#![expect(clippy::unwrap_used, reason = "Okay in test")]
#![expect(clippy::indexing_slicing, reason = "Okay in test")]
#![expect(
    clippy::arithmetic_side_effects,
    clippy::integer_division_remainder_used,
    reason = "Okay in test: tiny deterministic generator"
)]

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, LargeListArray, ListArray, StringArray, StructArray};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::cast;
use arrow_schema::{DataType, Field, Fields};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Int,
    Dict,
    List,
    LargeList,
    Struct,
    Struct2,
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
    /// Struct `{a: v}` (and `b: 7` for `Struct2`).
    St(Box<Self>),
}

pub const fn leaf(kind: Kind, nullable: bool) -> Node {
    Node {
        kind,
        nullable,
        child: None,
    }
}

pub fn wrap(kind: Kind, nullable: bool, child: &Node) -> Node {
    Node {
        kind,
        nullable,
        child: Some(Box::new(child.clone())),
    }
}

pub fn child(node: &Node) -> &Node {
    node.child.as_deref().unwrap()
}

pub fn trees(depth: usize) -> Vec<Node> {
    let mut out: Vec<Node> = [Kind::Int, Kind::Dict]
        .iter()
        .flat_map(|&k| [leaf(k, true), leaf(k, false)])
        .collect();
    if depth > 0 {
        for c in trees(depth - 1) {
            for kind in [Kind::List, Kind::Struct] {
                for nullable in [true, false] {
                    out.push(wrap(kind, nullable, &c));
                }
            }
        }
    }
    out
}

pub fn all_types() -> Vec<Node> {
    let mut set: BTreeSet<Node> = trees(3).into_iter().collect();
    for c in trees(1) {
        for nullable in [true, false] {
            set.insert(wrap(Kind::LargeList, nullable, &c));
            set.insert(wrap(Kind::Struct2, nullable, &c));
        }
    }
    set.into_iter().collect()
}

pub fn arrow_type(node: &Node) -> DataType {
    match node.kind {
        Kind::Int => DataType::Int32,
        Kind::Dict => DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        Kind::List => DataType::List(Arc::new(item_field(child(node)))),
        Kind::LargeList => DataType::LargeList(Arc::new(item_field(child(node)))),
        Kind::Struct | Kind::Struct2 => DataType::Struct(struct_fields(node)),
    }
}

pub fn item_field(c: &Node) -> Field {
    Field::new("item", arrow_type(c), c.nullable)
}

pub fn struct_fields(node: &Node) -> Fields {
    let c = child(node);
    let mut fields = vec![Field::new("a", arrow_type(c), c.nullable)];
    if node.kind == Kind::Struct2 {
        fields.push(Field::new("b", DataType::Int32, false));
    }
    Fields::from(fields)
}

/// Up to six distinct values of `node` (including null when nullable).
pub fn vals(node: &Node) -> Vec<Val> {
    let mut v: Vec<Val> = match node.kind {
        Kind::Int => vec![Val::I(1), Val::I(2)],
        Kind::Dict => vec![Val::S("x"), Val::S("y")],
        Kind::List | Kind::LargeList => {
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
        Kind::Struct | Kind::Struct2 => vals(child(node))
            .into_iter()
            .map(|x| Val::St(Box::new(x)))
            .collect(),
    };
    v.truncate(5);
    if node.nullable {
        v.insert(0, Val::Null);
    }
    v.truncate(6);
    v
}

/// A valid placeholder for a non-null child under a null parent.
pub fn placeholder(node: &Node) -> Val {
    if node.nullable {
        Val::Null
    } else {
        vals(node)
            .into_iter()
            .find(|v| !matches!(v, Val::Null))
            .unwrap()
    }
}

pub fn validity(rows: &[Val]) -> NullBuffer {
    NullBuffer::from(
        rows.iter()
            .map(|v| !matches!(v, Val::Null))
            .collect::<Vec<_>>(),
    )
}

pub fn build(node: &Node, rows: &[Val]) -> ArrayRef {
    let nulls = node.nullable.then(|| validity(rows));
    match node.kind {
        Kind::Int => Arc::new(Int32Array::from(
            rows.iter()
                .map(|v| match v {
                    Val::I(i) => Some(*i),
                    _ => None,
                })
                .collect::<Vec<_>>(),
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
        Kind::List | Kind::LargeList => {
            let c = child(node);
            let mut lengths = Vec::new();
            let mut flat = Vec::new();
            for v in rows {
                match v {
                    Val::L(items) => {
                        lengths.push(items.len());
                        flat.extend(items.iter().cloned());
                    }
                    _ => lengths.push(0),
                }
            }
            let values = build(c, &flat);
            let item = Arc::new(item_field(c));
            if node.kind == Kind::List {
                Arc::new(ListArray::new(
                    item,
                    OffsetBuffer::<i32>::from_lengths(lengths),
                    values,
                    nulls,
                ))
            } else {
                Arc::new(LargeListArray::new(
                    item,
                    OffsetBuffer::<i64>::from_lengths(lengths),
                    values,
                    nulls,
                ))
            }
        }
        Kind::Struct | Kind::Struct2 => {
            let c = child(node);
            let inner: Vec<Val> = rows
                .iter()
                .map(|v| match v {
                    Val::St(x) => (**x).clone(),
                    _ => placeholder(c),
                })
                .collect();
            let mut columns = vec![build(c, &inner)];
            if node.kind == Kind::Struct2 {
                columns.push(Arc::new(Int32Array::from(vec![7; rows.len()])));
            }
            Arc::new(StructArray::new(struct_fields(node), columns, nulls))
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
        let row: Vec<Val> = vec![vs[next()].clone(), vs[next()].clone(), vs[next()].clone()];
        sets.push(row);
    }
    sets
}
