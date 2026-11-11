//! Collision search over nested container types (ITL-898).
//!
//! Every column type built from list / large list / struct / struct-with-a-second-field, each
//! nullable or not, around `Int32` and `Dictionary<Int8, Utf8>` leaves, up to three container
//! levels (420 types). For each type, ~60 small tables (0–3 rows) are hashed; two tables with
//! different logical values must never share a hash. v0.4.0 lost information in 262 of these
//! types (nullable structs; lists whose element is a nullable list or struct); v0.5 fixes them
//! (byte-layout spec Section 10). Checked in the default mode and with both order options on.

#[path = "common/nested_types.rs"]
mod nested_types;

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]

    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow_schema::{Field, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    use crate::nested_types::{all_types, arrow_type, build, row_sets, vals};

    #[test]
    fn nested_types_never_collide() {
        let types = all_types();
        assert_eq!(types.len(), 420);
        let mut failures = Vec::new();
        for node in &types {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "c",
                arrow_type(node),
                node.nullable,
            )]));
            let sets = row_sets(&vals(node));
            for (flags, config) in [
                (0_u8, HasherConfig::default()),
                (
                    3,
                    HasherConfig {
                        ordered_columns: true,
                        ordered_struct_fields: true,
                        ..HasherConfig::default()
                    },
                ),
            ] {
                let mut seen: HashMap<Vec<u8>, String> = HashMap::new();
                for rows in &sets {
                    let column = build(node, rows);
                    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
                    let key = format!("{rows:?}");
                    let hash = ArrowDigester::hash_record_batch(&batch, config);
                    if let Some(other) = seen.get(&hash) {
                        if *other != key {
                            failures.push(format!("{node:?} flags {flags}: {other} == {key}"));
                        }
                    } else {
                        seen.insert(hash, key);
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} collisions:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
