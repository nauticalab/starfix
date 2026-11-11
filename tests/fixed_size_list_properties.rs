//! Collision and canonicality search over column types containing `FixedSizeList` (ITL-927).
//!
//! Every type of up to three container levels built from list / struct / fixed-size list of
//! size 2 (each nullable or not, around `Int32` and `Dictionary<Int8, Utf8>` leaves) that
//! contains a fixed-size list, plus size-0 fixed-size lists and dictionary-encoded fixed-size
//! lists at every container position (`tests/common/fsl_family.rs`). For each type ~60 small
//! tables (0–3 rows) are hashed. Checked:
//!
//! * **collisions**: two tables with different logical values never share a record-batch hash
//!   (default mode and both order options) or a `hash_array` hash;
//! * **canonicality**: a second physical layout (other content under null slots / null rows /
//!   null list values, other dictionary keys), a slice with non-zero offset, one batch per row,
//!   and the plain (non-dictionary) twin type all hash like the original;
//! * **panics**: no hash call panics.

#[path = "common/fsl_family.rs"]
mod fsl_family;

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::print_stderr, reason = "Report the search volume")]

    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use arrow::array::{ArrayRef, RecordBatch};
    use arrow_schema::{Field, Schema};
    use starfix::{ArrowDigester, HasherConfig};

    use crate::fsl_family::{arrow_type, build, plain, row_sets, types, vals, Node, Val};

    fn batch(node: &Node, column: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c",
            arrow_type(node),
            node.nullable,
        )]));
        RecordBatch::try_new(schema, vec![column]).unwrap()
    }

    const ORDERED: HasherConfig = HasherConfig {
        include_metadata: false,
        ordered_columns: true,
        ordered_struct_fields: true,
    };

    #[derive(Default)]
    struct Tally {
        types: usize,
        tables: usize,
        hashes: usize,
        failures: Vec<String>,
    }

    impl Tally {
        fn hash_rb(&mut self, b: &RecordBatch, config: HasherConfig) -> Vec<u8> {
            self.hashes = self.hashes.checked_add(1).unwrap();
            ArrowDigester::hash_record_batch(b, config)
        }

        fn hash_array(&mut self, column: &ArrayRef) -> Vec<u8> {
            self.hashes = self.hashes.checked_add(1).unwrap();
            ArrowDigester::hash_array_with_config(column.as_ref(), HasherConfig::default())
        }

        fn stream(&mut self, node: &Node, rows: &[Val]) -> Vec<u8> {
            self.hashes = self.hashes.checked_add(1).unwrap();
            let schema = batch(node, build(node, &[], false)).schema();
            let mut digester = ArrowDigester::new(&schema, HasherConfig::default());
            for row in rows.chunks(1) {
                digester.update(&batch(node, build(node, row, false)));
            }
            digester.finalize()
        }
    }

    /// The hashes of one table: record batch (default, ordered), `hash_array`.
    type Hashes = (Vec<u8>, Vec<u8>, Vec<u8>);

    fn check_table(tally: &mut Tally, node: &Node, rows: &[Val]) -> Hashes {
        let base = build(node, rows, false);
        let b = batch(node, Arc::clone(&base));
        let hashes = (
            tally.hash_rb(&b, HasherConfig::default()),
            tally.hash_rb(&b, ORDERED),
            tally.hash_array(&base),
        );
        let label = format!("{node:?} {rows:?}");

        // other content in positions that are not elements
        let alt = build(node, rows, true);
        if tally.hash_rb(&batch(node, Arc::clone(&alt)), HasherConfig::default()) != hashes.0
            || tally.hash_array(&alt) != hashes.2
        {
            tally
                .failures
                .push(format!("hidden content matters: {label}"));
        }

        // a slice with a non-zero offset
        let pad = vals(node).last().unwrap().clone();
        let mut padded = vec![pad.clone()];
        padded.extend(rows.iter().cloned());
        padded.push(pad);
        let padded_batch = batch(node, build(node, &padded, false));
        let sliced = padded_batch.slice(1, rows.len());
        if tally.hash_rb(&sliced, HasherConfig::default()) != hashes.0
            || tally.hash_array(sliced.column(0)) != hashes.2
        {
            tally.failures.push(format!("slice differs: {label}"));
        }

        // one batch per row
        if tally.stream(node, rows) != hashes.0 {
            tally.failures.push(format!("batch split differs: {label}"));
        }

        // dictionary-encoded fixed-size lists hash like the plain values
        let twin = plain(node);
        if twin != *node {
            let twin_column = build(&twin, rows, false);
            if tally.hash_rb(
                &batch(&twin, Arc::clone(&twin_column)),
                HasherConfig::default(),
            ) != hashes.0
                || tally.hash_array(&twin_column) != hashes.2
            {
                tally
                    .failures
                    .push(format!("dictionary differs from plain: {label}"));
            }
        }
        hashes
    }

    #[test]
    fn fixed_size_list_types_never_collide_and_are_canonical() {
        let mut tally = Tally::default();
        for node in &types() {
            tally.types = tally.types.checked_add(1).unwrap();
            let (mut seen0, mut seen3, mut seen_array) =
                <(HashMap<Vec<u8>, String>, HashMap<_, _>, HashMap<_, _>)>::default();
            for rows in &row_sets(&vals(node)) {
                tally.tables = tally.tables.checked_add(1).unwrap();
                let key = format!("{rows:?}");
                let outcome =
                    catch_unwind(AssertUnwindSafe(|| check_table(&mut tally, node, rows)));
                let Ok((h0, h3, ha)) = outcome else {
                    tally.failures.push(format!("panic: {node:?} {key}"));
                    continue;
                };
                for (map, hash, what) in [
                    (&mut seen0, h0, "record batch"),
                    (&mut seen3, h3, "record batch (ordered)"),
                    (&mut seen_array, ha, "hash_array"),
                ] {
                    match map.get(&hash) {
                        Some(other) if *other != key => tally
                            .failures
                            .push(format!("{what} collision {node:?}: {other} == {key}")),
                        Some(_) => {}
                        None => {
                            map.insert(hash, key.clone());
                        }
                    }
                }
            }
        }
        eprintln!(
            "fixed-size-list family: {} types, {} tables, {} hashes, {} failures",
            tally.types,
            tally.tables,
            tally.hashes,
            tally.failures.len()
        );
        assert!(tally.types > 900, "type count {}", tally.types);
        assert!(
            tally.failures.is_empty(),
            "{} failures:\n{}",
            tally.failures.len(),
            tally.failures.join("\n")
        );
    }
}
