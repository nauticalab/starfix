//! Prints the reference digests of the legacy 0.4 differential corpus, computed with the
//! **published** starfix 0.4.0 crate:
//!
//! ```bash
//! cd tools/legacy_v0_4_reference
//! cargo run --release > ../../tests/golden/legacy_v0_4_reference_digests.json
//! ```
//!
//! `tests/legacy_v0_4_differential.rs` recomputes the same digests with
//! `starfix::legacy::v0_4` and requires them to be equal.

#[path = "../../../tests/common/leaf_types.rs"]
mod leaf_types;
#[path = "../../../tests/common/legacy_corpus.rs"]
mod legacy_corpus;
#[path = "../../../tests/common/nested_types.rs"]
mod nested_types;

use std::path::Path;

use arrow::array::{Array, RecordBatch};
use arrow_schema::{Schema, SchemaRef};
use starfix::{ArrowDigester, HasherConfig};

fn schema(s: &Schema, include_metadata: bool) -> Vec<u8> {
    ArrowDigester::hash_schema(s, HasherConfig { include_metadata })
}

fn batch(b: &RecordBatch, include_metadata: bool) -> Vec<u8> {
    ArrowDigester::hash_record_batch(b, HasherConfig { include_metadata })
}

fn stream(s: &SchemaRef, batches: &[RecordBatch], include_metadata: bool) -> Vec<u8> {
    let mut d = ArrowDigester::new(s, HasherConfig { include_metadata });
    for b in batches {
        d.update(b);
    }
    d.finalize()
}

fn array(a: &dyn Array) -> Vec<u8> {
    ArrowDigester::hash_array(a)
}

fn main() {
    let api = legacy_corpus::Api {
        schema,
        batch,
        stream,
        array,
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let groups: Vec<serde_json::Value> = legacy_corpus::group_digests(&api, &root)
        .into_iter()
        .map(|(name, cases, digest)| {
            serde_json::json!({ "name": name, "cases": cases, "digest": digest })
        })
        .collect();
    let doc = serde_json::json!({
        "description": "Digests of the legacy 0.4 differential corpus (tests/common/legacy_corpus.rs) computed with the published starfix 0.4.0 crate; tests/legacy_v0_4_differential.rs requires starfix::legacy::v0_4 to produce the same.",
        "generated_by": "cd tools/legacy_v0_4_reference && cargo run --release",
        "starfix_release": "0.4.0",
        "groups": groups,
    });
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
}
