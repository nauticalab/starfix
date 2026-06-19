//! Regression guard for the golden metadata fixture.
//!
//! Reads `tests/golden/include_metadata_v0.3.json`, re-runs `ArrowDigester` on each
//! vector's IPC blob, and asserts that the output matches the committed `expected_hash`.
//! If this test fails, either the fixture is stale (regenerate with
//! `cargo run --bin emit_golden_metadata`) or the hasher has changed unexpectedly.

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::expect_used, reason = "Okay in test")]
    #![expect(clippy::panic, reason = "Okay in test")]

    use std::collections::HashMap;
    use std::fs;
    use std::io::Cursor;
    use std::result::Result;
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow_ipc::reader::StreamReader;
    use arrow_schema::Schema;
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;
    use hex::encode;
    use serde::Deserialize;
    use starfix::{ArrowDigester, HasherConfig};

    #[derive(Deserialize)]
    struct GoldenFixture {
        vectors: Vec<GoldenVector>,
    }

    #[derive(Deserialize)]
    struct GoldenVector {
        id: String,
        description: String,
        method: String,
        include_metadata: bool,
        ipc_b64: String,
        expected_hash: String,
    }

    fn load_fixture() -> GoldenFixture {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/include_metadata_v0.3.json"
        );
        let raw = fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("fixture not found at {path} — run `cargo run --bin emit_golden_metadata > tests/golden/include_metadata_v0.3.json`"));
        serde_json::from_str(&raw).unwrap()
    }

    fn decode_ipc(ipc_b64: &str) -> (Arc<Schema>, Option<RecordBatch>) {
        let bytes = STANDARD.decode(ipc_b64).unwrap();
        let cursor = Cursor::new(bytes);
        let mut reader = StreamReader::try_new(cursor, None).unwrap();
        let schema = reader.schema();
        let batch = reader.next().and_then(Result::ok);
        (schema, batch)
    }

    #[test]
    fn golden_vectors_match() {
        let fixture = load_fixture();
        assert!(
            !fixture.vectors.is_empty(),
            "fixture must contain at least one vector"
        );

        for vector in &fixture.vectors {
            let (schema, maybe_batch) = decode_ipc(&vector.ipc_b64);
            let config = HasherConfig {
                include_metadata: vector.include_metadata,
            };

            let result = match vector.method.as_str() {
                "hash_schema" => encode(ArrowDigester::hash_schema(&schema, config)),
                "hash_record_batch" => {
                    let batch = maybe_batch.unwrap_or_else(|| {
                        panic!(
                            "vector '{}': method is hash_record_batch but IPC has no batch",
                            vector.id
                        )
                    });
                    encode(ArrowDigester::hash_record_batch(&batch, config))
                }
                other => panic!("vector '{}': unknown method '{other}'", vector.id),
            };

            assert_eq!(
                result, vector.expected_hash,
                "vector '{}' mismatch: {}\n  got:      {result}\n  expected: {}",
                vector.id, vector.description, vector.expected_hash
            );
        }
    }

    #[test]
    fn empty_metadata_invariant_both_flags() {
        let fixture = load_fixture();
        let vector = fixture
            .vectors
            .iter()
            .find(|v| v.id == "empty_metadata_invariant")
            .expect("empty_metadata_invariant vector must exist in fixture");

        let (schema, _) = decode_ipc(&vector.ipc_b64);

        let hash_false = encode(ArrowDigester::hash_schema(
            &schema,
            HasherConfig {
                include_metadata: false,
            },
        ));
        let hash_true = encode(ArrowDigester::hash_schema(
            &schema,
            HasherConfig {
                include_metadata: true,
            },
        ));

        assert_eq!(
            hash_false, vector.expected_hash,
            "empty_metadata_invariant: include_metadata=false must match pinned hash"
        );
        assert_eq!(
            hash_true, vector.expected_hash,
            "empty_metadata_invariant: include_metadata=true must match pinned hash"
        );
        assert_eq!(
            hash_false, hash_true,
            "empty_metadata_invariant: both flag values must produce the same hash"
        );
    }

    #[test]
    fn key_reorder_hashes_are_identical() {
        let fixture = load_fixture();
        let vecs: HashMap<&str, &GoldenVector> =
            fixture.vectors.iter().map(|v| (v.id.as_str(), v)).collect();

        let canonical = vecs
            .get("key_reorder_canonical")
            .expect("key_reorder_canonical must exist");
        let shuffled = vecs
            .get("key_reorder_shuffled")
            .expect("key_reorder_shuffled must exist");

        assert_eq!(
            canonical.expected_hash, shuffled.expected_hash,
            "key_reorder_canonical and key_reorder_shuffled must have identical expected_hash"
        );
    }
}
