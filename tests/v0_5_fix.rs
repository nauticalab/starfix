//! Permanent check of the v0.5 hash fix record, `tests/golden/fixed_in_v0.5.json` (ITL-898).
//!
//! For every reference vector: the current code reproduces its `v0_5_hash`; the
//! affected-type rule (byte-layout spec Section 10) computed here agrees with the record; and
//! every vector whose type is NOT affected keeps its v0.4.0 digest byte for byte (the 32 bytes
//! after the 3-byte version prefix, so this also holds if the prefix is bumped).

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "Okay in test")]
    #![expect(clippy::expect_used, reason = "Okay in test")]
    #![expect(clippy::panic, reason = "Okay in test")]

    use std::fs;
    use std::io::Cursor;

    use arrow::array::{Array as _, RecordBatch};
    use arrow_ipc::reader::StreamReader;
    use arrow_schema::{DataType, SchemaRef};
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;
    use serde::Deserialize;
    use starfix::{ArrowDigester, HasherConfig};

    #[derive(Deserialize)]
    struct FixRecord {
        reference_vectors: Vec<ReferenceVector>,
        existing_tests: Vec<ExistingTest>,
    }

    #[derive(Deserialize)]
    struct ReferenceVector {
        id: String,
        method: String,
        include_metadata: bool,
        ipc_b64: String,
        v0_4_0_hash: String,
        v0_5_hash: String,
        status: String,
        affected_type: bool,
    }

    #[derive(Deserialize)]
    struct ExistingTest {
        test: String,
        v0_4_0_hash: String,
        v0_5_hash: String,
    }

    fn load() -> FixRecord {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/fixed_in_v0.5.json"
        );
        serde_json::from_str(&fs::read_to_string(path).expect("fix record")).expect("parse")
    }

    fn decode(ipc_b64: &str) -> (SchemaRef, Vec<RecordBatch>) {
        let reader =
            StreamReader::try_new(Cursor::new(STANDARD.decode(ipc_b64).unwrap()), None).unwrap();
        let schema = reader.schema();
        (schema, reader.map(|b| b.unwrap()).collect())
    }

    /// The affected-type rule of byte-layout spec Section 10.
    fn affected(data_type: &DataType, nullable: bool) -> bool {
        match data_type {
            DataType::Dictionary(_, value) => affected(value, nullable),
            DataType::Boolean => true,
            DataType::Struct(fields) => {
                nullable
                    || fields
                        .iter()
                        .any(|f| affected(f.data_type(), f.is_nullable()))
            }
            DataType::List(element) | DataType::LargeList(element) => {
                let element_is_container = matches!(
                    element.data_type(),
                    DataType::List(_) | DataType::LargeList(_) | DataType::Struct(_)
                );
                (element.is_nullable() && element_is_container)
                    || affected(element.data_type(), element.is_nullable())
            }
            _ => false,
        }
    }

    fn digest(hash_hex: &str) -> &str {
        hash_hex
            .get(6..)
            .expect("hash longer than the version prefix")
    }

    #[test]
    fn fix_record_reference_vectors() {
        let record = load();
        assert!(!record.reference_vectors.is_empty());
        let mut unaffected = 0_usize;
        for v in &record.reference_vectors {
            let config = HasherConfig {
                include_metadata: v.include_metadata,
                ..HasherConfig::default()
            };
            let (schema, batches) = decode(&v.ipc_b64);
            let (hash, is_affected) = match v.method.as_str() {
                "hash_schema" => (ArrowDigester::hash_schema(&schema, config), false),
                "hash_record_batch" | "hash_stream" => {
                    let mut d = ArrowDigester::new(&schema, config);
                    for b in &batches {
                        d.update(b);
                    }
                    let hit = schema
                        .fields()
                        .iter()
                        .any(|f| affected(f.data_type(), f.is_nullable()));
                    (d.finalize(), hit)
                }
                "hash_array" => {
                    let array = batches.first().expect("one batch").column(0);
                    (
                        ArrowDigester::hash_array(array.as_ref()),
                        affected(array.data_type(), array.null_count() > 0),
                    )
                }
                other => panic!("{}: unknown method {other}", v.id),
            };
            let hash_hex = hex::encode(hash);
            assert_eq!(hash_hex, v.v0_5_hash, "{}: v0.5 value", v.id);
            assert_eq!(is_affected, v.affected_type, "{}: affected-type rule", v.id);
            let moved = digest(&v.v0_4_0_hash) != digest(&hash_hex);
            assert_eq!(v.status == "fixed", moved, "{}: status", v.id);
            if !is_affected {
                unaffected += 1;
                assert!(
                    !moved,
                    "{}: unaffected type must keep its v0.4.0 digest",
                    v.id
                );
            }
        }
        assert!(unaffected > 0);
    }

    #[test]
    fn fix_record_existing_tests_all_moved() {
        let record = load();
        assert_eq!(record.existing_tests.len(), 7);
        for t in &record.existing_tests {
            assert_ne!(digest(&t.v0_4_0_hash), digest(&t.v0_5_hash), "{}", t.test);
        }
    }
}
