//! Golden metadata fixture generator for PLT-1735.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin emit_golden_metadata > tests/golden/include_metadata_v0.3.json
//! ```
//!
//! Then copy the output file to `starfix-python/tests/golden/include_metadata_v0.3.json`.
//!
//! # When to regenerate
//!
//! Only regenerate when the hash algorithm changes intentionally. The committed fixture
//! is the authoritative source — `cargo test` reads it and will fail if the hasher
//! output no longer matches.

#![expect(clippy::unwrap_used, reason = "CLI tool — panics are acceptable")]

use std::collections::HashMap;
use std::process::Command;

use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde::Serialize;
use starfix::{ArrowDigester, HasherConfig};

#[derive(Serialize)]
struct GoldenFixture {
    version: String,
    generated_by: String,
    rust_commit: String,
    vectors: Vec<GoldenVector>,
}

#[derive(Serialize)]
struct GoldenVector {
    id: String,
    description: String,
    method: String,
    include_metadata: bool,
    ipc_b64: String,
    expected_hash: String,
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_or_else(
            |_| "unknown".to_owned(),
            |o| {
                if o.status.success() {
                    String::from_utf8_lossy(&o.stdout).trim().to_owned()
                } else {
                    "unknown".to_owned()
                }
            },
        )
}

fn schema_to_ipc_b64(schema: &Schema) -> String {
    let mut buf: Vec<u8> = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, schema).unwrap();
    writer.finish().unwrap();
    BASE64_STANDARD.encode(&buf)
}

fn hash_schema_hex(schema: &Schema, include_metadata: bool) -> String {
    let config = HasherConfig { include_metadata };
    hex::encode(ArrowDigester::hash_schema(schema, config))
}

fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|&(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "each golden vector is a self-contained fixture block; splitting would reduce clarity"
)]
fn build_vectors() -> Vec<GoldenVector> {
    let mut vectors: Vec<GoldenVector> = Vec::new();

    // ── 1. no_metadata_include_false ────────────────────────────────────────
    {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::LargeUtf8, true),
        ]);
        vectors.push(GoldenVector {
            id: "no_metadata_include_false".to_owned(),
            description: "Schema {id: Int64, name: LargeUtf8}, no metadata, include_metadata=false"
                .to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: false,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, false),
        });
    };

    // ── 2. schema_level_metadata ─────────────────────────────────────────────
    {
        let schema = Schema::new_with_metadata(
            vec![Field::new("id", DataType::Int64, false)],
            meta(&[("version", "2")]),
        );
        vectors.push(GoldenVector {
            id: "schema_level_metadata".to_owned(),
            description: "Schema-level metadata {version: 2}, include_metadata=true".to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, true),
        });
    };

    // ── 3. field_metadata_single_field ───────────────────────────────────────
    {
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int32, false).with_metadata(meta(&[("unit", "kg")]))
        ]);
        vectors.push(GoldenVector {
            id: "field_metadata_single_field".to_owned(),
            description: "Single field x: Int32 with metadata {unit: kg}, include_metadata=true"
                .to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, true),
        });
    };

    // ── 4. field_metadata_multiple_fields ────────────────────────────────────
    {
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int32, false).with_metadata(meta(&[("unit", "kg")])),
            Field::new("y", DataType::Float64, false).with_metadata(meta(&[("unit", "m")])),
        ]);
        vectors.push(GoldenVector {
            id: "field_metadata_multiple_fields".to_owned(),
            #[expect(
                clippy::literal_string_with_formatting_args,
                reason = "description text contains metadata key syntax, not format args"
            )]
            description: "Two fields x:{unit:kg}, y:{unit:m}, include_metadata=true".to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, true),
        });
    };

    // ── 5. schema_and_field_metadata ─────────────────────────────────────────
    {
        let schema = Schema::new_with_metadata(
            vec![Field::new("x", DataType::Int32, false).with_metadata(meta(&[("unit", "kg")]))],
            meta(&[("version", "1")]),
        );
        vectors.push(GoldenVector {
            id: "schema_and_field_metadata".to_owned(),
            #[expect(
                clippy::literal_string_with_formatting_args,
                reason = "description text contains metadata key syntax, not format args"
            )]
            description:
                "Schema metadata {version:1} + field metadata {unit:kg}, include_metadata=true"
                    .to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, true),
        });
    };

    // ── 6. unicode_metadata ───────────────────────────────────────────────────
    {
        let schema = Schema::new(vec![Field::new("data", DataType::LargeUtf8, false)
            .with_metadata(meta(&[
                ("emoji_key_\u{1F511}", "value_\u{2713}"),
                ("\u{4E2D}\u{6587}", "\u{65E5}\u{672C}\u{8A9E}"),
            ]))]);
        vectors.push(GoldenVector {
            id: "unicode_metadata".to_owned(),
            description: "Field metadata with emoji and CJK keys/values, include_metadata=true"
                .to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, true),
        });
    };

    // ── 7. key_reorder_canonical ──────────────────────────────────────────────
    {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)
            .with_metadata(meta(&[("alpha", "1"), ("beta", "2"), ("gamma", "3")]))]);
        let expected = hash_schema_hex(&schema, true);
        vectors.push(GoldenVector {
            id: "key_reorder_canonical".to_owned(),
            description: "Field metadata {alpha,beta,gamma} inserted in alphabetical order, include_metadata=true".to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: expected,
        });
    };

    // ── 8. key_reorder_shuffled ───────────────────────────────────────────────
    // Same logical metadata as canonical; HashMap iteration may produce same IPC bytes
    // within one process run, but the hash is always identical — that is the invariant.
    {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)
            .with_metadata(meta(&[("gamma", "3"), ("alpha", "1"), ("beta", "2")]))]);
        vectors.push(GoldenVector {
            id: "key_reorder_shuffled".to_owned(),
            description: "Same metadata {alpha,beta,gamma} inserted in shuffled order — expected_hash must equal key_reorder_canonical".to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: true,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, true),
        });
    };

    // ── 9. empty_metadata_invariant ───────────────────────────────────────────
    // No metadata at all. Hash with include_metadata=false and include_metadata=true
    // must be identical. We pin include_metadata=false as the fixture entry;
    // both Rust and Python tests additionally assert the true variant equals this.
    {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::LargeUtf8, true),
        ]);
        vectors.push(GoldenVector {
            id: "empty_metadata_invariant".to_owned(),
            description: "Schema with no metadata — hash(include_metadata=false) must equal hash(include_metadata=true)".to_owned(),
            method: "hash_schema".to_owned(),
            include_metadata: false,
            ipc_b64: schema_to_ipc_b64(&schema),
            expected_hash: hash_schema_hex(&schema, false),
        });
    };

    vectors
}

fn main() {
    let fixture = GoldenFixture {
        version: "0.3".to_owned(),
        generated_by: "cargo run --bin emit_golden_metadata".to_owned(),
        rust_commit: git_sha(),
        vectors: build_vectors(),
    };

    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
