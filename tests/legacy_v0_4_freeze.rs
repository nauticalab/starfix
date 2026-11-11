//! Freeze guard for the legacy 0.4 mode (`starfix::legacy::v0_4`).
//!
//! The legacy module must reproduce starfix 0.4.0 byte for byte, forever. Its source, the
//! restored v0.4.0 tests and the format-1 golden file are pinned by SHA-256 here, so any edit
//! fails loudly. Do not update a hash below unless the change has been reviewed as preserving
//! every 0.4.0 output (the differential test `legacy_v0_4_differential.rs` must still pass).

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "Okay in test")]

    use std::fs;

    use sha2::{Digest as _, Sha256};

    const FROZEN: &[(&str, &str)] = &[
        (
            "src/legacy/v0_4.rs",
            "0fb07968ee1bd3086ba6a4f23baa0b21ac8e3a60d97d9728bd2de1c20e39ce5b",
        ),
        (
            "src/legacy/v0_4/arrow_digester_core.rs",
            "e1d0ab3fa3ff47ef1f856c1f7e70cb939114199b751aece2ae407803a7af9448",
        ),
        (
            "tests/legacy_v0_4_arrow_digester.rs",
            "d9e9ce8d3f68028b4a4442166e5b63213e95ca7d320a0c10ed4bb320a7d9302e",
        ),
        (
            "tests/legacy_v0_4_digest_bytes.rs",
            "6aca7d3271dc7110ab2a9f6e3e12e9e649d819f2f038922825b7c02ccfecaba1",
        ),
        (
            "tests/legacy_v0_4_golden_vectors.rs",
            "2cf3f1aad7e385890e53380faa0692129e5168949556665766f289eeb3a424aa",
        ),
        (
            "tests/golden/include_metadata_v0.3.json",
            "505d646b7898009f42b1a466a699f7d08774049a14d84fb3b2c75aff232c4bff",
        ),
    ];

    #[test]
    fn legacy_v0_4_files_are_frozen() {
        for (path, expected) in FROZEN {
            let full = format!("{}/{path}", env!("CARGO_MANIFEST_DIR"));
            let bytes = fs::read(&full).expect("frozen legacy file exists");
            let actual = hex::encode(Sha256::digest(&bytes));
            assert_eq!(
                &actual, expected,
                "{path} changed: the legacy 0.4 mode is frozen (see this file's header)"
            );
        }
    }
}
