# include_metadata Hashing Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use sensei:subagent-driven-development (recommended) or sensei:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `HasherConfig { include_metadata: bool }` option that incorporates Arrow schema- and field-level metadata into the hash when `true`, while producing byte-for-byte identical hashes to v0.0.x when `false`.

**Architecture:** `HasherConfig` is a public struct in `lib.rs`. `ArrowDigesterCore` gains two private helpers (`update_metadata_hash`, `build_schema_equality_key`), a stored `include_metadata: bool` field (needed so `update()` can recompute the incoming schema's equality key), and a renamed field (`serialized_schema` → `schema_equality_key`). The three schema-touching methods on `ArrowDigester` (`new`, `hash_schema`, `hash_record_batch`) each gain a `config: HasherConfig` parameter. `pyarrow.rs` FFI entry points pass `HasherConfig::default()`. `VERSION_BYTES` is not changed.

**Tech Stack:** Rust, arrow-schema 57.0.0, serde_json, sha2/digest

---

## File Map

| File | Change |
|---|---|
| `src/lib.rs` | Add `HasherConfig`; update `ArrowDigester::new`, `hash_schema`, `hash_record_batch` |
| `src/arrow_digester_core.rs` | Rename field `serialized_schema` → `schema_equality_key`; add `include_metadata: bool` field; add `update_metadata_hash` and `build_schema_equality_key` helpers; update `new`, `hash_schema`, `hash_record_batch`, `update` |
| `src/pyarrow.rs` | Import `HasherConfig`; pass `HasherConfig::default()` at 3 call sites |
| `Cargo.toml` | Version `0.0.2` → `0.1.0` |
| `tests/arrow_digester.rs` | Import `HasherConfig`; update existing call sites; add 8 new metadata tests |
| `tests/digest_bytes.rs` | Import `HasherConfig`; update existing call sites |

---

### Task 1: Add `HasherConfig` to `lib.rs`

**Files:** Modify `src/lib.rs`

- [ ] **Step 1: Add `HasherConfig` after the `VERSION_BYTES` constant (line 11)**

In `src/lib.rs`, after the line `const VERSION_BYTES: [u8; 3] = [0_u8, 0_u8, 1_u8]; // Version 0.0.1`, insert:

```rust
/// Configuration for the Arrow hasher.
#[derive(Clone, Copy, Debug)]
pub struct HasherConfig {
    /// When `true`, schema-level and per-field Arrow metadata are included in the hash.
    /// Default is `false`, preserving pre-v0.1.0 hash stability.
    pub include_metadata: bool,
}

impl Default for HasherConfig {
    fn default() -> Self {
        Self {
            include_metadata: false,
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo build 2>&1
```

Expected: no errors. Existing code is untouched — this is a pure addition.

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs
git commit -m "feat(core): add HasherConfig struct with include_metadata option (PLT-1733)"
```

---

### Task 2: Update `ArrowDigesterCore`, `ArrowDigester`, and `pyarrow.rs`

All code changes in one atomic task so the codebase compiles at the end. Existing tests will fail to compile until Task 3 updates their call sites — that is expected.

**Files:** Modify `src/arrow_digester_core.rs`, `src/lib.rs`, `src/pyarrow.rs`

#### 2a: `ArrowDigesterCore` struct

- [ ] **Step 1: Replace the struct definition**

In `src/arrow_digester_core.rs`, replace:

```rust
#[derive(Clone)]
pub struct ArrowDigesterCore<D: Digest> {
    schema_digest: Vec<u8>,
    serialized_schema: String,
    fields_digest_buffer: BTreeMap<String, DigestBufferType<D>>,
}
```

with:

```rust
#[derive(Clone)]
pub struct ArrowDigesterCore<D: Digest> {
    schema_digest: Vec<u8>,
    schema_equality_key: String,
    fields_digest_buffer: BTreeMap<String, DigestBufferType<D>>,
    include_metadata: bool,
}
```

#### 2b: `ArrowDigesterCore::new`

- [ ] **Step 2: Replace `new()` — remove the `shadow_reuse` expect, accept `include_metadata`**

Replace the entire `pub fn new` block:

```rust
    #[expect(
        clippy::shadow_reuse,
        reason = "Intentional: shadow input with normalized version so all downstream code uses canonical types"
    )]
    pub fn new(schema: &Schema) -> Self {
        // Normalize the schema so all internal state uses canonical large types
        let schema = normalize_schema(schema);

        // Hash the normalized schema
        let schema_digest = Self::hash_schema(&schema);

        // Flatten all nested fields into a single map, this allows us to hash each field individually and efficiently
        let mut fields_digest_buffer = BTreeMap::new();
        schema.fields.into_iter().for_each(|field| {
            Self::extract_fields_name(field, "", &mut fields_digest_buffer);
        });

        let serialized_schema = Self::serialized_schema(&schema);

        // Store it in the new struct for now
        Self {
            schema_digest,
            serialized_schema,
            fields_digest_buffer,
        }
    }
```

with:

```rust
    /// Create a new instance of `ArrowDigesterCore` with the schema, which will be enforced through each update.
    pub fn new(schema: &Schema, include_metadata: bool) -> Self {
        // Normalize the schema to canonical large types (normalize_schema preserves metadata)
        let normalized = normalize_schema(schema);

        // Hash the schema — hash_schema normalizes internally; passing original is equivalent
        let schema_digest = Self::hash_schema(schema, include_metadata);

        // Build the equality key used in update() to enforce schema identity
        let schema_equality_key = Self::build_schema_equality_key(&normalized, include_metadata);

        // Flatten all nested fields into a single map for per-field hashing
        let mut fields_digest_buffer = BTreeMap::new();
        for field in normalized.fields.into_iter() {
            Self::extract_fields_name(field, "", &mut fields_digest_buffer);
        }

        Self {
            schema_digest,
            schema_equality_key,
            fields_digest_buffer,
            include_metadata,
        }
    }
```

#### 2c: `ArrowDigesterCore::update`

- [ ] **Step 3: Replace `update()` to use `schema_equality_key` and `build_schema_equality_key`**

Replace the assertion at the top of `update()`:

```rust
    pub fn update(&mut self, record_batch: &RecordBatch) {
        assert!(
            Self::serialized_schema(record_batch.schema().as_ref()) == self.serialized_schema,
            "Record batch schema does not match ArrowDigester schema"
        );
```

with:

```rust
    pub fn update(&mut self, record_batch: &RecordBatch) {
        let rb_schema = record_batch.schema();
        let rb_normalized = normalize_schema(&rb_schema);
        let rb_equality_key = Self::build_schema_equality_key(
            &rb_normalized,
            rb_schema.as_ref(),
            self.include_metadata,
        );
        assert!(
            rb_equality_key == self.schema_equality_key,
            "Record batch schema does not match ArrowDigester schema"
        );
```

#### 2d: `ArrowDigesterCore::hash_record_batch`

- [ ] **Step 4: Update `hash_record_batch` to accept and thread `include_metadata`**

Replace:

```rust
    /// Hash a record batch directly without needing to create an `ArrowDigester` instance on the user side.
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        let mut digester = Self::new(record_batch.schema().as_ref());
        digester.update(record_batch);
        digester.finalize()
    }
```

with:

```rust
    /// Hash a record batch directly without needing to create an `ArrowDigester` instance on the user side.
    pub fn hash_record_batch(record_batch: &RecordBatch, include_metadata: bool) -> Vec<u8> {
        let mut digester = Self::new(record_batch.schema().as_ref(), include_metadata);
        digester.update(record_batch);
        digester.finalize()
    }
```

#### 2e: `ArrowDigesterCore::hash_schema`

- [ ] **Step 5: Replace `hash_schema` with two-phase implementation**

Replace:

```rust
    /// Hash the schema by serializing it to a canonical JSON string and computing its digest.
    pub fn hash_schema(schema: &Schema) -> Vec<u8> {
        // Hash the entire thing to the digest
        D::digest(Self::serialized_schema(schema)).to_vec()
    }
```

with:

```rust
    /// Hash the schema in two phases.
    ///
    /// Phase 1 (always): canonical JSON of field names, data types, nullability — identical to v0.0.x.
    /// Phase 2 (when `include_metadata` is `true`): per-field and schema-level metadata fed into the
    /// same hasher via `update_metadata_hash`. Phase 2 adds nothing when all metadata maps are empty,
    /// preserving the empty-metadata invariant.
    ///
    /// `schema` must be the original (pre-normalization) schema so that metadata is available for
    /// Phase 2. Normalization for Phase 1 is handled internally.
    pub fn hash_schema(schema: &Schema, include_metadata: bool) -> Vec<u8> {
        let normalized = normalize_schema(schema);
        let mut hasher = D::new();
        hasher.update(Self::serialized_schema(&normalized));
        if include_metadata {
            Self::update_metadata_hash(&mut hasher, schema);
        }
        hasher.finalize().to_vec()
    }
```

#### 2f: Add `update_metadata_hash` and `build_schema_equality_key` helpers

- [ ] **Step 6: Add the two private helpers inside the `impl ArrowDigesterCore<D>` block**

Add immediately after `hash_schema`:

```rust
    /// Feed per-field and schema-level metadata into `hasher` for Phase 2 of schema hashing.
    ///
    /// Builds a single JSON object with up to two keys — `"fields"` (BTreeMap of field path to
    /// sorted metadata map) and `"schema"` (sorted schema-level metadata map) — and feeds the
    /// serialized JSON as one `hasher.update()` call. `"fields"` sorts before `"schema"` in
    /// BTreeMap alphabetical order, so the key ordering is deterministic.
    ///
    /// Nothing is written when both the per-field and schema-level metadata maps are empty,
    /// preserving the empty-metadata invariant. JSON is self-delimiting so no length prefixes
    /// are needed.
    fn update_metadata_hash(hasher: &mut D, schema: &Schema) {
        let mut meta_doc: BTreeMap<&str, serde_json::Value> = BTreeMap::new();

        let field_meta: BTreeMap<&str, BTreeMap<&str, &str>> = schema
            .fields()
            .iter()
            .filter(|f| !f.metadata().is_empty())
            .map(|f| {
                let sorted: BTreeMap<&str, &str> =
                    f.metadata().iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                (f.name().as_str(), sorted)
            })
            .collect();

        if !field_meta.is_empty() {
            meta_doc.insert(
                "fields",
                serde_json::to_value(&field_meta)
                    .expect("Failed to serialize field metadata"),
            );
        }

        if !schema.metadata().is_empty() {
            let sorted_schema_meta: BTreeMap<&str, &str> = schema
                .metadata()
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            meta_doc.insert(
                "schema",
                serde_json::to_value(&sorted_schema_meta)
                    .expect("Failed to serialize schema metadata"),
            );
        }

        if !meta_doc.is_empty() {
            let json = serde_json::to_string(&meta_doc)
                .expect("Failed to serialize metadata document to string");
            hasher.update(json.as_bytes());
        }
    }

    /// Build a canonical string key for schema identity enforcement in `update()`.
    ///
    /// When `include_metadata` is `false`, or when the schema has no metadata anywhere,
    /// returns the structure-only JSON (v0.0.x format) — preserving the empty-metadata invariant.
    ///
    /// When `include_metadata` is `true` and metadata is present, appends `|` followed by a
    /// canonical JSON object `{"field_meta": {...}, "schema_meta": {...}}`. The `|` separator is
    /// unambiguous because JSON objects never end with `|`.
    fn build_schema_equality_key(
        normalized: &Schema,
        original: &Schema,
        include_metadata: bool,
    ) -> String {
        let structure = Self::serialized_schema(normalized);
        if !include_metadata {
            return structure;
        }
        let has_any_metadata = !original.metadata().is_empty()
            || original.fields().iter().any(|f| !f.metadata().is_empty());
        if !has_any_metadata {
            return structure; // empty-metadata invariant: same key as include_metadata=false
        }

        let field_meta: BTreeMap<&str, BTreeMap<&str, &str>> = original
            .fields()
            .iter()
            .filter(|f| !f.metadata().is_empty())
            .map(|f| {
                let sorted: BTreeMap<&str, &str> =
                    f.metadata().iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                (f.name().as_str(), sorted)
            })
            .collect();

        let schema_meta: BTreeMap<&str, &str> = original
            .metadata()
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let meta_json = serde_json::to_string(&serde_json::json!({
            "field_meta": field_meta,
            "schema_meta": schema_meta,
        }))
        .expect("Failed to serialize metadata equality key to string");

        format!("{structure}|{meta_json}")
    }
```

You'll also need `Arc` in scope for `Vec<&Arc<Field>>`. The file already imports `use std::sync::Arc;` at line 6 — no new import needed.

#### 2g: Update `ArrowDigester` in `lib.rs`

- [ ] **Step 7: Update the three schema-touching methods on `ArrowDigester`**

In `src/lib.rs`, replace:

```rust
    /// Create a new instance of `ArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    pub fn new(schema: &Schema) -> Self {
        Self {
            digester: ArrowDigesterCore::<Sha256>::new(schema),
        }
    }
```

with:

```rust
    /// Create a new instance of `ArrowDigester` with SHA-256 as the digest algorithm. The schema will be enforced on each update.
    pub fn new(schema: &Schema, config: HasherConfig) -> Self {
        Self {
            digester: ArrowDigesterCore::<Sha256>::new(schema, config.include_metadata),
        }
    }
```

Replace:

```rust
    /// Hash a complete `RecordBatch` in one go.
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_record_batch(record_batch))
    }
```

with:

```rust
    /// Hash a complete `RecordBatch` in one go.
    pub fn hash_record_batch(record_batch: &RecordBatch, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(
            ArrowDigesterCore::<Sha256>::hash_record_batch(record_batch, config.include_metadata),
        )
    }
```

Replace:

```rust
    /// Hash a schema only.
    pub fn hash_schema(schema: &Schema) -> Vec<u8> {
        Self::prepend_version_bytes(ArrowDigesterCore::<Sha256>::hash_schema(schema))
    }
```

with:

```rust
    /// Hash a schema only.
    pub fn hash_schema(schema: &Schema, config: HasherConfig) -> Vec<u8> {
        Self::prepend_version_bytes(
            ArrowDigesterCore::<Sha256>::hash_schema(schema, config.include_metadata),
        )
    }
```

#### 2h: Update `pyarrow.rs`

- [ ] **Step 8: Add `HasherConfig` import and update 3 call sites in `src/pyarrow.rs`**

Replace:

```rust
use crate::ArrowDigester;
```

with:

```rust
use crate::{ArrowDigester, HasherConfig};
```

Then update the three call sites:

Line ~39 — replace:
```rust
    ArrowDigester::hash_record_batch(&RecordBatch::from(StructArray::from(array_data)))
```
with:
```rust
    ArrowDigester::hash_record_batch(
        &RecordBatch::from(StructArray::from(array_data)),
        HasherConfig::default(),
    )
```

Line ~60 — replace:
```rust
    ArrowDigester::hash_schema(&schema)
```
with:
```rust
    ArrowDigester::hash_schema(&schema, HasherConfig::default())
```

Line ~84 — replace:
```rust
        Self {
            digester: Arc::new(Mutex::new(ArrowDigester::new(&schema))),
        }
```
with:
```rust
        Self {
            digester: Arc::new(Mutex::new(ArrowDigester::new(&schema, HasherConfig::default()))),
        }
```

- [ ] **Step 9: Verify `src/` compiles (tests will fail to compile — expected)**

```bash
cargo build 2>&1
```

Expected: `src/` builds cleanly. Test compilation errors about missing `HasherConfig` argument are expected and will be fixed in Task 3.

- [ ] **Step 10: Commit**

```bash
git add src/lib.rs src/arrow_digester_core.rs src/pyarrow.rs
git commit -m "feat(core): implement include_metadata two-phase schema hashing (PLT-1733)"
```

---

### Task 3: Update Existing Test Call Sites and Verify Regression

**Files:** Modify `tests/arrow_digester.rs`, `tests/digest_bytes.rs`

- [ ] **Step 1: Add `HasherConfig` import to both test files**

In `tests/arrow_digester.rs`, the imports block starts with `use starfix::ArrowDigester;`. Add:
```rust
use starfix::HasherConfig;
```

In `tests/digest_bytes.rs`, similarly add:
```rust
use starfix::HasherConfig;
```

- [ ] **Step 2: Update all `ArrowDigester::new(&schema)` call sites**

In both test files, replace every occurrence of:
```rust
ArrowDigester::new(&schema)
```
with:
```rust
ArrowDigester::new(&schema, HasherConfig::default())
```

Also catch the pattern `ArrowDigester::new(schema.as_ref())`:
```rust
ArrowDigester::new(schema.as_ref(), HasherConfig::default())
```

- [ ] **Step 3: Update all `ArrowDigester::hash_record_batch(...)` call sites**

Replace every occurrence of:
```rust
ArrowDigester::hash_record_batch(&batch)
ArrowDigester::hash_record_batch(batch1.as_ref().unwrap())
ArrowDigester::hash_record_batch(&batch1)
ArrowDigester::hash_record_batch(&batch2)
ArrowDigester::hash_record_batch(&combined)
// etc.
```
with the same but adding `, HasherConfig::default()` as the last argument. For example:
```rust
ArrowDigester::hash_record_batch(&batch, HasherConfig::default())
```

- [ ] **Step 4: Update all `ArrowDigester::hash_schema(...)` call sites**

Replace every occurrence of:
```rust
ArrowDigester::hash_schema(&schema)
ArrowDigester::hash_schema(&nullable_schema)
ArrowDigester::hash_schema(&non_nullable_schema)
ArrowDigester::hash_schema(&schema1)
ArrowDigester::hash_schema(&schema2)
// etc.
```
with the same but adding `, HasherConfig::default()`. For example:
```rust
ArrowDigester::hash_schema(&schema, HasherConfig::default())
```

Note: `ArrowDigester::hash_array(...)` call sites are **not changed** — `hash_array` has no `config` parameter.

- [ ] **Step 5: Run the full test suite and verify all existing tests pass**

```bash
cargo test 2>&1
```

Expected: all existing tests pass with the same golden hex values as before. No hash values change because `HasherConfig::default()` has `include_metadata: false`, which follows the same v0.0.x code path.

If any golden hash value changes, **stop** — there is a regression in the core hashing logic.

- [ ] **Step 6: Commit**

```bash
git add tests/arrow_digester.rs tests/digest_bytes.rs
git commit -m "test: update existing call sites for HasherConfig API (PLT-1733)"
```

---

### Task 4: Metadata Excluded by Default

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add failing test**

Inside the `mod tests` block in `tests/arrow_digester.rs`, add:

```rust
    #[test]
    fn metadata_excluded_by_default() {
        // Two schemas with identical structure but different field metadata
        // must hash identically when include_metadata = false (the default).
        let schema_no_meta = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let schema_with_meta = Schema::new(vec![
            Field::new("x", DataType::Int32, false).with_metadata(
                [("ARROW:extension:name".to_string(), "my_ext".to_string())].into(),
            ),
        ]);

        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema_no_meta, HasherConfig::default())),
            encode(ArrowDigester::hash_schema(&schema_with_meta, HasherConfig::default())),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test metadata_excluded_by_default -- --nocapture 2>&1
```

Expected: PASS. The default config never hashes metadata.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: metadata_excluded_by_default — hash unchanged when include_metadata=false (PLT-1733)"
```

---

### Task 5: Field Metadata Changes Hash

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn field_metadata_changes_hash() {
        // With include_metadata=true, adding field metadata must change the hash.
        let schema_no_meta = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let schema_with_meta = Schema::new(vec![
            Field::new("x", DataType::Int32, false).with_metadata(
                [("ARROW:extension:name".to_string(), "my_ext".to_string())].into(),
            ),
        ]);

        let config = HasherConfig { include_metadata: true };
        assert_ne!(
            encode(ArrowDigester::hash_schema(&schema_no_meta, config)),
            encode(ArrowDigester::hash_schema(&schema_with_meta, config)),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test field_metadata_changes_hash -- --nocapture 2>&1
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: field_metadata_changes_hash — hash differs when include_metadata=true (PLT-1733)"
```

---

### Task 6: Schema-Level Metadata Changes Hash

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn schema_metadata_changes_hash() {
        // With include_metadata=true, schema-level metadata must change the hash.
        let schema_no_meta = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let schema_with_meta = Schema::new_with_metadata(
            vec![Field::new("x", DataType::Int32, false)],
            [("version".to_string(), "2".to_string())].into(),
        );

        let config = HasherConfig { include_metadata: true };
        assert_ne!(
            encode(ArrowDigester::hash_schema(&schema_no_meta, config)),
            encode(ArrowDigester::hash_schema(&schema_with_meta, config)),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test schema_metadata_changes_hash -- --nocapture 2>&1
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: schema_metadata_changes_hash — schema-level metadata affects hash (PLT-1733)"
```

---

### Task 7: Metadata Key Ordering Is Deterministic

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn metadata_key_ordering_is_deterministic() {
        // Same metadata key-value pairs, inserted in different HashMap order.
        // BTreeMap sorting guarantees the same hash regardless of insertion order.
        use std::collections::HashMap;

        let mut meta_a: HashMap<String, String> = HashMap::new();
        meta_a.insert("alpha".to_string(), "1".to_string());
        meta_a.insert("beta".to_string(), "2".to_string());
        meta_a.insert("gamma".to_string(), "3".to_string());

        let mut meta_b: HashMap<String, String> = HashMap::new();
        meta_b.insert("gamma".to_string(), "3".to_string());
        meta_b.insert("alpha".to_string(), "1".to_string());
        meta_b.insert("beta".to_string(), "2".to_string());

        let schema_a = Schema::new(vec![
            Field::new("x", DataType::Int32, false).with_metadata(meta_a),
        ]);
        let schema_b = Schema::new(vec![
            Field::new("x", DataType::Int32, false).with_metadata(meta_b),
        ]);

        let config = HasherConfig { include_metadata: true };
        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema_a, config)),
            encode(ArrowDigester::hash_schema(&schema_b, config)),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test metadata_key_ordering_is_deterministic -- --nocapture 2>&1
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: metadata_key_ordering_is_deterministic — BTreeMap sort ensures stable hash (PLT-1733)"
```

---

### Task 8: Empty-Metadata Invariant

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn empty_metadata_invariant() {
        // A schema with no metadata must produce the same hash regardless of include_metadata.
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::LargeUtf8, true),
        ]);

        assert_eq!(
            encode(ArrowDigester::hash_schema(&schema, HasherConfig::default())),
            encode(ArrowDigester::hash_schema(
                &schema,
                HasherConfig { include_metadata: true }
            )),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test empty_metadata_invariant -- --nocapture 2>&1
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: empty_metadata_invariant — no metadata means same hash regardless of flag (PLT-1733)"
```

---

### Task 9: Unicode Metadata Keys and Values

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn unicode_metadata() {
        // Unicode keys and values must hash without panic, and the hash must differ
        // from a schema with no metadata.
        let schema_with_meta = Schema::new(vec![
            Field::new("data", DataType::LargeUtf8, false).with_metadata(
                [
                    ("emoji_key_\u{1F511}".to_string(), "value_\u{2713}".to_string()),
                    ("\u{4E2D}\u{6587}".to_string(), "\u{65E5}\u{672C}\u{8A9E}".to_string()),
                ]
                .into(),
            ),
        ]);
        let schema_no_meta =
            Schema::new(vec![Field::new("data", DataType::LargeUtf8, false)]);

        let config = HasherConfig { include_metadata: true };
        let hash_with = ArrowDigester::hash_schema(&schema_with_meta, config);
        let hash_without = ArrowDigester::hash_schema(&schema_no_meta, config);
        assert_ne!(hash_with, hash_without);
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test unicode_metadata -- --nocapture 2>&1
```

Expected: PASS (no panic, hashes differ).

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: unicode_metadata — unicode keys/values hash without panic (PLT-1733)"
```

---

### Task 10: Large Metadata Value

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn large_metadata_value() {
        // A kilobyte-scale metadata value must hash without panic and change the hash
        // versus a schema with no metadata.
        let large_value = "x".repeat(10_000);
        let schema_large = Schema::new(vec![
            Field::new("col", DataType::Int32, false)
                .with_metadata([("big".to_string(), large_value)].into()),
        ]);
        let schema_no_meta = Schema::new(vec![Field::new("col", DataType::Int32, false)]);

        let config = HasherConfig { include_metadata: true };
        assert_ne!(
            ArrowDigester::hash_schema(&schema_large, config),
            ArrowDigester::hash_schema(&schema_no_meta, config),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test large_metadata_value -- --nocapture 2>&1
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: large_metadata_value — kilobyte metadata value hashes correctly (PLT-1733)"
```

---

### Task 11: Schema-Level vs Field-Level Metadata Independence

**Files:** Modify `tests/arrow_digester.rs`

- [ ] **Step 1: Add test**

```rust
    #[test]
    fn schema_vs_field_metadata_independence() {
        // The same key-value pair on field-level vs schema-level must produce different hashes
        // with include_metadata=true, confirming both layers are encoded distinctly.
        let schema_field_meta = Schema::new(vec![
            Field::new("x", DataType::Int32, false)
                .with_metadata([("key".to_string(), "value".to_string())].into()),
        ]);
        let schema_schema_meta = Schema::new_with_metadata(
            vec![Field::new("x", DataType::Int32, false)],
            [("key".to_string(), "value".to_string())].into(),
        );

        let config = HasherConfig { include_metadata: true };
        assert_ne!(
            encode(ArrowDigester::hash_schema(&schema_field_meta, config)),
            encode(ArrowDigester::hash_schema(&schema_schema_meta, config)),
        );
    }
```

- [ ] **Step 2: Run the test**

```bash
cargo test schema_vs_field_metadata_independence -- --nocapture 2>&1
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/arrow_digester.rs
git commit -m "test: schema_vs_field_metadata_independence — field and schema metadata encode distinctly (PLT-1733)"
```

---

### Task 12: Run Full Test Suite + Version Bump

**Files:** Modify `Cargo.toml`

- [ ] **Step 1: Run the complete test suite**

```bash
cargo test 2>&1
```

Expected: all tests pass. Count should be all prior tests + 8 new metadata tests.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy -- -D warnings 2>&1
```

Expected: no warnings or errors.

- [ ] **Step 3: Run formatter**

```bash
cargo fmt 2>&1
```

Expected: no output (or only whitespace adjustments). Stage any formatting changes.

- [ ] **Step 4: Bump the crate version in `Cargo.toml`**

In `Cargo.toml`, replace:

```toml
version = "0.0.2"
```

with:

```toml
version = "0.1.0"
```

`VERSION_BYTES` in `src/lib.rs` must **not** change — it encodes the hash format version and is independent of the crate package version.

- [ ] **Step 5: Final test run**

```bash
cargo test 2>&1
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml
git commit -m "chore: bump crate version to 0.1.0 (PLT-1733)"
```
