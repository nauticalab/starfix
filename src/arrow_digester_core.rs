#![expect(
    clippy::expect_used,
    clippy::todo,
    reason = "First iteration of code, will add proper error handling later. Allow for unsupported data types for now"
)]
use std::{
    collections::{BTreeMap, HashSet},
    iter::repeat_n,
    sync::Arc,
};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, GenericListArray, LargeBinaryArray, LargeListArray,
        LargeStringArray, ListArray, OffsetSizeTrait, RecordBatch, StructArray,
    },
    buffer::NullBuffer,
    compute::{cast, concat},
    datatypes::{DataType, Schema},
};
use arrow_schema::Field;
use bitvec::prelude::*;
use digest::Digest;

use crate::{HasherConfig, VERSION_BYTES};

const DELIMITER_FOR_NESTED_FIELD: &str = "/";

/// Arrow field-metadata keys of extension types.
const EXTENSION_NAME_KEY: &str = "ARROW:extension:name";
const EXTENSION_METADATA_KEY: &str = "ARROW:extension:metadata";

/// Format-2 header magic (byte-layout spec Section 1): `0x00 "starfix"`, followed by the 3-byte
/// format version and the flags byte. Canonical JSON never starts with `0x00`.
const HEADER_MAGIC: &[u8; 8] = b"\x00starfix";

/// The option flags byte: bit 0 `ordered_columns`, bit 1 `ordered_struct_fields`.
/// Bits 2–7 are reserved and always 0 (bit 2 was briefly a provisional, never released switch;
/// it must not be reused). `include_metadata` is not a flag.
const fn option_flags(config: HasherConfig) -> u8 {
    (config.ordered_columns as u8) | ((config.ordered_struct_fields as u8) << 1_u8)
}

/// `0x00 "starfix" || format version (3 bytes) || entry-point kind || flags` — the first input
/// of every digest, in every mode (13 bytes).
fn header(kind: u8, flags: u8) -> Vec<u8> {
    let mut h = HEADER_MAGIC.to_vec();
    h.extend_from_slice(&VERSION_BYTES);
    h.push(kind);
    h.push(flags);
    h
}

/// Entry-point kind byte of the header (byte-layout spec Section 1): the schema digest (used by
/// `hash_schema` and inside every record-batch / stream hash) ...
const KIND_SCHEMA: u8 = b'S';
/// ... the final digest of a record batch or stream ...
const KIND_RECORD_BATCH: u8 = b'R';
/// ... and `hash_array`.
const KIND_ARRAY: u8 = b'A';

#[derive(Clone)]
struct DigestBufferType<D: Digest> {
    null_bits: Option<BitVec<u8, Lsb0>>,
    structural: Option<D>,
    data: Option<D>,
}

impl<D: Digest> DigestBufferType<D> {
    /// Create a buffer for a leaf field (data + optional `null_bits`).
    fn new_data_only(nullable: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: None,
            data: Some(D::new()),
        }
    }

    /// Create a buffer for a list node (lengths + optional `null_bits`).
    fn new_list(nullable: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: Some(D::new()),
            data: None,
        }
    }

    /// Create a buffer for a nullable struct node (`null_bits` only).
    fn new_validity_only() -> Self {
        Self {
            null_bits: Some(BitVec::<u8, Lsb0>::new()),
            structural: None,
            data: None,
        }
    }

    /// Get a mutable reference to the data digest, panicking if absent.
    #[expect(clippy::panic, reason = "Const fn cannot use expect/unwrap")]
    const fn data_mut(&mut self) -> &mut D {
        match &mut self.data {
            Some(d) => d,
            None => panic!("data digest not present on this entry"),
        }
    }
}

/// Recursively normalize a `DataType` to its canonical large equivalent.
///
/// - `Utf8` → `LargeUtf8`
/// - `Binary` → `LargeBinary`
/// - `List(field)` → `LargeList(normalized_field)`
/// - `Dictionary(_, value_type)` → `normalize_data_type(value_type)`
/// - `Struct`, `LargeList`, `FixedSizeList`, `Map` have their inner fields normalized recursively.
fn normalize_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Utf8 => DataType::LargeUtf8,
        DataType::Binary => DataType::LargeBinary,
        DataType::List(field) | DataType::LargeList(field) => {
            DataType::LargeList(Arc::new(normalize_field(field)))
        }
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Arc::new(normalize_field(f)))
                .collect(),
        ),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(Arc::new(normalize_field(field)), *size)
        }
        DataType::Map(field, sorted) => DataType::Map(Arc::new(normalize_field(field)), *sorted),
        DataType::Dictionary(_, value_type) => normalize_data_type(value_type),
        other => other.clone(),
    }
}

/// Normalize a single field: keep name, nullability, and metadata; normalize the data type recursively.
fn normalize_field(field: &Field) -> Field {
    let mut normalized = Field::new(
        field.name(),
        normalize_data_type(field.data_type()),
        field.is_nullable(),
    );
    if !field.metadata().is_empty() {
        normalized = normalized.with_metadata(field.metadata().clone());
    }
    normalized
}

/// Normalize all fields in a schema to their canonical large equivalents, preserving metadata.
fn normalize_schema(schema: &Schema) -> Schema {
    let fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .map(|f| Arc::new(normalize_field(f)))
        .collect();
    if schema.metadata().is_empty() {
        Schema::new(fields)
    } else {
        Schema::new_with_metadata(fields, schema.metadata().clone())
    }
}

#[derive(Clone)]
pub struct ArrowDigesterCore<D: Digest> {
    schema_digest: Vec<u8>,
    schema_equality_key: String,
    fields_digest_buffer: BTreeMap<String, DigestBufferType<D>>,
    config: HasherConfig,
    row_count: u64,
}

impl<D: Digest> ArrowDigesterCore<D> {
    /// Test shim: v0.4.0 constructor taking only `include_metadata`.
    #[cfg(test)]
    pub fn new(schema: &Schema, include_metadata: bool) -> Self {
        Self::new_with_config(
            schema,
            HasherConfig {
                include_metadata,
                ..HasherConfig::default()
            },
        )
    }

    /// Create a new instance of `ArrowDigesterCore` with the schema, which will be enforced through each update.
    pub fn new_with_config(schema: &Schema, config: HasherConfig) -> Self {
        // Normalize the schema to canonical large types (metadata is preserved by normalize_schema)
        let normalized = normalize_schema(schema);

        // Hash the schema — hash_schema normalizes internally, so passing the original here is equivalent
        let schema_digest = Self::hash_schema_with_config(schema, config);

        // Build the equality key used in update() to enforce schema identity
        let schema_equality_key = Self::build_schema_equality_key(&normalized, config);

        // Reject data types whose data cannot be hashed, before any state exists
        for field in &normalized.fields {
            Self::validate_data_type(field.data_type());
        }

        // Flatten all nested fields into a single map for per-field hashing
        let mut fields_digest_buffer = BTreeMap::new();
        for field in &normalized.fields {
            Self::extract_fields_name(field, &mut fields_digest_buffer);
        }

        Self {
            schema_digest,
            schema_equality_key,
            fields_digest_buffer,
            config,
            row_count: 0,
        }
    }

    /// Hash a record batch and update the internal digests.
    pub fn update(&mut self, record_batch: &RecordBatch) {
        let rb_schema = record_batch.schema();

        // Fast path (include_metadata = false): serialized_schema normalizes types internally
        // via data_type_to_value, so normalize_schema is not needed and we can compare
        // directly — avoiding the allocation of a normalized Schema/Fields/Arc<Field> tree.
        //
        // Metadata path (include_metadata = true): normalize first (normalize_schema preserves
        // metadata), then build the full equality key that includes the metadata JSON.
        let flags = option_flags(self.config);
        if self.config.include_metadata {
            let rb_normalized = normalize_schema(&rb_schema);
            let rb_equality_key = Self::build_schema_equality_key(&rb_normalized, self.config);
            assert!(
                rb_equality_key == self.schema_equality_key,
                "Record batch schema does not match ArrowDigester schema"
            );
        } else if flags == 0 {
            assert!(
                Self::serialized_schema(&rb_schema) == self.schema_equality_key,
                "Record batch schema does not match ArrowDigester schema"
            );
        } else {
            assert!(
                Self::canonical_schema_json(&rb_schema, self.config) == self.schema_equality_key,
                "Record batch schema does not match ArrowDigester schema"
            );
        }
        // An unordered top level is keyed by name, so a batch with duplicate column names
        // can match the digester's key; reject it like the digester's own schema.
        Self::validate_schema_paths(&rb_schema);
        self.row_count = self
            .row_count
            .checked_add(record_batch.num_rows() as u64)
            .expect("row count overflow");

        let schema = record_batch.schema();
        for col_idx in 0..record_batch.num_columns() {
            let field = schema.field(col_idx);
            let array = record_batch.column(col_idx);
            let path = field.name().to_owned();

            Self::traverse_and_update(
                field.data_type(),
                field.is_nullable(),
                array.as_ref(),
                &path,
                None, // no ancestor struct nulls at top level
                &mut self.fields_digest_buffer,
            );
        }
    }

    /// Hash an array directly without needing to create an `ArrowDigester` instance on the user side.
    /// Unlike full table hashing, we don't have a schema to hash; however, we do have the field data type.
    /// Similar to schema hashing, we hash based on the data type to encode metadata information into the digest.
    ///
    /// Uses the same recursive decomposition as the record-batch path so that data hashing
    /// is consistent regardless of which API is used.
    ///
    /// # Panics
    ///
    /// This function will panic if JSON serialization of the data type fails.
    ///
    #[cfg(test)]
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        Self::hash_array_with_config(array, HasherConfig::default())
    }

    /// `hash_array` with options (byte-layout spec Section 4). `include_metadata` has no effect.
    pub fn hash_array_with_config(array: &dyn Array, config: HasherConfig) -> Vec<u8> {
        let flags = option_flags(config);
        let effective_type = array.data_type().clone();
        let effective_array = array;

        // Normalize to canonical large types
        let normalized_type = normalize_data_type(&effective_type);

        Self::validate_type_paths(&normalized_type);
        Self::validate_data_type(&normalized_type);

        let mut final_digest = D::new();
        final_digest.update(header(KIND_ARRAY, flags));

        // Use canonical type serialization for metadata (data_type_to_value also normalizes,
        // but we pass the already-normalized type for consistency)
        let canonical_type =
            Self::data_type_to_value(&normalized_type, config.ordered_struct_fields);
        let data_type_serialized = serde_json::to_string(&canonical_type)
            .expect("Failed to serialize data type to string");

        // Update the digest with array metadata, then the element count
        final_digest.update(data_type_serialized);
        final_digest.update((effective_array.len() as u64).to_le_bytes());

        // Build BTreeMap entries from the type tree (same decomposition as record-batch path)
        let mut fields = BTreeMap::new();
        // The root is always nullable: its validity is always written (byte-layout spec Section 4)
        Self::extract_type_entries(&effective_type, true, "", &mut fields);

        // Traverse and populate entries
        Self::traverse_and_update(
            &effective_type,
            true,
            effective_array,
            "",
            None,
            &mut fields,
        );

        // Finalize all entries into the digest (same order as record-batch finalize)
        for (_, digest) in fields {
            Self::finalize_digest(&mut final_digest, digest);
        }

        final_digest.finalize().to_vec()
    }

    /// Hash a record batch directly without needing to create an `ArrowDigester` instance on the user side.
    pub fn hash_record_batch(record_batch: &RecordBatch, config: HasherConfig) -> Vec<u8> {
        let mut digester = Self::new_with_config(record_batch.schema().as_ref(), config);
        digester.update(record_batch);
        digester.finalize()
    }

    /// Consume the digester and produce the final digest: the schema digest, the row count, then
    /// every node's entry in byte-wise order of node paths (byte-layout spec Section 3.3).
    pub fn finalize(self) -> Vec<u8> {
        // Finalize all the sub digest and combine them into a single digest
        let mut final_digest = D::new();

        // header, the schema digest, then the number of rows
        final_digest.update(header(KIND_RECORD_BATCH, option_flags(self.config)));
        final_digest.update(&self.schema_digest);
        final_digest.update(self.row_count.to_le_bytes());

        // Then digest each field digest in order
        self.fields_digest_buffer
            .into_iter()
            .for_each(|(_, digest)| Self::finalize_digest(&mut final_digest, digest));

        final_digest.finalize().to_vec()
    }

    /// Finalize a single field digest into the final digest.
    /// Helper to reduce code duplication.
    fn finalize_digest(final_digest: &mut D, digest: DigestBufferType<D>) {
        // Null bits first (if nullable)
        if let Some(null_bit_vec) = &digest.null_bits {
            final_digest.update((null_bit_vec.len() as u64).to_le_bytes());
            for &word in null_bit_vec.as_raw_slice() {
                final_digest.update(word.to_le_bytes());
            }
        }
        // Structural digest (if list type) — sizes separated from leaf data
        if let Some(structural) = digest.structural {
            final_digest.update(structural.finalize());
        }
        // Data/leaf digest (if present)
        if let Some(data) = digest.data {
            final_digest.update(data.finalize());
        }
    }

    /// Serialize the schema into a canonical JSON string keyed by field name.
    ///
    /// # Panics
    /// This function will panic if JSON serialization of the schema fails.
    fn serialized_schema(schema: &Schema) -> String {
        let fields_digest: BTreeMap<String, serde_json::Value> = schema
            .fields
            .iter()
            .map(|field| {
                let value = serde_json::json!({
                    "data_type": Self::data_type_to_value(field.data_type(), false),
                    "nullable": field.is_nullable(),
                });
                (field.name().clone(), Self::sort_json_value(value))
            })
            .collect();

        serde_json::to_string(&fields_digest).expect("Failed to serialize field_digest to bytes")
    }

    /// Phase 1 canonical schema JSON under `config` (byte-layout spec Sections 2.1 and 2.2).
    ///
    /// With both order options off this is exactly `serialized_schema`. With
    /// `ordered_columns` the top level is a JSON array of `{data_type, name, nullable}` in schema
    /// order; with `ordered_struct_fields` struct children keep their declaration order.
    fn canonical_schema_json(schema: &Schema, config: HasherConfig) -> String {
        if !config.ordered_columns && !config.ordered_struct_fields {
            return Self::serialized_schema(schema);
        }
        let ordered_struct_fields = config.ordered_struct_fields;
        if config.ordered_columns {
            let columns: Vec<serde_json::Value> = schema
                .fields
                .iter()
                .map(|field| {
                    Self::sort_json_value(Self::inner_field_to_value(field, ordered_struct_fields))
                })
                .collect();
            serde_json::to_string(&columns).expect("Failed to serialize ordered columns")
        } else {
            let fields_digest: BTreeMap<String, serde_json::Value> = schema
                .fields
                .iter()
                .map(|field| {
                    let value = serde_json::json!({
                        "data_type": Self::data_type_to_value(field.data_type(), ordered_struct_fields),
                        "nullable": field.is_nullable(),
                    });
                    (field.name().clone(), Self::sort_json_value(value))
                })
                .collect();
            serde_json::to_string(&fields_digest)
                .expect("Failed to serialize field_digest to bytes")
        }
    }

    /// Convert a `DataType` to a JSON value, recursively converting any inner `Field`
    /// references to only include `name`, `data_type`, and `nullable`.
    ///
    /// Types are first normalized via `normalize_data_type` (Utf8→LargeUtf8, Binary→LargeBinary,
    /// List→LargeList, Dictionary→value type) so the JSON always reflects canonical forms.
    fn data_type_to_value(data_type: &DataType, ordered_struct_fields: bool) -> serde_json::Value {
        // Normalize first so all downstream serialization uses canonical types
        let canonical = normalize_data_type(data_type);
        let value = match &canonical {
            DataType::Struct(fields) => {
                // Unordered (default): children sorted by name. Ordered: declaration order.
                let mut children: Vec<_> = fields.iter().collect();
                if !ordered_struct_fields {
                    children.sort_by_key(|f| f.name().clone());
                }
                let fields_json: Vec<serde_json::Value> = children
                    .iter()
                    .map(|f| Self::inner_field_to_value(f, ordered_struct_fields))
                    .collect();
                serde_json::json!({ "Struct": fields_json })
            }
            // After normalization, all list types are LargeList
            DataType::LargeList(field) => {
                serde_json::json!({ "LargeList": Self::element_type_to_value(field, ordered_struct_fields) })
            }
            DataType::FixedSizeList(field, size) => {
                serde_json::json!({ "FixedSizeList": [Self::element_type_to_value(field, ordered_struct_fields), size] })
            }
            // Map: key and value element forms by position; the entries/key/value field names
            // and the entries field's metadata are not hashed (byte-layout spec Section 2.1).
            DataType::Map(field, keys_sorted) => {
                let kv = match field.data_type() {
                    DataType::Struct(kv) => kv.clone(),
                    _ => arrow_schema::Fields::empty(),
                };
                let key = kv
                    .first()
                    .expect("Arrow map entries are a struct with a key field");
                let value = kv
                    .get(1)
                    .expect("Arrow map entries are a struct with a value field");
                serde_json::json!({ "Map": {
                    "key": Self::element_type_to_value(key, ordered_struct_fields),
                    "keys_sorted": keys_sorted,
                    "value": Self::element_type_to_value(value, ordered_struct_fields),
                } })
            }
            // For all non-nested types (including LargeUtf8, LargeBinary after normalization),
            // Arrow's default serde is sufficient
            other => serde_json::to_value(other).expect("Failed to serialize data type"),
        };
        Self::sort_json_value(value)
    }

    /// Convert an inner field (e.g., struct child) to a JSON value
    /// with `name`, `data_type`, and `nullable`.
    fn inner_field_to_value(field: &Field, ordered_struct_fields: bool) -> serde_json::Value {
        serde_json::json!({
            "name": field.name(),
            "data_type": Self::data_type_to_value(field.data_type(), ordered_struct_fields),
            "nullable": field.is_nullable(),
        })
    }

    /// Convert a container element field (e.g., list item) to a JSON value
    /// with only `data_type` and `nullable`, omitting the Arrow-internal field name.
    fn element_type_to_value(field: &Field, ordered_struct_fields: bool) -> serde_json::Value {
        serde_json::json!({
            "data_type": Self::data_type_to_value(field.data_type(), ordered_struct_fields),
            "nullable": field.is_nullable(),
        })
    }

    /// Recursively sort all JSON object keys for deterministic serialization.
    fn sort_json_value(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let sorted: serde_json::Map<String, serde_json::Value> = map
                    .into_iter()
                    .map(|(k, v)| (k, Self::sort_json_value(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect();
                serde_json::Value::Object(sorted)
            }
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.into_iter().map(Self::sort_json_value).collect())
            }
            other => other,
        }
    }

    /// Hash the schema in two phases.
    ///
    /// Phase 1 (always): canonical JSON of field names, data types, nullability (byte-layout spec Section 2.2).
    /// Phase 2 (when `include_metadata` is `true`): per-field and schema-level metadata fed into the
    /// same hasher via `update_metadata_hash`. Phase 2 adds nothing when all metadata maps are empty,
    /// preserving the empty-metadata invariant.
    ///
    /// `normalize_schema` preserves metadata, so the normalized schema is used for both phases.
    pub fn hash_schema_with_config(schema: &Schema, config: HasherConfig) -> Vec<u8> {
        let include_metadata = config.include_metadata;
        let flags = option_flags(config);
        Self::validate_schema_paths(schema);
        let normalized = normalize_schema(schema);
        let mut hasher = D::new();
        hasher.update(header(KIND_SCHEMA, flags));
        hasher.update(Self::canonical_schema_json(&normalized, config));
        if include_metadata {
            Self::update_metadata_hash(&mut hasher, &normalized);
        }
        hasher.finalize().to_vec()
    }

    /// Recursively collect per-field metadata from a field and all its nested children.
    ///
    /// `path` is the **full, already-computed path** for `field` — the caller is responsible
    /// for computing it before the call. This matches the path-naming conventions used in
    /// `extract_type_entries` / `traverse_list`:
    ///
    /// - **Struct children**: path = `"<parent>{DELIMITER}<child_name>"` (e.g. `"s/child"`)
    /// - **List / map element fields**: path = `"<list_field>/"` (trailing slash, **no element
    ///   field name**) — because `element_type_to_value` in Phase 1 also omits the
    ///   Arrow-internal element field name (e.g. `"item"`). Using the same convention means
    ///   renaming only the internal element field name has no effect on the metadata hash.
    ///
    /// Returns a `BTreeMap` keyed by field path. Only fields with non-empty metadata are
    /// included. Because the result is a `BTreeMap`, entries are automatically ordered by
    /// path, giving deterministic traversal regardless of struct child insertion order.
    fn collect_nested_field_metadata(
        field: &Field,
        path: &str,
    ) -> BTreeMap<String, BTreeMap<String, String>> {
        let mut result: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        Self::collect_field_metadata(field, path, &mut result);
        result
    }

    /// Record `field`'s metadata (if any) at `path`, then recurse into its type.
    fn collect_field_metadata(
        field: &Field,
        path: &str,
        result: &mut BTreeMap<String, BTreeMap<String, String>>,
    ) {
        if !field.metadata().is_empty() {
            let mut sorted_meta: BTreeMap<String, String> = field
                .metadata()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            // An extension type without `ARROW:extension:metadata` is the same as one with an
            // empty value (byte-layout spec Section 2.3).
            if sorted_meta.contains_key(EXTENSION_NAME_KEY) {
                sorted_meta
                    .entry(EXTENSION_METADATA_KEY.to_owned())
                    .or_default();
            }
            result.insert(path.to_owned(), sorted_meta);
        }
        Self::collect_type_metadata(field.data_type(), path, result);
    }

    /// Metadata of the fields nested in `data_type`, keyed by node path (byte-layout spec
    /// Section 2.3): struct child `join(path, name)`, list element `path/`, map key and value
    /// `path//key` and `path//value` (fixed names; the map's entries field contributes nothing).
    /// The schema is normalized first, so dictionaries have already been replaced by their values.
    fn collect_type_metadata(
        data_type: &DataType,
        path: &str,
        result: &mut BTreeMap<String, BTreeMap<String, String>>,
    ) {
        match data_type {
            DataType::Struct(children) => {
                for child in children {
                    let child_path = Self::construct_field_name_hierarchy(path, child.name());
                    Self::collect_field_metadata(child, &child_path, result);
                }
            }
            DataType::List(inner)
            | DataType::LargeList(inner)
            | DataType::FixedSizeList(inner, _) => {
                let element_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
                Self::collect_field_metadata(inner, &element_path, result);
            }
            DataType::Map(entries, _) => {
                let entries_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
                if let DataType::Struct(kv) = entries.data_type() {
                    for (child, name) in kv.iter().zip(["key", "value"]) {
                        let child_path = Self::construct_field_name_hierarchy(&entries_path, name);
                        Self::collect_field_metadata(child, &child_path, result);
                    }
                }
            }
            _ => {}
        }
    }

    /// Feed per-field and schema-level metadata into `hasher` for Phase 2 of schema hashing.
    ///
    /// All fields are visited recursively (struct children, list/map element fields, etc.) so
    /// that metadata on nested fields is included. Field paths use
    /// [`DELIMITER_FOR_NESTED_FIELD`] as a separator (e.g. `"parent/child"`). Metadata keys
    /// within each map are sorted via `BTreeMap` to guarantee determinism regardless of
    /// `HashMap` iteration order. Nothing is written when all metadata maps are empty,
    /// preserving the empty-metadata invariant.
    ///
    /// The entire Phase 2 contribution is serialized as a **single JSON object** and fed into
    /// the hasher in one update. JSON is self-delimiting, so no length prefixes are needed:
    ///
    /// ```text
    /// hasher.update(serde_json::to_string({
    ///     "fields": { "<path>": { "<key>": "<value>", ... }, ... },  // present when any field has metadata
    ///     "schema": { "<key>": "<value>", ... },                     // present when schema has metadata
    /// }))
    /// ```
    ///
    /// Keys within each metadata map are sorted (via `BTreeMap`). Field paths are sorted
    /// alphabetically (via `BTreeMap`). The top-level keys `"fields"` and `"schema"` are
    /// fixed, so `"fields"` always sorts before `"schema"`. Nothing is written when both
    /// `"fields"` and `"schema"` would be absent.
    fn update_metadata_hash(hasher: &mut D, schema: &Schema) {
        // Collect metadata from all fields recursively (BTreeMap gives deterministic order).
        let mut all_field_meta: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for field in schema.fields() {
            all_field_meta.extend(Self::collect_nested_field_metadata(field, field.name()));
        }

        // Build a BTreeMap so "fields" sorts before "schema" deterministically.
        let mut meta_doc: BTreeMap<&str, serde_json::Value> = BTreeMap::new();
        if !all_field_meta.is_empty() {
            meta_doc.insert(
                "fields",
                serde_json::to_value(&all_field_meta)
                    .expect("Failed to serialize field metadata to JSON value"),
            );
        }
        if !schema.metadata().is_empty() {
            let sorted_meta: BTreeMap<&String, &String> = schema.metadata().iter().collect();
            meta_doc.insert(
                "schema",
                serde_json::to_value(&sorted_meta)
                    .expect("Failed to serialize schema metadata to JSON value"),
            );
        }

        if !meta_doc.is_empty() {
            let json = serde_json::to_string(&meta_doc)
                .expect("Failed to serialize metadata doc to JSON string");
            hasher.update(json.as_bytes());
        }
    }

    /// Build a canonical string key for schema identity enforcement in `update()`.
    ///
    /// When `include_metadata` is `false`, or when the schema has no metadata anywhere
    /// (including nested fields), returns the structure-only JSON (Phase 1) — preserving
    /// the empty-metadata invariant.
    ///
    /// When `include_metadata` is `true` and metadata is present, appends `|` followed by the
    /// same JSON object format used in `update_metadata_hash`:
    /// `{"fields": {...}, "schema": {...}}`. The `|` separator is unambiguous because JSON
    /// objects never end with `|`.
    fn build_schema_equality_key(normalized: &Schema, config: HasherConfig) -> String {
        let structure = Self::canonical_schema_json(normalized, config);
        if !config.include_metadata {
            return structure;
        }

        // Collect all field metadata recursively (includes nested fields).
        let mut field_meta: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for field in normalized.fields() {
            field_meta.extend(Self::collect_nested_field_metadata(field, field.name()));
        }

        let has_any_metadata = !normalized.metadata().is_empty() || !field_meta.is_empty();
        if !has_any_metadata {
            return structure;
        }

        // Mirror the same BTreeMap structure used in update_metadata_hash so the equality
        // key reflects exactly what was hashed.
        let mut meta_doc: BTreeMap<&str, serde_json::Value> = BTreeMap::new();
        if !field_meta.is_empty() {
            meta_doc.insert(
                "fields",
                serde_json::to_value(&field_meta)
                    .expect("Failed to serialize field metadata to JSON value"),
            );
        }
        if !normalized.metadata().is_empty() {
            let sorted_schema_meta: BTreeMap<&String, &String> =
                normalized.metadata().iter().collect();
            meta_doc.insert(
                "schema",
                serde_json::to_value(&sorted_schema_meta)
                    .expect("Failed to serialize schema metadata to JSON value"),
            );
        }

        let meta_json = serde_json::to_string(&meta_doc)
            .expect("Failed to serialize metadata equality key to JSON string");

        format!("{structure}|{meta_json}")
    }

    /// Top-down recursive traversal that routes data to `BTreeMap` entries.
    fn traverse_and_update(
        data_type: &DataType,
        nullable: bool,
        array: &dyn Array,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        // Normalize small variants
        let (normalized_type, cast_array);
        let (effective_type, effective_array): (&DataType, &dyn Array) = match data_type {
            DataType::Utf8 => {
                normalized_type = DataType::LargeUtf8;
                cast_array = cast(array, &normalized_type).expect("cast Utf8");
                (&normalized_type, cast_array.as_ref())
            }
            DataType::Binary => {
                normalized_type = DataType::LargeBinary;
                cast_array = cast(array, &normalized_type).expect("cast Binary");
                (&normalized_type, cast_array.as_ref())
            }
            // A dictionary is its decoded values, resolved recursively (nested dictionaries)
            DataType::Dictionary(_, value_type) => {
                let decoded = cast(array, value_type.as_ref()).expect("decode dictionary");
                return Self::traverse_and_update(
                    value_type,
                    nullable,
                    decoded.as_ref(),
                    path,
                    ancestor_struct_nulls,
                    fields,
                );
            }
            _ => (data_type, array),
        };

        let canonical = normalize_data_type(effective_type);

        match &canonical {
            DataType::LargeList(value_field) => {
                Self::traverse_list(
                    effective_array,
                    value_field,
                    nullable,
                    path,
                    ancestor_struct_nulls,
                    fields,
                );
            }
            DataType::Struct(_) => {
                Self::traverse_struct(
                    effective_array,
                    nullable,
                    path,
                    ancestor_struct_nulls,
                    fields,
                );
            }
            _ => {
                Self::traverse_leaf(
                    effective_type,
                    effective_array,
                    nullable,
                    path,
                    ancestor_struct_nulls,
                    fields,
                );
            }
        }
    }

    /// A list node: [validity] and the lengths of its valid elements go into its own entry; the
    /// items of all valid elements are then traversed once, as the element node at `path/`.
    /// `List` and `LargeList` are traversed alike, without casting (byte-layout spec Section 3.2):
    /// only the items of valid list values are visited, so nulls hidden under null list values
    /// (or outside the offsets) are never seen.
    fn traverse_list(
        array: &dyn Array,
        value_field: &Field,
        nullable: bool,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
            Self::traverse_list_of(
                list,
                value_field,
                nullable,
                path,
                ancestor_struct_nulls,
                fields,
            );
        } else {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .expect("downcast to LargeListArray");
            Self::traverse_list_of(
                list,
                value_field,
                nullable,
                path,
                ancestor_struct_nulls,
                fields,
            );
        }
    }

    fn traverse_list_of<O>(
        list_array: &GenericListArray<O>,
        value_field: &Field,
        nullable: bool,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) where
        O: OffsetSizeTrait,
    {
        Self::assert_nulls_allowed(nullable, list_array.nulls(), ancestor_struct_nulls);

        let effective_nulls = Self::combine_nulls(list_array.nulls(), ancestor_struct_nulls);
        let entry = fields
            .get_mut(path)
            .expect("entry must exist for list path");
        if let Some(ref mut null_bits) = entry.null_bits {
            Self::push_validity(null_bits, effective_nulls.as_ref(), list_array.len());
        }

        // Lengths of the valid elements, and the item ranges (start, length) they cover, with
        // adjacent ranges merged
        let offsets = list_array.value_offsets();
        let mut lengths: Vec<u8> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for i in 0..list_array.len() {
            if effective_nulls.as_ref().is_none_or(|nb| nb.is_valid(i)) {
                let start = Self::offset_at(offsets, i);
                let end = Self::offset_at(offsets, i.checked_add(1).expect("index overflow"));
                let len = end.checked_sub(start).expect("offsets are monotonic");
                lengths.extend_from_slice(&(len as u64).to_le_bytes());
                match ranges.last_mut() {
                    _ if len == 0 => {}
                    Some(last) if last.0.checked_add(last.1) == Some(start) => {
                        last.1 = last.1.checked_add(len).expect("range length overflow");
                    }
                    _ => ranges.push((start, len)),
                }
            }
        }
        entry
            .structural
            .as_mut()
            .expect("list entry has lengths")
            .update(&lengths);

        let values = list_array.values();
        let items: ArrayRef = match ranges.as_slice() {
            [] => values.slice(0, 0),
            [(start, len)] => values.slice(*start, *len),
            many => {
                let parts: Vec<ArrayRef> = many
                    .iter()
                    .map(|(start, len)| values.slice(*start, *len))
                    .collect();
                let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
                concat(&refs).expect("concatenate list items")
            }
        };
        let element_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
        Self::traverse_and_update(
            items.data_type(),
            value_field.is_nullable(),
            items.as_ref(),
            &element_path,
            None, // list elements don't have ancestor struct nulls
            fields,
        );
    }

    fn offset_at<O>(offsets: &[O], i: usize) -> usize
    where
        O: OffsetSizeTrait,
    {
        offsets.get(i).expect("offset index in range").as_usize()
    }

    /// Push one validity bit per element: `validity`, or all valid when absent.
    fn push_validity(null_bits: &mut BitVec<u8, Lsb0>, validity: Option<&NullBuffer>, len: usize) {
        match validity {
            Some(nb) => {
                for i in 0..len {
                    null_bits.push(nb.is_valid(i));
                }
            }
            None => null_bits.extend(repeat_n(true, len)),
        }
    }

    /// A non-nullable node may only have nulls where an enclosing struct is null (byte-layout
    /// spec Section 7).
    ///
    /// # Panics
    /// When a non-nullable field's array has a null that no enclosing struct null masks.
    fn assert_nulls_allowed(
        nullable: bool,
        own_nulls: Option<&NullBuffer>,
        ancestor: Option<&NullBuffer>,
    ) {
        if nullable {
            return;
        }
        let Some(own) = own_nulls.filter(|nb| nb.null_count() > 0) else {
            return;
        };
        let masked =
            ancestor.is_some_and(|anc| (own.inner() | &!anc.inner()).count_set_bits() == own.len());
        assert!(masked, "A non-nullable field contains null values");
    }

    fn traverse_struct(
        array: &dyn Array,
        nullable: bool,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("downcast to StructArray");

        Self::assert_nulls_allowed(nullable, struct_array.nulls(), ancestor_struct_nulls);

        // Combine struct's own nulls with ancestor nulls (AND propagation)
        let combined_nulls = if nullable {
            Self::combine_nulls(struct_array.nulls(), ancestor_struct_nulls)
        } else {
            ancestor_struct_nulls.cloned()
        };

        // A nullable struct records its own (combined) validity at its path — in its
        // validity-only entry, or, for a list element, in the list's element entry
        // (byte-layout spec Section 3.2; format 1 recorded nothing).
        if nullable {
            let null_bits = fields
                .get_mut(path)
                .and_then(|entry| entry.null_bits.as_mut())
                .expect("nullable struct must have a validity entry");
            Self::push_validity(null_bits, combined_nulls.as_ref(), struct_array.len());
        }

        // Use the ORIGINAL struct array's fields (not the canonical ones from
        // the type tree) so that data_type matches the actual child array.
        // traverse_and_update will normalize types internally.
        let original_fields = struct_array.fields();
        let mut sorted_children: Vec<(usize, &Field)> = original_fields
            .iter()
            .enumerate()
            .map(|(i, f)| (i, f.as_ref()))
            .collect();
        sorted_children.sort_by_key(|(_, f)| f.name().clone());

        for (idx, child_field) in sorted_children {
            let child_array = struct_array.column(idx);
            let child_path = Self::construct_field_name_hierarchy(path, child_field.name());

            Self::traverse_and_update(
                child_field.data_type(),
                child_field.is_nullable(),
                child_array.as_ref(),
                &child_path,
                combined_nulls.as_ref(),
                fields,
            );
        }
    }

    fn traverse_leaf(
        data_type: &DataType,
        array: &dyn Array,
        nullable: bool,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        Self::assert_nulls_allowed(nullable, array.nulls(), ancestor_struct_nulls);
        let entry = fields
            .get_mut(path)
            .expect("entry must exist for leaf path");

        // Effective validity: own nulls AND enclosing struct nulls
        let effective_nulls = Self::combine_nulls(array.nulls(), ancestor_struct_nulls);
        if let Some(ref mut null_bits) = entry.null_bits {
            Self::push_validity(null_bits, effective_nulls.as_ref(), array.len());
        }
        Self::hash_leaf_values(data_type, array, effective_nulls.as_ref(), entry.data_mut());
    }

    fn combine_nulls(
        own_nulls: Option<&NullBuffer>,
        ancestor_nulls: Option<&NullBuffer>,
    ) -> Option<NullBuffer> {
        match (own_nulls, ancestor_nulls) {
            (Some(own), Some(ancestor)) => Some(NullBuffer::new(own.inner() & ancestor.inner())),
            (Some(own), None) => Some(own.clone()),
            (None, Some(ancestor)) => Some(ancestor.clone()),
            (None, None) => None,
        }
    }

    /// Feed the values of the valid elements of a leaf array into `data` (byte-layout spec
    /// Section 3.2). `validity` is the effective validity (own AND enclosing structs); the array's
    /// own null buffer is not consulted, so no array is rebuilt.
    #[expect(
        clippy::unreachable,
        reason = "Unsupported data types are rejected before any data is hashed"
    )]
    fn hash_leaf_values(
        data_type: &DataType,
        array: &dyn Array,
        validity: Option<&NullBuffer>,
        data: &mut D,
    ) {
        let valid = |i: usize| validity.is_none_or(|nb| nb.is_valid(i));
        match data_type {
            DataType::Boolean => {
                let bools = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("Failed to downcast to BooleanArray");
                let bytes: Vec<u8> = (0..bools.len())
                    .filter(|&i| valid(i))
                    .map(|i| u8::from(bools.value(i)))
                    .collect();
                data.update(&bytes);
            }
            DataType::Int8 | DataType::UInt8 => Self::hash_fixed_size(array, validity, 1, data),
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => {
                Self::hash_fixed_size(array, validity, 2, data);
            }
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Decimal32(_, _) => Self::hash_fixed_size(array, validity, 4, data),
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Decimal64(_, _) => Self::hash_fixed_size(array, validity, 8, data),
            DataType::Decimal128(_, _) => Self::hash_fixed_size(array, validity, 16, data),
            DataType::Decimal256(_, _) => Self::hash_fixed_size(array, validity, 32, data),
            DataType::FixedSizeBinary(size) => {
                let width = usize::try_from(*size).expect("non-negative FixedSizeBinary size");
                Self::hash_fixed_size(array, validity, width, data);
            }
            DataType::Binary | DataType::LargeBinary => {
                let large = cast(array, &DataType::LargeBinary).expect("cast to LargeBinary");
                let bin = large
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .expect("Failed to downcast to LargeBinaryArray");
                for i in (0..bin.len()).filter(|&i| valid(i)) {
                    let value = bin.value(i);
                    data.update((value.len() as u64).to_le_bytes());
                    data.update(value);
                }
            }
            DataType::Utf8 | DataType::LargeUtf8 => {
                let large = cast(array, &DataType::LargeUtf8).expect("cast to LargeUtf8");
                let strings = large
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Failed to downcast to LargeStringArray");
                for i in (0..strings.len()).filter(|&i| valid(i)) {
                    let value = strings.value(i);
                    data.update((value.len() as u64).to_le_bytes());
                    data.update(value.as_bytes());
                }
            }
            other => unreachable!("data of type {other} is rejected before hashing"),
        }
    }

    /// Fixed-size values: the whole value buffer when every element is valid, else the bytes of
    /// each valid element.
    fn hash_fixed_size(
        array: &dyn Array,
        validity: Option<&NullBuffer>,
        element_size: usize,
        data: &mut D,
    ) {
        if element_size == 0 {
            return; // zero-width values (FixedSizeBinary(0)) contribute no bytes
        }
        let array_data = array.to_data();
        let start = array_data
            .offset()
            .checked_mul(element_size)
            .expect("Offset multiplication overflow");
        let end = start
            .checked_add(
                array_data
                    .len()
                    .checked_mul(element_size)
                    .expect("Length multiplication overflow"),
            )
            .expect("End position overflow");
        let slice = array_data
            .buffers()
            .first()
            .expect("fixed-size array has a value buffer")
            .as_slice()
            .get(start..end)
            .expect("value buffer covers the array");
        match validity.filter(|nb| nb.null_count() > 0) {
            None => data.update(slice),
            Some(nb) => {
                for (i, value) in slice.chunks_exact(element_size).enumerate() {
                    if nb.is_valid(i) {
                        data.update(value);
                    }
                }
            }
        }
    }

    /// Create the entries of a column (the node at its bare name) and of every node below it;
    /// see `extract_type_entries`.
    fn extract_fields_name(
        field: &Field,
        fields_digest_buffer: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        Self::extract_type_entries(
            field.data_type(),
            field.is_nullable(),
            field.name(),
            fields_digest_buffer,
        );
    }

    /// Core recursive type walker — one entry per node of the type tree (byte-layout spec
    /// Section 3): a leaf gets [validity] + values, a list gets [validity] + lengths and its
    /// element is the node at `path/`, a nullable struct gets validity, and its children are
    /// the nodes at `join(path, name)`. `nullable` comes from the node's `Field`.
    fn extract_type_entries(
        data_type: &DataType,
        nullable: bool,
        path: &str,
        fields_digest_buffer: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        match normalize_data_type(data_type) {
            DataType::Struct(fields) => {
                if nullable {
                    fields_digest_buffer
                        .insert(path.to_owned(), DigestBufferType::new_validity_only());
                }
                for child_field in &fields {
                    let child_path = Self::construct_field_name_hierarchy(path, child_field.name());
                    Self::extract_type_entries(
                        child_field.data_type(),
                        child_field.is_nullable(),
                        &child_path,
                        fields_digest_buffer,
                    );
                }
            }
            DataType::LargeList(value_field) => {
                fields_digest_buffer.insert(path.to_owned(), DigestBufferType::new_list(nullable));
                let element_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
                Self::extract_type_entries(
                    value_field.data_type(),
                    value_field.is_nullable(),
                    &element_path,
                    fields_digest_buffer,
                );
            }
            _ => {
                fields_digest_buffer
                    .insert(path.to_owned(), DigestBufferType::new_data_only(nullable));
            }
        }
    }

    /// Reject types whose data cannot be hashed (byte-layout spec Section 8), before any data is
    /// hashed, whatever the number of rows.
    ///
    /// # Panics
    /// On `Null`, `FixedSizeList`, `Map`, `Interval`, view and other unsupported types anywhere in
    /// the type tree.
    fn validate_data_type(data_type: &DataType) {
        match normalize_data_type(data_type) {
            DataType::Struct(children) => {
                for child in &children {
                    Self::validate_data_type(child.data_type());
                }
            }
            DataType::LargeList(element) => Self::validate_data_type(element.data_type()),
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::FixedSizeBinary(_)
            | DataType::LargeBinary
            | DataType::LargeUtf8 => {}
            other => todo!("hashing data of type {other} is not supported"),
        }
    }

    /// Reject schemas that cannot be hashed unambiguously (byte-layout spec Section 7), in every
    /// mode.
    ///
    /// # Panics
    /// When two nodes of the type tree share a node path (duplicate names, or a name containing
    /// `/` that aliases a nested field), or when the schema contains a `ListView`,
    /// `LargeListView`, `Union` or `RunEndEncoded` type.
    fn validate_schema_paths(schema: &Schema) {
        let mut seen = HashSet::new();
        for field in schema.fields() {
            Self::claim_node_path(field.name(), &mut seen);
            Self::collect_node_paths(field.data_type(), field.name(), &mut seen);
        }
    }

    /// `hash_array` variant of `validate_schema_paths`: the array itself is the node at `""`.
    fn validate_type_paths(data_type: &DataType) {
        let mut seen = HashSet::new();
        Self::claim_node_path("", &mut seen);
        Self::collect_node_paths(data_type, "", &mut seen);
    }

    fn claim_node_path(path: &str, seen: &mut HashSet<String>) {
        assert!(
            seen.insert(path.to_owned()),
            "Ambiguous field path {path:?}: duplicate field names, or a field name containing \
             {DELIMITER_FOR_NESTED_FIELD:?} that collides with a nested field, cannot be hashed"
        );
    }

    fn collect_node_paths(data_type: &DataType, path: &str, seen: &mut HashSet<String>) {
        let canonical = normalize_data_type(data_type);
        assert!(
            !matches!(
                canonical,
                DataType::ListView(_)
                    | DataType::LargeListView(_)
                    | DataType::Union(_, _)
                    | DataType::RunEndEncoded(_, _)
            ),
            "ListView, LargeListView, Union and RunEndEncoded types cannot be hashed"
        );
        match canonical {
            DataType::Struct(children) => {
                for child in &children {
                    let child_path = Self::construct_field_name_hierarchy(path, child.name());
                    Self::claim_node_path(&child_path, seen);
                    Self::collect_node_paths(child.data_type(), &child_path, seen);
                }
            }
            DataType::LargeList(element) | DataType::FixedSizeList(element, _) => {
                let element_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
                Self::claim_node_path(&element_path, seen);
                Self::collect_node_paths(element.data_type(), &element_path, seen);
            }
            DataType::Map(entries, _) => {
                let entries_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
                Self::claim_node_path(&entries_path, seen);
                if let DataType::Struct(kv) = entries.data_type() {
                    for (child, name) in kv.iter().zip(["key", "value"]) {
                        let child_path = Self::construct_field_name_hierarchy(&entries_path, name);
                        Self::claim_node_path(&child_path, seen);
                        Self::collect_node_paths(child.data_type(), &child_path, seen);
                    }
                }
            }
            _ => {}
        }
    }

    /// Path of a struct child: always `parent + "/" + name` (byte-layout spec Section 3.1), so
    /// empty names stay unambiguous. Columns are at their bare name.
    fn construct_field_name_hierarchy(parent_field_name: &str, field_name: &str) -> String {
        format!("{parent_field_name}{DELIMITER_FOR_NESTED_FIELD}{field_name}")
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::panic,
        clippy::indexing_slicing,
        reason = "Tests require panics and unwraps for assertions"
    )]

    use std::{
        f64::{self, consts},
        sync::Arc,
    };

    use arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
            Decimal32Array, DurationMicrosecondArray, FixedSizeBinaryBuilder, Float16Array,
            Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
            LargeBinaryArray, LargeListArray, LargeListBuilder, LargeStringArray, ListBuilder,
            PrimitiveBuilder, RecordBatch, StringArray, StructArray, Time32SecondArray,
            Time64MicrosecondArray, TimestampMicrosecondArray, UInt16Array, UInt32Array,
            UInt64Array, UInt8Array,
        },
        datatypes::Int32Type,
    };
    use arrow_schema::{DataType, Field, Schema, TimeUnit};

    use hex::encode;
    use pretty_assertions::assert_eq;
    use sha2::{Digest as _, Sha256};

    use crate::arrow_digester_core::ArrowDigesterCore;
    use arrow::array::{Decimal256Array, Decimal64Array};
    use arrow::buffer::OffsetBuffer;
    use arrow_buffer::i256;

    #[expect(
        clippy::too_many_lines,
        reason = "Comprehensive test of schema serialization and nested field name extraction"
    )]
    #[test]
    fn schema() {
        let schema = Schema::new(vec![
            Field::new("bool_name", DataType::Boolean, true),
            Field::new("int8_name", DataType::Int8, false),
            Field::new("uint8_name", DataType::UInt8, false),
            Field::new("int16_name", DataType::Int16, false),
            Field::new("uint16_name", DataType::UInt16, false),
            Field::new("int32_name", DataType::Int32, false),
            Field::new("uint32_name", DataType::UInt32, false),
            Field::new("int64_name", DataType::Int64, false),
            Field::new("uint64_name", DataType::UInt64, false),
            Field::new("float32_name", DataType::Float32, false),
            Field::new("float64_name", DataType::Float64, false),
            Field::new("date32_name", DataType::Date32, false),
            Field::new("date64_name", DataType::Date64, false),
            Field::new(
                "time32_second_name",
                DataType::Time32(TimeUnit::Second),
                false,
            ),
            Field::new(
                "time32_millis_name",
                DataType::Time32(TimeUnit::Millisecond),
                false,
            ),
            Field::new(
                "time64_micro_name",
                DataType::Time64(TimeUnit::Microsecond),
                false,
            ),
            Field::new(
                "time64_nano_name",
                DataType::Time64(TimeUnit::Nanosecond),
                false,
            ),
            Field::new("binary_name", DataType::Binary, true),
            Field::new("large_binary_name", DataType::LargeBinary, true),
            Field::new("utf8_name", DataType::Utf8, true),
            Field::new("large_utf8_name", DataType::LargeUtf8, true),
            Field::new(
                "list_name",
                DataType::List(Box::new(Field::new("item", DataType::Int32, true)).into()),
                true,
            ),
            Field::new(
                "large_list_name",
                DataType::LargeList(Box::new(Field::new("item", DataType::Int32, true)).into()),
                true,
            ),
            Field::new("decimal32_name", DataType::Decimal32(9, 2), true),
            Field::new("decimal64_name", DataType::Decimal64(18, 3), true),
            Field::new("decimal128_name", DataType::Decimal128(38, 5), true),
            Field::new(
                "struct_name",
                DataType::Struct(
                    vec![
                        Field::new("struct_field1", DataType::Int32, false),
                        Field::new("struct_field2", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ),
            Field::new(
                "doubly_nested_struct_name",
                DataType::Struct(
                    vec![
                        Field::new("outer_field", DataType::Int32, false),
                        Field::new(
                            "middle",
                            DataType::Struct(
                                vec![
                                    Field::new("middle_field", DataType::Utf8, true),
                                    Field::new(
                                        "inner",
                                        DataType::Struct(
                                            vec![
                                                Field::new("inner_field1", DataType::Int64, false),
                                                Field::new("inner_field2", DataType::Boolean, true),
                                            ]
                                            .into(),
                                        ),
                                        false,
                                    ),
                                ]
                                .into(),
                            ),
                            false,
                        ),
                    ]
                    .into(),
                ),
                true,
            ),
        ]);

        // Serialize the schema and covert it over to pretty json for comparison
        let compact_json: serde_json::Value =
            serde_json::from_str(&ArrowDigesterCore::<Sha256>::serialized_schema(&schema)).unwrap();
        let mut pretty_json = serde_json::to_string_pretty(&compact_json).unwrap();
        pretty_json.push('\n');

        println!("{pretty_json}");

        assert_eq!(
            pretty_json,
            include_str!("../tests/golden_files/schema_serialization_pretty.json")
        );
    }

    #[test]
    fn nested_fields() {
        // Test nested struct field name extraction
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "nested",
                DataType::Struct(
                    vec![
                        Field::new("name", DataType::Utf8, true),
                        Field::new(
                            "deep",
                            DataType::Struct(
                                vec![Field::new("value", DataType::Int64, false)].into(),
                            ),
                            false,
                        ),
                    ]
                    .into(),
                ),
                false,
            ),
        ]);

        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        assert_eq!(field_names.len(), 3);
        assert!(field_names.contains(&&"id".to_owned()));
        assert!(field_names.contains(&&"nested/name".to_owned()));
        assert!(field_names.contains(&&"nested/deep/value".to_owned()));

        let compact_json: serde_json::Value =
            serde_json::from_str(&ArrowDigesterCore::<Sha256>::serialized_schema(&schema)).unwrap();
        let mut pretty_json = serde_json::to_string_pretty(&compact_json).unwrap();
        pretty_json.push('\n');
        print!("{pretty_json}");

        // Test the nested field update by creating record_batch and using the update method
        let id_array = Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef;
        let name_array = Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])) as ArrayRef;
        let value_array = Arc::new(Int64Array::from(vec![Some(100), Some(200)])) as ArrayRef;

        let schema_ref = Arc::new(schema);

        let nested_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, true)),
                name_array,
            ),
            (
                Arc::new(Field::new(
                    "deep",
                    DataType::Struct(vec![Field::new("value", DataType::Int64, false)].into()),
                    false,
                )),
                Arc::new(StructArray::from(vec![(
                    Arc::new(Field::new("value", DataType::Int64, false)),
                    value_array,
                )])) as ArrayRef,
            ),
        ]);

        let record_batch = RecordBatch::try_new(
            Arc::clone(&schema_ref),
            vec![id_array, Arc::new(nested_struct)],
        )
        .unwrap();

        digester.update(&record_batch);

        // Check the digest
        assert_eq!(
            encode(digester.finalize()),
            // format 2 (ITL-898); format 1 value was
            // 9b52ad7430dea81b35f14a04d828b2424080fbc210570081c6e6cb62b6566c42
            "e87004b857524cdf72c644487132ae03f113a88662ca6ac32d90ab1a4d0c0c58"
        );
    }

    // ── Boolean ───────────────────────────────────────────────────────────

    #[test]
    fn digest_bool_nullable_bytes() {
        // [true, None, false, true] — one byte per valid value, null skipped
        // (v0.5 fix, ITL-898: v0.4.0 bit-packed the valid values into 0x05)
        let array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Boolean, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Boolean,
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 4);
        assert!(null_bit_vec[0], "index 0 (true) should be valid");
        assert!(!null_bit_vec[1], "index 1 (None) should be null");
        assert!(null_bit_vec[2], "index 2 (false) should be valid");
        assert!(null_bit_vec[3], "index 3 (true) should be valid");

        // Valid values [true, false, true] → bytes 01 00 01
        let mut manual = Sha256::new();
        manual.update([0x01_u8, 0x00, 0x01]);
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_bool_non_nullable_bytes() {
        // [false, true, false] — one byte per value, no nulls
        // (v0.5 fix, ITL-898: v0.4.0 bit-packed them into 0x02)
        let array = BooleanArray::from(vec![false, true, false]);
        let schema = Schema::new(vec![Field::new("col", DataType::Boolean, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Boolean,
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        // [false, true, false] → bytes 00 01 00
        let mut manual = Sha256::new();
        manual.update([0x00_u8, 0x01, 0x00]);
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Int8 / UInt8 (1-byte fixed) ───────────────────────────────────────

    #[test]
    fn digest_int8_nullable_bytes() {
        // [10, None, -3] — valid bytes: 0x0A, 0xFD
        let array = Int8Array::from(vec![Some(10_i8), None, Some(-3_i8)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Int8, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int8, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(10_i8.to_le_bytes()); // 0a
        manual.update((-3_i8).to_le_bytes()); // fd
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_uint8_non_nullable_bytes() {
        // [1, 2, 255]
        let array = UInt8Array::from(vec![1_u8, 2_u8, 255_u8]);
        let schema = Schema::new(vec![Field::new("col", DataType::UInt8, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::UInt8, false)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update([0x01_u8, 0x02_u8, 0xFF_u8]);
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Int16 / UInt16 (2-byte fixed) ─────────────────────────────────────

    #[test]
    fn digest_int16_nullable_bytes() {
        // [1000, None, -512]
        // 1000 LE  = e8 03
        // -512 LE  = 00 fe
        let array = Int16Array::from(vec![Some(1000_i16), None, Some(-512_i16)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Int16, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int16, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(1000_i16.to_le_bytes());
        manual.update((-512_i16).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_uint16_non_nullable_bytes() {
        // [100, 200, 65535]
        let array = UInt16Array::from(vec![100_u16, 200_u16, 0xFFFF_u16]);
        let schema = Schema::new(vec![Field::new("col", DataType::UInt16, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::UInt16,
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(100_u16.to_le_bytes());
        manual.update(200_u16.to_le_bytes());
        manual.update(0xFFFF_u16.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Float16 (2-byte float) ─────────────────────────────────────────────

    #[test]
    fn digest_float16_non_nullable_bytes() {
        // [1.0, 2.5, -0.5]
        let array = Float16Array::from(vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.5),
            half::f16::from_f32(-0.5),
        ]);
        let schema = Schema::new(vec![Field::new("col", DataType::Float16, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Float16,
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(half::f16::from_f32(1.0).to_le_bytes());
        manual.update(half::f16::from_f32(2.5).to_le_bytes());
        manual.update(half::f16::from_f32(-0.5).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Int32 / UInt32 (4-byte fixed) ─────────────────────────────────────

    // Tests to check if the digest is bytes that are being fed into the digest in the correct format
    #[test]
    fn digest_int32_nullable_bytes() {
        // Given this array, we will compare our manual hashing of the bytes according to our spec
        // with the output of the digest to make sure they are consistent.

        let int_array = Int32Array::from(vec![Some(42), None, Some(-7), Some(0)]);

        let schema = Schema::new(vec![Field::new("int32_col", DataType::Int32, true)]);

        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);

        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "int32_col",
                    DataType::Int32,
                    true,
                )])),
                vec![Arc::new(int_array)],
            )
            .unwrap(),
        );

        let buf = digester
            .fields_digest_buffer
            .get("int32_col")
            .expect("int32_col field should exist in digest buffer");
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        // The null bit vector should be [true, false, true, true] for [Some(42), None, Some(-7), Some(0)]
        assert_eq!(null_bit_vec.len(), 4);
        assert!(null_bit_vec[0], "index 0 (42) should be valid");
        assert!(!null_bit_vec[1], "index 1 (None) should be null");
        assert!(null_bit_vec[2], "index 2 (-7) should be valid");
        assert!(null_bit_vec[3], "index 3 (0) should be valid");

        // For the valid values [42, -7, 0], we hash only their little-endian bytes (nulls are skipped):
        // 42  -> 2a 00 00 00
        // -7  -> f9 ff ff ff
        // 0   -> 00 00 00 00
        let mut manual_digest = Sha256::new();
        manual_digest.update([0x2a, 0x00, 0x00, 0x00]);
        manual_digest.update([0xf9, 0xff, 0xff, 0xff]);
        manual_digest.update([0x00, 0x00, 0x00, 0x00]);

        assert_eq!(data_digest.clone().finalize(), manual_digest.finalize());
    }

    #[test]
    fn digest_uint32_nullable_bytes() {
        // [0, None, u32::MAX]
        let array = UInt32Array::from(vec![Some(0_u32), None, Some(u32::MAX)]);
        let schema = Schema::new(vec![Field::new("col", DataType::UInt32, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::UInt32, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(0_u32.to_le_bytes());
        manual.update(u32::MAX.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Float32 (4-byte float) ─────────────────────────────────────────────

    #[test]
    fn digest_float32_nullable_bytes() {
        // [1.0, None, 2.5]
        // 1.0f32 LE: 00 00 80 3f
        // 2.5f32 LE: 00 00 20 40
        let array = Float32Array::from(vec![Some(1.0_f32), None, Some(2.5_f32)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Float32, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Float32,
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(1.0_f32.to_le_bytes());
        manual.update(2.5_f32.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Decimal32 (4-byte decimal) ────────────────────────────────────────

    #[test]
    fn digest_decimal32_nullable_bytes() {
        // Values are stored as i32 in 4 little-endian bytes.
        // [12345, None, -999]
        let array = Decimal32Array::from(vec![Some(12_345_i32), None, Some(-999_i32)])
            .with_precision_and_scale(9, 2)
            .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal32(9, 2), true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Decimal32(9, 2),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(12_345_i32.to_le_bytes());
        manual.update((-999_i32).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_decimal32_non_nullable_bytes() {
        // [0, 1000, -1]
        let array = Decimal32Array::from(vec![Some(0_i32), Some(1_000_i32), Some(-1_i32)])
            .with_precision_and_scale(9, 2)
            .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal32(9, 2), false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Decimal32(9, 2),
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(0_i32.to_le_bytes());
        manual.update(1_000_i32.to_le_bytes());
        manual.update((-1_i32).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Int64 / UInt64 (8-byte fixed) ─────────────────────────────────────

    #[test]
    fn digest_int64_nullable_bytes() {
        // [i64::MIN, None, 9_876_543_210]
        let array = Int64Array::from(vec![Some(i64::MIN), None, Some(9_876_543_210_i64)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Int64, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int64, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(i64::MIN.to_le_bytes());
        manual.update(9_876_543_210_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_uint64_nullable_bytes() {
        // [0, None, u64::MAX]
        let array = UInt64Array::from(vec![Some(0_u64), None, Some(u64::MAX)]);
        let schema = Schema::new(vec![Field::new("col", DataType::UInt64, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::UInt64, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(0_u64.to_le_bytes());
        manual.update(u64::MAX.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Float64 (8-byte float) ─────────────────────────────────────────────

    #[test]
    fn digest_float64_non_nullable_bytes() {
        // [1.0, -0.5, π]
        let array = Float64Array::from(vec![1.0_f64, -0.5_f64, f64::consts::PI]);
        let schema = Schema::new(vec![Field::new("col", DataType::Float64, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Float64,
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(1.0_f64.to_le_bytes());
        manual.update((-0.5_f64).to_le_bytes());
        manual.update(consts::PI.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Decimal64 (8-byte decimal) ────────────────────────────────────────

    #[test]
    fn digest_decimal64_nullable_bytes() {
        // Values are stored as i64 in 8 little-endian bytes.
        // [987654321, None, -42]
        let array = Decimal64Array::from(vec![Some(987_654_321_i64), None, Some(-42_i64)])
            .with_precision_and_scale(18, 3)
            .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal64(18, 3), true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Decimal64(18, 3),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(987_654_321_i64.to_le_bytes());
        manual.update((-42_i64).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_decimal64_non_nullable_bytes() {
        // [0, 1000000, -1]
        let array = Decimal64Array::from(vec![Some(0_i64), Some(1_000_000_i64), Some(-1_i64)])
            .with_precision_and_scale(18, 3)
            .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal64(18, 3), false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Decimal64(18, 3),
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(1_000_000_i64.to_le_bytes());
        manual.update((-1_i64).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Date32 / Date64 ───────────────────────────────────────────────────

    #[test]
    fn digest_date32_nullable_bytes() {
        // Days since Unix epoch: [0, None, 19000]
        let array = Date32Array::from(vec![Some(0_i32), None, Some(19000_i32)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Date32, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Date32, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(0_i32.to_le_bytes());
        manual.update(19000_i32.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_date64_nullable_bytes() {
        // Milliseconds since Unix epoch: [0, None, 1_000_000]
        let array = Date64Array::from(vec![Some(0_i64), None, Some(1_000_000_i64)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Date64, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Date64, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(1_000_000_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Time32 / Time64 ───────────────────────────────────────────────────

    #[test]
    fn digest_time32_nullable_bytes() {
        // Seconds since midnight: [0, None, 3600]
        let array = Time32SecondArray::from(vec![Some(0_i32), None, Some(3600_i32)]);
        let schema = Schema::new(vec![Field::new(
            "col",
            DataType::Time32(TimeUnit::Second),
            true,
        )]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Time32(TimeUnit::Second),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(0_i32.to_le_bytes());
        manual.update(3600_i32.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_time64_nullable_bytes() {
        // Microseconds since midnight: [0, None, 3_600_000_000]
        let array = Time64MicrosecondArray::from(vec![Some(0_i64), None, Some(3_600_000_000_i64)]);
        let schema = Schema::new(vec![Field::new(
            "col",
            DataType::Time64(TimeUnit::Microsecond),
            true,
        )]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Time64(TimeUnit::Microsecond),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(3_600_000_000_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Timestamp / Duration ──────────────────────────────────────────────

    #[test]
    fn digest_timestamp_nullable_bytes() {
        // Microseconds since epoch: [0, None, 3_600_000_000]
        let array =
            TimestampMicrosecondArray::from(vec![Some(0_i64), None, Some(3_600_000_000_i64)])
                .with_timezone("UTC");
        let mut digester = ArrowDigesterCore::<Sha256>::new(
            &Schema::new(vec![Field::new(
                "col",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            )]),
            false,
        );
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0], "index 0 should be valid");
        assert!(!null_bit_vec[1], "index 1 (None) should be null");
        assert!(null_bit_vec[2], "index 2 should be valid");

        // Physical storage: int64 LE bytes for valid elements only
        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(3_600_000_000_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_duration_nullable_bytes() {
        // Duration microseconds: [0, None, 3_600_000_000]
        let array =
            DurationMicrosecondArray::from(vec![Some(0_i64), None, Some(3_600_000_000_i64)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(
            &Schema::new(vec![Field::new(
                "col",
                DataType::Duration(TimeUnit::Microsecond),
                true,
            )]),
            false,
        );
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Duration(TimeUnit::Microsecond),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0], "index 0 should be valid");
        assert!(!null_bit_vec[1], "index 1 (None) should be null");
        assert!(null_bit_vec[2], "index 2 should be valid");

        // Physical storage: int64 LE bytes for valid elements only
        let mut manual = Sha256::new();
        manual.update(0_i64.to_le_bytes());
        manual.update(3_600_000_000_i64.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Decimal128 (16-byte decimal) ──────────────────────────────────────

    #[test]
    fn digest_decimal128_nullable_bytes() {
        // Values are stored as i128 in 16 little-endian bytes.
        // [123456, None, -1]
        let array = Decimal128Array::from(vec![Some(123_456_i128), None, Some(-1_i128)])
            .with_precision_and_scale(38, 5)
            .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal128(38, 5), true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Decimal128(38, 5),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(123_456_i128.to_le_bytes());
        manual.update((-1_i128).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Decimal256 (32-byte decimal) ──────────────────────────────────────

    #[test]
    fn digest_decimal256_nullable_bytes() {
        // Values are stored as i256 in 32 little-endian bytes.
        // [123456, None, -1]
        let array = Decimal256Array::from(vec![
            Some(i256::from_i128(123_456_i128)),
            None,
            Some(i256::from_i128(-1_i128)),
        ])
        .with_precision_and_scale(76, 10)
        .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal256(76, 10), true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::Decimal256(76, 10),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(i256::from_i128(123_456_i128).to_le_bytes());
        manual.update(i256::from_i128(-1_i128).to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── FixedSizeBinary ───────────────────────────────────────────────────

    #[test]
    fn digest_fixed_size_binary_nullable_bytes() {
        // 4-byte fixed-width blobs: [0x01020304, None, 0xDEADBEEF]
        let mut builder = FixedSizeBinaryBuilder::with_capacity(3, 4);
        builder.append_value([0x01, 0x02, 0x03, 0x04]).unwrap();
        builder.append_null();
        builder.append_value([0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        let array = builder.finish();

        let schema = Schema::new(vec![Field::new("col", DataType::FixedSizeBinary(4), true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::FixedSizeBinary(4),
                    true,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        // Null bytes are skipped entirely; only valid slots' raw bytes are hashed.
        let mut manual = Sha256::new();
        manual.update([0x01_u8, 0x02_u8, 0x03_u8, 0x04_u8]);
        manual.update([0xDE_u8, 0xAD_u8, 0xBE_u8, 0xEF_u8]);
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Binary / LargeBinary ──────────────────────────────────────────────

    #[test]
    fn digest_binary_nullable_bytes() {
        // [b"hello", None, b"world"]
        // Valid entries: (length as u64 LE) ++ bytes.
        // Null entries are skipped entirely in the data digest.
        let array = BinaryArray::from(vec![Some(b"hello".as_ref()), None, Some(b"world".as_ref())]);
        let schema = Schema::new(vec![Field::new("col", DataType::Binary, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Binary, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(5_u64.to_le_bytes()); // len("hello")
        manual.update(b"hello");
        // null entry skipped — no sentinel bytes
        manual.update(5_u64.to_le_bytes()); // len("world")
        manual.update(b"world");
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_large_binary_non_nullable_bytes() {
        // [b"ab", b"cde"] — all valid, length prefix is usize LE
        let array = LargeBinaryArray::from(vec![b"ab".as_ref(), b"cde".as_ref()]);
        let schema = Schema::new(vec![Field::new("col", DataType::LargeBinary, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::LargeBinary,
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(2_u64.to_le_bytes());
        manual.update(b"ab");
        manual.update(3_u64.to_le_bytes());
        manual.update(b"cde");
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Utf8 / LargeUtf8 ──────────────────────────────────────────────────

    #[test]
    fn digest_utf8_nullable_bytes() {
        // ["foo", None, "ba"]
        // Valid entries: (length as u64 LE) ++ UTF-8 bytes.
        // Null entries are skipped entirely in the data digest.
        let array = StringArray::from(vec![Some("foo"), None, Some("ba")]);
        let schema = Schema::new(vec![Field::new("col", DataType::Utf8, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Utf8, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec[0]);
        assert!(!null_bit_vec[1]);
        assert!(null_bit_vec[2]);

        let mut manual = Sha256::new();
        manual.update(3_u64.to_le_bytes()); // len("foo")
        manual.update(b"foo");
        // null entry skipped — no sentinel bytes
        manual.update(2_u64.to_le_bytes()); // len("ba")
        manual.update(b"ba");
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_large_utf8_non_nullable_bytes() {
        // ["x", "yz"] — all valid, length prefix is u64 LE
        let array = LargeStringArray::from(vec!["x", "yz"]);
        let schema = Schema::new(vec![Field::new("col", DataType::LargeUtf8, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::LargeUtf8,
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        let mut manual = Sha256::new();
        manual.update(1_u64.to_le_bytes());
        manual.update(b"x");
        manual.update(2_u64.to_le_bytes());
        manual.update(b"yz");
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── List<Int32> / LargeList<Int32> ─────────────────────────────────────
    //
    // Format 2: a non-nullable List<Int32 nullable> column has the list node "col" (lengths)
    // and the element node "col/" (item validity and values).

    #[test]
    fn digest_list_non_nullable_bytes() {
        // [[10, 20, 30]] — single outer element, non-nullable List<Int32 nullable>
        let mut builder = ListBuilder::new(PrimitiveBuilder::<Int32Type>::new());
        builder.values().append_value(10);
        builder.values().append_value(20);
        builder.values().append_value(30);
        builder.append(true);
        let array = builder.finish();

        let item_field = Arc::new(Field::new("item", DataType::Int32, true));
        let schema = Schema::new(vec![Field::new(
            "col",
            DataType::List(Arc::clone(&item_field)),
            false,
        )]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::List(Arc::clone(&item_field)),
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        // Format 2: list node "col" (non-nullable: lengths only); element node "col/"
        // (nullable items: validity + values)
        let list_buf = &digester.fields_digest_buffer["col"];
        assert!(list_buf.null_bits.is_none() && list_buf.data.is_none());
        let buf = &digester.fields_digest_buffer["col/"];
        // Items are nullable → null_bits present (all valid in this case)
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable items");
        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec.iter().all(|b| *b), "All items should be valid");

        let structural_digest = list_buf
            .structural
            .as_ref()
            .expect("Expected structural digest for list");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        // Structural digest: element count (sizes separated from leaf data)
        let mut manual_structural = Sha256::new();
        manual_structural.update(3_u64.to_le_bytes());
        assert_eq!(
            structural_digest.clone().finalize(),
            manual_structural.finalize()
        );

        // Data/leaf digest: only the raw leaf values
        let mut manual_data = Sha256::new();
        manual_data.update(10_i32.to_le_bytes());
        manual_data.update(20_i32.to_le_bytes());
        manual_data.update(30_i32.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual_data.finalize());
    }

    #[test]
    fn digest_large_list_non_nullable_bytes() {
        // [[1, 2, 3]] — single outer element, non-nullable LargeList<Int32 nullable>
        let mut builder = LargeListBuilder::new(PrimitiveBuilder::<Int32Type>::new());
        builder.values().append_value(1);
        builder.values().append_value(2);
        builder.values().append_value(3);
        builder.append(true);
        let array = builder.finish();

        let item_field = Arc::new(Field::new("item", DataType::Int32, true));
        let schema = Schema::new(vec![Field::new(
            "col",
            DataType::LargeList(Arc::clone(&item_field)),
            false,
        )]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "col",
                    DataType::LargeList(Arc::clone(&item_field)),
                    false,
                )])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        // Format 2: list node "col" (non-nullable: lengths only); element node "col/"
        // (nullable items: validity + values)
        let list_buf = &digester.fields_digest_buffer["col"];
        assert!(list_buf.null_bits.is_none() && list_buf.data.is_none());
        let buf = &digester.fields_digest_buffer["col/"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable items");
        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec.iter().all(|b| *b), "All items should be valid");

        let structural_digest = list_buf
            .structural
            .as_ref()
            .expect("Expected structural digest for list");
        let data_digest = buf.data.as_ref().expect("Expected data digest");

        // Structural digest: element count (sizes separated from leaf data)
        let mut manual_structural = Sha256::new();
        manual_structural.update(3_u64.to_le_bytes());
        assert_eq!(
            structural_digest.clone().finalize(),
            manual_structural.finalize()
        );

        // Data/leaf digest: only the raw leaf values
        let mut manual_data = Sha256::new();
        manual_data.update(1_i32.to_le_bytes());
        manual_data.update(2_i32.to_le_bytes());
        manual_data.update(3_i32.to_le_bytes());
        assert_eq!(data_digest.clone().finalize(), manual_data.finalize());
    }

    #[test]
    fn digest_buffer_type_list() {
        let buf = super::DigestBufferType::<Sha256>::new_list(true);
        assert!(buf.null_bits.is_some());
        assert!(buf.structural.is_some());
        assert!(buf.data.is_none());
    }

    #[test]
    fn digest_buffer_type_data_only() {
        let buf = super::DigestBufferType::<Sha256>::new_data_only(false);
        assert!(buf.null_bits.is_none());
        assert!(buf.structural.is_none());
        assert!(buf.data.is_some());
    }

    #[test]
    fn digest_buffer_type_validity_only() {
        let buf = super::DigestBufferType::<Sha256>::new_validity_only();
        assert!(buf.null_bits.is_some());
        assert!(buf.structural.is_none());
        assert!(buf.data.is_none());
    }

    #[test]
    fn extract_fields_list_of_struct() {
        // List<Struct<a: Int32, b: String>>
        let schema = Schema::new(vec![Field::new(
            "x",
            DataType::LargeList(Arc::new(Field::new(
                "item",
                DataType::Struct(
                    vec![
                        Field::new("a", DataType::Int32, false),
                        Field::new("b", DataType::LargeUtf8, false),
                    ]
                    .into(),
                ),
                false,
            ))),
            true, // column is nullable
        )]);

        let digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        // Format 2: "x" (list node: validity + lengths), "x//a", "x//b" (leaves); the
        // non-nullable struct element "x/" has no entry
        assert_eq!(
            field_names.len(),
            3,
            "Expected 3 entries, got: {field_names:?}"
        );
        assert!(field_names.contains(&&"x".to_owned()));
        assert!(!field_names.contains(&&"x/".to_owned()));
        assert!(field_names.contains(&&"x//a".to_owned()));
        assert!(field_names.contains(&&"x//b".to_owned()));
    }

    #[test]
    fn extract_fields_nested_list_struct_list() {
        // x: Nullable<List<Struct<a: Nullable<Int32>, b: Struct<g: Nullable<List<Int32>>, h: Int32>>>>
        let schema = Schema::new(vec![Field::new(
            "x",
            DataType::LargeList(Arc::new(Field::new(
                "item",
                DataType::Struct(
                    vec![
                        Field::new("a", DataType::Int32, true),
                        Field::new(
                            "b",
                            DataType::Struct(
                                vec![
                                    Field::new(
                                        "g",
                                        DataType::LargeList(Arc::new(Field::new(
                                            "item",
                                            DataType::Int32,
                                            false,
                                        ))),
                                        true,
                                    ),
                                    Field::new("h", DataType::Int32, false),
                                ]
                                .into(),
                            ),
                            false,
                        ),
                    ]
                    .into(),
                ),
                false,
            ))),
            true,
        )]);

        let digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        // Format 2: "x", "x//a", "x//b/g", "x//b/g/", "x//b/h" (non-nullable structs have no entry)
        assert_eq!(
            field_names.len(),
            5,
            "Expected 5 entries, got: {field_names:?}"
        );
        assert!(field_names.contains(&&"x".to_owned()));
        assert!(field_names.contains(&&"x//a".to_owned()));
        assert!(field_names.contains(&&"x//b/g".to_owned()));
        assert!(field_names.contains(&&"x//b/g/".to_owned()));
        assert!(field_names.contains(&&"x//b/h".to_owned()));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "One byte-level expectation for a deep nested type"
    )]
    fn recursive_list_struct_decomposition() {
        use crate::arrow_digester_core::normalize_schema;

        // Schema: x: Nullable<List<Struct<
        //     a: Nullable<Int32>,
        //     b: Struct<
        //         g: Nullable<List<Int32>>,
        //         h: Int32
        //     >
        // >>>
        let g_field = Field::new(
            "g",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, false))),
            true, // g is nullable
        );
        let h_field = Field::new("h", DataType::Int32, false);
        let b_field = Field::new(
            "b",
            DataType::Struct(vec![g_field.clone(), h_field.clone()].into()),
            false, // b is non-nullable
        );
        let a_field = Field::new("a", DataType::Int32, true); // a is nullable
        let struct_type = DataType::Struct(vec![a_field.clone(), b_field.clone()].into());
        let item_field = Field::new("item", struct_type, false);
        let x_field = Field::new(
            "x",
            DataType::LargeList(Arc::new(item_field.clone())),
            true, // column is nullable
        );
        let schema = Schema::new(vec![x_field]);

        // Build the data:
        // Row 0: [{a: 1, b: {g: [10, 20], h: 100}}, {a: null, b: {g: [30], h: 200}}]
        // Row 1: null
        // Row 2: [{a: 3, b: {g: null, h: 300}}, {a: 4, b: {g: [], h: 400}}, {a: 5, b: {g: [50], h: 500}}]

        // Inner g values: [10, 20, 30, 50] (across all non-null g lists)
        let g_values = Int32Array::from(vec![10, 20, 30, 50]);
        // g list offsets: elem0=[10,20](len2), elem1=[30](len1), elem2=null, elem3=[](len0), elem4=[50](len1)
        // For 5 struct elements, g has offsets [0, 2, 3, 3, 3, 4]
        // with validity [true, true, false, true, true]
        let g_list = LargeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            OffsetBuffer::new(vec![0_i64, 2, 3, 3, 3, 4].into()),
            Arc::new(g_values) as ArrayRef,
            Some(vec![true, true, false, true, true].into()), // g null at struct element 2
        );

        let h_values = Int32Array::from(vec![100, 200, 300, 400, 500]);

        let b_struct = StructArray::from(vec![
            (Arc::new(g_field), Arc::new(g_list) as ArrayRef),
            (Arc::new(h_field), Arc::new(h_values) as ArrayRef),
        ]);

        let a_values = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);

        let inner_struct = StructArray::from(vec![
            (Arc::new(a_field), Arc::new(a_values) as ArrayRef),
            (Arc::new(b_field), Arc::new(b_struct) as ArrayRef),
        ]);

        // Outer list: Row 0 has 2 elements, Row 1 is null, Row 2 has 3 elements
        // Offsets: [0, 2, 2, 5] (row 1 is null but offset still present)
        let outer_list = LargeListArray::new(
            Arc::new(item_field),
            OffsetBuffer::new(vec![0_i64, 2, 2, 5].into()),
            Arc::new(inner_struct) as ArrayRef,
            Some(vec![true, false, true].into()), // row 1 is null
        );

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(outer_list) as ArrayRef],
        )
        .unwrap();

        // ── Compute expected hash manually ──
        // Format 2 entries (in path order), after the row count 3:
        // "x"       → validity V,I,V (3 bits), lengths [2, 3]
        // "x//a"    → validity V,I,V,V,V (5 bits), values [1, 3, 4, 5] as i32 LE
        // "x//b/g"  → validity V,V,I,V,V (5 bits), lengths [2, 1, 0, 1]
        // "x//b/g/" → values [10, 20, 30, 50] as i32 LE
        // "x//b/h"  → values [100, 200, 300, 400, 500] as i32 LE

        let mut schema_hasher = Sha256::new();
        schema_hasher.update(super::header(super::KIND_SCHEMA, 0));
        schema_hasher.update(
            ArrowDigesterCore::<Sha256>::serialized_schema(&normalize_schema(&schema)).as_bytes(),
        );
        let schema_digest = schema_hasher.finalize();

        let mut final_digest = Sha256::new();
        final_digest.update(super::header(super::KIND_RECORD_BATCH, 0));
        final_digest.update(schema_digest);
        final_digest.update(3_u64.to_le_bytes()); // row count

        // Entry "x": validity V,I,V → bit_count=3, validity=0b101=5; lengths [2, 3]
        final_digest.update(3_u64.to_le_bytes());
        final_digest.update(5_u8.to_le_bytes());
        let mut x_structural = Sha256::new();
        x_structural.update(2_u64.to_le_bytes());
        x_structural.update(3_u64.to_le_bytes());
        final_digest.update(x_structural.finalize());

        // Entry "x//a": null_bits V,I,V,V,V → bit_count=5, validity=0b11101=29
        //   data: [1, 3, 4, 5] as i32 LE
        final_digest.update(5_u64.to_le_bytes());
        final_digest.update(29_u8.to_le_bytes());
        let mut xa_data = Sha256::new();
        xa_data.update(1_i32.to_le_bytes());
        xa_data.update(3_i32.to_le_bytes());
        xa_data.update(4_i32.to_le_bytes());
        xa_data.update(5_i32.to_le_bytes());
        final_digest.update(xa_data.finalize());

        // Entry "x//b/g": validity V,V,I,V,V → bit_count=5, validity=0b11011=27; lengths
        final_digest.update(5_u64.to_le_bytes());
        final_digest.update(27_u8.to_le_bytes());
        let mut xbg_structural = Sha256::new();
        xbg_structural.update(2_u64.to_le_bytes());
        xbg_structural.update(1_u64.to_le_bytes());
        xbg_structural.update(0_u64.to_le_bytes());
        xbg_structural.update(1_u64.to_le_bytes());
        final_digest.update(xbg_structural.finalize());
        // Entry "x//b/g/": values [10, 20, 30, 50] as i32 LE
        let mut xbg_data = Sha256::new();
        xbg_data.update(10_i32.to_le_bytes());
        xbg_data.update(20_i32.to_le_bytes());
        xbg_data.update(30_i32.to_le_bytes());
        xbg_data.update(50_i32.to_le_bytes());
        final_digest.update(xbg_data.finalize());

        // Entry "x//b/h": data only [100, 200, 300, 400, 500] as i32 LE
        let mut h_leaf_data = Sha256::new();
        h_leaf_data.update(100_i32.to_le_bytes());
        h_leaf_data.update(200_i32.to_le_bytes());
        h_leaf_data.update(300_i32.to_le_bytes());
        h_leaf_data.update(400_i32.to_le_bytes());
        h_leaf_data.update(500_i32.to_le_bytes());
        final_digest.update(h_leaf_data.finalize());

        let expected_hash = final_digest.finalize().to_vec();

        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema, false);
        digester.update(&batch);

        let actual_hash = digester.finalize();

        assert_eq!(
            encode(&actual_hash),
            encode(&expected_hash),
            "Recursive list/struct decomposition hash mismatch"
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Test builds multiple complex batches for batch-split independence verification"
    )]
    #[test]
    fn recursive_list_struct_batch_split_independence() {
        // Same schema and data as recursive_list_struct_decomposition,
        // split into two batches: rows 0-1 and row 2.
        // Verify: hash(batch1 + batch2) == hash(combined)

        let g_field = Field::new(
            "g",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, false))),
            true,
        );
        let h_field = Field::new("h", DataType::Int32, false);
        let b_field = Field::new(
            "b",
            DataType::Struct(vec![g_field.clone(), h_field.clone()].into()),
            false,
        );
        let a_field = Field::new("a", DataType::Int32, true);
        let struct_type = DataType::Struct(vec![a_field.clone(), b_field.clone()].into());
        let item_field = Field::new("item", struct_type, false);
        let x_field = Field::new("x", DataType::LargeList(Arc::new(item_field.clone())), true);
        let schema = Arc::new(Schema::new(vec![x_field]));

        // ── Build combined batch (all 3 rows) ──
        let g_values = Int32Array::from(vec![10, 20, 30, 50]);
        let g_list = LargeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            OffsetBuffer::new(vec![0_i64, 2, 3, 3, 3, 4].into()),
            Arc::new(g_values) as ArrayRef,
            Some(vec![true, true, false, true, true].into()),
        );
        let h_values = Int32Array::from(vec![100, 200, 300, 400, 500]);
        let b_struct = StructArray::from(vec![
            (Arc::new(g_field.clone()), Arc::new(g_list) as ArrayRef),
            (Arc::new(h_field.clone()), Arc::new(h_values) as ArrayRef),
        ]);
        let a_values = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let inner_struct = StructArray::from(vec![
            (Arc::new(a_field.clone()), Arc::new(a_values) as ArrayRef),
            (Arc::new(b_field.clone()), Arc::new(b_struct) as ArrayRef),
        ]);
        let outer_list = LargeListArray::new(
            Arc::new(item_field.clone()),
            OffsetBuffer::new(vec![0_i64, 2, 2, 5].into()),
            Arc::new(inner_struct) as ArrayRef,
            Some(vec![true, false, true].into()),
        );
        let combined_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(outer_list) as ArrayRef])
                .unwrap();

        // ── Build batch 1: rows 0-1 ──
        let g_values_1 = Int32Array::from(vec![10, 20, 30]);
        let g_list_1 = LargeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            OffsetBuffer::new(vec![0_i64, 2, 3].into()),
            Arc::new(g_values_1) as ArrayRef,
            Some(vec![true, true].into()),
        );
        let h_values_1 = Int32Array::from(vec![100, 200]);
        let b_struct_1 = StructArray::from(vec![
            (Arc::new(g_field.clone()), Arc::new(g_list_1) as ArrayRef),
            (Arc::new(h_field.clone()), Arc::new(h_values_1) as ArrayRef),
        ]);
        let a_values_1 = Int32Array::from(vec![Some(1), None]);
        let inner_struct_1 = StructArray::from(vec![
            (Arc::new(a_field.clone()), Arc::new(a_values_1) as ArrayRef),
            (Arc::new(b_field.clone()), Arc::new(b_struct_1) as ArrayRef),
        ]);
        let outer_list_1 = LargeListArray::new(
            Arc::new(item_field.clone()),
            OffsetBuffer::new(vec![0_i64, 2, 2].into()),
            Arc::new(inner_struct_1) as ArrayRef,
            Some(vec![true, false].into()),
        );
        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(outer_list_1) as ArrayRef],
        )
        .unwrap();

        // ── Build batch 2: row 2 ──
        let g_values_2 = Int32Array::from(vec![50]);
        let g_list_2 = LargeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            OffsetBuffer::new(vec![0_i64, 0, 0, 1].into()),
            Arc::new(g_values_2) as ArrayRef,
            Some(vec![false, true, true].into()),
        );
        let h_values_2 = Int32Array::from(vec![300, 400, 500]);
        let b_struct_2 = StructArray::from(vec![
            (Arc::new(g_field), Arc::new(g_list_2) as ArrayRef),
            (Arc::new(h_field), Arc::new(h_values_2) as ArrayRef),
        ]);
        let a_values_2 = Int32Array::from(vec![Some(3), Some(4), Some(5)]);
        let inner_struct_2 = StructArray::from(vec![
            (Arc::new(a_field), Arc::new(a_values_2) as ArrayRef),
            (Arc::new(b_field), Arc::new(b_struct_2) as ArrayRef),
        ]);
        let outer_list_2 = LargeListArray::new(
            Arc::new(item_field),
            OffsetBuffer::new(vec![0_i64, 3].into()),
            Arc::new(inner_struct_2) as ArrayRef,
            Some(vec![true].into()),
        );
        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(outer_list_2) as ArrayRef],
        )
        .unwrap();

        // ── Compare ──
        let mut single = ArrowDigesterCore::<Sha256>::new(schema.as_ref(), false);
        single.update(&combined_batch);
        let single_hash = single.finalize();

        let mut split = ArrowDigesterCore::<Sha256>::new(schema.as_ref(), false);
        split.update(&batch1);
        split.update(&batch2);
        let split_hash = split.finalize();

        assert_eq!(
            encode(&single_hash),
            encode(&split_hash),
            "Batch split independence failed for recursive list/struct decomposition"
        );
    }

    #[test]
    fn hash_array_list_of_struct() {
        // Verify hash_array works with List<Struct<...>> using the same recursive
        // decomposition as the record-batch path.
        let inner_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("b", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            ),
        ]);

        let list_array = LargeListArray::new(
            Arc::new(Field::new(
                "item",
                DataType::Struct(
                    vec![
                        Field::new("a", DataType::Int32, false),
                        Field::new("b", DataType::Int32, false),
                    ]
                    .into(),
                ),
                false,
            )),
            OffsetBuffer::new(vec![0_i64, 2, 3].into()),
            Arc::new(inner_struct) as ArrayRef,
            Some(vec![true, true].into()),
        );

        let hash1 = ArrowDigesterCore::<Sha256>::hash_array(&list_array);
        let hash2 = ArrowDigesterCore::<Sha256>::hash_array(&list_array);
        assert_eq!(hash1, hash2, "hash_array should be deterministic");
        assert_eq!(
            hash1.len(),
            32,
            "core hash_array should return 32 bytes (SHA-256)"
        );
    }
}
