#![expect(
    clippy::expect_used,
    clippy::todo,
    reason = "First iteration of code, will add proper error handling later. Allow for unsupported data types for now"
)]
#![expect(
    clippy::big_endian_bytes,
    reason = "Validity bytes are deliberately written in big-endian order for cross-platform consistency"
)]
use std::{collections::BTreeMap, iter::repeat_n, sync::Arc};

use arrow::{
    array::{
        make_array, Array, BooleanArray, GenericBinaryArray, GenericListArray, GenericStringArray,
        LargeBinaryArray, LargeListArray, LargeStringArray, OffsetSizeTrait, RecordBatch,
        StructArray,
    },
    buffer::NullBuffer,
    compute::cast,
    datatypes::{DataType, Schema},
};
use arrow_schema::Field;
use bitvec::prelude::*;
use digest::Digest;

const DELIMITER_FOR_NESTED_FIELD: &str = "/";

#[derive(Clone)]
struct DigestBufferType<D: Digest> {
    null_bits: Option<BitVec<u8, Lsb0>>,
    structural: Option<D>,
    data: Option<D>,
}

impl<D: Digest> DigestBufferType<D> {
    /// Create a buffer with all components present (legacy constructor).
    #[deprecated(
        note = "Use new_data_only, new_structural_only, new_list_leaf, or new_validity_only"
    )]
    fn new(nullable: bool, structured: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: structured.then(D::new),
            data: Some(D::new()),
        }
    }

    /// Create a buffer for a leaf field (data + optional `null_bits`).
    fn new_data_only(nullable: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: None,
            data: Some(D::new()),
        }
    }

    /// Create a buffer for a list-level-only entry (structural + optional `null_bits`, no data).
    fn new_structural_only(nullable: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: Some(D::new()),
            data: None,
        }
    }

    /// Create a buffer for a leaf that is itself a list type (structural + data + optional `null_bits`).
    fn new_list_leaf(nullable: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: Some(D::new()),
            data: Some(D::new()),
        }
    }

    /// Create a buffer for a column-level nullable entry (`null_bits` only).
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

const fn is_list_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::List(_) | DataType::LargeList(_))
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

/// Normalize a single field: keep name and nullability, normalize the data type recursively.
fn normalize_field(field: &Field) -> Field {
    Field::new(
        field.name(),
        normalize_data_type(field.data_type()),
        field.is_nullable(),
    )
}

/// Normalize all fields in a schema to their canonical large equivalents.
fn normalize_schema(schema: &Schema) -> Schema {
    Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| Arc::new(normalize_field(f)))
            .collect::<Vec<_>>(),
    )
}

#[derive(Clone)]
pub struct ArrowDigesterCore<D: Digest> {
    schema_digest: Vec<u8>,
    serialized_schema: String,
    fields_digest_buffer: BTreeMap<String, DigestBufferType<D>>,
}

impl<D: Digest> ArrowDigesterCore<D> {
    /// Create a new instance of `ArrowDigesterCore` with the schema, which will be enforced through each update.
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

    /// Hash a record batch and update the internal digests.
    pub fn update(&mut self, record_batch: &RecordBatch) {
        assert!(
            Self::serialized_schema(record_batch.schema().as_ref()) == self.serialized_schema,
            "Record batch schema does not match ArrowDigester schema"
        );

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
    /// # Panics
    ///
    /// This function will panic if JSON serialization of the data type fails.
    ///
    pub fn hash_array(array: &dyn Array) -> Vec<u8> {
        // Resolve dictionary arrays to their plain value type
        let (effective_type, resolved_array);
        let effective_array: &dyn Array =
            if let DataType::Dictionary(_, value_type) = array.data_type() {
                resolved_array = cast(array, value_type.as_ref())
                    .expect("Failed to cast dictionary to plain array");
                effective_type = value_type.as_ref().clone();
                resolved_array.as_ref()
            } else {
                effective_type = array.data_type().clone();
                array
            };

        // Normalize to canonical large types
        let normalized_type = normalize_data_type(&effective_type);

        let mut final_digest = D::new();

        // Use canonical type serialization for metadata (data_type_to_value also normalizes,
        // but we pass the already-normalized type for consistency)
        let canonical_type = Self::data_type_to_value(&normalized_type);
        let data_type_serialized = serde_json::to_string(&canonical_type)
            .expect("Failed to serialize data type to string");

        // Update the digest buffer with the array metadata and field data
        final_digest.update(data_type_serialized);

        // Now we update it with the actual array data
        // Note: array_digest_update will cast the array to match the normalized type
        let mut digest_buffer = DigestBufferType::new(
            effective_array.is_nullable(),
            is_list_type(&normalized_type),
        );
        Self::array_digest_update(&effective_type, effective_array, &mut digest_buffer);
        Self::finalize_digest(&mut final_digest, digest_buffer);

        // Finalize and return the digest
        final_digest.finalize().to_vec()
    }

    /// Hash a record batch directly without needing to create an `ArrowDigester` instance on the user side.
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        let mut digester = Self::new(record_batch.schema().as_ref());
        digester.update(record_batch);
        digester.finalize()
    }

    /// This will consume the `ArrowDigester` and produce the final combined digest where the schema
    /// digest is fed in first, followed by each field digest in alphabetical order of field names.
    pub fn finalize(self) -> Vec<u8> {
        // Finalize all the sub digest and combine them into a single digest
        let mut final_digest = D::new();

        // digest the schema first
        final_digest.update(&self.schema_digest);

        // Then digest each field digest in order
        self.fields_digest_buffer
            .into_iter()
            .for_each(|(_, digest)| Self::finalize_digest(&mut final_digest, digest));

        final_digest.finalize().to_vec()
    }

    #[expect(
        clippy::big_endian_bytes,
        reason = "Use for bit packing the null_bit_values"
    )]
    /// Finalize a single field digest into the final digest.
    /// Helper to reduce code duplication.
    fn finalize_digest(final_digest: &mut D, digest: DigestBufferType<D>) {
        // Null bits first (if nullable)
        if let Some(null_bit_vec) = &digest.null_bits {
            final_digest.update((null_bit_vec.len() as u64).to_le_bytes());
            for &word in null_bit_vec.as_raw_slice() {
                final_digest.update(word.to_be_bytes());
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
                    "data_type": Self::data_type_to_value(field.data_type()),
                    "nullable": field.is_nullable(),
                });
                (field.name().clone(), Self::sort_json_value(value))
            })
            .collect();

        serde_json::to_string(&fields_digest).expect("Failed to serialize field_digest to bytes")
    }

    /// Convert a `DataType` to a JSON value, recursively converting any inner `Field`
    /// references to only include `name`, `data_type`, and `nullable`.
    ///
    /// Types are first normalized via `normalize_data_type` (Utf8→LargeUtf8, Binary→LargeBinary,
    /// List→LargeList, Dictionary→value type) so the JSON always reflects canonical forms.
    fn data_type_to_value(data_type: &DataType) -> serde_json::Value {
        // Normalize first so all downstream serialization uses canonical types
        let canonical = normalize_data_type(data_type);
        let value = match &canonical {
            DataType::Struct(fields) => {
                let mut sorted_fields: Vec<_> = fields.iter().collect();
                sorted_fields.sort_by_key(|f| f.name().clone());
                let fields_json: Vec<serde_json::Value> = sorted_fields
                    .iter()
                    .map(|f| Self::inner_field_to_value(f))
                    .collect();
                serde_json::json!({ "Struct": fields_json })
            }
            // After normalization, all list types are LargeList
            DataType::LargeList(field) => {
                serde_json::json!({ "LargeList": Self::element_type_to_value(field) })
            }
            DataType::FixedSizeList(field, size) => {
                serde_json::json!({ "FixedSizeList": [Self::element_type_to_value(field), size] })
            }
            DataType::Map(field, sorted) => {
                serde_json::json!({ "Map": [Self::inner_field_to_value(field), sorted] })
            }
            // For all non-nested types (including LargeUtf8, LargeBinary after normalization),
            // Arrow's default serde is sufficient
            other => serde_json::to_value(other).expect("Failed to serialize data type"),
        };
        Self::sort_json_value(value)
    }

    /// Convert an inner field (e.g., struct child) to a JSON value
    /// with `name`, `data_type`, and `nullable`.
    fn inner_field_to_value(field: &Field) -> serde_json::Value {
        serde_json::json!({
            "name": field.name(),
            "data_type": Self::data_type_to_value(field.data_type()),
            "nullable": field.is_nullable(),
        })
    }

    /// Convert a container element field (e.g., list item) to a JSON value
    /// with only `data_type` and `nullable`, omitting the Arrow-internal field name.
    fn element_type_to_value(field: &Field) -> serde_json::Value {
        serde_json::json!({
            "data_type": Self::data_type_to_value(field.data_type()),
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

    /// Hash the schema by serializing it to a canonical JSON string and computing its digest.
    pub fn hash_schema(schema: &Schema) -> Vec<u8> {
        // Hash the entire thing to the digest
        D::digest(Self::serialized_schema(schema)).to_vec()
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
            DataType::List(field) => {
                normalized_type = DataType::LargeList(Arc::clone(field));
                cast_array = cast(array, &normalized_type).expect("cast List");
                (&normalized_type, cast_array.as_ref())
            }
            DataType::Dictionary(_, value_type) => {
                cast_array = cast(array, value_type.as_ref()).expect("cast Dict");
                (value_type.as_ref(), cast_array.as_ref())
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
            DataType::Struct(struct_fields) => {
                Self::traverse_struct(
                    effective_array,
                    struct_fields,
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
                    path,
                    ancestor_struct_nulls,
                    fields,
                );
            }
        }
    }

    fn traverse_list(
        array: &dyn Array,
        value_field: &Field,
        nullable: bool,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        let list_array = array
            .as_any()
            .downcast_ref::<LargeListArray>()
            .expect("downcast to LargeListArray");

        // If the field is nullable, record column/field-level validity at `path`
        if nullable {
            if let Some(entry) = fields.get_mut(path) {
                if let Some(ref mut null_bits) = entry.null_bits {
                    let effective_nulls =
                        Self::combine_nulls(list_array.nulls(), ancestor_struct_nulls);
                    match &effective_nulls {
                        Some(nb) => {
                            for i in 0..list_array.len() {
                                null_bits.push(nb.is_valid(i));
                            }
                        }
                        None => null_bits.extend(repeat_n(true, list_array.len())),
                    }
                }
            }
        }

        let list_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");

        // Determine effective null buffer (field null AND ancestor struct null)
        let effective_nulls = Self::combine_nulls(list_array.nulls(), ancestor_struct_nulls);

        // For each row, write structural info and recurse into non-null elements
        for i in 0..list_array.len() {
            let is_valid = effective_nulls.as_ref().is_none_or(|nb| nb.is_valid(i));
            if is_valid {
                let sub_array = list_array.value(i);
                let sub_len = sub_array.len() as u64;

                // Write list length to structural digest at list_path
                if let Some(entry) = fields.get_mut(&list_path) {
                    if let Some(ref mut structural) = entry.structural {
                        structural.update(sub_len.to_le_bytes());
                    }
                }

                // Recurse into the sub-array using the ORIGINAL value type
                // (not canonical) so traverse_and_update can normalize internally.
                let original_value_type = sub_array.data_type();
                Self::traverse_and_update(
                    original_value_type,
                    value_field.is_nullable(),
                    sub_array.as_ref(),
                    &list_path,
                    None, // list elements don't have ancestor struct nulls
                    fields,
                );
            }
        }
    }

    fn traverse_struct(
        array: &dyn Array,
        _struct_fields: &arrow_schema::Fields,
        nullable: bool,
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("downcast to StructArray");

        // Combine struct's own nulls with ancestor nulls (AND propagation)
        let combined_nulls = if nullable {
            Self::combine_nulls(struct_array.nulls(), ancestor_struct_nulls)
        } else {
            ancestor_struct_nulls.cloned()
        };

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
        path: &str,
        ancestor_struct_nulls: Option<&NullBuffer>,
        fields: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        let entry = fields
            .get_mut(path)
            .expect("entry must exist for leaf path");

        // Compute effective validity (own nulls AND ancestor struct nulls)
        let effective_nulls = Self::combine_nulls(array.nulls(), ancestor_struct_nulls);

        // Handle null_bits
        if let Some(ref mut null_bits) = entry.null_bits {
            match &effective_nulls {
                Some(nb) => {
                    for i in 0..array.len() {
                        null_bits.push(nb.is_valid(i));
                    }
                }
                None => null_bits.extend(repeat_n(true, array.len())),
            }
        }

        // Hash leaf data with combined null buffer
        if let Some(effective) = &effective_nulls {
            let child_data = array.to_data();
            let null_count = effective.null_count();
            let new_data = child_data
                .into_builder()
                .null_count(null_count)
                .null_bit_buffer(Some(effective.clone().into_inner().into_inner()))
                .build()
                .expect("rebuild array with combined null buffer");
            let combined_array = make_array(new_data);
            Self::hash_leaf_data(data_type, combined_array.as_ref(), entry);
        } else {
            Self::hash_leaf_data(data_type, array, entry);
        }
    }

    /// Hash leaf data into the entry's data digest, without modifying `null_bits`
    /// (which are already handled by `traverse_leaf`).
    fn hash_leaf_data(data_type: &DataType, array: &dyn Array, entry: &mut DigestBufferType<D>) {
        // Save and restore null_bits so array_digest_update's handle_null_bits
        // pushes don't pollute the real null_bits (which traverse_leaf manages).
        // We keep null_bits in place during the call so hash functions use
        // the null-aware code path (checking array.nulls() to skip null values).
        let saved = entry.null_bits.take();
        // Put a temporary empty bitvec so hash functions use the null-aware path
        // when the array actually has nulls
        if array.nulls().is_some() {
            entry.null_bits = Some(BitVec::<u8, Lsb0>::new());
        }
        Self::array_digest_update(data_type, array, entry);
        // Restore the real null_bits
        entry.null_bits = saved;
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

    #[expect(
        clippy::too_many_lines,
        reason = "Comprehensive match on all data types"
    )]
    #[expect(
        clippy::unreachable,
        reason = "Small type variants are normalized to large equivalents at the top of this function"
    )]
    fn array_digest_update(
        data_type: &DataType,
        array: &dyn Array,
        digest: &mut DigestBufferType<D>,
    ) {
        // Normalize small variants to their large equivalents so every code path
        // goes through a single canonical representation.  The cast only widens
        // offsets (i32 → i64); inner element types are normalised recursively
        // when hash_list_array re-enters array_digest_update for each sub-array.
        // These variables extend the lifetime of cast results. They are only
        // initialized (and read) in branches that perform a cast; the default
        // branch never touches them, which Rust's initialization analysis accepts.
        let (normalized_type, cast_array);
        let (effective_type, effective_array): (&DataType, &dyn Array) = match data_type {
            DataType::Utf8 => {
                normalized_type = DataType::LargeUtf8;
                cast_array =
                    cast(array, &normalized_type).expect("Failed to cast Utf8 to LargeUtf8");
                (&normalized_type, cast_array.as_ref())
            }
            DataType::Binary => {
                normalized_type = DataType::LargeBinary;
                cast_array =
                    cast(array, &normalized_type).expect("Failed to cast Binary to LargeBinary");
                (&normalized_type, cast_array.as_ref())
            }
            DataType::List(field) => {
                normalized_type = DataType::LargeList(Arc::clone(field));
                cast_array =
                    cast(array, &normalized_type).expect("Failed to cast List to LargeList");
                (&normalized_type, cast_array.as_ref())
            }
            _ => (data_type, array),
        };

        match effective_type {
            DataType::Null => todo!(),
            DataType::Boolean => {
                // Bool Array is stored a bit differently, so we can't use the standard fixed buffer approach
                let bool_array = effective_array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("Failed to downcast to BooleanArray");

                if let Some(ref mut null_bits) = digest.null_bits {
                    // Handle null bits first
                    Self::handle_null_bits(bool_array, null_bits);

                    // Handle the data — only valid bits
                    let mut bit_vec = BitVec::<u8, Lsb0>::with_capacity(bool_array.len());
                    for i in 0..bool_array.len() {
                        if bool_array.is_valid(i) {
                            bit_vec.push(bool_array.value(i));
                        }
                    }
                    digest.data_mut().update(bit_vec.as_raw_slice());
                } else {
                    // Non-nullable: pack all boolean values
                    let mut bit_vec = BitVec::<u8, Lsb0>::with_capacity(bool_array.len());
                    for i in 0..bool_array.len() {
                        bit_vec.push(bool_array.value(i));
                    }
                    digest.data_mut().update(bit_vec.as_raw_slice());
                }
            }
            DataType::Int8 | DataType::UInt8 => {
                Self::hash_fixed_size_array(effective_array, digest, 1);
            }
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => {
                Self::hash_fixed_size_array(effective_array, digest, 2);
            }
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Decimal32(_, _) => {
                Self::hash_fixed_size_array(effective_array, digest, 4);
            }
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Decimal64(_, _) => {
                Self::hash_fixed_size_array(effective_array, digest, 8);
            }
            DataType::Timestamp(_, _) => todo!(),
            DataType::Time32(_) => Self::hash_fixed_size_array(effective_array, digest, 4),
            DataType::Time64(_) => Self::hash_fixed_size_array(effective_array, digest, 8),
            DataType::Duration(_) => todo!(),
            DataType::Interval(_) => todo!(),
            // Small variants are normalized above — these arms are unreachable
            DataType::Binary | DataType::Utf8 | DataType::List(_) => {
                unreachable!("Normalized to Large variant at the top of array_digest_update")
            }
            DataType::FixedSizeBinary(element_size) => {
                Self::hash_fixed_size_array(effective_array, digest, *element_size);
            }
            DataType::LargeBinary => Self::hash_binary_array(
                effective_array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .expect("Failed to downcast to LargeBinaryArray"),
                digest,
            ),
            DataType::BinaryView => todo!(),
            DataType::LargeUtf8 => Self::hash_string_array(
                effective_array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Failed to downcast to LargeStringArray"),
                digest,
            ),
            DataType::Utf8View => todo!(),
            DataType::ListView(_) => todo!(),
            DataType::FixedSizeList(_, _) => todo!(),
            DataType::LargeList(field) => {
                Self::hash_list_array(
                    effective_array
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .expect("Failed to downcast to LargeListArray"),
                    field.data_type(),
                    digest,
                );
            }
            DataType::LargeListView(_) => todo!(),
            DataType::Struct(fields) => {
                let struct_array = effective_array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .expect("Failed to downcast to StructArray");

                // Push struct-level nulls to parent's BitVec (same pattern as other types)
                if let Some(ref mut null_bits) = digest.null_bits {
                    Self::handle_null_bits(struct_array, null_bits);
                }

                // Sort children alphabetically by field name
                let mut sorted_fields: Vec<_> = fields.iter().enumerate().collect();
                sorted_fields.sort_by_key(|(_, f)| f.name().clone());

                for (idx, child_field) in &sorted_fields {
                    let child_array = struct_array.column(*idx);

                    // Child is effectively nullable if the child field is nullable
                    // OR the struct itself has nulls (struct-level nulls propagate down)
                    let effectively_nullable =
                        child_field.is_nullable() || struct_array.nulls().is_some();

                    let mut child_digest = DigestBufferType::new(
                        effectively_nullable,
                        is_list_type(child_field.data_type()),
                    );

                    if let Some(struct_nulls) = struct_array.nulls() {
                        // Propagate struct-level nulls into the child array by combining
                        // struct validity with child validity: combined = struct AND child
                        let combined_nulls = child_array.nulls().map_or_else(
                            || struct_nulls.clone(),
                            |child_nulls| {
                                NullBuffer::new(struct_nulls.inner() & child_nulls.inner())
                            },
                        );
                        let child_data = child_array.to_data();
                        let null_count = combined_nulls.null_count();
                        let new_data = child_data
                            .into_builder()
                            .null_count(null_count)
                            .null_bit_buffer(Some(combined_nulls.into_inner().into_inner()))
                            .build()
                            .expect("Failed to rebuild child array with combined null buffer");
                        let combined_child = make_array(new_data);
                        Self::array_digest_update(
                            child_field.data_type(),
                            combined_child.as_ref(),
                            &mut child_digest,
                        );
                    } else {
                        Self::array_digest_update(
                            child_field.data_type(),
                            child_array.as_ref(),
                            &mut child_digest,
                        );
                    }

                    // Finalize child digest into parent's data stream
                    Self::finalize_child_into_data(digest, child_digest);
                }
            }
            DataType::Union(_, _) => todo!(),
            DataType::Dictionary(_, value_type) => {
                let resolved = cast(effective_array, value_type.as_ref())
                    .expect("Failed to cast dictionary to plain array");
                Self::array_digest_update(value_type.as_ref(), resolved.as_ref(), digest);
            }
            DataType::Decimal128(_, _) => {
                Self::hash_fixed_size_array(effective_array, digest, 16);
            }
            DataType::Decimal256(_, _) => {
                Self::hash_fixed_size_array(effective_array, digest, 32);
            }
            DataType::Map(_, _) => todo!(),
            DataType::RunEndEncoded(_, _) => todo!(),
        }
    }

    #[expect(clippy::cast_sign_loss, reason = "element_size is always positive")]
    fn hash_fixed_size_array(
        array: &dyn Array,
        digest_buffer: &mut DigestBufferType<D>,
        element_size: i32,
    ) {
        let array_data = array.to_data();
        let element_size_usize = element_size as usize;

        // Get the slice with offset and length accounted for
        let start = array_data
            .offset()
            .checked_mul(element_size_usize)
            .expect("Offset multiplication overflow");
        let end = start
            .checked_add(
                array_data
                    .len()
                    .checked_mul(element_size_usize)
                    .expect("Length multiplication overflow"),
            )
            .expect("End position overflow");
        let slice = array_data
            .buffers()
            .first()
            .expect("Unable to get first buffer to determine offset")
            .as_slice()
            .get(start..end)
            .expect("Failed to get buffer slice for FixedSizeBinaryArray");

        if let Some(ref mut null_bits) = digest_buffer.null_bits {
            // Handle null bits first
            Self::handle_null_bits(array, null_bits);

            match array_data.nulls() {
                Some(null_buffer) => {
                    // There are nulls, so we need to incrementally hash each value
                    for i in 0..array_data.len() {
                        if null_buffer.is_valid(i) {
                            let data_pos = i
                                .checked_mul(element_size_usize)
                                .expect("Data position multiplication overflow");
                            let end_pos = data_pos
                                .checked_add(element_size_usize)
                                .expect("End position addition overflow");

                            digest_buffer.data_mut().update(
                                slice
                                    .get(data_pos..end_pos)
                                    .expect("Failed to get data_slice"),
                            );
                        }
                    }
                }
                None => {
                    // No nulls, we can hash the entire buffer directly
                    digest_buffer.data_mut().update(slice);
                }
            }
        } else {
            // No nulls, we can hash the entire buffer directly
            digest_buffer.data_mut().update(slice);
        }
    }

    fn hash_binary_array(
        array: &GenericBinaryArray<impl OffsetSizeTrait>,
        digest: &mut DigestBufferType<D>,
    ) {
        if let Some(ref mut null_bits) = digest.null_bits {
            Self::handle_null_bits(array, null_bits);
        }

        let null_buf = array.nulls();
        for i in 0..array.len() {
            if null_buf.is_none_or(|nb| nb.is_valid(i)) {
                let value = array.value(i);
                digest.data_mut().update((value.len() as u64).to_le_bytes());
                digest.data_mut().update(value);
            }
        }
    }

    fn hash_string_array(
        array: &GenericStringArray<impl OffsetSizeTrait>,
        digest: &mut DigestBufferType<D>,
    ) {
        if let Some(ref mut null_bits) = digest.null_bits {
            Self::handle_null_bits(array, null_bits);
        }

        let null_buf = array.nulls();
        for i in 0..array.len() {
            if null_buf.is_none_or(|nb| nb.is_valid(i)) {
                let value = array.value(i);
                digest.data_mut().update((value.len() as u64).to_le_bytes());
                digest.data_mut().update(value.as_bytes());
            }
        }
    }

    fn hash_list_array(
        array: &GenericListArray<impl OffsetSizeTrait>,
        field_data_type: &DataType,
        digest: &mut DigestBufferType<D>,
    ) {
        // Handle null bits first (if nullable)
        if let Some(ref mut null_bits) = digest.null_bits {
            Self::handle_null_bits(array, null_bits);
        }

        let null_buf = array.nulls();
        for i in 0..array.len() {
            if null_buf.is_none_or(|nb| nb.is_valid(i)) {
                let sub = array.value(i);
                let size_bytes = (sub.len() as u64).to_le_bytes();

                // Write element count to structural digest (separating structure from leaf data).
                // If no structural digest exists, fall back to data digest for backward compat.
                if let Some(ref mut structural) = digest.structural {
                    structural.update(size_bytes);
                } else {
                    digest.data_mut().update(size_bytes);
                }

                // Recurse into sub-array — leaf data goes to data digest
                Self::array_digest_update(field_data_type, sub.as_ref(), digest);
            }
        }
    }

    /// Recursively extract field entries from the type tree.
    ///
    /// - **List**: creates a structural-only entry at `path/`, then recurses into
    ///   the value type. If the column field is nullable, also creates a
    ///   validity-only entry at the field path (before the `/`).
    /// - **Struct**: transparent — recurses into each child field with `path/childname`.
    ///   No entry for the struct itself. Struct null propagation is handled at
    ///   traversal time.
    /// - **Leaf (non-list, non-struct)**: creates a data entry at the current path.
    fn extract_fields_name(
        field: &Field,
        parent_field_name: &str,
        fields_digest_buffer: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        let path = Self::construct_field_name_hierarchy(parent_field_name, field.name());
        Self::extract_type_entries(
            field.data_type(),
            field.is_nullable(),
            &path,
            fields_digest_buffer,
        );
    }

    /// Core recursive type walker — creates `BTreeMap` entries based on the type tree.
    ///
    /// `nullable` reflects whether the current position is nullable (from the `Field`).
    fn extract_type_entries(
        data_type: &DataType,
        nullable: bool,
        path: &str,
        fields_digest_buffer: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        let canonical = normalize_data_type(data_type);

        match &canonical {
            DataType::Struct(fields) => {
                // Struct is transparent — no entry, just recurse into children.
                for child_field in fields {
                    let child_path = Self::construct_field_name_hierarchy(path, child_field.name());
                    Self::extract_type_entries(
                        child_field.data_type(),
                        child_field.is_nullable(),
                        &child_path,
                        fields_digest_buffer,
                    );
                }
            }
            DataType::LargeList(value_field) | DataType::List(value_field) => {
                // For a nullable field that is a list, create a validity-only entry
                // at the field path (column-level or field-level null tracking).
                if nullable {
                    fields_digest_buffer
                        .insert(path.to_owned(), DigestBufferType::new_validity_only());
                }

                // List level: create entry at path + "/"
                let list_path = format!("{path}{DELIMITER_FOR_NESTED_FIELD}");
                let inner_type = value_field.data_type();
                let inner_canonical = normalize_data_type(inner_type);

                match &inner_canonical {
                    DataType::Struct(_) => {
                        // List<Struct<...>>: list entry is structural-only,
                        // struct children become separate entries
                        fields_digest_buffer.insert(
                            list_path.clone(),
                            DigestBufferType::new_structural_only(value_field.is_nullable()),
                        );
                        // Recurse into the struct's children
                        Self::extract_type_entries(
                            inner_type,
                            value_field.is_nullable(),
                            &list_path,
                            fields_digest_buffer,
                        );
                    }
                    DataType::LargeList(_) | DataType::List(_) => {
                        // List<List<...>>: list entry is structural-only,
                        // recurse into the inner list
                        fields_digest_buffer.insert(
                            list_path.clone(),
                            DigestBufferType::new_structural_only(value_field.is_nullable()),
                        );
                        Self::extract_type_entries(
                            inner_type,
                            value_field.is_nullable(),
                            &list_path,
                            fields_digest_buffer,
                        );
                    }
                    _ => {
                        // List<Primitive>: list entry is both structural + data (leaf)
                        fields_digest_buffer.insert(
                            list_path,
                            DigestBufferType::new_list_leaf(value_field.is_nullable()),
                        );
                    }
                }
            }
            _ => {
                // Leaf type (non-struct, non-list): create data entry
                fields_digest_buffer
                    .insert(path.to_owned(), DigestBufferType::new_data_only(nullable));
            }
        }
    }

    fn construct_field_name_hierarchy(parent_field_name: &str, field_name: &str) -> String {
        if parent_field_name.is_empty() {
            field_name.to_owned()
        } else {
            format!("{parent_field_name}{DELIMITER_FOR_NESTED_FIELD}{field_name}")
        }
    }

    /// Write bytes directly into the data/leaf digest portion of the buffer, bypassing null-bit tracking.
    /// Used to write length prefixes that sit in the data stream but are not nullable values.
    fn update_data_digest(digest: &mut DigestBufferType<D>, data: impl AsRef<[u8]>) {
        digest.data_mut().update(data);
    }

    /// Finalize a child's digest and write the resulting bytes into the parent's data stream.
    /// Used for composite types (structs) where each child is independently hashed and then
    /// its finalized representation is fed into the parent digest.
    #[expect(
        clippy::big_endian_bytes,
        reason = "Use for bit packing the null_bit_values"
    )]
    fn finalize_child_into_data(parent: &mut DigestBufferType<D>, child: DigestBufferType<D>) {
        // Null bits first (if nullable child)
        if let Some(null_bit_vec) = &child.null_bits {
            Self::update_data_digest(parent, (null_bit_vec.len() as u64).to_le_bytes());
            for &word in null_bit_vec.as_raw_slice() {
                Self::update_data_digest(parent, word.to_be_bytes());
            }
        }
        // Structural digest (if list child)
        if let Some(structural) = child.structural {
            Self::update_data_digest(parent, structural.finalize());
        }
        // Data/leaf digest (if present)
        if let Some(data) = child.data {
            Self::update_data_digest(parent, data.finalize());
        }
    }

    fn handle_null_bits(array: &dyn Array, null_bit_vec: &mut BitVec<u8, Lsb0>) {
        match array.nulls() {
            Some(null_buf) => {
                // We would need to iterate through the null buffer and push it into the null_bit_vec
                for i in 0..array.len() {
                    null_bit_vec.push(null_buf.is_valid(i));
                }
            }
            None => {
                // All valid, therefore we can extend the bit vector with all true values
                null_bit_vec.extend(repeat_n(true, array.len()));
            }
        }
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
            Decimal32Array, FixedSizeBinaryBuilder, Float16Array, Float32Array, Float64Array,
            Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeListArray,
            LargeListBuilder, LargeStringArray, ListBuilder, PrimitiveBuilder, RecordBatch,
            StringArray, StructArray, Time32SecondArray, Time64MicrosecondArray, UInt16Array,
            UInt32Array, UInt64Array, UInt8Array,
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

        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
            "9b52ad7430dea81b35f14a04d828b2424080fbc210570081c6e6cb62b6566c42"
        );
    }

    // ── Boolean ───────────────────────────────────────────────────────────

    #[test]
    fn digest_bool_nullable_bytes() {
        // [true, None, false, true] — valid values bit-packed Lsb0, null skipped
        let array = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Boolean, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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

        // Valid values [true, false, true] packed Lsb0 into one byte:
        // bit0=1, bit1=0, bit2=1 → 0000_0101 = 0x05
        let mut manual = Sha256::new();
        manual.update([0x05_u8]);
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    #[test]
    fn digest_bool_non_nullable_bytes() {
        // [false, true, false] — all values bit-packed, no nulls
        let array = BooleanArray::from(vec![false, true, false]);
        let schema = Schema::new(vec![Field::new("col", DataType::Boolean, false)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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

        // [false, true, false] packed Lsb0: bit0=0, bit1=1, bit2=0 → 0000_0010 = 0x02
        let mut manual = Sha256::new();
        manual.update([0x02_u8]);
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── Int8 / UInt8 (1-byte fixed) ───────────────────────────────────────

    #[test]
    fn digest_int8_nullable_bytes() {
        // [10, None, -3] — valid bytes: 0x0A, 0xFD
        let array = Int8Array::from(vec![Some(10_i8), None, Some(-3_i8)]);
        let schema = Schema::new(vec![Field::new("col", DataType::Int8, true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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

        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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

    // ── Decimal128 (16-byte decimal) ──────────────────────────────────────

    #[test]
    fn digest_decimal128_nullable_bytes() {
        // Values are stored as i128 in 16 little-endian bytes.
        // [123456, None, -1]
        let array = Decimal128Array::from(vec![Some(123_456_i128), None, Some(-1_i128)])
            .with_precision_and_scale(38, 5)
            .unwrap();
        let schema = Schema::new(vec![Field::new("col", DataType::Decimal128(38, 5), true)]);
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
    // With recursive decomposition, a non-nullable List<Int32 nullable> column
    // creates a single entry at "col/" (list_leaf) with structural (element counts),
    // data (leaf values), and null_bits (item nullability).

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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

        // Non-nullable column → no "col" entry; list_leaf entry at "col/"
        let buf = &digester.fields_digest_buffer["col/"];
        // Items are nullable → null_bits present (all valid in this case)
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable items");
        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec.iter().all(|b| *b), "All items should be valid");

        let structural_digest = buf
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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

        // Non-nullable column → no "col" entry; list_leaf entry at "col/"
        let buf = &digester.fields_digest_buffer["col/"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable items");
        assert_eq!(null_bit_vec.len(), 3);
        assert!(null_bit_vec.iter().all(|b| *b), "All items should be valid");

        let structural_digest = buf
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
    fn digest_buffer_type_structural_only() {
        let buf = super::DigestBufferType::<Sha256>::new_structural_only(true);
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
    fn digest_buffer_type_list_leaf() {
        let buf = super::DigestBufferType::<Sha256>::new_list_leaf(true);
        assert!(buf.null_bits.is_some());
        assert!(buf.structural.is_some());
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

        let digester = ArrowDigesterCore::<Sha256>::new(&schema);
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        // Should have: "x" (validity-only), "x/" (structural), "x//a" (data), "x//b" (data)
        assert_eq!(
            field_names.len(),
            4,
            "Expected 4 entries, got: {field_names:?}"
        );
        assert!(field_names.contains(&&"x".to_owned()));
        assert!(field_names.contains(&&"x/".to_owned()));
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

        let digester = ArrowDigesterCore::<Sha256>::new(&schema);
        let field_names: Vec<&String> = digester.fields_digest_buffer.keys().collect();

        // Expected entries: "x", "x/", "x//a", "x//b/g", "x//b/g/", "x//b/h"
        assert_eq!(
            field_names.len(),
            6,
            "Expected 6 entries, got: {field_names:?}"
        );
        assert!(field_names.contains(&&"x".to_owned()));
        assert!(field_names.contains(&&"x/".to_owned()));
        assert!(field_names.contains(&&"x//a".to_owned()));
        assert!(field_names.contains(&&"x//b/g".to_owned()));
        assert!(field_names.contains(&&"x//b/g/".to_owned()));
        assert!(field_names.contains(&&"x//b/h".to_owned()));
    }

    #[test]
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
        // BTreeMap entries (in sorted order):
        // "x"       → null_bits: V,I,V (3 bits)
        // "x/"      → structural: [2, 3]
        // "x//a"    → null_bits: V,I,V,V,V (5 bits), data: [1, 3, 4, 5] as i32 LE
        // "x//b/g"  → null_bits: V,V,I,V,V (5 bits)
        // "x//b/g/" → structural: [2, 1, 0, 1], data: [10, 20, 30, 50] as i32 LE
        // "x//b/h"  → data: [100, 200, 300, 400, 500] as i32 LE

        let schema_digest = Sha256::digest(
            ArrowDigesterCore::<Sha256>::serialized_schema(&normalize_schema(&schema)).as_bytes(),
        );

        let mut final_digest = Sha256::new();
        final_digest.update(schema_digest);

        // Entry "x": null_bits V,I,V → bit_count=3, validity=0b101=5
        final_digest.update(3_u64.to_le_bytes());
        final_digest.update(5_u8.to_be_bytes());

        // Entry "x/": structural only [2, 3]
        let mut x_structural = Sha256::new();
        x_structural.update(2_u64.to_le_bytes());
        x_structural.update(3_u64.to_le_bytes());
        final_digest.update(x_structural.finalize());

        // Entry "x//a": null_bits V,I,V,V,V → bit_count=5, validity=0b11101=29
        //   data: [1, 3, 4, 5] as i32 LE
        final_digest.update(5_u64.to_le_bytes());
        final_digest.update(29_u8.to_be_bytes());
        let mut xa_data = Sha256::new();
        xa_data.update(1_i32.to_le_bytes());
        xa_data.update(3_i32.to_le_bytes());
        xa_data.update(4_i32.to_le_bytes());
        xa_data.update(5_i32.to_le_bytes());
        final_digest.update(xa_data.finalize());

        // Entry "x//b/g": null_bits V,V,I,V,V → bit_count=5, validity=0b11011=27
        final_digest.update(5_u64.to_le_bytes());
        final_digest.update(27_u8.to_be_bytes());

        // Entry "x//b/g/": structural [2, 1, 0, 1], data [10, 20, 30, 50] as i32 LE
        let mut xbg_structural = Sha256::new();
        xbg_structural.update(2_u64.to_le_bytes());
        xbg_structural.update(1_u64.to_le_bytes());
        xbg_structural.update(0_u64.to_le_bytes());
        xbg_structural.update(1_u64.to_le_bytes());
        final_digest.update(xbg_structural.finalize());
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

        let mut digester = ArrowDigesterCore::<Sha256>::new(&schema);
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
        let mut single = ArrowDigesterCore::<Sha256>::new(schema.as_ref());
        single.update(&combined_batch);
        let single_hash = single.finalize();

        let mut split = ArrowDigesterCore::<Sha256>::new(schema.as_ref());
        split.update(&batch1);
        split.update(&batch2);
        let split_hash = split.finalize();

        assert_eq!(
            encode(&single_hash),
            encode(&split_hash),
            "Batch split independence failed for recursive list/struct decomposition"
        );
    }
}
