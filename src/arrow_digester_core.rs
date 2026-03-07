#![expect(
    clippy::expect_used,
    clippy::todo,
    reason = "First iteration of code, will add proper error handling later. Allow for unsupported data types for now"
)]
use std::{collections::BTreeMap, iter::repeat_n};

use arrow::{
    array::{
        make_array, Array, BinaryArray, BooleanArray, GenericBinaryArray, GenericListArray,
        GenericStringArray, LargeBinaryArray, LargeListArray, LargeStringArray, ListArray,
        OffsetSizeTrait, RecordBatch, StringArray, StructArray,
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
    data: D,
}

impl<D: Digest> DigestBufferType<D> {
    fn new(nullable: bool, structured: bool) -> Self {
        Self {
            null_bits: nullable.then(BitVec::<u8, Lsb0>::new),
            structural: structured.then(D::new),
            data: D::new(),
        }
    }
}

const fn is_list_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::List(_) | DataType::LargeList(_))
}

#[derive(Clone)]
pub struct ArrowDigesterCore<D: Digest> {
    schema: Schema,
    schema_digest: Vec<u8>,
    fields_digest_buffer: BTreeMap<String, DigestBufferType<D>>,
}

impl<D: Digest> ArrowDigesterCore<D> {
    /// Create a new instance of `ArrowDigesterCore` with the schema which will be enforce through each update.
    pub fn new(schema: Schema) -> Self {
        // Hash the schema first
        let schema_digest = Self::hash_schema(&schema);

        // Flatten all nested fields into a single map, this allows us to hash each field individually and efficiently
        let mut fields_digest_buffer = BTreeMap::new();
        schema.fields.into_iter().for_each(|field| {
            Self::extract_fields_name(field, "", &mut fields_digest_buffer);
        });

        // Store it in the new struct for now
        Self {
            schema,
            schema_digest,
            fields_digest_buffer,
        }
    }

    /// Hash a record batch and update the internal digests.
    pub fn update(&mut self, record_batch: &RecordBatch) {
        // Verify schema matches logically (same fields regardless of order, with type canonicalization)
        assert!(
            Self::serialized_schema(record_batch.schema().as_ref())
                == Self::serialized_schema(&self.schema),
            "Record batch schema does not match ArrowDigester schema"
        );

        // Iterate through each field and update its digest
        self.fields_digest_buffer
            .iter_mut()
            .for_each(|(field_name, digest)| {
                // Determine if field name is nested
                let field_name_hierarchy = field_name
                    .split(DELIMITER_FOR_NESTED_FIELD)
                    .collect::<Vec<_>>();

                if field_name_hierarchy.len() == 1 {
                    Self::array_digest_update(
                        record_batch
                            .schema()
                            .field_with_name(field_name)
                            .expect("Failed to get field with name")
                            .data_type(),
                        record_batch
                            .column_by_name(field_name)
                            .expect("Failed to get column by name"),
                        digest,
                    );
                } else {
                    Self::update_nested_field(
                        &field_name_hierarchy,
                        0,
                        record_batch
                            .column_by_name(
                                field_name_hierarchy
                                    .first()
                                    .expect("Failed to get field name at idx 0, list is empty!"),
                            )
                            .expect("Failed to get column by name")
                            .as_any()
                            .downcast_ref::<StructArray>()
                            .expect("Failed to downcast to StructArray"),
                        digest,
                    );
                }
            });
    }

    /// Hash an array directly without needing to create an `ArrowDigester` instance on the user side
    /// For hash array, we don't have a schema to hash, however we do have field data type.
    /// So similar to schema, we will hash based on datatype to encode the metadata information into the digest.....
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

        let mut final_digest = D::new();

        // Use canonical type serialization for metadata
        let canonical_type = Self::data_type_to_value(&effective_type);
        let data_type_serialized = serde_json::to_string(&canonical_type)
            .expect("Failed to serialize data type to string");

        // Update the digest buffer with the array metadata and field data
        final_digest.update(data_type_serialized);

        // Now we update it with the actual array data
        let mut digest_buffer =
            DigestBufferType::new(effective_array.is_nullable(), is_list_type(&effective_type));
        Self::array_digest_update(&effective_type, effective_array, &mut digest_buffer);
        Self::finalize_digest(&mut final_digest, digest_buffer);

        // Finalize and return the digest
        final_digest.finalize().to_vec()
    }

    /// Hash record batch directly without needing to create an `ArrowDigester` instance on the user side.
    pub fn hash_record_batch(record_batch: &RecordBatch) -> Vec<u8> {
        let mut digester = Self::new(record_batch.schema().as_ref().clone());
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
    /// Helpers to reduce code duplication.
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
        // Data/leaf digest
        final_digest.update(digest.data.finalize());
    }

    /// Serialize the schema into a `BTreeMap` for field name and its digest.
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
    fn data_type_to_value(data_type: &DataType) -> serde_json::Value {
        let value = match data_type {
            DataType::Struct(fields) => {
                let mut sorted_fields: Vec<_> = fields.iter().collect();
                sorted_fields.sort_by_key(|f| f.name().clone());
                let fields_json: Vec<serde_json::Value> = sorted_fields
                    .iter()
                    .map(|f| Self::inner_field_to_value(f))
                    .collect();
                serde_json::json!({ "Struct": fields_json })
            }
            // Canonicalize List → LargeList; drop Arrow-internal field name ("item")
            DataType::List(field) | DataType::LargeList(field) => {
                serde_json::json!({ "LargeList": Self::element_type_to_value(field) })
            }
            DataType::FixedSizeList(field, size) => {
                serde_json::json!({ "FixedSizeList": [Self::element_type_to_value(field), size] })
            }
            DataType::Map(field, sorted) => {
                serde_json::json!({ "Map": [Self::inner_field_to_value(field), sorted] })
            }
            // Canonicalize Binary → LargeBinary
            DataType::Binary => {
                serde_json::to_value(&DataType::LargeBinary).expect("Failed to serialize data type")
            }
            // Canonicalize Utf8 → LargeUtf8
            DataType::Utf8 => {
                serde_json::to_value(&DataType::LargeUtf8).expect("Failed to serialize data type")
            }
            // Canonicalize Dictionary → value type
            DataType::Dictionary(_, value_type) => Self::data_type_to_value(value_type.as_ref()),
            // For all non-nested types, Arrow's default serde is sufficient
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

    /// Serialize the schema into a `BTreeMap` for field name and its digest.
    pub fn hash_schema(schema: &Schema) -> Vec<u8> {
        // Hash the entire thing to the digest
        D::digest(Self::serialized_schema(schema)).to_vec()
    }

    /// Recursive function to update nested field digests (structs within structs).
    fn update_nested_field(
        field_name_hierarchy: &[&str],
        current_level: usize,
        array: &StructArray,
        digest: &mut DigestBufferType<D>,
    ) {
        let current_level_plus_one = current_level
            .checked_add(1)
            .expect("Field nesting level overflow");

        if field_name_hierarchy
            .len()
            .checked_sub(1)
            .expect("field_name_hierarchy underflow")
            == current_level_plus_one
        {
            let array_data = array
                .column_by_name(
                    field_name_hierarchy
                        .last()
                        .expect("Failed to get field name at idx 0, list is empty!"),
                )
                .expect("Failed to get column by name");
            // Base case, it should be a non-struct field
            Self::array_digest_update(array_data.data_type(), array_data.as_ref(), digest);
        } else {
            // Recursive case, it should be a struct field
            let next_array = array
                .column_by_name(
                    field_name_hierarchy
                        .get(current_level_plus_one)
                        .expect("Failed to get field name at current level"),
                )
                .expect("Failed to get column by name")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("Failed to downcast to StructArray");

            Self::update_nested_field(
                field_name_hierarchy,
                current_level_plus_one,
                next_array,
                digest,
            );
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Comprehensive match on all data types"
    )]
    fn array_digest_update(
        data_type: &DataType,
        array: &dyn Array,
        digest: &mut DigestBufferType<D>,
    ) {
        match data_type {
            DataType::Null => todo!(),
            DataType::Boolean => {
                // Bool Array is stored a bit differently, so we can't use the standard fixed buffer approach
                let bool_array = array
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
                    digest.data.update(bit_vec.as_raw_slice());
                } else {
                    // Non-nullable: pack all boolean values
                    let mut bit_vec = BitVec::<u8, Lsb0>::with_capacity(bool_array.len());
                    for i in 0..bool_array.len() {
                        bit_vec.push(bool_array.value(i));
                    }
                    digest.data.update(bit_vec.as_raw_slice());
                }
            }
            DataType::Int8 | DataType::UInt8 => Self::hash_fixed_size_array(array, digest, 1),
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => {
                Self::hash_fixed_size_array(array, digest, 2);
            }
            DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Decimal32(_, _) => {
                Self::hash_fixed_size_array(array, digest, 4);
            }
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Decimal64(_, _) => {
                Self::hash_fixed_size_array(array, digest, 8);
            }
            DataType::Timestamp(_, _) => todo!(),
            DataType::Time32(_) => Self::hash_fixed_size_array(array, digest, 4),
            DataType::Time64(_) => Self::hash_fixed_size_array(array, digest, 8),
            DataType::Duration(_) => todo!(),
            DataType::Interval(_) => todo!(),
            DataType::Binary => Self::hash_binary_array(
                array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .expect("Failed to downcast to BinaryArray"),
                digest,
            ),
            DataType::FixedSizeBinary(element_size) => {
                Self::hash_fixed_size_array(array, digest, *element_size);
            }
            DataType::LargeBinary => Self::hash_binary_array(
                array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .expect("Failed to downcast to LargeBinaryArray"),
                digest,
            ),
            DataType::BinaryView => todo!(),
            DataType::Utf8 => Self::hash_string_array(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Failed to downcast to StringArray"),
                digest,
            ),
            DataType::LargeUtf8 => Self::hash_string_array(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Failed to downcast to LargeStringArray"),
                digest,
            ),
            DataType::Utf8View => todo!(),
            DataType::List(field) => {
                Self::hash_list_array(
                    array
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .expect("Failed to downcast to ListArray"),
                    field.data_type(),
                    digest,
                );
            }
            DataType::ListView(_) => todo!(),
            DataType::FixedSizeList(_, _) => todo!(),
            DataType::LargeList(field) => {
                Self::hash_list_array(
                    array
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .expect("Failed to downcast to LargeListArray"),
                    field.data_type(),
                    digest,
                );
            }
            DataType::LargeListView(_) => todo!(),
            DataType::Struct(fields) => {
                let struct_array = array
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
                let resolved = cast(array, value_type.as_ref())
                    .expect("Failed to cast dictionary to plain array");
                Self::array_digest_update(value_type.as_ref(), resolved.as_ref(), digest);
            }
            DataType::Decimal128(_, _) => {
                Self::hash_fixed_size_array(array, digest, 16);
            }
            DataType::Decimal256(_, _) => {
                Self::hash_fixed_size_array(array, digest, 32);
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

        // Get the slice with offset accounted for if there is any
        let slice = array_data
            .buffers()
            .first()
            .expect("Unable to get first buffer to determine offset")
            .as_slice()
            .get(
                array_data
                    .offset()
                    .checked_mul(element_size_usize)
                    .expect("Offset multiplication overflow")..,
            )
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

                            digest_buffer.data.update(
                                slice
                                    .get(data_pos..end_pos)
                                    .expect("Failed to get data_slice"),
                            );
                        }
                    }
                }
                None => {
                    // No nulls, we can hash the entire buffer directly
                    digest_buffer.data.update(slice);
                }
            }
        } else {
            // No nulls, we can hash the entire buffer directly
            digest_buffer.data.update(slice);
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
                digest.data.update((value.len() as u64).to_le_bytes());
                digest.data.update(value);
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
                digest.data.update((value.len() as u64).to_le_bytes());
                digest.data.update(value.as_bytes());
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
                    digest.data.update(size_bytes);
                }

                // Recurse into sub-array — leaf data goes to data digest
                Self::array_digest_update(field_data_type, sub.as_ref(), digest);
            }
        }
    }

    /// Internal recursive function to extract field names from nested structs effectively flattening the schema.
    /// The format is `parent__child__grandchild__etc`... for nested fields and will be stored in `fields_digest_buffer`.
    fn extract_fields_name(
        field: &Field,
        parent_field_name: &str,
        fields_digest_buffer: &mut BTreeMap<String, DigestBufferType<D>>,
    ) {
        // Check if field is a nested type of struct
        if let DataType::Struct(fields) = field.data_type() {
            // We will add fields in alphabetical order
            fields.into_iter().for_each(|field_inner| {
                Self::extract_fields_name(
                    field_inner,
                    Self::construct_field_name_hierarchy(parent_field_name, field.name()).as_str(),
                    fields_digest_buffer,
                );
            });
        } else {
            // Base case, just add the the combine field name to the map
            fields_digest_buffer.insert(
                Self::construct_field_name_hierarchy(parent_field_name, field.name()),
                DigestBufferType::new(field.is_nullable(), is_list_type(field.data_type())),
            );
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
        digest.data.update(data);
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
        // Data/leaf digest
        Self::update_data_digest(parent, child.data.finalize());
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
            Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeListBuilder,
            LargeStringArray, ListBuilder, PrimitiveBuilder, RecordBatch, StringArray, StructArray,
            Time32SecondArray, Time64MicrosecondArray, UInt16Array, UInt32Array, UInt64Array,
            UInt8Array,
        },
        datatypes::Int32Type,
    };
    use arrow_schema::{DataType, Field, Schema, TimeUnit};

    use hex::encode;
    use pretty_assertions::assert_eq;
    use sha2::{Digest as _, Sha256};

    use crate::arrow_digester_core::ArrowDigesterCore;
    use arrow::array::{Decimal256Array, Decimal64Array};
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

        let mut digester = ArrowDigesterCore::<Sha256>::new(schema.clone());
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int8, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::UInt8, false)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int16, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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

        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);

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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::UInt32, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int64, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::UInt64, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Date32, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Date64, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Binary, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
        digester.update(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("col", DataType::Utf8, true)])),
                vec![Arc::new(array)],
            )
            .unwrap(),
        );

        let buf = &digester.fields_digest_buffer["col"];
        let null_bit_vec = buf.null_bits.as_ref().expect("Expected nullable");
        let data_digest = &buf.data;

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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
        let data_digest = &buf.data;

        let mut manual = Sha256::new();
        manual.update(1_u64.to_le_bytes());
        manual.update(b"x");
        manual.update(2_u64.to_le_bytes());
        manual.update(b"yz");
        assert_eq!(data_digest.clone().finalize(), manual.finalize());
    }

    // ── List<Int32> / LargeList<Int32> ─────────────────────────────────────
    //
    // Each outer element is prefixed by its inner element count (u64 LE), then the
    // raw bytes of the inner array (no length limit — the implementation hashes from
    // the element's offset to the end of the shared child buffer).
    // Using a single outer element avoids buffer-bleed from preceding elements.

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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let structural_digest = buf
            .structural
            .as_ref()
            .expect("Expected structural digest for list");
        let data_digest = &buf.data;

        // Structural digest: element count (sizes separated from leaf data)
        let mut manual_structural = Sha256::new();
        manual_structural.update(3_u64.to_le_bytes()); // element count prefix
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
        let mut digester = ArrowDigesterCore::<Sha256>::new(schema);
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

        let buf = &digester.fields_digest_buffer["col"];
        assert!(buf.null_bits.is_none(), "Expected non-nullable");
        let structural_digest = buf
            .structural
            .as_ref()
            .expect("Expected structural digest for list");
        let data_digest = &buf.data;

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
}
