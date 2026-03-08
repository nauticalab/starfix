"""Pure-Python implementation of the starfix Arrow logical hasher.

Produces identical hashes to the Rust implementation for all supported types.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

VERSION_BYTES = b"\x00\x00\x01"
DELIMITER = "/"
NULL_BYTES = b"NULL"


# ---------------------------------------------------------------------------
# Bit-vector helper (MSB-first packing, matching bitvec<u8, Msb0>)
# ---------------------------------------------------------------------------

class _BitVec:
    """Minimal LSB-first u8 bit vector compatible with Rust bitvec<u8, Lsb0>.

    Matches Arrow's native validity bitmap layout.
    """

    __slots__ = ("_bytes", "_len")

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._len = 0

    def push(self, bit: bool) -> None:
        byte_idx = self._len >> 3
        bit_idx = self._len & 7  # LSB-first: bit 0 is least significant
        if byte_idx >= len(self._bytes):
            self._bytes.append(0)
        if bit:
            self._bytes[byte_idx] |= 1 << bit_idx
        self._len += 1

    def extend_true(self, count: int) -> None:
        for _ in range(count):
            self.push(True)

    def __len__(self) -> int:
        return self._len

    def raw_bytes(self) -> bytes:
        return bytes(self._bytes)


# ---------------------------------------------------------------------------
# Schema / DataType serialization  (matches Rust `serialized_schema`)
# ---------------------------------------------------------------------------

def _data_type_to_value(dt: pa.DataType) -> object:
    """Convert a pyarrow DataType to the JSON-compatible value that matches
    the Rust ``data_type_to_value`` output."""
    import pyarrow as pa

    # Normalize first
    dt = _normalize_data_type(dt)

    if pa.types.is_struct(dt):
        # Sort children alphabetically by field name
        children = [dt.field(i) for i in range(dt.num_fields)]
        children.sort(key=lambda f: f.name)
        fields_json = [_inner_field_to_value(f) for f in children]
        return {"Struct": fields_json}
    if pa.types.is_large_list(dt):
        return {"LargeList": _element_type_to_value(dt.value_field)}
    if pa.types.is_list(dt):
        # After normalization this shouldn't happen, but handle it
        return {"List": _element_type_to_value(dt.value_field)}
    if pa.types.is_fixed_size_list(dt):
        return {"FixedSizeList": [_element_type_to_value(dt.value_field), dt.list_size]}
    if pa.types.is_map(dt):
        return {"Map": [_inner_field_to_value(dt.key_field.with_name("entries")), False]}

    # Primitive / leaf types – must match Arrow-Rust serde
    return _primitive_data_type_string(dt)


def _element_type_to_value(field: pa.Field) -> dict:
    """Convert a container element field to a JSON value with only data_type and nullable."""
    return {
        "data_type": _data_type_to_value(field.type),
        "nullable": field.nullable,
    }


def _normalize_data_type(dt: pa.DataType) -> pa.DataType:
    """Normalize a DataType to its canonical large equivalent."""
    import pyarrow as pa

    if dt == pa.utf8():
        return pa.large_utf8()
    if dt == pa.binary():
        return pa.large_binary()
    if pa.types.is_list(dt) and not pa.types.is_large_list(dt):
        new_field = _normalize_field(dt.value_field)
        return pa.large_list(new_field)
    if pa.types.is_large_list(dt):
        new_field = _normalize_field(dt.value_field)
        return pa.large_list(new_field)
    if pa.types.is_struct(dt):
        new_fields = [_normalize_field(dt.field(i)) for i in range(dt.num_fields)]
        return pa.struct_(new_fields)
    if pa.types.is_fixed_size_list(dt):
        new_field = _normalize_field(dt.value_field)
        return pa.list_(new_field, dt.list_size)
    return dt


def _normalize_field(field: pa.Field) -> pa.Field:
    """Normalize a single field."""
    import pyarrow as pa
    return pa.field(field.name, _normalize_data_type(field.type), nullable=field.nullable)


def _primitive_data_type_string(dt: pa.DataType) -> object:
    """Return the serde_json representation that arrow-rs produces."""
    import pyarrow as pa

    _simple = {
        pa.bool_(): "Boolean",
        pa.int8(): "Int8",
        pa.uint8(): "UInt8",
        pa.int16(): "Int16",
        pa.uint16(): "UInt16",
        pa.int32(): "Int32",
        pa.uint32(): "UInt32",
        pa.int64(): "Int64",
        pa.uint64(): "UInt64",
        pa.float16(): "Float16",
        pa.float32(): "Float32",
        pa.float64(): "Float64",
        pa.date32(): "Date32",
        pa.date64(): "Date64",
        pa.utf8(): "Utf8",
        pa.large_utf8(): "LargeUtf8",
        pa.binary(): "Binary",
        pa.large_binary(): "LargeBinary",
    }
    if dt in _simple:
        return _simple[dt]

    if pa.types.is_decimal(dt):
        if dt.bit_width == 32:
            return {"Decimal32": [dt.precision, dt.scale]}
        if dt.bit_width == 64:
            return {"Decimal64": [dt.precision, dt.scale]}
        if dt.bit_width == 128:
            return {"Decimal128": [dt.precision, dt.scale]}
        if dt.bit_width == 256:
            return {"Decimal256": [dt.precision, dt.scale]}

    if pa.types.is_time32(dt):
        unit = "Second" if dt.unit == "s" else "Millisecond"
        return {"Time32": unit}
    if pa.types.is_time64(dt):
        unit = "Microsecond" if dt.unit == "us" else "Nanosecond"
        return {"Time64": unit}

    if pa.types.is_timestamp(dt):
        unit_map = {"s": "Second", "ms": "Millisecond", "us": "Microsecond", "ns": "Nanosecond"}
        unit = unit_map[dt.unit]
        if dt.tz is None:
            return {"Timestamp": [unit, None]}
        return {"Timestamp": [unit, dt.tz]}

    if pa.types.is_duration(dt):
        unit_map = {"s": "Second", "ms": "Millisecond", "us": "Microsecond", "ns": "Nanosecond"}
        return {"Duration": unit_map[dt.unit]}

    if pa.types.is_fixed_size_binary(dt):
        return {"FixedSizeBinary": dt.byte_width}

    raise NotImplementedError(f"Unsupported data type: {dt}")


def _inner_field_to_value(field: pa.Field) -> dict:
    return {
        "name": field.name,
        "data_type": _data_type_to_value(field.type),
        "nullable": field.nullable,
    }


def _raw_serde_field(field) -> dict:
    """Produce the full arrow-rs serde Field representation (used in hash_array).

    Arrow-rs Field serializes all struct fields in declaration order:
    name, data_type, nullable, dict_id, dict_is_ordered, metadata
    """
    result = OrderedDict()
    result["name"] = field.name
    result["data_type"] = _raw_serde_data_type(field.type)
    result["nullable"] = field.nullable
    result["dict_id"] = 0
    result["dict_is_ordered"] = False
    if field.metadata:
        result["metadata"] = {k.decode() if isinstance(k, bytes) else k:
                              v.decode() if isinstance(v, bytes) else v
                              for k, v in field.metadata.items()}
    else:
        result["metadata"] = {}
    return result


def _raw_serde_data_type(dt) -> object:
    """Produce the arrow-rs serde DataType representation (used in hash_array).

    This matches serde_json::to_string(&data_type) in Rust exactly.
    """
    import pyarrow as pa

    if pa.types.is_struct(dt):
        return {"Struct": [_raw_serde_field(dt.field(i)) for i in range(dt.num_fields)]}
    if pa.types.is_list(dt):
        return {"List": _raw_serde_field(dt.value_field)}
    if pa.types.is_large_list(dt):
        return {"LargeList": _raw_serde_field(dt.value_field)}
    if pa.types.is_fixed_size_list(dt):
        return {"FixedSizeList": [_raw_serde_field(dt.value_field), dt.list_size]}
    if pa.types.is_map(dt):
        return {"Map": [_raw_serde_field(dt.key_field.with_name("entries")), False]}

    return _primitive_data_type_string(dt)


def _sort_json_value(value: object) -> object:
    """Recursively sort JSON object keys (matching Rust ``sort_json_value``)."""
    if isinstance(value, dict):
        return OrderedDict(sorted((k, _sort_json_value(v)) for k, v in value.items()))
    if isinstance(value, list):
        return [_sort_json_value(v) for v in value]
    return value


def _serialized_schema(schema: pa.Schema) -> str:
    # Normalize the schema first
    import pyarrow as pa
    normalized_fields = [_normalize_field(schema.field(i)) for i in range(len(schema))]
    normalized_schema = pa.schema(normalized_fields)

    fields: dict[str, object] = {}
    for i in range(len(normalized_schema)):
        field = normalized_schema.field(i)
        value = {
            "data_type": _data_type_to_value(field.type),
            "nullable": field.nullable,
        }
        fields[field.name] = _sort_json_value(value)
    # Sort by field name (BTreeMap ordering)
    sorted_fields = OrderedDict(sorted(fields.items()))
    return json.dumps(sorted_fields, separators=(",", ":"))


def _hash_schema(schema: pa.Schema) -> bytes:
    return hashlib.sha256(_serialized_schema(schema).encode()).digest()


# ---------------------------------------------------------------------------
# Field extraction  (recursive decomposition into BTreeMap<path, entry>)
# ---------------------------------------------------------------------------

def _is_list_type(dt) -> bool:
    import pyarrow as pa
    return pa.types.is_list(dt) or pa.types.is_large_list(dt)


def _extract_fields(field, parent: str, out: dict):
    """Extract fields for a top-level schema field. Uses _extract_type_entries internally."""
    path = f"{parent}{DELIMITER}{field.name}" if parent else field.name
    _extract_type_entries(field.type, field.nullable, path, out)


def _extract_type_entries(data_type, nullable: bool, path: str, out: dict):
    """Recursively decompose types into BTreeMap entries.

    Entry format: {"null_bits": _BitVec or None, "structural": sha256 or None, "data": sha256 or None}
    """
    import pyarrow as pa

    canonical = _normalize_data_type(data_type)

    if pa.types.is_struct(canonical):
        # Struct is transparent — no entry for struct itself, recurse into children
        children = [canonical.field(i) for i in range(canonical.num_fields)]
        for child in children:
            child_path = f"{path}{DELIMITER}{child.name}"
            _extract_type_entries(child.type, child.nullable, child_path, out)
    elif _is_list_type(canonical):
        # If the field is nullable, create a validity-only entry at path
        if nullable:
            out[path] = {"null_bits": _BitVec(), "structural": None, "data": None}

        # List level entry at path + "/"
        list_path = f"{path}{DELIMITER}"
        value_field = canonical.value_field
        inner_type = value_field.type
        inner_canonical = _normalize_data_type(inner_type)

        if pa.types.is_struct(inner_canonical):
            # List<Struct>: structural-only entry, recurse into struct children
            out[list_path] = {
                "null_bits": _BitVec() if value_field.nullable else None,
                "structural": hashlib.sha256(),
                "data": None,
            }
            _extract_type_entries(inner_type, value_field.nullable, list_path, out)
        elif _is_list_type(inner_canonical):
            # List<List>: structural-only entry, recurse
            out[list_path] = {
                "null_bits": _BitVec() if value_field.nullable else None,
                "structural": hashlib.sha256(),
                "data": None,
            }
            _extract_type_entries(inner_type, value_field.nullable, list_path, out)
        else:
            # List<Primitive>: list-leaf entry (structural + data)
            out[list_path] = {
                "null_bits": _BitVec() if value_field.nullable else None,
                "structural": hashlib.sha256(),
                "data": hashlib.sha256(),
            }
    else:
        # Leaf type: data entry
        out[path] = {
            "null_bits": _BitVec() if nullable else None,
            "structural": None,
            "data": hashlib.sha256(),
        }


# ---------------------------------------------------------------------------
# Array data hashing (used by hash_array path — legacy composite approach)
# ---------------------------------------------------------------------------

def _handle_null_bits(arr, bit_vec: _BitVec) -> None:
    """Push validity bits for *arr* into *bit_vec*."""
    for i in range(len(arr)):
        bit_vec.push(arr[i].is_valid)


def _hash_fixed_size_array(arr, digest_entry, element_size: int) -> None:
    """Hash a fixed-width array by reading raw buffers (matching Rust behaviour)."""
    nullable, bit_vec, data_digest = _unpack_legacy_entry(digest_entry)

    bufs = arr.buffers()
    data_buf = bufs[1]
    offset = arr.offset

    raw = data_buf.to_pybytes()
    start = offset * element_size
    sliced = raw[start:]

    if not nullable:
        end = start + len(arr) * element_size
        data_digest.update(raw[start:end])
    else:
        _handle_null_bits(arr, bit_vec)
        if arr.null_count > 0:
            for i in range(len(arr)):
                if arr[i].is_valid:
                    pos = i * element_size
                    data_digest.update(sliced[pos:pos + element_size])
        else:
            end = len(arr) * element_size
            data_digest.update(sliced[:end])


def _hash_boolean_array(arr, digest_entry) -> None:
    nullable, bit_vec, data_digest = _unpack_legacy_entry(digest_entry)

    if not nullable:
        bv = _BitVec()
        for i in range(len(arr)):
            bv.push(arr[i].as_py())
        data_digest.update(bv.raw_bytes())
    else:
        _handle_null_bits(arr, bit_vec)
        bv = _BitVec()
        for i in range(len(arr)):
            if arr[i].is_valid:
                bv.push(arr[i].as_py())
        data_digest.update(bv.raw_bytes())


def _hash_binary_array(arr, digest_entry) -> None:
    """Hash Binary / LargeBinary arrays."""
    nullable, bit_vec, data_digest = _unpack_legacy_entry(digest_entry)

    if not nullable:
        for i in range(len(arr)):
            val = arr[i].as_py()
            data_digest.update(struct.pack("<Q", len(val)))
            data_digest.update(val)
    else:
        if arr.null_count > 0:
            for i in range(len(arr)):
                bit_vec.push(arr[i].is_valid)
            for i in range(len(arr)):
                if arr[i].is_valid:
                    val = arr[i].as_py()
                    data_digest.update(struct.pack("<Q", len(val)))
                    data_digest.update(val)
                else:
                    data_digest.update(NULL_BYTES)
        else:
            bit_vec.extend_true(len(arr))
            for i in range(len(arr)):
                val = arr[i].as_py()
                data_digest.update(struct.pack("<Q", len(val)))
                data_digest.update(val)


def _hash_string_array(arr, digest_entry) -> None:
    """Hash Utf8 / LargeUtf8 arrays."""
    nullable, bit_vec, data_digest = _unpack_legacy_entry(digest_entry)

    if not nullable:
        for i in range(len(arr)):
            val = arr[i].as_py().encode("utf-8")
            data_digest.update(struct.pack("<Q", len(val)))
            data_digest.update(val)
    else:
        _handle_null_bits(arr, bit_vec)
        if arr.null_count > 0:
            for i in range(len(arr)):
                if arr[i].is_valid:
                    val = arr[i].as_py().encode("utf-8")
                    data_digest.update(struct.pack("<Q", len(val)))
                    data_digest.update(val)
                else:
                    data_digest.update(NULL_BYTES)
        else:
            for i in range(len(arr)):
                val = arr[i].as_py().encode("utf-8")
                data_digest.update(struct.pack("<Q", len(val)))
                data_digest.update(val)


def _update_data_digest(digest_entry, data: bytes) -> None:
    digest_entry[2].update(data)


def _hash_list_array(arr, field_data_type, digest_entry) -> None:
    import pyarrow as pa
    nullable, bit_vec, data_digest = _unpack_legacy_entry(digest_entry)

    if not nullable:
        for i in range(len(arr)):
            sub = arr[i]
            sub_arr = pa.array(sub.values) if hasattr(sub, 'values') else sub
            sub_arr = arr.value(i) if hasattr(arr, 'value') else arr[i].values
            data_digest.update(struct.pack("<Q", len(sub_arr)))
            _array_digest_update(field_data_type, sub_arr, digest_entry)
    else:
        _handle_null_bits(arr, bit_vec)
        if arr.null_count > 0:
            for i in range(len(arr)):
                if arr[i].is_valid:
                    sub_arr = arr.value(i) if hasattr(arr, 'value') else arr[i].values
                    data_digest.update(struct.pack("<Q", len(sub_arr)))
                    _array_digest_update(field_data_type, sub_arr, digest_entry)
        else:
            for i in range(len(arr)):
                sub_arr = arr.value(i) if hasattr(arr, 'value') else arr[i].values
                data_digest.update(struct.pack("<Q", len(sub_arr)))
                _array_digest_update(field_data_type, sub_arr, digest_entry)


def _element_size_for_type(dt: pa.DataType) -> int | None:
    """Return byte width for fixed-size types, or None for variable-length."""
    import pyarrow as pa

    _sizes = {
        pa.int8(): 1, pa.uint8(): 1,
        pa.int16(): 2, pa.uint16(): 2, pa.float16(): 2,
        pa.int32(): 4, pa.uint32(): 4, pa.float32(): 4, pa.date32(): 4,
        pa.int64(): 8, pa.uint64(): 8, pa.float64(): 8, pa.date64(): 8,
    }
    if dt in _sizes:
        return _sizes[dt]
    if pa.types.is_time32(dt):
        return 4
    if pa.types.is_time64(dt):
        return 8
    if pa.types.is_decimal(dt):
        return dt.bit_width // 8
    if pa.types.is_fixed_size_binary(dt):
        return dt.byte_width
    if pa.types.is_decimal32(dt):
        return 4
    if pa.types.is_decimal64(dt):
        return 8
    return None


def _unpack_legacy_entry(entry):
    """Unpack an entry that may be either old-style tuple or new-style dict."""
    if isinstance(entry, dict):
        nullable = entry["null_bits"] is not None
        return nullable, entry["null_bits"], entry["data"]
    # Old tuple format (nullable, bit_vec, data_digest)
    return entry[0], entry[1], entry[2]


def _array_digest_update(data_type, arr, digest_entry) -> None:
    import pyarrow as pa

    if pa.types.is_boolean(data_type):
        _hash_boolean_array(arr, digest_entry)
    elif pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        _hash_binary_array(arr, digest_entry)
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        _hash_string_array(arr, digest_entry)
    elif pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        _hash_list_array(arr, data_type.value_type, digest_entry)
    elif pa.types.is_struct(data_type):
        raise NotImplementedError("Struct arrays in array_digest_update not supported")
    else:
        element_size = _element_size_for_type(data_type)
        if element_size is not None:
            _hash_fixed_size_array(arr, digest_entry, element_size)
        else:
            raise NotImplementedError(f"Unsupported data type: {data_type}")


# ---------------------------------------------------------------------------
# Null combination helper
# ---------------------------------------------------------------------------

def _get_validity_bools(arr, length: int):
    """Get validity as a list of booleans, or None if all valid."""
    if arr.null_count == 0 and (not hasattr(arr, 'buffers') or arr.buffers()[0] is None):
        return None
    if arr.null_count == 0:
        return None
    return [arr[i].is_valid for i in range(length)]


def _combine_nulls(array_validity, ancestor_nulls):
    """Combine array validity (list of bools or None) with ancestor nulls (list of bools or None).

    Returns a list of booleans or None if all valid.
    """
    if array_validity is None and ancestor_nulls is None:
        return None
    if array_validity is None:
        return ancestor_nulls
    if ancestor_nulls is None:
        return array_validity
    # AND combine
    return [a and b for a, b in zip(array_validity, ancestor_nulls)]


def _array_validity_bools(arr):
    """Extract validity as list of bools or None from a pyarrow array."""
    if arr.null_count == 0:
        return None
    return [arr[i].is_valid for i in range(len(arr))]


# ---------------------------------------------------------------------------
# Record-batch traversal (top-down recursive, mirrors Rust)
# ---------------------------------------------------------------------------

def _hash_leaf_data_rb(data_type, arr, effective_nulls, entry):
    """Hash leaf data into the entry's data digest for the record-batch path.

    effective_nulls: list of bools or None.
    This only writes to the data digest, not null_bits.
    """
    import pyarrow as pa

    data_digest = entry["data"]

    # Build an array with the effective null mask if needed
    if effective_nulls is not None:
        # We need to create an array where nulls match effective_nulls
        # Convert to python, apply mask, rebuild
        has_nulls = not all(effective_nulls)
    else:
        has_nulls = arr.null_count > 0

    if pa.types.is_boolean(data_type):
        bv = _BitVec()
        if has_nulls:
            nulls = effective_nulls if effective_nulls is not None else [arr[i].is_valid for i in range(len(arr))]
            for i in range(len(arr)):
                if nulls[i]:
                    bv.push(arr[i].as_py())
        else:
            for i in range(len(arr)):
                bv.push(arr[i].as_py())
        data_digest.update(bv.raw_bytes())
    elif pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        nulls = effective_nulls if effective_nulls is not None else (
            [arr[i].is_valid for i in range(len(arr))] if arr.null_count > 0 else None
        )
        if nulls is not None and not all(nulls):
            for i in range(len(arr)):
                if nulls[i]:
                    val = arr[i].as_py()
                    data_digest.update(struct.pack("<Q", len(val)))
                    data_digest.update(val)
                else:
                    data_digest.update(NULL_BYTES)
        else:
            for i in range(len(arr)):
                val = arr[i].as_py()
                data_digest.update(struct.pack("<Q", len(val)))
                data_digest.update(val)
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        nulls = effective_nulls if effective_nulls is not None else (
            [arr[i].is_valid for i in range(len(arr))] if arr.null_count > 0 else None
        )
        if nulls is not None and not all(nulls):
            for i in range(len(arr)):
                if nulls[i]:
                    val = arr[i].as_py().encode("utf-8")
                    data_digest.update(struct.pack("<Q", len(val)))
                    data_digest.update(val)
                else:
                    data_digest.update(NULL_BYTES)
        else:
            for i in range(len(arr)):
                val = arr[i].as_py().encode("utf-8")
                data_digest.update(struct.pack("<Q", len(val)))
                data_digest.update(val)
    else:
        element_size = _element_size_for_type(data_type)
        if element_size is None:
            raise NotImplementedError(f"Unsupported data type: {data_type}")

        bufs = arr.buffers()
        data_buf = bufs[1]
        offset = arr.offset
        raw = data_buf.to_pybytes()
        start = offset * element_size
        sliced = raw[start:]

        nulls = effective_nulls if effective_nulls is not None else (
            [arr[i].is_valid for i in range(len(arr))] if arr.null_count > 0 else None
        )
        if nulls is not None and not all(nulls):
            for i in range(len(arr)):
                if nulls[i]:
                    pos = i * element_size
                    data_digest.update(sliced[pos:pos + element_size])
        else:
            end = len(arr) * element_size
            data_digest.update(sliced[:end])


def _traverse_and_update(data_type, nullable, array, path, ancestor_struct_nulls, fields):
    """Top-down recursive traversal dispatching to list/struct/leaf."""
    import pyarrow as pa

    # Normalize small variants
    effective_type = data_type
    effective_array = array

    if data_type == pa.utf8():
        effective_type = pa.large_utf8()
        effective_array = array.cast(pa.large_utf8())
    elif data_type == pa.binary():
        effective_type = pa.large_binary()
        effective_array = array.cast(pa.large_binary())
    elif pa.types.is_list(data_type) and not pa.types.is_large_list(data_type):
        value_field = data_type.value_field
        effective_type = pa.large_list(value_field)
        effective_array = array.cast(pa.large_list(value_field))

    canonical = _normalize_data_type(effective_type)

    if pa.types.is_large_list(canonical):
        _traverse_list(effective_array, canonical.value_field, nullable, path, ancestor_struct_nulls, fields)
    elif pa.types.is_struct(canonical):
        _traverse_struct(effective_array, nullable, path, ancestor_struct_nulls, fields)
    else:
        _traverse_leaf(effective_type, effective_array, path, ancestor_struct_nulls, fields)


def _traverse_list(list_array, value_field, nullable, path, ancestor_struct_nulls, fields):
    """Handle list arrays in record-batch traversal."""
    import pyarrow as pa

    arr_len = len(list_array)

    # If field is nullable, record column/field-level validity at path
    if nullable:
        if path in fields:
            entry = fields[path]
            if entry["null_bits"] is not None:
                null_bits = entry["null_bits"]
                own_nulls = _array_validity_bools(list_array)
                effective_nulls = _combine_nulls(own_nulls, ancestor_struct_nulls)
                if effective_nulls is not None:
                    for i in range(arr_len):
                        null_bits.push(effective_nulls[i])
                else:
                    null_bits.extend_true(arr_len)

    list_path = f"{path}{DELIMITER}"

    # Determine effective null buffer
    own_nulls = _array_validity_bools(list_array)
    effective_nulls = _combine_nulls(own_nulls, ancestor_struct_nulls)

    # For each row, write structural info and recurse into non-null elements
    for i in range(arr_len):
        is_valid = effective_nulls is None or effective_nulls[i]
        if is_valid:
            sub_array = list_array.value(i)
            sub_len = len(sub_array)

            # Write list length to structural digest at list_path
            if list_path in fields:
                entry = fields[list_path]
                if entry["structural"] is not None:
                    entry["structural"].update(struct.pack("<Q", sub_len))

            # Recurse into the sub-array using original value type
            original_value_type = sub_array.type
            _traverse_and_update(
                original_value_type,
                value_field.nullable,
                sub_array,
                list_path,
                None,  # list elements don't have ancestor struct nulls
                fields,
            )


def _traverse_struct(struct_array, nullable, path, ancestor_struct_nulls, fields):
    """Handle struct arrays in record-batch traversal."""
    # Combine struct's own nulls with ancestor nulls (AND propagation)
    if nullable:
        own_nulls = _array_validity_bools(struct_array)
        combined_nulls = _combine_nulls(own_nulls, ancestor_struct_nulls)
    else:
        combined_nulls = ancestor_struct_nulls

    # Get original fields from struct array and sort alphabetically
    original_fields = struct_array.type
    children = [(i, original_fields.field(i)) for i in range(original_fields.num_fields)]
    children.sort(key=lambda x: x[1].name)

    for idx, child_field in children:
        child_array = struct_array.field(idx)
        child_path = f"{path}{DELIMITER}{child_field.name}"

        _traverse_and_update(
            child_field.type,
            child_field.nullable,
            child_array,
            child_path,
            combined_nulls,
            fields,
        )


def _traverse_leaf(data_type, array, path, ancestor_struct_nulls, fields):
    """Handle leaf arrays in record-batch traversal."""
    entry = fields[path]

    # Compute effective validity (own nulls AND ancestor struct nulls)
    own_nulls = _array_validity_bools(array)
    effective_nulls = _combine_nulls(own_nulls, ancestor_struct_nulls)

    # Handle null_bits
    if entry["null_bits"] is not None:
        null_bits = entry["null_bits"]
        if effective_nulls is not None:
            for i in range(len(array)):
                null_bits.push(effective_nulls[i])
        else:
            null_bits.extend_true(len(array))

    # Hash leaf data with combined null buffer
    _hash_leaf_data_rb(data_type, array, effective_nulls, entry)


# ---------------------------------------------------------------------------
# Finalization helpers
# ---------------------------------------------------------------------------

def _finalize_digest(final_digest, entry) -> None:
    """Finalize a single field entry into the final digest."""
    if isinstance(entry, dict):
        # New-style entry
        if entry["null_bits"] is not None:
            bv = entry["null_bits"]
            final_digest.update(struct.pack("<Q", len(bv)))
            for b in bv.raw_bytes():
                final_digest.update(bytes([b]))
        if entry["structural"] is not None:
            final_digest.update(entry["structural"].digest())
        if entry["data"] is not None:
            final_digest.update(entry["data"].digest())
    else:
        # Old tuple format for hash_array
        nullable, bit_vec, data_digest = entry
        if not nullable:
            final_digest.update(data_digest.digest())
        else:
            final_digest.update(struct.pack("<Q", len(bit_vec)))
            for b in bit_vec.raw_bytes():
                final_digest.update(bytes([b]))
            final_digest.update(data_digest.digest())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ArrowDigester:
    """Pure-Python equivalent of the Rust ``ArrowDigester``.

    Produces identical SHA-256 hashes with a 3-byte version prefix.
    """

    def __init__(self, schema: pa.Schema) -> None:
        self._schema = schema
        self._schema_digest = _hash_schema(schema)
        # BTreeMap<path, entry dict> – sorted by key
        self._fields: dict[str, dict] = {}
        for i in range(len(schema)):
            _extract_fields(schema.field(i), "", self._fields)
        # Ensure sorted order (Python 3.7+ dicts are insertion-ordered)
        self._fields = dict(sorted(self._fields.items()))

    def update(self, record_batch: pa.RecordBatch) -> None:
        """Feed a RecordBatch into the running digest."""
        for col_idx in range(record_batch.num_columns):
            field = record_batch.schema.field(col_idx)
            array = record_batch.column(col_idx)
            path = field.name

            _traverse_and_update(
                field.type,
                field.nullable,
                array,
                path,
                None,  # no ancestor struct nulls at top level
                self._fields,
            )

    def finalize(self) -> bytes:
        """Consume the digester and return the versioned hash."""
        final_digest = hashlib.sha256()
        final_digest.update(self._schema_digest)
        for _path, entry in sorted(self._fields.items()):
            _finalize_digest(final_digest, entry)
        return VERSION_BYTES + final_digest.digest()

    # -- Convenience class methods ------------------------------------------

    @staticmethod
    def hash_schema(schema: pa.Schema) -> bytes:
        return VERSION_BYTES + _hash_schema(schema)

    @staticmethod
    def hash_record_batch(record_batch: pa.RecordBatch) -> bytes:
        d = ArrowDigester(record_batch.schema)
        d.update(record_batch)
        return d.finalize()

    @staticmethod
    def hash_table(table: pa.Table) -> bytes:
        """Hash a full table (iterates over all batches)."""
        d = ArrowDigester(table.schema)
        for batch in table.to_batches():
            d.update(batch)
        return d.finalize()

    @staticmethod
    def hash_array(array: pa.Array) -> bytes:
        """Hash a single array (matches Rust ``hash_array``)."""
        dt_value = _raw_serde_data_type(array.type)
        dt_json = json.dumps(dt_value, separators=(",", ":"))

        final_digest = hashlib.sha256()
        final_digest.update(dt_json.encode())

        nullable = array.null_count > 0 or (hasattr(array, 'buffers') and array.buffers()[0] is not None)
        if nullable:
            entry = (True, _BitVec(), hashlib.sha256())
        else:
            entry = (False, None, hashlib.sha256())

        _array_digest_update(array.type, array, entry)
        _finalize_digest(final_digest, entry)

        return VERSION_BYTES + final_digest.digest()
