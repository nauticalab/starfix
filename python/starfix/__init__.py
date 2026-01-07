from . import _internal as sfr  # import rust binding
from typing import TYPE_CHECKING
import ctypes

if TYPE_CHECKING:
    import pyarrow as pa


def hash_record_batch(table: "pa.Table") -> bytes:
    # Covert table to record batch first (so we can extract the pointers), since the default behavior is 1 batch, we can just get the first element
    # After that we can extract the PyCapsules
    schema_capsule, array_capsule = table.to_batches()[0].__arrow_c_array__()

    # Extract raw pointers from capsules due to uniffi limitations
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    PyCapsule_GetPointer.restype = ctypes.c_void_p

    return sfr.hash_record_batch(
        PyCapsule_GetPointer(array_capsule, b"arrow_array"),
        PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
    )


def hash_schema(schema: "pa.Schema") -> bytes:
    schema_capsule = schema.__arrow_c_schema__()

    # Extract raw pointers from capsules due to uniffi limitations
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    PyCapsule_GetPointer.restype = ctypes.c_void_p

    return sfr.hash_schema(
        PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
    )


class PyArrowDigester:
    def __init__(self, schema: "pa.Schema") -> None:

        schema_capsule = schema.__arrow_c_schema__()

        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        PyCapsule_GetPointer.restype = ctypes.c_void_p

        schema_ptr = PyCapsule_GetPointer(schema_capsule, b"arrow_schema")

        self._internal = sfr.InternalPyArrowDigester(schema_ptr)

    def update(self, table: "pa.Table") -> None:
        # Covert table to record batch first (so we can extract the pointers), since the default behavior is 1 batch, we can just get the first element
        # After that we can extract the PyCapsules
        schema_capsule, array_capsule = table.to_batches()[0].__arrow_c_array__()

        # Extract raw pointers from capsules due to uniffi limitations
        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        PyCapsule_GetPointer.restype = ctypes.c_void_p

        self._internal.update(
            PyCapsule_GetPointer(array_capsule, b"arrow_array"),
            PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
        )

    def finalize(self) -> bytes:
        return self._internal.finalize()
