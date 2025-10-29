from . import _internal as sfr  # import rust binding
from typing import TYPE_CHECKING
import ctypes

if TYPE_CHECKING:
    import pyarrow as pa


def hash_arrow_table(table: "pa.Table") -> bytes:
    # Covert table to record batch first (so we can extract the pointers), since the default behavior is 1 batch, we can just get the first element
    # After that we can extract the PyCapsules
    schema_capsule, array_capsule = table.to_batches()[0].__arrow_c_array__()

    # Extract raw pointers from capsules due to uniffi limitations
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    PyCapsule_GetPointer.restype = ctypes.c_void_p

    return sfr.process_arrow_table(
        PyCapsule_GetPointer(array_capsule, b"arrow_array"),
        PyCapsule_GetPointer(schema_capsule, b"arrow_schema"),
    )
