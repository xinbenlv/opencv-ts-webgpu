#!/usr/bin/env python3
"""
Convert SMPL .pkl model to a fast-loading binary format for the browser.

Usage:
    python scripts/convert-smpl.py
    python scripts/convert-smpl.py path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl

Output: demo/smpl/smpl-neutral.smpl.bin (~40 MB, much faster to parse than pkl)

Binary format:
    magic:      b"SMPL"   (4 bytes)
    version:    uint32    (4 bytes) = 1
    num_arrays: uint32    (4 bytes)
    for each array:
        name_len: uint32  (4 bytes)
        name:     utf-8   (name_len bytes)
        dtype:    uint8   (1 byte)  0=float32, 1=int32, 2=uint32
        rank:     uint32  (4 bytes)
        shape:    uint32[rank]
        data:     dtype[product(shape)]
"""

import sys
import os
import struct
import pickle
import numpy as np
import pathlib

DTYPE_FLOAT32 = 0
DTYPE_INT32   = 1
DTYPE_UINT32  = 2

SMPL_KEYS = ['v_template', 'f', 'shapedirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights']

def find_pkl():
    """Find the SMPL neutral pkl in the expected location."""
    candidates = [
        'demo/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
        'demo/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def encode_array(name: str, arr: np.ndarray) -> bytes:
    """Encode a single numpy array into the binary format."""
    # Normalise dtype
    if arr.dtype == np.float64 or arr.dtype == np.float32:
        arr = arr.astype(np.float32, copy=False)
        dtype_byte = DTYPE_FLOAT32
    elif arr.dtype in (np.int32, np.int64):
        arr = arr.astype(np.int32, copy=False)
        dtype_byte = DTYPE_INT32
    elif arr.dtype in (np.uint32, np.uint64):
        arr = arr.astype(np.uint32, copy=False)
        dtype_byte = DTYPE_UINT32
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint32, copy=False)
        dtype_byte = DTYPE_UINT32
    else:
        raise ValueError(f"Unsupported dtype {arr.dtype} for array '{name}'")

    # Ensure C-contiguous (row-major)
    arr = np.ascontiguousarray(arr)

    name_bytes = name.encode('utf-8')
    header = struct.pack('<I', len(name_bytes))
    header += name_bytes
    header += struct.pack('<B', dtype_byte)
    header += struct.pack('<I', arr.ndim)
    for s in arr.shape:
        header += struct.pack('<I', int(s))

    return header + arr.tobytes()

class _Stub:
    """Generic stub that absorbs any __init__ args and stores them."""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
    def __reduce__(self):
        return (_Stub, ())

class SMPLUnpickler(pickle.Unpickler):
    """Custom unpickler that handles chumpy types without that lib."""
    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return _Stub
        # Let scipy.sparse load normally — it IS installed
        return super().find_class(module, name)

def _unwrap(obj):
    """Recursively extract numpy arrays from various wrapper types."""
    if isinstance(obj, np.ndarray):
        return obj
    # scipy sparse matrix
    if hasattr(obj, 'toarray'):
        return obj.toarray()
    if isinstance(obj, _Stub):
        for attr in ('x', 'r', '_x', 'v'):
            v = getattr(obj, attr, None)
            if v is not None and isinstance(v, np.ndarray):
                return v
        try:
            return np.array(obj)
        except Exception:
            return None
    return obj


def convert(pkl_path: str, out_path: str):
    print(f"Loading {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        data = SMPLUnpickler(f, encoding='latin1').load()

    print(f"  Keys in pkl: {list(data.keys())}")

    chunks = []
    found = []
    for key in SMPL_KEYS:
        aliases = {'f': ['faces', 'F'], 'v_template': ['v_shaped']}
        if key in data:
            raw = data[key]
        else:
            raw = next((data[a] for a in aliases.get(key, []) if a in data), None)
        if raw is None:
            print(f"  WARNING: key '{key}' not found, skipping")
            continue
        arr = _unwrap(raw)
        if arr is None:
            arr = np.array(raw)
        arr = np.array(arr)
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")

        chunks.append(encode_array(key, arr))
        found.append(key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'wb') as f:
        f.write(b'SMPL')
        f.write(struct.pack('<I', 1))          # version
        f.write(struct.pack('<I', len(found))) # num_arrays
        for chunk in chunks:
            f.write(chunk)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB, {len(found)} arrays)")
    print("Arrays included:", found)

if __name__ == '__main__':
    # Change working directory to project root regardless of where script is run from
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = find_pkl()
        if not pkl_path:
            print("ERROR: Could not find SMPL .pkl file.")
            print("Expected: demo/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
            sys.exit(1)

    out_path = 'demo/smpl/smpl-neutral.smpl.bin'
    convert(pkl_path, out_path)
