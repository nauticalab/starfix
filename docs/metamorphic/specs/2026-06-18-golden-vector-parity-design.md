# Cross-Language Hash Parity via Golden Vectors

**Issue:** PLT-1735  
**Date:** 2026-06-18  
**Status:** Approved  
**Repos:** `nauticalab/starfix` (authoritative), `nauticalab/starfix-python` (consumer)

---

## Overview

Both `starfix` (Rust) and `starfix-python` now implement `include_metadata` hashing (PLT-1733,
PLT-1734). This spec establishes a shared golden-vector fixture that proves the two
implementations produce bit-for-bit identical hashes for the same Arrow inputs. Rust is the
authoritative source; Python must match it exactly.

---

## Fixture Format

A single JSON file committed to both repos at:

```
tests/golden/include_metadata_v0.3.json
```

Top-level structure:

```json
{
  "version": "0.2",
  "generated_by": "cargo run --bin emit_golden_metadata",
  "rust_commit": "<output of `git rev-parse HEAD` at generation time>",
  "vectors": [ ... ]
}
```

Each entry in `vectors`:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique slug (used as pytest test ID) |
| `description` | string | Human-readable summary of what this vector tests |
| `method` | string | `"hash_schema"` or `"hash_record_batch"` |
| `include_metadata` | bool | Value passed to the hasher |
| `ipc_b64` | string | Base64-encoded Arrow IPC stream (schema + optional rows) |
| `expected_hash` | string | Rust-authoritative hex-encoded hash digest |

Arrow IPC is used for `ipc_b64` because it captures the exact bytes — including metadata key
insertion order — that were fed to the Rust hasher. This eliminates any risk of Python
constructing subtly different Arrow data.

---

## Required Vectors

| id | Scenario | `include_metadata` |
|---|---|---|
| `no_metadata_include_false` | `{id: Int64, name: LargeUtf8}`, no metadata | `false` |
| `schema_level_metadata` | Schema with `{"version": "2"}` at schema level | `true` |
| `field_metadata_single_field` | One field with `{"unit": "kg"}` | `true` |
| `field_metadata_multiple_fields` | Two fields each with distinct metadata | `true` |
| `schema_and_field_metadata` | Both schema-level and field-level metadata | `true` |
| `unicode_metadata` | Emoji + CJK keys/values | `true` |
| `key_reorder_canonical` | Field metadata keys in alphabetical order | `true` |
| `key_reorder_shuffled` | Same keys, different insertion order — **same `expected_hash` as `key_reorder_canonical`** | `true` |
| `empty_metadata_invariant` | No metadata at all — tested with `include_metadata=false`; `expected_hash` must equal that of the same schema hashed with `include_metadata=true` | `false` |

The `key_reorder_canonical` / `key_reorder_shuffled` pair encodes the key-ordering determinism
invariant directly in the fixture: two different IPC blobs (different insertion orders) map to
the same `expected_hash`.

The `empty_metadata_invariant` entry pins the empty-metadata fixed point: a schema with no
metadata must produce the same hash regardless of `include_metadata`. Only one entry is needed
because both flag values produce the same hash by definition; a second entry would be
redundant. The Rust and Python tests assert `hash(schema, false) == hash(schema, true) ==
expected_hash`.

---

## Rust Side (`nauticalab/starfix`)

### `src/bin/emit_golden_metadata.rs`

Developer tool. Generates the fixture to stdout:

```
cargo run --bin emit_golden_metadata > tests/golden/include_metadata_v0.3.json
cargo fmt
```

Responsibilities:
- Constructs each Arrow schema/batch for the 9 vectors above
- Serialises each to an Arrow IPC stream, base64-encodes it
- Calls `ArrowDigester` to produce the authoritative hash
- Writes the complete JSON to stdout
- Embeds a `rust_commit` field via `env!("CARGO_PKG_VERSION")` or a build-time `GIT_SHA` env var

The file header contains a comment documenting the full regeneration procedure (see
§ Regeneration Workflow below).

### `tests/golden_vectors.rs`

Regression guard. Runs as part of `cargo test` (covered by the `test` job in
`maturin-release.yml`).

For each entry in the committed fixture:
1. Decodes `ipc_b64` → Arrow IPC stream
2. Reads schema (and batch, if present)
3. Calls `ArrowDigester::hash_schema` or `ArrowDigester::hash_record_batch` with `include_metadata`
4. Asserts `hex::encode(result) == entry.expected_hash`

On failure, the panic message includes `id` and `description` for immediate identification.

Additionally, the test explicitly verifies the empty-metadata invariant by asserting:

```rust
assert_eq!(
    hash(schema, include_metadata=false),
    hash(schema, include_metadata=true),
    "empty_metadata_invariant: hash must be equal regardless of include_metadata"
);
```

---

## Python Side (`nauticalab/starfix-python`)

### `tests/golden/include_metadata_v0.3.json`

Exact copy of the Rust-generated fixture. Committed alongside existing test files. Updated
whenever the Rust fixture is regenerated (see § Regeneration Workflow).

### `tests/test_golden_parity_metadata.py`

Parametrized test file. Each vector becomes one pytest case, identified by its `id` slug:

```python
@pytest.mark.parametrize("vector", _load_vectors(), ids=lambda v: v["id"])
def test_golden_vector(vector):
    schema, batch = _deserialize_ipc(vector["ipc_b64"])
    include_metadata = vector["include_metadata"]
    if vector["method"] == "hash_schema":
        result = ArrowDigester.hash_schema(schema, include_metadata=include_metadata)
    else:
        result = ArrowDigester.hash_record_batch(batch, include_metadata=include_metadata)
    assert result.hex() == vector["expected_hash"], (
        f"Vector '{vector['id']}' mismatch: {vector['description']}"
    )
```

The `empty_metadata_invariant` vector is additionally tested with `include_metadata=True` in a
dedicated assertion that reads the same `expected_hash`:

```python
def test_empty_metadata_invariant_both_flags():
    # Load the empty_metadata_invariant vector and verify both flag values produce
    # the same Rust-authoritative hash.
    vector = _get_vector("empty_metadata_invariant")
    schema, _ = _deserialize_ipc(vector["ipc_b64"])
    hash_false = ArrowDigester.hash_schema(schema, include_metadata=False).hex()
    hash_true  = ArrowDigester.hash_schema(schema, include_metadata=True).hex()
    assert hash_false == vector["expected_hash"]
    assert hash_true  == vector["expected_hash"]
```

### `golden-sync-check` job in `.github/workflows/ci.yml`

Prevents the committed fixture from drifting from the Rust authoritative source. Runs on every
PR and push to `main`.

Uses `actions/create-github-app-token@v3` (GitHub-owned action) to generate a short-lived
installation token from a GitHub App with `contents:read` permission on `nauticalab/starfix`.

Required secrets in `starfix-python`:
- `STARFIX_APP_ID` — numeric GitHub App ID
- `STARFIX_APP_PRIVATE_KEY` — PEM private key

```yaml
golden-sync-check:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Generate GitHub App token
      id: app-token
      uses: actions/create-github-app-token@v3
      with:
        app-id: ${{ secrets.STARFIX_APP_ID }}
        private-key: ${{ secrets.STARFIX_APP_PRIVATE_KEY }}
        repositories: starfix

    - name: Fetch authoritative fixture from starfix
      run: |
        gh api repos/nauticalab/starfix/contents/tests/golden/include_metadata_v0.3.json \
          --jq '.content' | base64 -d > /tmp/upstream.json
      env:
        GH_TOKEN: ${{ steps.app-token.outputs.token }}

    - name: Fail on fixture drift
      run: diff tests/golden/include_metadata_v0.3.json /tmp/upstream.json
```

---

## Regeneration Workflow

When the Rust hasher changes and the fixture must be updated:

1. In `starfix`: `cargo run --bin emit_golden_metadata > tests/golden/include_metadata_v0.3.json`
2. Run `cargo fmt` and verify `cargo test` passes (the `golden_vectors` test will validate the new file)
3. Commit the updated fixture and merge to `main`
4. In `starfix-python`: copy the file to `tests/golden/include_metadata_v0.3.json` and commit
5. The `golden-sync-check` CI job gates the Python PR — it will fail until the committed copy matches `starfix` main

---

## Version Alignment

Both repos are bumped to `v0.3.0` as part of this work. The hash format byte prefix
(`[0, 0, 1]` — hash spec version 0.0.1) is unchanged; this is a package version bump only.

| Repo | Change | When |
|---|---|---|
| `nauticalab/starfix` | `Cargo.toml` `version` → `"0.3.0"` | In this PR |
| `nauticalab/starfix` | `v0.3.0` git tag | Created on merge to `main` |
| `nauticalab/starfix-python` | No `pyproject.toml` change (`hatch-vcs` reads from git tag) | — |
| `nauticalab/starfix-python` | `v0.3.0` git tag | Created on merge to `main` |

The Rust `Cargo.toml` version must always match the latest git tag. After bumping
`Cargo.toml` to `0.3.0` and merging, tag both repos simultaneously:

```bash
# in starfix
git tag v0.3.0 && git push origin v0.3.0

# in starfix-python
git tag v0.3.0 && git push origin v0.3.0
```

The `maturin-release.yml` workflow in `starfix` triggers on tags and handles wheel
building and PyPI publication automatically.

---

## Out of Scope

- The `include_metadata` implementation itself (PLT-1733, PLT-1734)
- Cross-version parity (v0.1.0 ↔ v0.2.0) — already covered by existing golden tests
- Future finer-grained metadata controls
- `hash_array` with `include_metadata` — arrays have no schema-level metadata; not applicable

---

## Risks

- **IPC metadata order:** Arrow IPC preserves key insertion order in its FlatBuffers encoding.
  This is load-bearing for the `key_reorder_*` vectors. If a future Arrow version changes this
  behaviour the vectors would need to be regenerated, but the fixture format itself remains
  valid.
- **Fixture drift:** Mitigated by the `golden-sync-check` CI job. If the GitHub App secret
  expires or is revoked, the drift check will fail loudly rather than silently passing.
