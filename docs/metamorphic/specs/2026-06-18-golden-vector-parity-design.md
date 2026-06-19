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
  "version": "0.3",
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

Arrow IPC is used for `ipc_b64` because it provides a stable, self-contained encoding of the
Arrow schema (including all metadata) that both Rust and Python can deserialize identically.
Note: `arrow-ipc`'s `metadata_to_fb` sorts metadata keys alphabetically before FlatBuffers
encoding (`ordered_keys.sort()` in convert.rs), so the IPC byte stream is deterministic
regardless of HashMap insertion order at the producer side.

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
invariant directly in the fixture. Because `arrow-ipc` sorts metadata keys alphabetically before
FlatBuffers encoding, both vectors produce **byte-identical IPC blobs** — the insertion-order
invariant is enforced at the IPC level, not the hasher level. Both vectors therefore share the
same `ipc_b64` and the same `expected_hash`. The test verifies that the hasher also produces
matching output when the live hasher is called directly on schemas built with different insertion
orders.

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
- Embeds a `rust_commit` field by running `git rev-parse HEAD` via `std::process::Command` at generation time

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

### Why `Cargo.toml` version and git tags are kept in sync

Cargo requires a hardcoded version in `Cargo.toml`; there is no `hatch-vcs`-style
auto-derivation from git tags. The invariant is enforced instead by using `cargo-release`,
which atomically bumps `Cargo.toml`, commits the change, creates the matching git tag, and
pushes both — making it structurally impossible to tag without also updating `Cargo.toml`.

### `release.toml` (new file, `nauticalab/starfix`)

```toml
pre-release-commit-message = "chore: release v{{version}}"
tag-name                   = "v{{version}}"
push                       = true
publish                    = false   # wheels go via maturin, not crates.io
```

### `cargo-release` CI enforcement (new job in `maturin-release.yml`)

A lightweight check that runs on every tag push and fails if the tag name does not match
the version in `Cargo.toml`:

```yaml
verify-version-tag-sync:
  runs-on: ubuntu-latest
  if: startsWith(github.ref, 'refs/tags/')
  steps:
    - uses: actions/checkout@v4
    - name: Verify Cargo.toml version matches tag
      run: |
        TAG="${GITHUB_REF#refs/tags/v}"
        CARGO_VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*= *"\(.*\)"/\1/')
        if [ "$TAG" != "$CARGO_VERSION" ]; then
          echo "Tag $TAG does not match Cargo.toml version $CARGO_VERSION"
          exit 1
        fi
```

### Manually-triggered release workflows

#### `nauticalab/starfix` — new `.github/workflows/release.yml`

```yaml
name: release
on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g. 0.3.0)'
        required: true
        type: string
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Generate GitHub App token
        id: app-token
        uses: actions/create-github-app-token@v3
        with:
          app-id: ${{ secrets.RELEASE_APP_ID }}
          private-key: ${{ secrets.RELEASE_APP_PRIVATE_KEY }}

      - uses: actions/checkout@v4
        with:
          token: ${{ steps.app-token.outputs.token }}
          fetch-depth: 0

      - uses: actions-rust-lang/setup-rust-toolchain@v1
        with:
          toolchain: 1.91.1

      - run: cargo install cargo-release

      - name: Configure git
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Release
        run: cargo release ${{ inputs.version }} --execute --no-confirm
```

A GitHub App token (rather than `GITHUB_TOKEN`) is required for checkout so that the tag
push from `cargo-release` triggers the downstream `maturin-release.yml` workflow.
`GITHUB_TOKEN`-pushed events do not trigger other workflows (GitHub's recursion guard).

Required secrets: `RELEASE_APP_ID`, `RELEASE_APP_PRIVATE_KEY` — a GitHub App with
`contents:write` on `nauticalab/starfix`.

**What this workflow does end-to-end:**
1. `cargo-release` bumps `Cargo.toml` → commits → creates `v{version}` tag → pushes both
2. Tag push fires `maturin-release.yml` → builds wheels → publishes to PyPI

#### `nauticalab/starfix-python` — new `.github/workflows/release.yml`

`hatch-vcs` reads the version from git tags automatically; there is no version file to
bump. The release workflow only needs to create and push the tag:

```yaml
name: release
on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g. 0.3.0)'
        required: true
        type: string
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Generate GitHub App token
        id: app-token
        uses: actions/create-github-app-token@v3
        with:
          app-id: ${{ secrets.RELEASE_APP_ID }}
          private-key: ${{ secrets.RELEASE_APP_PRIVATE_KEY }}

      - uses: actions/checkout@v4
        with:
          token: ${{ steps.app-token.outputs.token }}

      - name: Configure git
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Tag and push release
        run: |
          git tag v${{ inputs.version }}
          git push origin v${{ inputs.version }}
```

Tag push fires the existing `publish.yml` → pure-Python package published to PyPI.

### Release procedure (coordinated across both repos)

1. Merge the PLT-1735 PRs in both repos to `main`
2. Trigger `starfix` → Actions → **release** → Run workflow → version: `0.3.0`
3. Trigger `starfix-python` → Actions → **release** → Run workflow → version: `0.3.0`
4. Confirm both PyPI packages show `0.3.0`

---

## Out of Scope

- The `include_metadata` implementation itself (PLT-1733, PLT-1734)
- Cross-version parity (v0.1.0 ↔ v0.2.0) — already covered by existing golden tests
- Future finer-grained metadata controls
- `hash_array` with `include_metadata` — arrays have no schema-level metadata; not applicable

---

## Risks

- **IPC metadata order:** Arrow IPC does **not** preserve key insertion order — `metadata_to_fb`
  in arrow-ipc sorts keys alphabetically before FlatBuffers encoding (`ordered_keys.sort()` in
  convert.rs). As a result, the `key_reorder_canonical` and `key_reorder_shuffled` vectors
  produce byte-identical IPC blobs. If a future Arrow version changes this sorting behaviour,
  the `key_reorder_*` IPC blobs would diverge and the vectors would need to be regenerated;
  the fixture format itself remains valid.
- **Fixture drift:** Mitigated by the `golden-sync-check` CI job. If the GitHub App secret
  expires or is revoked, the drift check will fail loudly rather than silently passing.
- **Version/tag sync:** The `verify-version-tag-sync` CI job enforces the invariant on every
  tag push. If someone bypasses `cargo-release` and creates a tag manually without bumping
  `Cargo.toml`, this job will catch it and fail the `maturin-release.yml` run before any
  wheels are built.
- **GitHub App token for release:** Both release workflows require `RELEASE_APP_ID` and
  `RELEASE_APP_PRIVATE_KEY` secrets. If these expire or are revoked the workflows will fail
  at the token-generation step with a clear error — no silent failure.
