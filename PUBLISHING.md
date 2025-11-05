# Publishing to crates.io

This guide walks you through publishing the blue-noise crate to crates.io.

## Prerequisites

### 1. Create a crates.io Account

1. Go to https://crates.io
2. Click "Log in with GitHub"
3. Authorize the application

### 2. Get Your API Token

1. Go to https://crates.io/me
2. Click "New Token"
3. Give it a name (e.g., "blue-noise-publish")
4. Copy the token (you'll only see it once!)

### 3. Login with Cargo

```bash
cargo login <your-api-token>
```

This stores your token in `~/.cargo/credentials.toml`

---

## Pre-Publication Checklist

### ✅ Verify Cargo.toml

Check that your `Cargo.toml` has all required fields:

```toml
[package]
name = "blue-noise"
version = "0.2.0"                    # ✓ Semantic versioning
authors = ["Matthew Blode <m@blode.co>"]
edition = "2021"                      # ✓ Current edition
description = "High-quality blue noise generation and dithering using the void-and-cluster algorithm"  # ✓ Under 200 chars
documentation = "https://docs.rs/blue-noise"  # ✓ Auto-generated
readme = "README.md"                  # ✓ Exists
repository = "https://github.com/mblode/blue-noise/"  # ✓ Valid URL
license = "MIT"                       # ✓ Valid license
keywords = ["ditherpunk", "noise", "dithering", "dither", "blue-noise"]  # ✓ Max 5
categories = ["command-line-utilities", "multimedia"]  # ✓ Valid categories
```

**Current Status:** ✅ All fields present

### ✅ Run Tests

```bash
# All tests must pass
cargo test --all-features

# Check for warnings
cargo clippy --all-features -- -D warnings

# Check formatting
cargo fmt --all -- --check
```

### ✅ Verify Documentation

```bash
# Build and view docs
cargo doc --open --no-deps

# Check for missing documentation
cargo doc --all-features
```

### ✅ Test Installation

```bash
# Build release binary
cargo build --release

# Try running it
./target/release/blue-noise --help
./target/release/blue-noise generate --size 16 -o test.png
```

### ✅ Update Version-Specific Files

If not already done:

1. **Update version in Cargo.toml**
   ```toml
   version = "0.2.0"
   ```

2. **Update CHANGELOG.md**
   ```markdown
   ## [0.2.0] - 2025-01-05
   ### Added
   - Complete void-and-cluster implementation
   - FFT optimization
   - Modern CLI with subcommands
   ...
   ```

3. **Update lib.rs doc URL if needed**
   ```rust
   #![doc(html_root_url = "https://docs.rs/blue-noise/0.2.0")]
   ```

---

## Publishing Steps

### Step 1: Dry Run

First, do a dry run to check for issues:

```bash
cargo publish --dry-run
```

This will:
- ✓ Verify `Cargo.toml` metadata
- ✓ Check that all files are included
- ✓ Build documentation
- ✓ Package the crate
- ✗ But NOT actually publish

**Expected output:**
```
   Packaging blue-noise v0.2.0
   Verifying blue-noise v0.2.0
   Compiling blue-noise v0.2.0
    Finished dev [unoptimized + debuginfo] target(s)
```

### Step 2: Review Package Contents

See what will be published:

```bash
cargo package --list
```

**Should include:**
- All `.rs` files in `src/`
- All files in `examples/`
- All files in `benches/`
- `Cargo.toml` and `Cargo.lock`
- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `LICENSE.md`
- `.gitignore`

**Should NOT include:**
- `/target` directory
- `.DS_Store` files
- Test output files
- Git history

### Step 3: Publish!

When ready, publish for real:

```bash
cargo publish
```

**This will:**
1. Package your crate
2. Upload to crates.io
3. Build documentation (auto-hosted at docs.rs)

**Expected output:**
```
    Updating crates.io index
   Packaging blue-noise v0.2.0
   Verifying blue-noise v0.2.0
   Compiling blue-noise v0.2.0
    Finished dev [unoptimized + debuginfo] target(s)
   Uploading blue-noise v0.2.0
```

---

## Post-Publication

### 1. Verify on crates.io

Visit: https://crates.io/crates/blue-noise

Check:
- ✓ Metadata looks correct
- ✓ README displays properly
- ✓ Links work

### 2. Wait for Documentation

Documentation builds automatically at https://docs.rs/blue-noise

This can take 5-15 minutes after publishing.

### 3. Test Installation

From a different directory:

```bash
# Install globally
cargo install blue-noise

# Test it works
blue-noise --version
blue-noise generate --size 32 -o test.png

# Uninstall
cargo uninstall blue-noise
```

### 4. Create GitHub Release

1. Go to https://github.com/mblode/blue-noise/releases
2. Click "Draft a new release"
3. Create tag: `v0.2.0`
4. Title: `v0.2.0`
5. Copy description from CHANGELOG.md
6. Publish release

### 5. Update Repository README

Add a badge to your README.md:

```markdown
[![Crates.io](https://img.shields.io/crates/v/blue-noise.svg)](https://crates.io/crates/blue-noise)
[![Documentation](https://docs.rs/blue-noise/badge.svg)](https://docs.rs/blue-noise)
```

---

## Updating to a New Version

When you want to publish an update:

### 1. Update Version

```bash
# In Cargo.toml
version = "0.2.1"  # or "0.3.0" or "1.0.0"
```

Follow [Semantic Versioning](https://semver.org/):
- **Patch** (0.2.0 → 0.2.1): Bug fixes
- **Minor** (0.2.0 → 0.3.0): New features (backward compatible)
- **Major** (0.2.0 → 1.0.0): Breaking changes

### 2. Update CHANGELOG.md

```markdown
## [0.2.1] - 2025-01-XX

### Fixed
- Bug fix description

### Changed
- Change description
```

### 3. Update lib.rs doc URL

```rust
#![doc(html_root_url = "https://docs.rs/blue-noise/0.2.1")]
```

### 4. Commit and Tag

```bash
git add Cargo.toml CHANGELOG.md src/lib.rs
git commit -m "Release v0.2.1"
git tag v0.2.1
git push origin main --tags
```

### 5. Publish

```bash
cargo publish
```

---

## Yanking a Version

If you need to yank a published version (doesn't delete it, just marks as "do not use"):

```bash
cargo yank --vers 0.2.0
```

To undo:

```bash
cargo yank --vers 0.2.0 --undo
```

---

## Troubleshooting

### "crate name already taken"

The crate name `blue-noise` must be available. Check: https://crates.io/crates/blue-noise

If taken, you'll need to choose a different name in `Cargo.toml`.

### "missing required field"

Ensure `Cargo.toml` has all required fields:
- `name`
- `version`
- `authors` or `author`
- `edition`
- `license` or `license-file`
- `description`

### "file not included in package"

Check `.gitignore` - cargo uses it to determine what to include.

List what will be packaged:
```bash
cargo package --list
```

### "documentation failed to build"

Test locally:
```bash
cargo doc --all-features
```

Fix any doc warnings or errors.

---

## Security Notes

- ✅ **Never commit** your API token
- ✅ API token is stored in `~/.cargo/credentials.toml`
- ✅ Add `credentials.toml` to global `.gitignore`
- ✅ You can revoke tokens at https://crates.io/me

---

## Quick Command Reference

```bash
# Login (one-time)
cargo login <token>

# Pre-publish checks
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo doc --all-features

# Dry run
cargo publish --dry-run

# Publish
cargo publish

# Install from crates.io
cargo install blue-noise

# Update version
# 1. Edit Cargo.toml version
# 2. Update CHANGELOG.md
# 3. Update lib.rs doc URL
git commit -am "Release vX.Y.Z"
git tag vX.Y.Z
git push --tags
cargo publish
```

---

## Additional Resources

- [Publishing on crates.io](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [Semantic Versioning](https://semver.org/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [crates.io Policies](https://crates.io/policies)

---

**Ready to publish?** Run `cargo publish --dry-run` to start! 🚀
