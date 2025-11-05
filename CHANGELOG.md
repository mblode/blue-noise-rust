# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-05

### Added

#### Generator
- Complete implementation of Ulichney's void-and-cluster algorithm
- FFT-optimized Gaussian blur for power-of-two dimensions (~50% faster)
- Spatial domain Gaussian blur fallback for non-power-of-two sizes
- Seeded random number generation (Mulberry32 PRNG) for reproducibility
- Progress indicators for long-running operations
- Comprehensive error handling with custom error types
- Support for rectangular (non-square) textures
- Configurable sigma and initial density parameters

#### Dithering
- Hex color parsing for foreground/background colors
- Image resizing during dithering
- Contrast adjustment support
- Blue noise texture loading and validation
- Ordered dithering with custom level support
- Seamless tiling with toroidal topology

#### CLI
- Modern CLI using clap v4 with derive macros
- `generate` subcommand for creating blue noise textures
  - Size, sigma, seed, and output path options
  - Verbose mode with progress indicators
- `dither` subcommand for applying dithering
  - Input/output paths
  - Custom foreground/background colors
  - Image resizing options
  - Contrast adjustment
  - Custom noise texture selection
- Comprehensive help messages and validation

#### Documentation
- Extensive inline documentation with algorithm explanations
- Comprehensive README with installation and usage examples
- CONTRIBUTING.md with development guidelines
- Four example programs demonstrating library usage
- Complete API documentation for all public items

#### Testing
- Unit tests for generator module (14 tests)
- Unit tests for dither module (11 tests)
- Integration tests for FFT vs spatial methods
- Reproducibility and determinism tests
- Validation and error handling tests

#### CI/CD
- GitHub Actions workflow for CI
- Multi-platform testing (Linux, Windows, macOS)
- Rust stable and beta testing
- Formatting and clippy checks
- Documentation building
- Code coverage with tarpaulin
- Build artifacts for all platforms

#### Performance
- Criterion.rs benchmarks for:
  - Generation performance (different sizes)
  - FFT vs spatial blur comparison
  - Dithering performance
  - Sigma value impacts
  - Hex color parsing
- HTML benchmark reports

#### Examples
- `generate_texture.rs` - Basic texture generation
- `dither_image.rs` - Image dithering with multiple color schemes
- `compare_sizes.rs` - Performance comparison of different sizes
- `library_integration.rs` - In-memory operations and batch processing

### Changed
- Updated from Rust 2018 to 2021 edition
- Upgraded all dependencies to latest versions:
  - clap 3.0.0-beta.2 → 4.5 (with derive features)
  - image 0.23 → 0.25
  - Added rustfft 6.4
  - Added thiserror 2.0
  - Added anyhow 1.0
  - Added indicatif 0.17
- Removed YAML-based CLI configuration
- Complete rewrite of main.rs with modern patterns

### Removed
- `cli.yaml` configuration file (replaced with derive macros)
- Legacy CLI interface (replaced with subcommands)

## [0.1.2] - Previous Release

### Features
- Basic blue noise dithering
- Simple CLI with positional arguments
- Black and white dithering only

---

## Migration Guide: 0.1.x → 0.2.0

### CLI Changes

**Old usage (0.1.x):**
```bash
blue-noise input.png output.png
```

**New usage (0.2.0):**
```bash
# Generate blue noise texture first
blue-noise generate --size 64 --output blue-noise.png

# Dither image
blue-noise dither --input input.png --output output.png
```

### Library Changes

The 0.2.0 release introduces the library API. If you were using the binary only, you can now import blue-noise as a library:

```toml
[dependencies]
blue-noise = "0.2"
```

```rust
use blue_noise::{BlueNoiseGenerator, BlueNoiseConfig};

let config = BlueNoiseConfig::default();
let generator = BlueNoiseGenerator::new(config)?;
let result = generator.generate()?;
```

### Breaking Changes

- CLI now requires subcommands (`generate` or `dither`)
- Removed hardcoded `img/noise.png` path - noise texture must be specified
- Changed from YAML configuration to clap derive macros

### New Features Available

- Generate custom blue noise textures (not just use pre-made ones)
- Custom color dithering (not just black & white)
- Image resizing and contrast adjustment
- Progress indicators for long operations
- Reproducible generation with seeds
- Library API for integration into other projects

[0.2.0]: https://github.com/mblode/blue-noise/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/mblode/blue-noise/releases/tag/v0.1.2
