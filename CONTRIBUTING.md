# Contributing to Blue Noise

Thank you for your interest in contributing to Blue Noise! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/blue-noise.git
   cd blue-noise
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/mblode/blue-noise.git
   ```

## Development Setup

### Prerequisites

- Rust 1.70 or later (latest stable recommended)
- Cargo (comes with Rust)

### Building

```bash
# Debug build
cargo build

# Release build (much faster for testing generation)
cargo build --release
```

### Running

```bash
# Generate a texture
cargo run --release -- generate --size 64

# Dither an image
cargo run --release -- dither -i input.jpg -o output.png
```

## Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the [Style Guidelines](#style-guidelines)

3. **Write tests** for new functionality

4. **Update documentation** as needed

5. **Ensure all tests pass**:
   ```bash
   cargo test
   ```

6. **Check formatting**:
   ```bash
   cargo fmt --all -- --check
   ```

7. **Run clippy** (Rust linter):
   ```bash
   cargo clippy --all-features -- -D warnings
   ```

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests in release mode (faster for algorithm tests)
cargo test --release

# Run a specific test
cargo test test_generate_small_texture

# Run tests with output
cargo test -- --nocapture
```

### Writing Tests

- Add unit tests in the same file as the code being tested using `#[cfg(test)]` modules
- Integration tests go in the `tests/` directory
- Use descriptive test names that explain what is being tested
- Test edge cases and error conditions

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#FF0000").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
    }
}
```

### Benchmarks

Run benchmarks to measure performance:

```bash
cargo bench
```

View detailed reports in `target/criterion/report/index.html`.

## Documentation

### Code Documentation

- All public APIs must have documentation comments (`///`)
- Include examples in doc comments when helpful
- Document panics, errors, and safety considerations

Example:
```rust
/// Parse a hex color string to RGB values.
///
/// # Arguments
///
/// * `hex` - Hex color string (e.g., "#FF0000" or "FF0000")
///
/// # Returns
///
/// Returns a `Color` on success, or `DitherError::InvalidHexColor` if parsing fails.
///
/// # Examples
///
/// ```
/// use blue_noise::Color;
///
/// let red = Color::from_hex("#FF0000").unwrap();
/// assert_eq!(red.r, 255);
/// ```
pub fn from_hex(hex: &str) -> Result<Self, DitherError> {
    // ...
}
```

### Building Documentation

```bash
# Generate and open documentation
cargo doc --open --no-deps
```

## Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

   Follow commit message conventions:
   - Use present tense ("Add feature" not "Added feature")
   - Keep first line under 72 characters
   - Reference issue numbers when applicable

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Provide a clear description of the changes
   - Reference any related issues
   - Include screenshots for visual changes
   - Ensure CI checks pass

## Style Guidelines

### Rust Code Style

- Follow the [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/)
- Use `cargo fmt` for automatic formatting
- Maximum line length: 100 characters
- Use meaningful variable and function names
- Prefer explicit types over `auto` for public APIs

### Naming Conventions

- **Types**: `PascalCase` (e.g., `BlueNoiseGenerator`)
- **Functions**: `snake_case` (e.g., `generate_blue_noise`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_SIGMA`)
- **Modules**: `snake_case` (e.g., `generator`, `dither`)

### Error Handling

- Use `Result` for operations that can fail
- Use `thiserror` for custom error types
- Provide helpful error messages
- Document error conditions

### Performance

- Prefer `&str` over `String` for function parameters
- Use `Vec::with_capacity` when size is known
- Profile code before optimizing
- Comment on performance-critical sections

## Project Structure

```
blue-noise/
├── src/
│   ├── main.rs         # CLI entry point
│   ├── lib.rs          # Library entry point
│   ├── generator.rs    # Blue noise generation
│   └── dither.rs       # Image dithering
├── examples/           # Usage examples
├── benches/            # Performance benchmarks
├── tests/              # Integration tests
└── .github/
    └── workflows/      # CI/CD pipelines
```

## Areas for Contribution

### Good First Issues

- Add more example color schemes to README
- Improve error messages
- Add more unit tests
- Update documentation

### Feature Ideas

- Support for non-square textures
- Additional output formats (SVG, WebP)
- Multi-level dithering (more than 2 colors)
- GPU acceleration
- Parallel generation for large textures
- Alternative dithering algorithms

### Performance Improvements

- Optimize Gaussian blur for non-power-of-2 sizes
- Parallelize phase iterations
- Memory usage optimizations
- SIMD optimizations

## Questions?

Feel free to open an issue for:
- Questions about the codebase
- Feature requests
- Bug reports
- Documentation improvements

Thank you for contributing to Blue Noise!
