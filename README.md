<h1 align="center">Blue Noise</h1>
<div align="center">
 <strong>
   High-quality blue noise generation and dithering using the void-and-cluster algorithm
 </strong>
</div>

<br />

<div align="center">
  <!-- Crates version -->
  <a href="https://crates.io/crates/blue-noise">
    <img src="https://img.shields.io/crates/v/blue-noise.svg?style=flat-square"
    alt="Crates.io version" />
  </a>
  <!-- Downloads -->
  <a href="https://crates.io/crates/blue-noise">
    <img src="https://img.shields.io/crates/d/blue-noise.svg?style=flat-square"
      alt="Download" />
  </a>
  <!-- docs.rs docs -->
  <a href="https://docs.rs/blue-noise">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square"
      alt="docs.rs docs" />
  </a>
</div>

<div align="center">
  <h3>
    <a href="https://docs.rs/blue-noise">
      API Docs
    </a>
    <span> | </span>
    <a href="https://github.com/mblode/blue-noise/blob/main/CONTRIBUTING.md">
      Contributing
    </a>
    <span> | </span>
    <a href="https://github.com/mblode/blue-noise">
      Repo
    </a>
  </h3>
</div>

A comprehensive Rust implementation of Robert Ulichney's **void-and-cluster algorithm** for generating blue noise textures and applying high-quality dithering to images. Blue noise produces evenly distributed, visually pleasing dithered images without the clustering artifacts of white noise or the repetitive patterns of Bayer dithering.

## ✨ Features

- 🎨 **Generate blue noise textures** using the void-and-cluster algorithm
- 🖼️ **Dither images** with customizable colors and contrast
- ⚡ **FFT-optimized** Gaussian blur for power-of-two dimensions (~50% faster)
- 🔄 **Seamless tiling** - all textures use toroidal topology
- 🎲 **Reproducible** - seeded random number generation
- 📊 **Progress indicators** for long-running operations
- 🦀 **Pure Rust** - fast, safe, and dependency-minimal

## 📦 Installation

### From crates.io

```bash
cargo install blue-noise
```

### From source

```bash
git clone https://github.com/mblode/blue-noise.git
cd blue-noise
cargo build --release
```

The binary will be in `target/release/blue-noise`.

## 🚀 Usage

### CLI Tool

#### Generate Blue Noise Textures

```bash
# Generate a 128x128 blue noise texture
blue-noise generate --size 128 --output blue-noise.png

# Generate with custom sigma and seed for reproducibility
blue-noise generate --size 64 --sigma 1.9 --seed 42 --verbose

# Quick 64x64 texture (2-5 seconds)
blue-noise generate -s 64 -o noise-64.png

# Larger 256x256 texture (several minutes)
blue-noise generate -s 256 -o noise-256.png --verbose
```

**Options:**
- `-s, --size <SIZE>` - Texture size (8-512, default: 128)
- `-o, --output <PATH>` - Output file path (default: blue-noise.png)
- `--sigma <SIGMA>` - Gaussian sigma (1.0-3.0, default: 1.9)
- `--seed <SEED>` - Random seed for reproducibility
- `-v, --verbose` - Show detailed progress

#### Dither Images

```bash
# Basic dithering
blue-noise dither -i input.jpg -o output.png

# Custom colors
blue-noise dither -i photo.jpg -o dithered.png \
  --foreground "#000000" \
  --background "#ffffff"

# Resize and adjust contrast
blue-noise dither -i image.png -o output.png \
  --width 800 \
  --contrast 1.2

# Use custom noise texture
blue-noise dither -i input.jpg -o output.png \
  --noise custom-noise.png \
  --foreground "#1a1a1a" \
  --background "#f0f0f0"
```

**Options:**
- `-i, --input <PATH>` - Input image path (required)
- `-o, --output <PATH>` - Output image path (required)
- `-n, --noise <PATH>` - Blue noise texture (default: blue-noise.png)
- `-f, --foreground <HEX>` - Foreground color (default: #000000)
- `-b, --background <HEX>` - Background color (default: #ffffff)
- `-w, --width <PIXELS>` - Output width (maintains aspect ratio)
- `--height <PIXELS>` - Output height (maintains aspect ratio)
- `-c, --contrast <FLOAT>` - Contrast adjustment (1.0 = normal)

### Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
blue-noise = "0.2"
```

#### Generate Blue Noise

```rust
use blue_noise::{BlueNoiseGenerator, BlueNoiseConfig, save_blue_noise_to_png};

// Create configuration
let config = BlueNoiseConfig {
    width: 128,
    height: 128,
    sigma: 1.9,
    seed: Some(42),
    verbose: true,
    ..Default::default()
};

// Generate texture
let generator = BlueNoiseGenerator::new(config)?;
let result = generator.generate()?;

// Save to file
save_blue_noise_to_png(&result, "blue-noise.png")?;
```

#### Apply Dithering

```rust
use blue_noise::{
    BlueNoiseTexture,
    Color,
    DitherOptions,
    apply_dithering
};

// Load blue noise texture
let noise = BlueNoiseTexture::load("blue-noise.png")?;

// Configure dithering
let options = DitherOptions {
    foreground: Color::from_hex("#000000")?,
    background: Color::from_hex("#ffffff")?,
    width: Some(800),
    height: None,
    contrast: Some(1.2),
};

// Apply dithering
apply_dithering("input.jpg", "output.png", &noise, options)?;
```

## 🎨 Examples

### Before and After

![Input dark](/img/dark.png)
![Output dark](/img/dark-noise.jpg)

-----

![Input light](/img/light.png)
![Output light](/img/light-noise.jpg)

### Different Colors

```bash
# Sepia tone
blue-noise dither -i photo.jpg -o sepia.png \
  --foreground "#704214" --background "#f4e8d8"

# Blue on white
blue-noise dither -i image.jpg -o blue.png \
  --foreground "#0066cc" --background "#ffffff"

# Green matrix style
blue-noise dither -i code.jpg -o matrix.png \
  --foreground "#00ff00" --background "#000000"
```

## 🔬 Algorithm

This implementation uses **Ulichney's void-and-cluster algorithm**, which produces high-quality blue noise with evenly distributed energy at high frequencies and minimal low-frequency content.

### Why Blue Noise?

- **White noise**: Random distribution creates visible clusters and voids
- **Bayer dithering**: Regular patterns create repetitive artifacts
- **Blue noise**: Evenly distributed with minimal low-frequency patterns ✓

### Generation Process

1. **Phase 0**: Generate initial binary pattern with random points
2. **Phase 1**: Serialize initial points by removing from tightest clusters
3. **Phase 2**: Fill to half capacity by adding to largest voids
4. **Phase 3**: Invert and fill to completion
5. **Phase 4**: Convert ranks to threshold map (0-255)

The algorithm uses **toroidal topology** (wraparound edges) to ensure seamless tiling, making it perfect for dithering large images with small noise textures.

### Performance

For power-of-two dimensions, Gaussian blur is performed in the **frequency domain using FFT**, providing ~50% performance improvement through the convolution theorem:

```
convolution(A, B) = IFFT(FFT(A) × FFT(B))
```

**Generation times:**
- 64×64: ~2-5 seconds
- 128×128: ~30-60 seconds
- 256×256: Several minutes

**Tip**: Pre-generate textures rather than generating at runtime.

## 📚 References

- Ulichney, R. (1993). "Void-and-cluster method for dither array generation"
  *Proceedings of SPIE 1913, Human Vision, Visual Processing, and Digital Display IV*
  https://doi.org/10.1117/12.152707

- Ulichney, R. (1988). "Dithering with blue noise"
  *Proceedings of the IEEE, 76(1), 56-79*

## 🛠️ Development

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Robert Ulichney for the void-and-cluster algorithm
- [Surma's Ditherpunk](https://surma.dev/things/ditherpunk/) for inspiration
- [TIGSource discussion](https://forums.tigsource.com/index.php?topic=40832.msg1363742#msg1363742)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
