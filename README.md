# Blue Noise

A Rust library and CLI for generating blue noise textures and dithering images using the void-and-cluster algorithm.

[![Crates.io](https://img.shields.io/crates/v/blue-noise.svg)](https://crates.io/crates/blue-noise)
[![Documentation](https://docs.rs/blue-noise/badge.svg)](https://docs.rs/blue-noise)

## Installation

```bash
cargo install blue-noise
```

Or build from source:

```bash
git clone https://github.com/mblode/blue-noise.git
cd blue-noise
cargo build --release
```

## Usage

### Generate a texture

```bash
blue-noise generate --size 128 --output blue-noise.png
blue-noise generate --width 256 --height 128 --output blue-noise-wide.png
```

Options:
- `--size` - Texture size in pixels (default: 128)
- `--width` - Texture width in pixels (overrides `--size`)
- `--height` - Texture height in pixels (overrides `--size`)
- `--output` - Output file path
- `--sigma` - Gaussian sigma, 1.0-3.0 (default: 1.9)
- `--seed` - Random seed for reproducibility
- `--verbose` - Show progress

### Dither an image

```bash
blue-noise dither -i input.jpg -o output.png
```

Options:
- `--input` - Input image
- `--output` - Output image
- `--noise` - Blue noise texture (default: blue-noise.png)
- `--foreground` - Foreground color as hex (default: #000000)
- `--background` - Background color as hex (default: #ffffff)
- `--width` - Output width
- `--height` - Output height
- `--contrast` - Contrast adjustment (default: 1.0)

## Library

Add to your `Cargo.toml`:

```toml
[dependencies]
blue-noise = "0.2"
```

Generate a texture:

```rust
use blue_noise::{BlueNoiseGenerator, BlueNoiseConfig, save_blue_noise_to_png};

let config = BlueNoiseConfig {
    width: 128,
    height: 128,
    sigma: 1.9,
    seed: Some(42),
    ..Default::default()
};

let generator = BlueNoiseGenerator::new(config)?;
let result = generator.generate()?;
save_blue_noise_to_png(&result, "blue-noise.png")?;
```

Apply dithering:

```rust
use blue_noise::{BlueNoiseTexture, Color, DitherOptions, apply_dithering};

let noise = BlueNoiseTexture::load("blue-noise.png")?;
let options = DitherOptions {
    foreground: Color::from_hex("#000000")?,
    background: Color::from_hex("#ffffff")?,
    width: Some(800),
    height: None,
    contrast: Some(1.2),
};

apply_dithering("input.jpg", "output.png", &noise, options)?;
```

## Examples

![Input dark](/img/dark.png)
![Output dark](/img/dark-noise.jpg)

![Input light](/img/light.png)
![Output light](/img/light-noise.jpg)

## References

- Ulichney, R. (1993). "Void-and-cluster method for dither array generation"
- Ulichney, R. (1988). "Dithering with blue noise"

## License

MIT
