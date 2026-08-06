<div align="center">

# Blue Noise

**Black and white image dithering using blue noise, plus a void-and-cluster generator for the noise itself**

Turn a photograph into two colors without the crosshatch of an ordered dither.

<p align="center">
  <a href="https://crates.io/crates/blue-noise">
    <img src="https://img.shields.io/crates/v/blue-noise?style=flat&colorA=000000&colorB=000000" />
  </a>
  <a href="https://github.com/mblode/blue-noise-rust/blob/main/LICENSE.md">
    <img src="https://img.shields.io/github/license/mblode/blue-noise-rust?style=flat&colorA=000000&colorB=000000" />
  </a>
</p>

</div>

<p align="center">
  <img alt="Source photograph" src="img/dark.png" width="320" />
  <img alt="The same photograph dithered with blue noise" src="img/dark-noise.jpg" width="320" />
</p>

## Install

```bash
cargo add blue-noise
```

## Quickstart

```rust
use blue_noise::{
    BlueNoiseConfig, BlueNoiseGenerator, BlueNoiseTexture, Color, DitherOptions, apply_dithering,
    save_blue_noise_to_png,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BlueNoiseConfig {
        width: 64,
        height: 64,
        seed: Some(42),
        ..Default::default()
    };
    let result = BlueNoiseGenerator::new(config)?.generate()?;
    save_blue_noise_to_png(&result, "blue-noise.png")?;

    let noise = BlueNoiseTexture::load("blue-noise.png")?;
    apply_dithering(
        "photo.jpg",
        "photo-dithered.png",
        &noise,
        DitherOptions {
            foreground: Color::from_hex("#1447e5")?,
            contrast: Some(1.2),
            ..Default::default()
        },
    )?;

    Ok(())
}
```

`apply_dithering_to_image` takes and returns an in-memory image if you would rather not touch the filesystem.

## CLI

```bash
cargo install blue-noise
```

```bash
# Write a 128 by 128 tileable texture to blue-noise.png
blue-noise generate --size 128 --verbose

# Threshold a photo against it
blue-noise dither -i photo.jpg -o photo-dithered.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma <f32>` | `1.9` | Gaussian sigma, higher spreads points further apart |
| `--seed <u32>` | | Seed for a reproducible texture |
| `--noise <path>` | `blue-noise.png` | Texture the dither thresholds against |
| `--contrast <f32>` | | Contrast adjustment, above 1 for more |

`--width` and `--height` override `--size` for a non-square texture, and resize the output when dithering. Full lists are in `blue-noise generate --help` and `blue-noise dither --help`.

## Notes

- Distances wrap at the edges, so a texture tiles seamlessly across an image of any size.
- Power-of-two sizes run their Gaussian blur through an FFT, roughly halving generation time. Generate once and reuse the file.
- Uses the void-and-cluster algorithm from [Ulichney (1993)](https://doi.org/10.1117/12.152707), building on [Ulichney (1988)](https://doi.org/10.1109/5.3288).
- [blue-noise-typescript](https://github.com/mblode/blue-noise-typescript) is the same dithering as a Node CLI, published on npm.
- API documentation is on [docs.rs](https://docs.rs/blue-noise).

## License

MIT

---

Crafted by [<img src="https://blode.co/avatar-circle.png" width="20" align="top" />](https://blode.co) [Matthew Blode](https://blode.co)
