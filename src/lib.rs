//! Blue Noise Library
//!
//! A comprehensive Rust implementation of Robert Ulichney's void-and-cluster
//! algorithm for generating blue noise textures and applying high-quality
//! dithering to images.
//!
//! # Features
//!
//! - Generate blue noise textures using the void-and-cluster algorithm
//! - Apply blue noise dithering to images with customizable colors
//! - FFT-optimized Gaussian blur for power-of-two dimensions
//! - Seamless tiling with toroidal topology
//! - Reproducible results with seeded random number generation
//!
//! # Quick Start
//!
//! ## Generating Blue Noise
//!
//! ```no_run
//! use blue_noise::{BlueNoiseGenerator, BlueNoiseConfig, save_blue_noise_to_png};
//!
//! let config = BlueNoiseConfig {
//!     width: 64,
//!     height: 64,
//!     sigma: 1.9,
//!     seed: Some(42),
//!     verbose: false,
//!     ..Default::default()
//! };
//!
//! let generator = BlueNoiseGenerator::new(config).unwrap();
//! let result = generator.generate().unwrap();
//! save_blue_noise_to_png(&result, "noise.png").unwrap();
//! ```
//!
//! ## Dithering Images
//!
//! ```no_run
//! use blue_noise::{
//!     BlueNoiseTexture,
//!     Color,
//!     DitherOptions,
//!     apply_dithering
//! };
//!
//! let noise = BlueNoiseTexture::load("noise.png").unwrap();
//! let options = DitherOptions {
//!     foreground: Color::from_hex("#000000").unwrap(),
//!     background: Color::from_hex("#ffffff").unwrap(),
//!     width: Some(800),
//!     height: None,
//!     contrast: Some(1.2),
//! };
//!
//! apply_dithering("input.jpg", "output.png", &noise, options).unwrap();
//! ```
//!
//! # Algorithm
//!
//! The void-and-cluster algorithm works in 5 phases:
//!
//! 1. **Phase 0**: Generate initial binary pattern with random points
//! 2. **Phase 1**: Serialize initial points by removing from tightest clusters
//! 3. **Phase 2**: Fill to half capacity by adding to largest voids
//! 4. **Phase 3**: Invert and fill to completion
//! 5. **Phase 4**: Convert ranks to threshold map (0-255)
//!
//! All operations use toroidal topology (wraparound edges) to ensure
//! seamless tiling.
//!
//! # Performance
//!
//! For power-of-two dimensions, Gaussian blur is performed using FFT,
//! providing approximately 50% performance improvement. Generation times:
//!
//! - 64×64: ~2-5 seconds
//! - 128×128: ~30-60 seconds
//! - 256×256: Several minutes
//!
//! # References
//!
//! - Ulichney, R. (1993). "Void-and-cluster method for dither array generation"
//! - Ulichney, R. (1988). "Dithering with blue noise"

#![doc(html_root_url = "https://docs.rs/blue-noise/0.2.0")]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

/// Image dithering module
pub mod dither;
/// Blue noise generation module
pub mod generator;

// Re-export main types for convenience
pub use dither::{
    apply_dithering, ordered_dither, BlueNoiseTexture, Color, DitherError, DitherOptions,
};
pub use generator::{
    generate_blue_noise, save_blue_noise_to_png, BlueNoiseConfig, BlueNoiseGenerator,
    BlueNoiseResult, GeneratorError,
};
