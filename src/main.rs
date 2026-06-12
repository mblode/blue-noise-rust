/**
 * Blue Noise CLI - Modern command-line interface for blue noise generation and dithering
 */
mod dither;
mod generator;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use dither::{BlueNoiseTexture, Color, DitherOptions, apply_dithering};
use generator::{BlueNoiseConfig, BlueNoiseGenerator, save_blue_noise_to_png};

/// Blue noise generation and dithering tools
#[derive(Parser)]
#[command(name = "blue-noise")]
#[command(author = "Matthew Blode <m@blode.co>")]
#[command(version)]
#[command(about = "Blue noise dithering and generation tools", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a blue noise texture using the void-and-cluster algorithm
    Generate {
        /// Output file path
        #[arg(short, long, default_value = "blue-noise.png")]
        output: PathBuf,

        /// Default square texture size
        #[arg(short, long, default_value = "128")]
        size: usize,

        /// Texture width in pixels (overrides --size)
        #[arg(long)]
        width: Option<usize>,

        /// Texture height in pixels (overrides --size)
        #[arg(long)]
        height: Option<usize>,

        /// Gaussian sigma value (1.5-2.5, higher = more spread)
        #[arg(long, default_value = "1.9")]
        sigma: f32,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u32>,

        /// Show detailed generation progress
        #[arg(short, long)]
        verbose: bool,
    },

    /// Apply blue noise dithering to an image
    Dither {
        /// Input image path
        #[arg(short, long)]
        input: PathBuf,

        /// Output image path
        #[arg(short, long)]
        output: PathBuf,

        /// Path to blue noise texture
        #[arg(short, long, default_value = "blue-noise.png")]
        noise: PathBuf,

        /// Foreground color (hex)
        #[arg(short, long, default_value = "#000000")]
        foreground: String,

        /// Background color (hex)
        #[arg(short, long, default_value = "#ffffff")]
        background: String,

        /// Output width in pixels (maintains aspect ratio if height not specified)
        #[arg(short, long)]
        width: Option<u32>,

        /// Output height in pixels (maintains aspect ratio if width not specified)
        #[arg(long)]
        height: Option<u32>,

        /// Contrast adjustment (1.0 = normal, >1 = more contrast, <1 = less)
        #[arg(short, long)]
        contrast: Option<f32>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            output,
            size,
            width,
            height,
            sigma,
            seed,
            verbose,
        } => {
            let width = width.unwrap_or(size);
            let height = height.unwrap_or(size);

            // Validate inputs
            if !(8..=512).contains(&size) {
                anyhow::bail!("Size must be between 8 and 512");
            }
            if !(8..=512).contains(&width) {
                anyhow::bail!("Width must be between 8 and 512");
            }
            if !(8..=512).contains(&height) {
                anyhow::bail!("Height must be between 8 and 512");
            }
            if !(1.0..=3.0).contains(&sigma) {
                anyhow::bail!("Sigma must be between 1.0 and 3.0");
            }

            if !verbose {
                println!("Generating {}×{} blue noise texture", width, height);
                println!("Sigma: {}", sigma);
                if let Some(s) = seed {
                    println!("Seed: {}", s);
                }
                println!("Output: {}", output.display());
                println!();
            }

            // Create output directory if it doesn't exist
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent).context("Failed to create output directory")?;
            }

            // Generate blue noise
            let config = BlueNoiseConfig {
                width,
                height,
                sigma,
                seed,
                verbose,
                ..Default::default()
            };

            let generator =
                BlueNoiseGenerator::new(config).context("Failed to create generator")?;
            let result = generator
                .generate()
                .context("Failed to generate blue noise")?;

            // Save to file
            save_blue_noise_to_png(&result, &output)
                .context("Failed to save blue noise texture")?;

            println!();
            println!("Done!");
        }

        Commands::Dither {
            input,
            output,
            noise,
            foreground,
            background,
            width,
            height,
            contrast,
        } => {
            // Validate inputs
            if !input.exists() {
                anyhow::bail!("Input file does not exist: {}", input.display());
            }
            if !noise.exists() {
                anyhow::bail!("Noise texture does not exist: {}", noise.display());
            }

            // Parse colors
            let fg = Color::from_hex(&foreground).context("Failed to parse foreground color")?;
            let bg = Color::from_hex(&background).context("Failed to parse background color")?;

            // Validate contrast
            if let Some(c) = contrast
                && c <= 0.0
            {
                anyhow::bail!("Contrast must be positive");
            }

            println!("Processing: {}", input.display());
            println!("Noise texture: {}", noise.display());
            println!("Output: {}", output.display());
            if let (Some(w), Some(h)) = (width, height) {
                println!("Dimensions: {}×{}", w, h);
            } else if let Some(w) = width {
                println!("Width: {} (maintaining aspect ratio)", w);
            } else if let Some(h) = height {
                println!("Height: {} (maintaining aspect ratio)", h);
            }
            if let Some(c) = contrast {
                println!("Contrast: {}", c);
            }
            println!("Foreground: {}", foreground);
            println!("Background: {}", background);
            println!();

            // Create output directory if it doesn't exist
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent).context("Failed to create output directory")?;
            }

            // Load noise texture
            let noise_texture =
                BlueNoiseTexture::load(&noise).context("Failed to load blue noise texture")?;

            // Apply dithering
            let options = DitherOptions {
                foreground: fg,
                background: bg,
                width,
                height,
                contrast,
            };

            apply_dithering(&input, &output, &noise_texture, options)
                .context("Failed to apply dithering")?;

            println!("Dithered image saved to: {}", output.display());
            println!();
            println!("Done!");
        }
    }

    Ok(())
}
