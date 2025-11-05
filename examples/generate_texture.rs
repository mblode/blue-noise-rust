/**
 * Example: Generate a blue noise texture
 *
 * This example demonstrates how to generate a blue noise texture
 * with custom parameters and save it to a PNG file.
 *
 * Run with:
 *   cargo run --example generate_texture
 */

use blue_noise::{BlueNoiseConfig, BlueNoiseGenerator, save_blue_noise_to_png};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating blue noise texture...\n");

    // Create configuration
    let config = BlueNoiseConfig {
        width: 64,
        height: 64,
        sigma: 1.9,
        seed: Some(42), // Use a seed for reproducibility
        verbose: true,
        ..Default::default()
    };

    // Create generator and generate texture
    let generator = BlueNoiseGenerator::new(config)?;
    let result = generator.generate()?;

    // Save to file
    save_blue_noise_to_png(&result, "example-noise-64.png")?;

    println!("\nTexture saved to example-noise-64.png");
    println!("Size: {}×{} pixels", result.width, result.height);
    println!("Data points: {}", result.data.len());

    Ok(())
}
