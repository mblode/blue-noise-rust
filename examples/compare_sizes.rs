/**
 * Example: Compare different blue noise texture sizes
 *
 * This example generates blue noise textures of different sizes
 * to demonstrate the trade-off between quality and generation time.
 *
 * Run with:
 *   cargo run --example compare_sizes --release
 *
 * Note: Use --release mode for faster generation!
 */

use blue_noise::{BlueNoiseConfig, BlueNoiseGenerator, save_blue_noise_to_png};
use std::time::Instant;

fn generate_and_time(size: usize, seed: u32) -> Result<f64, Box<dyn std::error::Error>> {
    let config = BlueNoiseConfig {
        width: size,
        height: size,
        sigma: 1.9,
        seed: Some(seed),
        verbose: false,
        ..Default::default()
    };

    let start = Instant::now();
    let generator = BlueNoiseGenerator::new(config)?;
    let result = generator.generate()?;
    let elapsed = start.elapsed().as_secs_f64();

    let filename = format!("example-noise-{}x{}.png", size, size);
    save_blue_noise_to_png(&result, &filename)?;

    Ok(elapsed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Blue Noise Size Comparison\n");
    println!("Generating textures of different sizes...\n");
    println!("{:<12} {:<15} {:<15} {:<15}", "Size", "FFT Optimized", "Time (sec)", "Output File");
    println!("{}", "-".repeat(65));

    let sizes: Vec<usize> = vec![16, 32, 64, 128];
    let seed = 42;

    for size in sizes {
        let uses_fft = size.is_power_of_two();
        let elapsed = generate_and_time(size, seed)?;

        println!(
            "{:<12} {:<15} {:<15.2} example-noise-{}x{}.png",
            format!("{}×{}", size, size),
            if uses_fft { "Yes" } else { "No" },
            elapsed,
            size,
            size
        );
    }

    println!("\n✓ All textures generated!");
    println!("\nKey observations:");
    println!("  - Smaller textures generate faster");
    println!("  - Power-of-2 sizes use FFT optimization (~50% faster)");
    println!("  - 64×64 is a good balance for most uses");
    println!("  - Larger textures provide better quality but take longer");

    Ok(())
}
