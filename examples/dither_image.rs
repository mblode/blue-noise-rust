/**
 * Example: Dither an image with blue noise
 *
 * This example shows how to apply blue noise dithering to an image
 * with custom colors and options.
 *
 * Run with:
 *   cargo run --example dither_image
 *
 * Note: This example requires an input image file and a blue noise texture.
 * You can generate a blue noise texture first with the generate_texture example.
 */

use blue_noise::{
    apply_dithering, BlueNoiseConfig, BlueNoiseGenerator, BlueNoiseTexture, Color, DitherOptions,
    save_blue_noise_to_png,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Blue Noise Dithering Example\n");

    // First, generate a blue noise texture if one doesn't exist
    println!("Step 1: Generating blue noise texture...");
    let config = BlueNoiseConfig {
        width: 64,
        height: 64,
        sigma: 1.9,
        seed: Some(42),
        verbose: false,
        ..Default::default()
    };

    let generator = BlueNoiseGenerator::new(config)?;
    let noise_result = generator.generate()?;
    save_blue_noise_to_png(&noise_result, "example-noise.png")?;
    println!("  ✓ Generated 64×64 blue noise texture\n");

    // Create a simple test gradient image
    println!("Step 2: Creating test gradient image...");
    let width = 256;
    let height = 256;
    let mut gradient_data = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            // Create a horizontal gradient from black to white
            let value = (x * 255 / (width - 1)) as u8;
            gradient_data.push(value);
            gradient_data.push(value);
            gradient_data.push(value);
        }
    }

    let gradient_img = image::RgbImage::from_vec(width, height, gradient_data)
        .expect("Failed to create gradient image");
    gradient_img.save("example-gradient.png")?;
    println!("  ✓ Created test gradient image\n");

    // Load the blue noise texture
    println!("Step 3: Loading blue noise texture...");
    let noise_texture = BlueNoiseTexture::load("example-noise.png")?;
    println!("  ✓ Loaded noise texture\n");

    // Apply dithering with black and white
    println!("Step 4: Applying dithering (black & white)...");
    let bw_options = DitherOptions {
        foreground: Color::from_hex("#000000")?,
        background: Color::from_hex("#ffffff")?,
        width: None,
        height: None,
        contrast: None,
    };

    apply_dithering(
        "example-gradient.png",
        "example-dithered-bw.png",
        &noise_texture,
        bw_options,
    )?;
    println!("  ✓ Saved to example-dithered-bw.png\n");

    // Apply dithering with custom colors (sepia)
    println!("Step 5: Applying dithering (sepia tone)...");
    let sepia_options = DitherOptions {
        foreground: Color::from_hex("#704214")?,
        background: Color::from_hex("#f4e8d8")?,
        width: None,
        height: None,
        contrast: Some(1.2),
    };

    apply_dithering(
        "example-gradient.png",
        "example-dithered-sepia.png",
        &noise_texture,
        sepia_options,
    )?;
    println!("  ✓ Saved to example-dithered-sepia.png\n");

    // Apply dithering with blue tones
    println!("Step 6: Applying dithering (blue tone)...");
    let blue_options = DitherOptions {
        foreground: Color::from_hex("#003366")?,
        background: Color::from_hex("#e6f2ff")?,
        width: Some(512), // Resize to 512 pixels wide
        height: None,
        contrast: None,
    };

    apply_dithering(
        "example-gradient.png",
        "example-dithered-blue.png",
        &noise_texture,
        blue_options,
    )?;
    println!("  ✓ Saved to example-dithered-blue.png\n");

    println!("✓ All examples completed!");
    println!("\nGenerated files:");
    println!("  - example-noise.png (blue noise texture)");
    println!("  - example-gradient.png (test gradient)");
    println!("  - example-dithered-bw.png (black & white)");
    println!("  - example-dithered-sepia.png (sepia tone)");
    println!("  - example-dithered-blue.png (blue tone, resized)");

    Ok(())
}
