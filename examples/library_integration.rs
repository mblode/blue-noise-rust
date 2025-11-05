/**
 * Example: Using blue-noise as a library
 *
 * This example demonstrates how to integrate blue-noise into your
 * own application as a library, including in-memory operations
 * without file I/O.
 *
 * Run with:
 *   cargo run --example library_integration
 */

use blue_noise::{
    BlueNoiseConfig, BlueNoiseGenerator, BlueNoiseTexture, Color, DitherOptions,
};
use image::{GrayImage, ImageBuffer, Luma, RgbImage};

fn create_test_image() -> RgbImage {
    // Create a 200×200 image with a radial gradient
    let size = 200;
    let mut img = RgbImage::new(size, size);

    let center_x = size as f32 / 2.0;
    let center_y = size as f32 / 2.0;
    let max_dist = ((center_x * center_x + center_y * center_y) as f32).sqrt();

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let value = ((1.0 - dist / max_dist) * 255.0) as u8;

            img.put_pixel(x, y, image::Rgb([value, value, value]));
        }
    }

    img
}

fn dither_in_memory(
    input: &RgbImage,
    noise_data: &[u8],
    noise_width: usize,
    noise_height: usize,
    fg: Color,
    bg: Color,
) -> RgbImage {
    let (width, height) = input.dimensions();
    let mut output = RgbImage::new(width, height);

    // Convert input to grayscale
    let gray: GrayImage = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = input.get_pixel(x, y);
        let avg = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
        Luma([avg as u8])
    });

    // Apply dithering
    for y in 0..height {
        for x in 0..width {
            let pixel_luma = gray.get_pixel(x, y)[0];

            // Get noise value with tiling
            let noise_x = (x as usize) % noise_width;
            let noise_y = (y as usize) % noise_height;
            let noise_luma = noise_data[noise_y * noise_width + noise_x];

            // Choose color based on threshold
            let color = if pixel_luma > noise_luma {
                bg
            } else {
                fg
            };

            output.put_pixel(x, y, image::Rgb([color.r, color.g, color.b]));
        }
    }

    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Blue Noise Library Integration Example\n");

    // Step 1: Generate blue noise in memory
    println!("1. Generating blue noise texture in memory...");
    let config = BlueNoiseConfig {
        width: 64,
        height: 64,
        sigma: 1.9,
        seed: Some(12345),
        verbose: false,
        ..Default::default()
    };

    let generator = BlueNoiseGenerator::new(config)?;
    let noise_result = generator.generate()?;
    println!("   ✓ Generated {}×{} texture", noise_result.width, noise_result.height);

    // Step 2: Create test image in memory
    println!("\n2. Creating test image (radial gradient)...");
    let test_image = create_test_image();
    println!("   ✓ Created {}×{} test image", test_image.width(), test_image.height());

    // Step 3: Perform in-memory dithering
    println!("\n3. Applying in-memory dithering...");
    let fg = Color::from_hex("#1a1a1a")?;
    let bg = Color::from_hex("#f5f5f5")?;

    let dithered = dither_in_memory(
        &test_image,
        &noise_result.data,
        noise_result.width,
        noise_result.height,
        fg,
        bg,
    );
    println!("   ✓ Dithering complete");

    // Step 4: Save results
    println!("\n4. Saving results...");
    test_image.save("example-radial-original.png")?;
    println!("   ✓ Saved original: example-radial-original.png");

    dithered.save("example-radial-dithered.png")?;
    println!("   ✓ Saved dithered: example-radial-dithered.png");

    // Step 5: Generate multiple variations with different seeds
    println!("\n5. Generating variations with different seeds...");
    for seed in [111, 222, 333] {
        let config = BlueNoiseConfig {
            width: 64,
            height: 64,
            seed: Some(seed),
            verbose: false,
            ..Default::default()
        };

        let generator = BlueNoiseGenerator::new(config)?;
        let result = generator.generate()?;

        let dithered = dither_in_memory(
            &test_image,
            &result.data,
            result.width,
            result.height,
            fg,
            bg,
        );

        let filename = format!("example-radial-seed-{}.png", seed);
        dithered.save(&filename)?;
        println!("   ✓ Saved variation: {}", filename);
    }

    println!("\n✓ Library integration example complete!");
    println!("\nThis example showed:");
    println!("  - Generating blue noise in memory (no file I/O)");
    println!("  - Custom in-memory dithering function");
    println!("  - Creating test images programmatically");
    println!("  - Batch processing with different seeds");
    println!("  - Integration into your own image processing pipeline");

    Ok(())
}
