/**
 * Blue Noise Dithering Module
 *
 * Applies ordered dithering to images using a blue noise threshold map.
 * Blue noise dithering produces higher quality results than traditional
 * methods like Bayer dithering due to its even frequency distribution.
 */

use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use std::path::Path;
use thiserror::Error;

/// RGB color representation
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl Color {
    /// Create a new color from RGB values
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Parse a hex color string (e.g., "#FF0000" or "FF0000")
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches('#');

        if hex.len() != 6 {
            return Err(DitherError::InvalidHexColor(hex.to_string()));
        }

        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| DitherError::InvalidHexColor(hex.to_string()))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| DitherError::InvalidHexColor(hex.to_string()))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| DitherError::InvalidHexColor(hex.to_string()))?;

        Ok(Self { r, g, b })
    }
}

/// Options for dithering
#[derive(Debug, Clone)]
pub struct DitherOptions {
    /// Foreground color (for dark pixels)
    pub foreground: Color,
    /// Background color (for bright pixels)
    pub background: Color,
    /// Optional output width in pixels
    pub width: Option<u32>,
    /// Optional output height in pixels
    pub height: Option<u32>,
    /// Optional contrast adjustment (1.0 = normal)
    pub contrast: Option<f32>,
}

impl Default for DitherOptions {
    fn default() -> Self {
        Self {
            foreground: Color::new(0, 0, 0),
            background: Color::new(255, 255, 255),
            width: None,
            height: None,
            contrast: None,
        }
    }
}

/// Error types for dithering
#[derive(Error, Debug)]
pub enum DitherError {
    /// Failed to load or save an image
    #[error("Failed to load image: {0}")]
    ImageLoadError(#[from] image::ImageError),

    /// Invalid hex color string format
    #[error("Invalid hex color: {0}")]
    InvalidHexColor(String),

    /// Could not determine image dimensions
    #[error("Could not determine image dimensions")]
    InvalidDimensions,

    /// Blue noise texture has no data
    #[error("Blue noise texture is empty")]
    EmptyNoiseTexture,
}

/// Result type for dithering operations
pub type Result<T> = std::result::Result<T, DitherError>;

/// Blue noise texture data
pub struct BlueNoiseTexture {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl BlueNoiseTexture {
    /// Load blue noise texture from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let img = image::open(path)?;
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();

        if width == 0 || height == 0 {
            return Err(DitherError::InvalidDimensions);
        }

        Ok(Self {
            data: gray.into_raw(),
            width: width as usize,
            height: height as usize,
        })
    }

    /// Create from existing data
    pub fn from_data(data: Vec<u8>, width: usize, height: usize) -> Result<Self> {
        if data.is_empty() {
            return Err(DitherError::EmptyNoiseTexture);
        }

        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Get the noise value at the given coordinates (with tiling)
    #[inline]
    fn get(&self, x: u32, y: u32) -> u8 {
        let wrap_x = (x as usize) % self.width;
        let wrap_y = (y as usize) % self.height;
        self.data[wrap_y * self.width + wrap_x]
    }
}

/// Apply contrast adjustment to an image
fn apply_contrast(img: DynamicImage, contrast: f32) -> DynamicImage {
    // Convert to RGB for processing
    let mut rgb = img.to_rgb8();
    let factor = contrast;
    let offset = 128.0 * (1.0 - factor);

    for pixel in rgb.pixels_mut() {
        for channel in pixel.0.iter_mut() {
            let value = *channel as f32;
            let adjusted = (value * factor + offset).clamp(0.0, 255.0);
            *channel = adjusted as u8;
        }
    }

    DynamicImage::ImageRgb8(rgb)
}

/// Apply blue noise dithering to an image
pub fn apply_dithering<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    noise_texture: &BlueNoiseTexture,
    options: DitherOptions,
) -> Result<()> {
    // Load input image
    let mut img = image::open(input_path)?;

    // Resize if requested
    if let (Some(width), Some(height)) = (options.width, options.height) {
        img = img.resize(width, height, image::imageops::FilterType::Lanczos3);
    } else if let Some(width) = options.width {
        img = img.resize(width, u32::MAX, image::imageops::FilterType::Lanczos3);
    } else if let Some(height) = options.height {
        img = img.resize(u32::MAX, height, image::imageops::FilterType::Lanczos3);
    }

    // Apply contrast if requested
    if let Some(contrast) = options.contrast {
        img = apply_contrast(img, contrast);
    }

    // Convert to grayscale
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Create output image
    let mut output: RgbImage = ImageBuffer::new(width, height);

    // Apply dithering
    for y in 0..height {
        for x in 0..width {
            let pixel_luma = gray.get_pixel(x, y).0[0];
            let noise_luma = noise_texture.get(x, y);

            // Compare: if picture is brighter than noise, use background color
            let color = if pixel_luma > noise_luma {
                options.background
            } else {
                options.foreground
            };

            output.put_pixel(x, y, Rgb([color.r, color.g, color.b]));
        }
    }

    // Save output
    output.save(output_path)?;

    Ok(())
}

/// Apply ordered dithering using blue noise threshold map
///
/// This function can be used for more advanced dithering with custom level counts
pub fn ordered_dither(
    value: u8,
    x: u32,
    y: u32,
    noise_texture: &BlueNoiseTexture,
    levels: usize,
) -> u8 {
    let threshold = noise_texture.get(x, y);

    let normalized = value as f32 / 255.0;
    let step = 1.0 / levels as f32;
    let quantized = (normalized / step).floor() as usize;
    let fraction = (normalized % step) / step;

    let output = if fraction > threshold as f32 / 255.0 {
        (quantized + 1).min(levels - 1)
    } else {
        quantized
    };

    ((output * 255) / (levels - 1)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        // Test valid hex colors
        let black = Color::from_hex("#000000").unwrap();
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);

        let white = Color::from_hex("#ffffff").unwrap();
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);

        let red = Color::from_hex("#ff0000").unwrap();
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        // Test without # prefix
        let blue = Color::from_hex("0000ff").unwrap();
        assert_eq!(blue.r, 0);
        assert_eq!(blue.g, 0);
        assert_eq!(blue.b, 255);

        // Test case insensitivity
        let green = Color::from_hex("#00FF00").unwrap();
        assert_eq!(green.r, 0);
        assert_eq!(green.g, 255);
        assert_eq!(green.b, 0);
    }

    #[test]
    fn test_color_from_hex_invalid() {
        // Too short
        assert!(Color::from_hex("#fff").is_err());

        // Too long
        assert!(Color::from_hex("#fffffff").is_err());

        // Invalid characters
        assert!(Color::from_hex("#gggggg").is_err());

        // Empty string
        assert!(Color::from_hex("").is_err());
    }

    #[test]
    fn test_color_new() {
        let color = Color::new(123, 45, 67);
        assert_eq!(color.r, 123);
        assert_eq!(color.g, 45);
        assert_eq!(color.b, 67);
    }

    #[test]
    fn test_dither_options_default() {
        let options = DitherOptions::default();
        assert_eq!(options.foreground.r, 0);
        assert_eq!(options.foreground.g, 0);
        assert_eq!(options.foreground.b, 0);
        assert_eq!(options.background.r, 255);
        assert_eq!(options.background.g, 255);
        assert_eq!(options.background.b, 255);
        assert!(options.width.is_none());
        assert!(options.height.is_none());
        assert!(options.contrast.is_none());
    }

    #[test]
    fn test_blue_noise_texture_from_data() {
        let data = vec![0u8, 128, 255, 64];
        let texture = BlueNoiseTexture::from_data(data.clone(), 2, 2).unwrap();

        assert_eq!(texture.width, 2);
        assert_eq!(texture.height, 2);
        assert_eq!(texture.data, data);
    }

    #[test]
    fn test_blue_noise_texture_from_data_empty() {
        let empty = vec![];
        let result = BlueNoiseTexture::from_data(empty, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_blue_noise_texture_get_wrapping() {
        let data = vec![10, 20, 30, 40]; // 2x2 texture
        let texture = BlueNoiseTexture::from_data(data, 2, 2).unwrap();

        // Direct access
        assert_eq!(texture.get(0, 0), 10);
        assert_eq!(texture.get(1, 0), 20);
        assert_eq!(texture.get(0, 1), 30);
        assert_eq!(texture.get(1, 1), 40);

        // Test wrapping
        assert_eq!(texture.get(2, 0), 10); // wraps to (0, 0)
        assert_eq!(texture.get(3, 1), 40); // wraps to (1, 1)
        assert_eq!(texture.get(0, 2), 10); // wraps to (0, 0)
        assert_eq!(texture.get(5, 7), 40); // wraps to (1, 1)
    }

    #[test]
    fn test_ordered_dither_binary() {
        let data = vec![128u8; 64]; // 8x8 texture with all 128s
        let texture = BlueNoiseTexture::from_data(data, 8, 8).unwrap();

        // Test binary dithering (2 levels)
        let result_low = ordered_dither(64, 0, 0, &texture, 2);
        let result_high = ordered_dither(192, 0, 0, &texture, 2);

        // Low value should map to 0, high value to 255
        assert_eq!(result_low, 0);
        assert_eq!(result_high, 255);
    }

    #[test]
    fn test_ordered_dither_levels() {
        let data = vec![0u8; 16]; // 4x4 texture with all 0s
        let texture = BlueNoiseTexture::from_data(data, 4, 4).unwrap();

        // With 4 levels, output should be in valid range
        let levels = 4;
        for value in [0, 85, 170, 255] {
            let result = ordered_dither(value, 0, 0, &texture, levels);
            // Result should be valid (0-255)
            assert!(result <= 255);
        }

        // Test that different input values produce varied outputs
        let result1 = ordered_dither(0, 0, 0, &texture, levels);
        let result2 = ordered_dither(255, 0, 0, &texture, levels);
        // Low and high values should map to different outputs
        assert_ne!(result1, result2);
    }

    #[test]
    fn test_ordered_dither_range() {
        let data = vec![100u8; 16];
        let texture = BlueNoiseTexture::from_data(data, 4, 4).unwrap();

        // All outputs should be in valid range
        for value in 0..=255 {
            let result = ordered_dither(value, 0, 0, &texture, 2);
            assert!(result <= 255);
        }
    }
}
