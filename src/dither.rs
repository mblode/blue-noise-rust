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
}
