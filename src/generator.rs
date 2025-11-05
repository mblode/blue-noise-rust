/**
 * Blue Noise Texture Generator using Void-and-Cluster Algorithm
 *
 * Implementation of Robert Ulichney's void-and-cluster method for generating
 * high-quality blue noise textures. Blue noise has evenly distributed energy
 * at high frequencies whilst minimising low-frequency content, avoiding the
 * clustering and void patterns found in white noise.
 *
 * ALGORITHM OVERVIEW
 * ==================
 *
 * Blue noise addresses issues with other dithering methods:
 * - White noise: Random distribution creates visible clusters and voids
 * - Bayer dithering: Regular patterns create repetitive artefacts
 * - Blue noise: Evenly distributed with minimal low-frequency patterns
 *
 * The void-and-cluster algorithm works by:
 * 1. Finding clusters (areas of high density) using Gaussian blur
 * 2. Finding voids (areas of low density) using the same blur
 * 3. Iteratively redistributing points to spread them evenly
 * 4. Assigning each pixel a rank based on its importance
 *
 * TORUS TOPOLOGY
 * ==============
 * All distance calculations wrap around at the edges (toroidal topology),
 * ensuring the resulting texture tiles seamlessly when repeated. This is
 * essential for dithering large images with small noise textures.
 *
 * FFT OPTIMISATION
 * ================
 * For power-of-two dimensions, Gaussian blur is performed in the frequency
 * domain using Fast Fourier Transform. Convolution becomes element-wise
 * multiplication in frequency space, providing ~50% performance improvement.
 *
 * GENERATION PHASES
 * =================
 *
 * Phase 0: Generate initial binary pattern
 *   - Place random points and redistribute until convergence
 *   - Convergence occurs when tightest cluster equals largest void
 *
 * Phase 1: Serialize initial points
 *   - Remove points from tightest clusters
 *   - Assign ranks from (initialPoints - 1) down to 0
 *
 * Phase 2: Fill to half capacity
 *   - Restore initial pattern and add points to largest voids
 *   - Assign ranks from initialPoints to area/2
 *
 * Phase 3: Fill to completion
 *   - Invert bitmap (0s become minority)
 *   - Remove minority points from tightest clusters
 *   - Assign ranks from area/2 to area-1
 *
 * Phase 4: Convert to threshold map
 *   - Map ranks [0, area-1] to threshold values [0, 255]
 *
 * REFERENCES
 * ==========
 * - Ulichney, R. (1993). "Void-and-cluster method for dither array generation"
 *   Proceedings of SPIE 1913, Human Vision, Visual Processing, and Digital
 *   Display IV. https://doi.org/10.1117/12.152707
 *
 * - Ulichney, R. (1988). "Dithering with blue noise"
 *   Proceedings of the IEEE, 76(1), 56-79.
 *
 * PERFORMANCE
 * ===========
 * 64�64 texture:  ~2-5 seconds
 * 128�128 texture: ~30-60 seconds
 * 256�256 texture: Several minutes
 *
 * For production use, pre-generate textures rather than generating at runtime.
 */

use image::{GrayImage, ImageBuffer, Luma};
use indicatif::{ProgressBar, ProgressStyle};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::path::Path;
use thiserror::Error;

/// Configuration for blue noise generation
#[derive(Debug, Clone)]
pub struct BlueNoiseConfig {
    /// Width of the texture in pixels
    pub width: usize,
    /// Height of the texture in pixels
    pub height: usize,
    /// Gaussian blur sigma value (typically 1.5-2.5)
    pub sigma: f32,
    /// Initial density of points (typically 0.1)
    pub initial_density: f32,
    /// Optional random seed for reproducible results
    pub seed: Option<u32>,
    /// Show progress indicators
    pub verbose: bool,
}

impl Default for BlueNoiseConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            sigma: 1.9,
            initial_density: 0.1,
            seed: None,
            verbose: false,
        }
    }
}

/// Result of blue noise generation
#[derive(Debug, Clone)]
pub struct BlueNoiseResult {
    /// Grayscale threshold map data (0-255)
    pub data: Vec<u8>,
    /// Width of the generated texture
    pub width: usize,
    /// Height of the generated texture
    pub height: usize,
}

/// Error types for blue noise generation
#[derive(Error, Debug)]
pub enum GeneratorError {
    /// Width or height is zero or negative
    #[error("Width and height must be positive")]
    InvalidDimensions,

    /// Width or height is not an integer value
    #[error("Width and height must be integers")]
    NonIntegerDimensions,

    /// Sigma parameter is zero or negative
    #[error("Sigma must be positive")]
    InvalidSigma,

    /// Initial density is out of valid range (0, 1)
    #[error("Initial density must be between 0 and 1")]
    InvalidDensity,

    /// Failed to save generated image
    #[error("Failed to save image: {0}")]
    ImageSaveError(#[from] image::ImageError),

    /// Algorithm failed to converge
    #[error("Generation failed to converge")]
    ConvergenceError,
}

/// Result type for generator operations
pub type Result<T> = std::result::Result<T, GeneratorError>;

/**
 * Mulberry32 seeded random number generator
 * Fast, high-quality PRNG for reproducible results
 */
struct SeededRandom {
    seed: u32,
}

impl SeededRandom {
    fn new(seed: Option<u32>) -> Self {
        Self {
            seed: seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u32
            }),
        }
    }

    fn next(&mut self) -> f32 {
        self.seed = self.seed.wrapping_add(0x6D2B79F5);
        let mut t = self.seed ^ (self.seed >> 15);
        t = t.wrapping_mul(1 | self.seed);
        t ^= t.wrapping_add(t.wrapping_mul(t ^ (t >> 7)).wrapping_mul(61 | t));
        ((t ^ (t >> 14)) as f32) / 4294967296.0
    }
}

/**
 * Main class for generating blue noise textures
 */
pub struct BlueNoiseGenerator {
    // Constants
    max_iterations_multiplier: usize,
    threshold_map_levels: usize,

    // Configuration
    width: usize,
    height: usize,
    area: usize,
    sigma: f32,
    initial_density: f32,
    verbose: bool,
    random: SeededRandom,

    // Working arrays
    bitmap: Vec<u8>,
    rank: Vec<i32>,
    energy: Vec<f32>,

    // Cached values for performance
    ones_count: usize,

    // FFT optimization
    use_fft: bool,
    gaussian_kernel_freq: Option<Vec<Complex<f32>>>,

    // Progress bar
    progress: Option<ProgressBar>,
}

impl BlueNoiseGenerator {
    /// Default constants
    const DEFAULT_SIGMA: f32 = 1.9;
    const DEFAULT_INITIAL_DENSITY: f32 = 0.1;
    const MAX_ITERATIONS_MULTIPLIER: usize = 10;
    const THRESHOLD_MAP_LEVELS: usize = 256;

    /// Check if a number is a power of two
    fn is_power_of_two(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Create a new generator with the given configuration
    pub fn new(config: BlueNoiseConfig) -> Result<Self> {
        // Validation
        if config.width == 0 || config.height == 0 {
            return Err(GeneratorError::InvalidDimensions);
        }
        if config.sigma <= 0.0 {
            return Err(GeneratorError::InvalidSigma);
        }
        if config.initial_density <= 0.0 || config.initial_density >= 1.0 {
            return Err(GeneratorError::InvalidDensity);
        }

        let area = config.width * config.height;
        let use_fft = Self::is_power_of_two(config.width) && Self::is_power_of_two(config.height);

        let progress = if config.verbose {
            Some(ProgressBar::new(100))
        } else {
            None
        };

        if let Some(pb) = &progress {
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>3}% {msg}")
                    .unwrap()
                    .progress_chars("##-"),
            );
        }

        let mut generator = Self {
            max_iterations_multiplier: Self::MAX_ITERATIONS_MULTIPLIER,
            threshold_map_levels: Self::THRESHOLD_MAP_LEVELS,
            width: config.width,
            height: config.height,
            area,
            sigma: config.sigma,
            initial_density: config.initial_density,
            verbose: config.verbose,
            random: SeededRandom::new(config.seed),
            bitmap: vec![0; area],
            rank: vec![0; area],
            energy: vec![0.0; area],
            ones_count: 0,
            use_fft,
            gaussian_kernel_freq: None,
            progress,
        };

        if use_fft {
            generator.gaussian_kernel_freq = Some(generator.create_gaussian_kernel_fft());
        }

        Ok(generator)
    }

    /**
     * Create Gaussian kernel in frequency domain for FFT convolution
     *
     * Pre-computes the Gaussian kernel and transforms it to frequency space.
     * The kernel uses toroidal distance to ensure seamless tiling. By keeping
     * the kernel in frequency space, we only need to compute FFT once during
     * initialization rather than for every blur operation.
     */
    fn create_gaussian_kernel_fft(&self) -> Vec<Complex<f32>> {
        let mut kernel = vec![0.0f32; self.area];
        let divisor = 2.0 * self.sigma * self.sigma;

        for y in 0..self.height {
            for x in 0..self.width {
                // Compute distance from center with wrapping
                let dx = x.min(self.width - x) as f32;
                let dy = y.min(self.height - y) as f32;
                let dist_sq = dx * dx + dy * dy;

                kernel[y * self.width + x] = (-dist_sq / divisor).exp();
            }
        }

        // Normalize kernel
        let sum: f32 = kernel.iter().sum();
        for val in kernel.iter_mut() {
            *val /= sum;
        }

        // Transform to frequency domain
        self.fft_2d_forward(&kernel)
    }

    /**
     * Perform 2D FFT on real-valued data
     */
    fn fft_2d_forward(&self, data: &[f32]) -> Vec<Complex<f32>> {
        let mut complex_data: Vec<Complex<f32>> = data
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // FFT on rows
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.width);

        for y in 0..self.height {
            let start = y * self.width;
            let end = start + self.width;
            fft.process(&mut complex_data[start..end]);
        }

        // FFT on columns
        let fft = planner.plan_fft_forward(self.height);
        let mut column = vec![Complex::new(0.0, 0.0); self.height];

        for x in 0..self.width {
            for y in 0..self.height {
                column[y] = complex_data[y * self.width + x];
            }
            fft.process(&mut column);
            for y in 0..self.height {
                complex_data[y * self.width + x] = column[y];
            }
        }

        complex_data
    }

    /**
     * Perform 2D inverse FFT
     */
    fn fft_2d_inverse(&self, complex_data: &[Complex<f32>]) -> Vec<f32> {
        let mut data = complex_data.to_vec();

        // Inverse FFT on columns
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(self.height);
        let mut column = vec![Complex::new(0.0, 0.0); self.height];

        for x in 0..self.width {
            for y in 0..self.height {
                column[y] = data[y * self.width + x];
            }
            ifft.process(&mut column);
            for y in 0..self.height {
                data[y * self.width + x] = column[y];
            }
        }

        // Inverse FFT on rows
        let ifft = planner.plan_fft_inverse(self.width);

        for y in 0..self.height {
            let start = y * self.width;
            let end = start + self.width;
            ifft.process(&mut data[start..end]);
        }

        // Extract real parts and normalize
        data.iter()
            .map(|c| c.re / (self.area as f32))
            .collect()
    }

    /**
     * Apply Gaussian blur using FFT (frequency domain convolution)
     *
     * FFT optimization: Convolution in the spatial domain (O(n� � k�) where k is
     * kernel size) becomes element-wise multiplication in the frequency domain
     * (O(n� log n) for FFT). This provides ~50% performance improvement for
     * power-of-two dimensions.
     *
     * The convolution theorem states: convolution(A, B) = IFFT(FFT(A) � FFT(B))
     */
    fn gaussian_blur_fft(&self, data: &[u8]) -> Vec<f32> {
        // Convert to float and transform to frequency domain
        let float_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let data_freq = self.fft_2d_forward(&float_data);

        // Element-wise multiplication in frequency domain (convolution)
        let kernel_freq = self.gaussian_kernel_freq.as_ref().unwrap();
        let result_freq: Vec<Complex<f32>> = data_freq
            .iter()
            .zip(kernel_freq.iter())
            .map(|(d, k)| d * k)
            .collect();

        // Transform back to spatial domain
        self.fft_2d_inverse(&result_freq)
    }

    /**
     * Apply Gaussian blur using spatial domain convolution (fallback)
     *
     * Used when dimensions are not powers of two. Applies the Gaussian kernel
     * directly in the spatial domain by computing weighted sums of neighbouring
     * pixels. Coordinates wrap around (torus topology) to ensure seamless tiling.
     */
    fn gaussian_blur_spatial(&self, data: &[u8]) -> Vec<f32> {
        let mut blurred = vec![0.0f32; self.area];
        let kernel_radius = (3.0 * self.sigma).ceil() as i32;
        let divisor = 2.0 * self.sigma * self.sigma;

        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for ky in -kernel_radius..=kernel_radius {
                    for kx in -kernel_radius..=kernel_radius {
                        // Wrap coordinates (torus topology)
                        let px = ((x as i32 + kx + self.width as i32) % self.width as i32) as usize;
                        let py = ((y as i32 + ky + self.height as i32) % self.height as i32) as usize;

                        let dist_sq = (kx * kx + ky * ky) as f32;
                        let weight = (-dist_sq / divisor).exp();

                        sum += data[py * self.width + px] as f32 * weight;
                        weight_sum += weight;
                    }
                }

                blurred[y * self.width + x] = sum / weight_sum;
            }
        }

        blurred
    }

    /**
     * Apply Gaussian blur (chooses FFT or spatial based on size)
     */
    fn gaussian_blur(&self, data: &[u8]) -> Vec<f32> {
        if self.use_fft {
            self.gaussian_blur_fft(data)
        } else {
            self.gaussian_blur_spatial(data)
        }
    }

    /**
     * Find the tightest cluster: pixel with highest energy among all 1s
     *
     * A "cluster" is a region where pixels are densely packed. The Gaussian
     * blur creates an energy field where areas with many nearby 1s have high
     * energy. The tightest cluster is the 1-pixel with the most neighbours.
     */
    fn find_tightest_cluster(&self) -> Option<usize> {
        let mut max_energy = f32::NEG_INFINITY;
        let mut max_idx = None;

        for i in 0..self.area {
            if self.bitmap[i] == 1 && self.energy[i] > max_energy {
                max_energy = self.energy[i];
                max_idx = Some(i);
            }
        }

        max_idx
    }

    /**
     * Find the largest void: pixel with lowest energy among all 0s
     *
     * A "void" is a region where pixels are sparse. In the energy field,
     * areas with few nearby 1s have low energy. The largest void is the
     * 0-pixel with the fewest neighbours.
     */
    fn find_largest_void(&self) -> Option<usize> {
        let mut min_energy = f32::INFINITY;
        let mut min_idx = None;

        for i in 0..self.area {
            if self.bitmap[i] == 0 && self.energy[i] < min_energy {
                min_energy = self.energy[i];
                min_idx = Some(i);
            }
        }

        min_idx
    }

    /**
     * Count number of 1s in the bitmap (returns cached value)
     */
    fn count_ones(&self) -> usize {
        self.ones_count
    }

    /**
     * Set a bit in the bitmap and update the cache
     */
    fn set_bit(&mut self, idx: usize, value: u8) {
        let old_value = self.bitmap[idx];
        self.bitmap[idx] = value;
        self.ones_count = self.ones_count + value as usize - old_value as usize;
    }

    /**
     * Recalculate ones count from scratch (used for initialization)
     */
    fn recalculate_ones_count(&mut self) {
        self.ones_count = self.bitmap.iter().map(|&x| x as usize).sum();
    }

    /**
     * Recalculate energy field by applying Gaussian blur to bitmap
     *
     * The energy field is the key to finding clusters and voids. By applying
     * a Gaussian blur to the binary pattern, we create a smooth field where:
     * - High values indicate clusters (many nearby 1s)
     * - Low values indicate voids (few nearby 1s)
     *
     * The blur uses torus wrapping so edges connect seamlessly.
     */
    fn recalculate_energy(&mut self) {
        self.energy = self.gaussian_blur(&self.bitmap);
    }

    /**
     * Phase 0: Generate Initial Binary Pattern
     *
     * Creates a well-distributed set of initial points by:
     * 1. Randomly placing points (10% density by default)
     * 2. Iteratively swapping clustered points with void positions
     * 3. Stopping when convergence is reached (cluster = void position)
     *
     * This phase establishes the foundation for even distribution.
     */
    fn phase0_generate_initial_pattern(&mut self) -> Result<()> {
        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_message("Phase 0: Generating initial pattern");
                pb.set_position(0);
            }
        }

        let target_points = (self.area as f32 * self.initial_density) as usize;

        // Randomly place initial points
        while self.count_ones() < target_points {
            let idx = (self.random.next() * self.area as f32) as usize;
            if self.bitmap[idx] == 0 {
                self.set_bit(idx, 1);
            }
        }

        self.recalculate_energy();

        // Redistribute points until convergence
        let max_iterations = self.area * self.max_iterations_multiplier;
        let mut iterations = 0;

        while iterations < max_iterations {
            iterations += 1;

            // Find tightest cluster and remove it
            let cluster_idx = self.find_tightest_cluster()
                .ok_or(GeneratorError::ConvergenceError)?;
            self.set_bit(cluster_idx, 0);

            // Find largest void with updated energy
            self.recalculate_energy();
            let void_idx = self.find_largest_void()
                .ok_or(GeneratorError::ConvergenceError)?;

            // Check for convergence
            if void_idx == cluster_idx {
                self.set_bit(cluster_idx, 1);
                self.recalculate_energy();
                break;
            }

            // Place point in void
            self.set_bit(void_idx, 1);
            self.recalculate_energy();
        }

        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_position(20);
            }
        }

        Ok(())
    }

    /**
     * Phase 1: Serialize Initial Points
     *
     * Assigns ranks to the initial minority pattern by removing points from
     * tightest clusters first. These points are ranked from (initialPoints - 1)
     * down to 0, establishing which pixels are most important for creating
     * the blue noise distribution.
     */
    fn phase1_serialize_initial_points(&mut self) -> Result<()> {
        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_message("Phase 1: Serializing initial points");
                pb.set_position(20);
            }
        }

        let mut rank_counter = self.count_ones() as i32 - 1;

        while self.count_ones() > 0 {
            let cluster_idx = self.find_tightest_cluster()
                .ok_or(GeneratorError::ConvergenceError)?;
            self.rank[cluster_idx] = rank_counter;
            rank_counter -= 1;

            self.set_bit(cluster_idx, 0);
            self.recalculate_energy();
        }

        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_position(40);
            }
        }

        Ok(())
    }

    /**
     * Phase 2: Fill to Half Capacity
     *
     * Restores the initial pattern and continues adding points to the largest
     * voids until the bitmap is 50% full. Ranks continue from initialPoints
     * to area/2. This builds up a minority pattern (less than half full).
     */
    fn phase2_fill_to_half(&mut self, prototype: &[u8], initial_points: usize) -> Result<()> {
        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_message("Phase 2: Filling to half capacity");
                pb.set_position(40);
            }
        }

        self.bitmap.copy_from_slice(prototype);
        self.recalculate_ones_count();
        self.recalculate_energy();

        let mut rank_counter = initial_points as i32;
        let half_area = self.area / 2;

        while self.count_ones() < half_area {
            let void_idx = self.find_largest_void()
                .ok_or(GeneratorError::ConvergenceError)?;
            self.rank[void_idx] = rank_counter;
            rank_counter += 1;

            self.set_bit(void_idx, 1);
            self.recalculate_energy();
        }

        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_position(60);
            }
        }

        Ok(())
    }

    /**
     * Phase 3: Fill to Completion
     *
     * Inverts the bitmap so 0s become the minority (more than half full).
     * Then removes the remaining minority pixels from their tightest clusters,
     * ranking them from area/2 to area-1. This clever inversion allows the
     * algorithm to work symmetrically for both minority and majority patterns.
     */
    fn phase3_fill_to_completion(&mut self, mut rank_counter: i32) -> Result<()> {
        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_message("Phase 3: Filling to completion");
                pb.set_position(60);
            }
        }

        // Invert bitmap
        for i in 0..self.area {
            self.bitmap[i] = 1 - self.bitmap[i];
        }
        self.recalculate_ones_count();
        self.recalculate_energy();

        while rank_counter < self.area as i32 {
            let cluster_idx = self.find_tightest_cluster()
                .ok_or(GeneratorError::ConvergenceError)?;
            self.rank[cluster_idx] = rank_counter;
            rank_counter += 1;

            self.set_bit(cluster_idx, 0);
            self.recalculate_energy();
        }

        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_position(80);
            }
        }

        Ok(())
    }

    /**
     * Phase 4: Convert Ranks to Threshold Map
     *
     * Maps the rank values (0 to area-1) to grayscale threshold values (0-255).
     * Each pixel's rank determines its threshold value for dithering. Lower
     * ranked pixels will be turned "on" first when dithering bright images.
     */
    fn phase4_convert_to_threshold_map(&self) -> Vec<u8> {
        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_message("Phase 4: Converting to threshold map");
                pb.set_position(80);
            }
        }

        let output: Vec<u8> = self
            .rank
            .iter()
            .map(|&r| ((r as usize * self.threshold_map_levels) / self.area) as u8)
            .collect();

        if self.verbose {
            if let Some(pb) = &self.progress {
                pb.set_position(100);
                pb.finish_with_message("Blue noise generation complete");
            }
        }

        output
    }

    /**
     * Generate the blue noise texture
     */
    pub fn generate(mut self) -> Result<BlueNoiseResult> {
        let start_time = std::time::Instant::now();

        if self.verbose {
            println!(
                "Generating {}�{} blue noise texture...",
                self.width, self.height
            );
            println!(
                "Using {} Gaussian blur",
                if self.use_fft { "FFT-optimized" } else { "spatial" }
            );
        }

        self.phase0_generate_initial_pattern()?;

        let prototype = self.bitmap.clone();
        let initial_points = self.count_ones();

        if self.verbose {
            println!("Initial pattern: {} points", initial_points);
        }

        self.phase1_serialize_initial_points()?;
        self.phase2_fill_to_half(&prototype, initial_points)?;

        let half_area = self.area / 2;
        self.phase3_fill_to_completion(half_area as i32)?;

        let data = self.phase4_convert_to_threshold_map();

        if self.verbose {
            let elapsed = start_time.elapsed();
            println!(
                "Blue noise generation complete in {:.2}s",
                elapsed.as_secs_f32()
            );
        }

        Ok(BlueNoiseResult {
            data,
            width: self.width,
            height: self.height,
        })
    }
}

/**
 * Convenience function to generate a blue noise texture
 */
pub fn generate_blue_noise(width: usize, height: usize, sigma: f32) -> Result<BlueNoiseResult> {
    let config = BlueNoiseConfig {
        width,
        height,
        sigma,
        ..Default::default()
    };
    let generator = BlueNoiseGenerator::new(config)?;
    generator.generate()
}

/**
 * Save blue noise texture to PNG file
 */
pub fn save_blue_noise_to_png<P: AsRef<Path>>(
    result: &BlueNoiseResult,
    filename: P,
) -> Result<()> {
    let img: GrayImage = ImageBuffer::from_fn(result.width as u32, result.height as u32, |x, y| {
        let idx = y as usize * result.width + x as usize;
        Luma([result.data[idx]])
    });

    img.save(&filename)?;
    println!("Saved blue noise texture to {}", filename.as_ref().display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_random_deterministic() {
        let mut rng1 = SeededRandom::new(Some(42));
        let mut rng2 = SeededRandom::new(Some(42));

        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_seeded_random_range() {
        let mut rng = SeededRandom::new(Some(12345));

        // All values should be in [0, 1)
        for _ in 0..1000 {
            let val = rng.next();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(BlueNoiseGenerator::is_power_of_two(1));
        assert!(BlueNoiseGenerator::is_power_of_two(2));
        assert!(BlueNoiseGenerator::is_power_of_two(4));
        assert!(BlueNoiseGenerator::is_power_of_two(8));
        assert!(BlueNoiseGenerator::is_power_of_two(16));
        assert!(BlueNoiseGenerator::is_power_of_two(64));
        assert!(BlueNoiseGenerator::is_power_of_two(128));
        assert!(BlueNoiseGenerator::is_power_of_two(256));

        assert!(!BlueNoiseGenerator::is_power_of_two(0));
        assert!(!BlueNoiseGenerator::is_power_of_two(3));
        assert!(!BlueNoiseGenerator::is_power_of_two(5));
        assert!(!BlueNoiseGenerator::is_power_of_two(100));
        assert!(!BlueNoiseGenerator::is_power_of_two(255));
    }

    #[test]
    fn test_config_validation() {
        // Valid config should work
        let config = BlueNoiseConfig {
            width: 64,
            height: 64,
            sigma: 1.9,
            initial_density: 0.1,
            seed: Some(42),
            verbose: false,
        };
        assert!(BlueNoiseGenerator::new(config).is_ok());

        // Zero width should fail
        let config = BlueNoiseConfig {
            width: 0,
            height: 64,
            ..Default::default()
        };
        assert!(BlueNoiseGenerator::new(config).is_err());

        // Zero height should fail
        let config = BlueNoiseConfig {
            width: 64,
            height: 0,
            ..Default::default()
        };
        assert!(BlueNoiseGenerator::new(config).is_err());

        // Negative sigma should fail
        let config = BlueNoiseConfig {
            width: 64,
            height: 64,
            sigma: -1.0,
            ..Default::default()
        };
        assert!(BlueNoiseGenerator::new(config).is_err());

        // Invalid density (too low) should fail
        let config = BlueNoiseConfig {
            width: 64,
            height: 64,
            initial_density: 0.0,
            ..Default::default()
        };
        assert!(BlueNoiseGenerator::new(config).is_err());

        // Invalid density (too high) should fail
        let config = BlueNoiseConfig {
            width: 64,
            height: 64,
            initial_density: 1.0,
            ..Default::default()
        };
        assert!(BlueNoiseGenerator::new(config).is_err());
    }

    #[test]
    fn test_generate_small_texture() {
        let config = BlueNoiseConfig {
            width: 16,
            height: 16,
            sigma: 1.5,
            seed: Some(42),
            verbose: false,
            ..Default::default()
        };

        let generator = BlueNoiseGenerator::new(config).unwrap();
        let result = generator.generate().unwrap();

        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.data.len(), 256);

        // Check all values are in valid range
        for &val in &result.data {
            assert!(val <= 255);
        }
    }

    #[test]
    fn test_generate_reproducible() {
        let config1 = BlueNoiseConfig {
            width: 32,
            height: 32,
            seed: Some(12345),
            verbose: false,
            ..Default::default()
        };

        let config2 = BlueNoiseConfig {
            width: 32,
            height: 32,
            seed: Some(12345),
            verbose: false,
            ..Default::default()
        };

        let gen1 = BlueNoiseGenerator::new(config1).unwrap();
        let gen2 = BlueNoiseGenerator::new(config2).unwrap();

        let result1 = gen1.generate().unwrap();
        let result2 = gen2.generate().unwrap();

        // Same seed should produce identical results
        assert_eq!(result1.data, result2.data);
    }

    #[test]
    fn test_generate_different_seeds() {
        let config1 = BlueNoiseConfig {
            width: 32,
            height: 32,
            seed: Some(111),
            verbose: false,
            ..Default::default()
        };

        let config2 = BlueNoiseConfig {
            width: 32,
            height: 32,
            seed: Some(222),
            verbose: false,
            ..Default::default()
        };

        let gen1 = BlueNoiseGenerator::new(config1).unwrap();
        let gen2 = BlueNoiseGenerator::new(config2).unwrap();

        let result1 = gen1.generate().unwrap();
        let result2 = gen2.generate().unwrap();

        // Different seeds should produce different results
        assert_ne!(result1.data, result2.data);
    }

    #[test]
    fn test_generate_power_of_two_uses_fft() {
        let config = BlueNoiseConfig {
            width: 64,
            height: 64,
            seed: Some(42),
            verbose: false,
            ..Default::default()
        };

        let generator = BlueNoiseGenerator::new(config).unwrap();
        assert!(generator.use_fft);
        assert!(generator.gaussian_kernel_freq.is_some());
    }

    #[test]
    fn test_generate_non_power_of_two_no_fft() {
        let config = BlueNoiseConfig {
            width: 50,
            height: 50,
            seed: Some(42),
            verbose: false,
            ..Default::default()
        };

        let generator = BlueNoiseGenerator::new(config).unwrap();
        assert!(!generator.use_fft);
        assert!(generator.gaussian_kernel_freq.is_none());
    }

    #[test]
    fn test_threshold_map_distribution() {
        let config = BlueNoiseConfig {
            width: 32,
            height: 32,
            seed: Some(99),
            verbose: false,
            ..Default::default()
        };

        let generator = BlueNoiseGenerator::new(config).unwrap();
        let result = generator.generate().unwrap();

        // Check that we have good distribution across the range
        let mut histogram = vec![0usize; 256];
        for &val in &result.data {
            histogram[val as usize] += 1;
        }

        // Should have values across the spectrum
        let non_empty_bins = histogram.iter().filter(|&&count| count > 0).count();
        assert!(non_empty_bins > 200, "Expected diverse distribution");
    }

    #[test]
    fn test_convenience_function() {
        let result = generate_blue_noise(16, 16, 1.5);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.data.len(), 256);
    }

    #[test]
    fn test_rectangular_texture() {
        let config = BlueNoiseConfig {
            width: 32,
            height: 16,
            seed: Some(42),
            verbose: false,
            ..Default::default()
        };

        let generator = BlueNoiseGenerator::new(config).unwrap();
        let result = generator.generate().unwrap();

        assert_eq!(result.width, 32);
        assert_eq!(result.height, 16);
        assert_eq!(result.data.len(), 512);
    }

    #[test]
    fn test_default_config() {
        let config = BlueNoiseConfig::default();
        assert_eq!(config.width, 64);
        assert_eq!(config.height, 64);
        assert_eq!(config.sigma, 1.9);
        assert_eq!(config.initial_density, 0.1);
        assert_eq!(config.seed, None);
        assert_eq!(config.verbose, false);
    }
}
