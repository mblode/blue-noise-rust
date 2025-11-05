/**
 * Performance benchmarks for blue-noise
 *
 * Run with:
 *   cargo bench
 *
 * View HTML reports in:
 *   target/criterion/report/index.html
 */

use blue_noise::{
    apply_dithering, BlueNoiseConfig, BlueNoiseGenerator, BlueNoiseTexture, Color, DitherOptions,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{GrayImage, ImageBuffer, Luma};

/// Benchmark blue noise generation for different sizes
fn bench_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation");

    // Test different sizes
    for size in [16, 32, 64].iter() {
        let config = BlueNoiseConfig {
            width: *size,
            height: *size,
            sigma: 1.9,
            seed: Some(42),
            verbose: false,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("power_of_two", size),
            size,
            |b, _| {
                b.iter(|| {
                    let generator = BlueNoiseGenerator::new(config.clone()).unwrap();
                    black_box(generator.generate().unwrap())
                });
            },
        );
    }

    // Test non-power-of-two size (no FFT optimization)
    let config_non_pot = BlueNoiseConfig {
        width: 50,
        height: 50,
        sigma: 1.9,
        seed: Some(42),
        verbose: false,
        ..Default::default()
    };

    group.bench_function("non_power_of_two_50", |b| {
        b.iter(|| {
            let generator = BlueNoiseGenerator::new(config_non_pot.clone()).unwrap();
            black_box(generator.generate().unwrap())
        });
    });

    group.finish();
}

/// Benchmark FFT vs spatial Gaussian blur
fn bench_gaussian_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur");

    // FFT-optimized (power of two)
    let config_fft = BlueNoiseConfig {
        width: 64,
        height: 64,
        sigma: 1.9,
        seed: Some(42),
        verbose: false,
        ..Default::default()
    };

    group.bench_function("fft_64x64", |b| {
        b.iter(|| {
            let generator = BlueNoiseGenerator::new(config_fft.clone()).unwrap();
            black_box(generator.generate().unwrap())
        });
    });

    // Spatial domain (non-power of two)
    let config_spatial = BlueNoiseConfig {
        width: 60,
        height: 60,
        sigma: 1.9,
        seed: Some(42),
        verbose: false,
        ..Default::default()
    };

    group.bench_function("spatial_60x60", |b| {
        b.iter(|| {
            let generator = BlueNoiseGenerator::new(config_spatial.clone()).unwrap();
            black_box(generator.generate().unwrap())
        });
    });

    group.finish();
}

/// Benchmark dithering performance
fn bench_dithering(c: &mut Criterion) {
    let mut group = c.benchmark_group("dithering");

    // Generate a test noise texture
    let noise_config = BlueNoiseConfig {
        width: 64,
        height: 64,
        seed: Some(42),
        verbose: false,
        ..Default::default()
    };
    let generator = BlueNoiseGenerator::new(noise_config).unwrap();
    let noise_result = generator.generate().unwrap();
    let noise_texture =
        BlueNoiseTexture::from_data(noise_result.data, noise_result.width, noise_result.height)
            .unwrap();

    // Create test images of different sizes
    let sizes = [100, 200, 400];

    for size in sizes.iter() {
        // Create a gradient test image
        let mut test_image_data = Vec::new();
        for y in 0..*size {
            for x in 0..*size {
                let value = ((x + y) % 256) as u8;
                test_image_data.push(value);
                test_image_data.push(value);
                test_image_data.push(value);
            }
        }

        let test_image =
            image::RgbImage::from_vec(*size, *size, test_image_data.clone()).unwrap();
        let filename = format!("/tmp/bench-test-{}.png", size);
        test_image.save(&filename).unwrap();

        group.bench_with_input(BenchmarkId::new("dither", size), size, |b, _| {
            let output = format!("/tmp/bench-output-{}.png", size);
            let options = DitherOptions {
                foreground: Color::new(0, 0, 0),
                background: Color::new(255, 255, 255),
                width: None,
                height: None,
                contrast: None,
            };

            b.iter(|| {
                apply_dithering(&filename, &output, &noise_texture, options.clone()).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark different sigma values
fn bench_sigma_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigma_values");

    for sigma in [1.5, 1.9, 2.5].iter() {
        let config = BlueNoiseConfig {
            width: 32,
            height: 32,
            sigma: *sigma,
            seed: Some(42),
            verbose: false,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("sigma", sigma), sigma, |b, _| {
            b.iter(|| {
                let generator = BlueNoiseGenerator::new(config.clone()).unwrap();
                black_box(generator.generate().unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark hex color parsing
fn bench_hex_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("hex_parsing");

    group.bench_function("with_hash", |b| {
        b.iter(|| black_box(Color::from_hex("#FF8800").unwrap()))
    });

    group.bench_function("without_hash", |b| {
        b.iter(|| black_box(Color::from_hex("FF8800").unwrap()))
    });

    group.bench_function("uppercase", |b| {
        b.iter(|| black_box(Color::from_hex("#AABBCC").unwrap()))
    });

    group.bench_function("lowercase", |b| {
        b.iter(|| black_box(Color::from_hex("#aabbcc").unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_generation,
    bench_gaussian_blur,
    bench_dithering,
    bench_sigma_values,
    bench_hex_parsing
);
criterion_main!(benches);
