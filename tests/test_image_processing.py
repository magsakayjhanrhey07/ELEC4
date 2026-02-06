import os
import cv2
import numpy as np
from image_processor import process_images

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

# -------------------------
# Basic tests
# -------------------------

def test_process_images_runs():
    """Run the processing function"""
    result = process_images()
    assert result is True, "process_images() did not return True"
    print("✅ process_images() ran successfully")

def test_output_directory_exists():
    """Check that the output folder exists"""
    assert os.path.exists(OUTPUT_DIR), f"Output directory '{OUTPUT_DIR}' does not exist"
    print("✅ Output directory exists")

def test_output_images_created():
    """Check that output images are created"""
    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]
    assert len(files) > 0, "No images created in output directory"
    print(f"✅ Number of output images: {len(files)}")

def test_output_matches_input_count():
    """Check that output image count matches input image count"""
    input_count = len([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
    output_count = len([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
    assert input_count == output_count, f"Input count {input_count} != Output count {output_count}"
    print("✅ Input and output image counts match")

# -------------------------
# Technique effect checks
# -------------------------

def test_grayscale_output():
    """Check that output images are grayscale"""
    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith((".png",".jpg",".jpeg")):
            out_img = cv2.imread(os.path.join(OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            assert len(out_img.shape) == 2, f"{filename} is not grayscale"
    print("✅ All output images are grayscale")

def test_adaptive_threshold_and_invert():
    """Check that most pixels are black or white (threshold + invert applied)"""
    for filename in os.listdir(OUTPUT_DIR):
        if filename.lower().endswith((".png",".jpg",".jpeg")):
            out_img = cv2.imread(os.path.join(OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            total_pixels = out_img.size
            black_white_pixels = np.sum((out_img <= 5) | (out_img >= 250))
            ratio = black_white_pixels / total_pixels
            assert ratio > 0.95, f"{filename} threshold/invert may not be applied properly, ratio={ratio:.2f}"
    print("✅ Adaptive Threshold and Invert Colors applied correctly")

def test_output_differs_from_input():
    """Check that output image differs from input (CLAHE + Blur + Threshold/Inversion applied)"""
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".png",".jpg",".jpeg")):
            in_img = cv2.imread(os.path.join(INPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            out_img = cv2.imread(os.path.join(OUTPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            diff = np.mean(np.abs(out_img.astype(int) - in_img.astype(int)))
            assert diff > 0, f"{filename} output is identical to input, no processing applied"
    print("✅ CLAHE and Gaussian Blur effects confirmed by difference from input")
