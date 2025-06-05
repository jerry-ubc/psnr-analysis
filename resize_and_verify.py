# This file is used to pre-process

import cv2
import os
import glob
import shutil
from collections import defaultdict

# Expected resolutions (width x height)
EXPECTED_RESOLUTIONS = {
    '2k': (2048, 1080),  # 2K
    '4k': (3840, 2160),  # 4K
    '8k': (7680, 4320)   # 8K
}

def get_total_pixels(width, height):
    """Calculate total pixels in an image."""
    return width * height

def meets_minimum_resolution(current_w, current_h, target_w, target_h):
    """Check if current resolution meets minimum requirements for target resolution."""
    current_pixels = get_total_pixels(current_w, current_h)
    target_pixels = get_total_pixels(target_w, target_h)
    return current_pixels >= target_pixels

def resize_image(img, target_width, target_height):
    """Resize image to target resolution using Lanczos interpolation."""
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def process_directory(resolution_prefix):
    """Process images for a given resolution prefix (2k, 4k, or 8k)."""
    print(f"\nProcessing {resolution_prefix} images...")
    
    # Setup directories
    input_dir = f"images/{resolution_prefix}"
    original_dir = f"images/{resolution_prefix}-original"
    uniform_dir = f"images/{resolution_prefix}-uniform"
    
    # Create directories if they don't exist
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(uniform_dir, exist_ok=True)
    
    # Get expected resolution
    expected_w, expected_h = EXPECTED_RESOLUTIONS[resolution_prefix]
    print(f"Target resolution: {expected_w}x{expected_h}")
    
    # Get all images from input directory
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                 glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # Statistics
    total = len(image_files)
    processed = 0
    deleted = 0
    errors = 0
    
    print(f"Found {total} images")
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        if i % 10 == 0:
            print(f"Processing image {i}/{total}")
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading {os.path.basename(img_path)}")
                errors += 1
                continue
                
            h, w = img.shape[:2]
            
            # Check if image meets minimum resolution requirements
            if not meets_minimum_resolution(w, h, expected_w, expected_h):
                print(f"Deleting {os.path.basename(img_path)} - too small ({w}x{h})")
                os.remove(img_path)
                deleted += 1
                continue
            
            # Copy to original directory
            original_path = os.path.join(original_dir, os.path.basename(img_path))
            shutil.copy2(img_path, original_path)
            
            # Create uniform version
            uniform_path = os.path.join(uniform_dir, os.path.basename(img_path))
            
            # Resize to exact resolution if needed
            if (w, h) != (expected_w, expected_h):
                resized = resize_image(img, expected_w, expected_h)
                cv2.imwrite(uniform_path, resized)
            else:
                # If already correct size, just copy
                shutil.copy2(img_path, uniform_path)
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
            errors += 1
    
    # Verify results
    print(f"\nResults for {resolution_prefix}:")
    print(f"Total images found: {total}")
    print(f"Successfully processed: {processed}")
    print(f"Deleted (too small): {deleted}")
    print(f"Errors: {errors}")
    
    # Verify uniform directory
    wrong_sizes = defaultdict(int)
    verification_errors = 0
    
    print("\nVerifying uniform directory resolutions...")
    uniform_files = glob.glob(os.path.join(uniform_dir, "*.jpg")) + \
                   glob.glob(os.path.join(uniform_dir, "*.jpeg")) + \
                   glob.glob(os.path.join(uniform_dir, "*.png"))
    
    for img_path in uniform_files:
        img = cv2.imread(img_path)
        if img is None:
            verification_errors += 1
            continue
            
        h, w = img.shape[:2]
        if (w, h) != (expected_w, expected_h):
            wrong_sizes[(w, h)] += 1
    
    if wrong_sizes:
        print(f"\nWarning: Some images in {uniform_dir} have incorrect resolutions:")
        for (w, h), count in sorted(wrong_sizes.items()):
            print(f"  {w}x{h}: {count} images")
    else:
        print(f"\nSuccess: All images in {uniform_dir} verified to be {expected_w}x{expected_h}")
    
    print("-" * 50)

def main():
    # Process each resolution
    for resolution in ['2k', '4k', '8k']:
        process_directory(resolution)

if __name__ == '__main__':
    main() 